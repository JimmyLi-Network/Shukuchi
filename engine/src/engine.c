#include "engine.h"
#include "model_loader.h"
#include "ops.h"
#include "kv_cache.h"
#include "prefetch.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>

struct engine_handle {
    struct engine_config cfg;
    model_handle_t *model;
    struct model_info info;
    kv_cache_t *kv;
    prefetcher_t *prefetch;
    struct streaming_stats stats;
    char *prompt;
};

static int debug_enabled(void) {
    const char *env = getenv("SHUKUCHI_DEBUG");
    if (!env || env[0] == '\0' || strcmp(env, "0") == 0) {
        return 0;
    }
    return 1;
}

static void debug_check(const char *name, const float *x, uint32_t n) {
    if (!x || n == 0) {
        printf("[DEBUG] %s: empty\n", name);
        return;
    }
    float min_v = x[0];
    float max_v = x[0];
    float sum = 0.0f;
    int has_nan = 0;
    int has_inf = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float v = x[i];
        if (isnan(v)) has_nan = 1;
        if (isinf(v)) has_inf = 1;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
    }
    printf("[DEBUG] %s: min=%.4f max=%.4f mean=%.4f nan=%d inf=%d\n",
           name, min_v, max_v, sum / (float)n, has_nan, has_inf);
}

static uint32_t q4k_rows_from_bytes(uint64_t size_bytes, uint32_t k) {
    const uint64_t block_bytes = 144; // sizeof(block_q4_k)
    if (k == 0 || size_bytes == 0) {
        return 0;
    }
    uint64_t blocks = size_bytes / block_bytes;
    uint64_t values = blocks * 256;
    return (uint32_t)(values / k);
}

static int matmul_quant(uint32_t dtype, const void *a, const float *b, float *c, uint32_t m, uint32_t k) {
    if (dtype == 12) {
        return op_matmul_q4_k(NULL, a, b, c, m, k);
    }
    if (dtype == 13) {
        return op_matmul_q5_k(NULL, a, b, c, m, k);
    }
    if (dtype == 14) {
        return op_matmul_q6_k(NULL, a, b, c, m, k);
    }
    return -1;
}

static int forward_layer_view(engine_handle_t *h, const struct layer_view *lv,
                              uint32_t layer_id, uint32_t pos, float *hidden) {
    if (!lv) {
        return -1;
    }

    uint32_t n_embd = h->info.n_embd;
    uint32_t n_heads = h->info.n_heads;
    uint32_t n_kv_heads = h->info.n_kv_heads;
    uint32_t head_dim = h->info.head_dim;
    float rope_theta = h->info.rope_theta > 0.0f ? h->info.rope_theta : 10000.0f;

    if (debug_enabled()) {
        printf("[DEBUG] dtypes: q=%u k=%u v=%u o=%u gate=%u up=%u down=%u\n",
               lv->attn_q_dtype, lv->attn_k_dtype, lv->attn_v_dtype, lv->attn_o_dtype,
               lv->ffn_gate_dtype, lv->ffn_up_dtype, lv->ffn_down_dtype);
    }

    float *normed = (float *)malloc((size_t)n_embd * sizeof(float));
    float *q = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    float *k = (float *)malloc((size_t)n_kv_heads * head_dim * sizeof(float));
    float *v = (float *)malloc((size_t)n_kv_heads * head_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    float *attn_proj = (float *)malloc((size_t)n_embd * sizeof(float));
    if (!normed || !q || !k || !v || !attn_out || !attn_proj) {
        free(normed); free(q); free(k); free(v); free(attn_out); free(attn_proj);
        return -1;
    }

    int dbg = (debug_enabled() && layer_id == 0 && pos == 0);
    // Attention block
    if (op_rmsnorm(NULL, hidden, (const float *)lv->attn_norm, normed, 1, n_embd) != 0) {
        goto fail;
    }
    if (dbg) debug_check("attn_norm", normed, n_embd);
    if (matmul_quant(lv->attn_q_dtype, lv->attn_q, normed, q, n_heads * head_dim, n_embd) != 0) {
        goto fail;
    }
    if (dbg) debug_check("Q", q, n_heads * head_dim);
    if (matmul_quant(lv->attn_k_dtype, lv->attn_k, normed, k, n_kv_heads * head_dim, n_embd) != 0) {
        goto fail;
    }
    if (dbg) debug_check("K", k, n_kv_heads * head_dim);
    if (matmul_quant(lv->attn_v_dtype, lv->attn_v, normed, v, n_kv_heads * head_dim, n_embd) != 0) {
        goto fail;
    }
    if (dbg) debug_check("V", v, n_kv_heads * head_dim);

    if (op_rope(NULL, q, n_heads, head_dim, pos, rope_theta) != 0) {
        goto fail;
    }
    if (op_rope(NULL, k, n_kv_heads, head_dim, pos, rope_theta) != 0) {
        goto fail;
    }
    if (dbg) debug_check("Q_rope", q, n_heads * head_dim);
    if (dbg) debug_check("K_rope", k, n_kv_heads * head_dim);

    if (kv_cache_append(h->kv, layer_id, pos, k, v) != 0) {
        goto fail;
    }

    uint32_t seq_len = pos + 1;
    uint32_t kv_dim = n_kv_heads * head_dim;
    float *k_cache = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *v_cache = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    if (!k_cache || !v_cache) {
        free(k_cache); free(v_cache);
        goto fail;
    }
    if (kv_cache_read_range(h->kv, layer_id, 0, seq_len, k_cache, v_cache) != 0) {
        free(k_cache); free(v_cache);
        goto fail;
    }
    if (dbg) debug_check("K_cache", k_cache, seq_len * kv_dim);
    if (dbg) debug_check("V_cache", v_cache, seq_len * kv_dim);
    float scale = 1.0f / sqrtf((float)head_dim);
    if (op_attention(NULL, q, k_cache, v_cache, attn_out, n_heads, n_kv_heads, head_dim, seq_len, scale, NULL) != 0) {
        free(k_cache); free(v_cache);
        goto fail;
    }
    if (dbg) debug_check("attn_out", attn_out, n_heads * head_dim);
    free(k_cache);
    free(v_cache);

    if (matmul_quant(lv->attn_o_dtype, lv->attn_o, attn_out, attn_proj, n_embd, n_heads * head_dim) != 0) {
        goto fail;
    }
    if (dbg) debug_check("attn_proj", attn_proj, n_embd);
    for (uint32_t i = 0; i < n_embd; ++i) {
        hidden[i] += attn_proj[i];
    }
    if (dbg) debug_check("hidden_after_attn", hidden, n_embd);

    // MLP block
    uint32_t d_ff = q4k_rows_from_bytes(lv->ffn_gate_size, n_embd);
    float *mlp_out = (float *)malloc((size_t)n_embd * sizeof(float));
    if (!mlp_out || d_ff == 0) {
        free(mlp_out);
        goto fail;
    }
    if (op_rmsnorm(NULL, hidden, (const float *)lv->ffn_norm, normed, 1, n_embd) != 0) {
        free(mlp_out);
        goto fail;
    }
    if (dbg) debug_check("ffn_norm", normed, n_embd);
    if (lv->ffn_gate_dtype == 12 && lv->ffn_up_dtype == 12 && lv->ffn_down_dtype == 12) {
        if (op_mlp_swiglu(NULL, normed, lv->ffn_gate, lv->ffn_up, lv->ffn_down, mlp_out, 1, n_embd, d_ff) != 0) {
            free(mlp_out);
            goto fail;
        }
    } else {
        float *gate = (float *)malloc((size_t)d_ff * sizeof(float));
        float *up = (float *)malloc((size_t)d_ff * sizeof(float));
        float *hidden_mlp = (float *)malloc((size_t)d_ff * sizeof(float));
        if (!gate || !up || !hidden_mlp) {
            free(gate); free(up); free(hidden_mlp);
            free(mlp_out);
            goto fail;
        }
        if (matmul_quant(lv->ffn_gate_dtype, lv->ffn_gate, normed, gate, d_ff, n_embd) != 0) {
            free(gate); free(up); free(hidden_mlp); free(mlp_out);
            goto fail;
        }
        if (matmul_quant(lv->ffn_up_dtype, lv->ffn_up, normed, up, d_ff, n_embd) != 0) {
            free(gate); free(up); free(hidden_mlp); free(mlp_out);
            goto fail;
        }
        for (uint32_t i = 0; i < d_ff; ++i) {
            float g = gate[i];
            float sig = 1.0f / (1.0f + expf(-g));
            float silu = g * sig;
            hidden_mlp[i] = silu * up[i];
        }
        if (matmul_quant(lv->ffn_down_dtype, lv->ffn_down, hidden_mlp, mlp_out, n_embd, d_ff) != 0) {
            free(gate); free(up); free(hidden_mlp); free(mlp_out);
            goto fail;
        }
        free(gate);
        free(up);
        free(hidden_mlp);
    }
    if (dbg) debug_check("mlp_out", mlp_out, n_embd);
    for (uint32_t i = 0; i < n_embd; ++i) {
        hidden[i] += mlp_out[i];
    }
    if (dbg) debug_check("hidden_after_mlp", hidden, n_embd);
    free(mlp_out);

    free(normed); free(q); free(k); free(v); free(attn_out); free(attn_proj);
    return 0;
fail:
    free(normed); free(q); free(k); free(v); free(attn_out); free(attn_proj);
    return -1;
}

static int forward_layer(engine_handle_t *h, uint32_t layer_id, uint32_t pos, float *hidden) {
    const struct layer_view *lv = NULL;
    if (model_get_layer_view(h->model, layer_id, &lv) != 0 || !lv) {
        return -1;
    }
    return forward_layer_view(h, lv, layer_id, pos, hidden);
}

engine_handle_t *engine_open(const char *model_path, const struct engine_config *cfg) {
    struct engine_handle *h = (struct engine_handle *)calloc(1, sizeof(*h));
    if (!h) {
        return NULL;
    }
    if (cfg) {
        h->cfg = *cfg;
    }
    struct model_config mcfg;
    mcfg.prefer_gguf = 1;
    mcfg.use_mmap = cfg ? cfg->use_mmap : 0;
    h->model = model_open(model_path, &mcfg);
    if (!h->model) {
        free(h);
        return NULL;
    }
    if (model_get_info(h->model, &h->info) != 0) {
        model_close(h->model);
        free(h);
        return NULL;
    }
    struct kv_cache_config kcfg;
    kcfg.n_layers = h->info.n_layers;
    kcfg.n_kv_heads = h->info.n_kv_heads;
    kcfg.head_dim = h->info.head_dim;
    kcfg.block_size = h->cfg.kv_block_size ? h->cfg.kv_block_size : 32;
    kcfg.max_seq_len = 2048;
    kcfg.quant = KV_Q8_0;
    h->kv = kv_cache_create(&kcfg);
    if (!h->kv) {
        model_close(h->model);
        free(h);
        return NULL;
    }
    memset(&h->stats, 0, sizeof(h->stats));
    struct prefetcher_config pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    pcfg.depth = h->cfg.prefetch_depth ? h->cfg.prefetch_depth : 2;
    pcfg.model = h->model;
    pcfg.buffer_size = 0;
    pcfg.stats = &h->stats;
    h->prefetch = prefetcher_create(&pcfg);
    if (h->prefetch) {
        prefetcher_start(h->prefetch);
    }
    return h;
}

int engine_set_prompt(engine_handle_t *h, const char *prompt) {
    if (!h) {
        return -1;
    }
    if (h->prompt) {
        free(h->prompt);
        h->prompt = NULL;
    }
    if (prompt) {
        size_t len = strlen(prompt);
        h->prompt = (char *)malloc(len + 1);
        if (!h->prompt) {
            return -1;
        }
        memcpy(h->prompt, prompt, len + 1);
    }
    return 0;
}

int engine_generate(engine_handle_t *h, uint32_t max_tokens) {
    if (!h || max_tokens == 0) {
        return -1;
    }
    struct resident_tensors resident;
    if (model_get_resident(h->model, &resident) != 0) {
        return -1;
    }
    uint32_t n_embd = h->info.n_embd;
    uint32_t n_vocab = 0;
    if (model_get_vocab_size(h->model, &n_vocab) != 0 || n_vocab == 0) {
        return -1;
    }
    float *hidden = (float *)calloc(n_embd, sizeof(float));
    if (!hidden) {
        return -1;
    }

    uint32_t *prompt_tokens = NULL;
    uint32_t prompt_len = 0;
    const char *prompt_text = h->prompt ? h->prompt : "";
    if (model_tokenize(h->model, prompt_text, &prompt_tokens, &prompt_len) != 0) {
        free(hidden);
        return -1;
    }
    if (prompt_len == 0) {
        prompt_len = 1;
        prompt_tokens = (uint32_t *)malloc(sizeof(uint32_t));
        if (!prompt_tokens) {
            free(hidden);
            return -1;
        }
        prompt_tokens[0] = 1;
    }
    uint32_t n_layers = h->info.n_layers;
    uint32_t pos = 0;

    for (uint32_t i = 0; i < prompt_len; ++i) {
        uint32_t tok = prompt_tokens[i];
        if (op_embed(NULL, resident.token_embd, resident.token_embd_dtype, &tok, hidden, 1, n_embd) != 0) {
            free(hidden);
            return -1;
        }
        if (i == 0 && debug_enabled()) {
            debug_check("embed", hidden, n_embd);
        }
        if (h->prefetch) {
            prefetch_request_t *req0 = prefetcher_request(h->prefetch, 0);
            prefetch_request_t *req1 = (n_layers > 1) ? prefetcher_request(h->prefetch, 1) : NULL;
            if (!req0) {
                free(hidden);
                return -1;
            }
            for (uint32_t l = 0; l < n_layers; ++l) {
                prefetch_request_t *next_req = NULL;
                uint32_t ahead = l + 2;
                if (ahead < n_layers) {
                    next_req = prefetcher_request(h->prefetch, ahead);
                }
                struct layer_buffer *buf = prefetcher_wait(req0);
                if (!buf) {
                    fprintf(stderr, "engine: prefetch wait failed at layer %u (prefill)\n", l);
                    free(hidden);
                    return -1;
                }
                if (forward_layer_view(h, &buf->view, l, pos, hidden) != 0) {
                    prefetcher_release(h->prefetch, buf);
                    fprintf(stderr, "engine: forward failed at layer %u (prefill)\n", l);
                    free(hidden);
                    return -1;
                }
                prefetcher_release(h->prefetch, buf);
                req0 = req1;
                req1 = next_req;
                if (l + 1 < n_layers && !req0) {
                    free(hidden);
                    return -1;
                }
            }
        } else {
            for (uint32_t l = 0; l < n_layers; ++l) {
                if (forward_layer(h, l, pos, hidden) != 0) {
                    free(hidden);
                    return -1;
                }
            }
        }
        pos++;
    }

    uint32_t n_vocab_use = n_vocab;
    float *logits = (float *)malloc((size_t)n_vocab_use * sizeof(float));
    if (!logits) {
        free(hidden);
        return -1;
    }

    for (uint32_t t = 0; t < max_tokens; ++t) {
        if (resident.lm_head_dtype == 12) {
            if (op_matmul_q4_k(NULL, resident.lm_head, hidden, logits, n_vocab_use, n_embd) != 0) {
                free(logits); free(hidden);
                return -1;
            }
        } else if (resident.lm_head_dtype == 13) {
            if (op_matmul_q5_k(NULL, resident.lm_head, hidden, logits, n_vocab_use, n_embd) != 0) {
                free(logits); free(hidden);
                return -1;
            }
        } else if (resident.lm_head_dtype == 14) {
            if (op_matmul_q6_k(NULL, resident.lm_head, hidden, logits, n_vocab_use, n_embd) != 0) {
                free(logits); free(hidden);
                return -1;
            }
        } else {
            free(logits); free(hidden);
            return -1;
        }

        uint32_t next = 0;
        float maxv = logits[0];
        for (uint32_t i = 1; i < n_vocab_use; ++i) {
            if (logits[i] > maxv) {
                maxv = logits[i];
                next = i;
            }
        }

        const char *tok_str = NULL;
        if (model_get_token_string(h->model, next, &tok_str) == 0 && tok_str) {
            printf("<%u>%s", next, tok_str);
        } else {
            printf("<%u>", next);
        }

        if (op_embed(NULL, resident.token_embd, resident.token_embd_dtype, &next, hidden, 1, n_embd) != 0) {
            free(logits); free(hidden);
            return -1;
        }
        if (h->prefetch) {
            prefetch_request_t *req0 = prefetcher_request(h->prefetch, 0);
            prefetch_request_t *req1 = (n_layers > 1) ? prefetcher_request(h->prefetch, 1) : NULL;
            if (!req0) {
                free(logits); free(hidden);
                return -1;
            }
            for (uint32_t l = 0; l < n_layers; ++l) {
                prefetch_request_t *next_req = NULL;
                uint32_t ahead = l + 2;
                if (ahead < n_layers) {
                    next_req = prefetcher_request(h->prefetch, ahead);
                }
                struct layer_buffer *buf = prefetcher_wait(req0);
                if (!buf) {
                    fprintf(stderr, "engine: prefetch wait failed at layer %u (decode)\n", l);
                    free(logits); free(hidden);
                    return -1;
                }
                if (forward_layer_view(h, &buf->view, l, pos, hidden) != 0) {
                    prefetcher_release(h->prefetch, buf);
                    fprintf(stderr, "engine: forward failed at layer %u (decode)\n", l);
                    free(logits); free(hidden);
                    return -1;
                }
                prefetcher_release(h->prefetch, buf);
                req0 = req1;
                req1 = next_req;
                if (l + 1 < n_layers && !req0) {
                    free(logits); free(hidden);
                    return -1;
                }
            }
        } else {
            for (uint32_t l = 0; l < n_layers; ++l) {
                if (forward_layer(h, l, pos, hidden) != 0) {
                    free(logits); free(hidden);
                    return -1;
                }
            }
        }
        pos++;
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
            size_t rss = (size_t)ru.ru_maxrss;
#else
            size_t rss = (size_t)ru.ru_maxrss * 1024u;
#endif
            model_update_peak_rss(h->model, rss);
        }
    }
    printf("\n");
    free(logits);
    free(hidden);
    free(prompt_tokens);
    return 0;
}

int engine_generate_stream(engine_handle_t *h, uint32_t max_tokens,
                           token_callback cb, void *user) {
    (void)h;
    (void)max_tokens;
    (void)cb;
    (void)user;
    return 0;
}

const char *engine_get_output(engine_handle_t *h) {
    (void)h;
    return 0;
}

const uint32_t *engine_get_tokens(engine_handle_t *h, size_t *n_tokens) {
    (void)h;
    (void)n_tokens;
    return 0;
}

void engine_cancel(engine_handle_t *h) {
    (void)h;
}

void engine_close(engine_handle_t *h) {
    if (!h) {
        return;
    }
    free(h->prompt);
    if (h->prefetch) {
        prefetcher_stop(h->prefetch);
    }
    kv_cache_destroy(h->kv);
    model_close(h->model);
    free(h);
}

int engine_get_streaming_stats(engine_handle_t *h, struct streaming_stats *out) {
    if (!h || !out) {
        return -1;
    }
    if (model_get_streaming_stats(h->model, out) != 0) {
        return -1;
    }
    if (h->stats.max_layer_size > out->max_layer_size) {
        out->max_layer_size = h->stats.max_layer_size;
    }
    if (h->stats.peak_buffer_usage > out->peak_buffer_usage) {
        out->peak_buffer_usage = h->stats.peak_buffer_usage;
    }
    out->max_concurrent_buffers = h->stats.max_concurrent_buffers;
    out->prefetch_hits = h->stats.prefetch_hits;
    out->prefetch_misses = h->stats.prefetch_misses;
    return 0;
}

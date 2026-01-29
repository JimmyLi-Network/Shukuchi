#include "model_loader.h"
#include "gguf_reader.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int build_layer_specs(gguf_file_t *f,
                      struct resident_spec *resident,
                      struct layer_spec **layers_out,
                      uint32_t *n_layers_out);

struct model_handle {
    gguf_file_t *gguf;
    struct resident_spec resident_spec;
    struct layer_spec *layers;
    struct resident_tensors resident_loaded;
    uint32_t n_layers;
    const char * const *tokens;
    uint32_t n_tokens;
    void *layer_buf;
    size_t layer_buf_size;
    struct layer_view layer_view;
    struct streaming_stats stats;
    uint32_t bos_token_id;
    int has_bos;
    void *layer_io_buf;
    size_t layer_io_buf_size;
};

struct prefetch_handle {
    int dummy;
};

static size_t align_up_size(size_t x, size_t a) {
    size_t mask = a - 1;
    return (x + mask) & ~mask;
}

static void resident_clear(struct resident_tensors *r) {
    if (!r) {
        return;
    }
    free((void *)r->token_embd);
    free((void *)r->output_norm);
    free((void *)r->lm_head);
    memset(r, 0, sizeof(*r));
}

static int ensure_io_buf(struct model_handle *m, size_t size) {
    if (m->layer_io_buf_size >= size) {
        return 0;
    }
    void *nbuf = realloc(m->layer_io_buf, size);
    if (!nbuf) {
        return -1;
    }
    m->layer_io_buf = nbuf;
    m->layer_io_buf_size = size;
    return 0;
}

model_handle_t *model_open(const char *path, const struct model_config *cfg) {
    int use_mmap = cfg ? cfg->use_mmap : 0;
    gguf_file_t *f = gguf_open(path, use_mmap);
    if (!f) {
        return NULL;
    }
    if (gguf_read_header(f) != 0) {
        gguf_close(f);
        return NULL;
    }

    struct model_handle *m = (struct model_handle *)calloc(1, sizeof(*m));
    if (!m) {
        gguf_close(f);
        return NULL;
    }
    m->gguf = f;
    memset(&m->stats, 0, sizeof(m->stats));

    // TODO: hardcoded llama mapping for now
    if (build_layer_specs(f, &m->resident_spec, &m->layers, &m->n_layers) != 0) {
        gguf_close(f);
        free(m);
        return NULL;
    }

    if (model_load_resident(m) != 0) {
        gguf_close(f);
        free(m->layers);
        free(m);
        return NULL;
    }

    struct gguf_kv_pair kv;
    if (gguf_find_kv(f, "tokenizer.ggml.tokens", &kv) == 0 && kv.type == GGUF_KV_ARRAY) {
        const struct gguf_array *arr = (const struct gguf_array *)kv.value;
        if (arr && arr->type == GGUF_KV_STRING) {
            m->tokens = arr->strs;
            if (arr->n > 0) {
                m->n_tokens = (uint32_t)arr->n;
            }
        }
    }

    if (gguf_find_kv(f, "tokenizer.ggml.bos_token_id", &kv) == 0 &&
        kv.type == GGUF_KV_UINT32) {
        m->bos_token_id = *(const uint32_t *)kv.value;
        m->has_bos = 1;
    }

    return m;
}

int model_load_resident(model_handle_t *m) {
    if (!m) {
        return -1;
    }
    struct gguf_tensor t;
    struct resident_tensors *r = &m->resident_loaded;
    memset(r, 0, sizeof(*r));

    t.offset = m->resident_spec.token_embd.offset;
    t.size = m->resident_spec.token_embd.size;
    t.dtype = m->resident_spec.token_embd.dtype;
    if (t.size == 0) {
        resident_clear(r);
        return -1;
    }
    r->token_embd = malloc((size_t)t.size);
    if (!r->token_embd) {
        resident_clear(r);
        return -1;
    }
    if (gguf_read_tensor_data(m->gguf, &t, (void *)r->token_embd, (size_t)t.size) != 0) {
        resident_clear(r);
        return -1;
    }
    r->token_embd_dtype = t.dtype;

    t.offset = m->resident_spec.output_norm.offset;
    t.size = m->resident_spec.output_norm.size;
    t.dtype = m->resident_spec.output_norm.dtype;
    if (t.size == 0) {
        resident_clear(r);
        return -1;
    }
    r->output_norm = malloc((size_t)t.size);
    if (!r->output_norm) {
        resident_clear(r);
        return -1;
    }
    if (gguf_read_tensor_data(m->gguf, &t, (void *)r->output_norm, (size_t)t.size) != 0) {
        resident_clear(r);
        return -1;
    }
    r->output_norm_dtype = t.dtype;

    t.offset = m->resident_spec.lm_head.offset;
    t.size = m->resident_spec.lm_head.size;
    t.dtype = m->resident_spec.lm_head.dtype;
    if (t.size == 0) {
        resident_clear(r);
        return -1;
    }
    r->lm_head = malloc((size_t)t.size);
    if (!r->lm_head) {
        resident_clear(r);
        return -1;
    }
    if (gguf_read_tensor_data(m->gguf, &t, (void *)r->lm_head, (size_t)t.size) != 0) {
        resident_clear(r);
        return -1;
    }
    r->lm_head_dtype = t.dtype;

    return 0;
}

int model_get_layer_buffer_size(model_handle_t *m, uint32_t layer_id, size_t *out) {
    if (!m || !out || layer_id >= m->n_layers) {
        return -1;
    }
    const struct layer_spec *ls = &m->layers[layer_id];
    size_t total = 0;
    const size_t align = 32;
    const struct tensor_ref *refs[] = {
        &ls->attn_norm, &ls->attn_q, &ls->attn_k, &ls->attn_v, &ls->attn_o,
        &ls->ffn_norm, &ls->ffn_gate, &ls->ffn_up, &ls->ffn_down
    };
    for (size_t i = 0; i < sizeof(refs) / sizeof(refs[0]); ++i) {
        if (refs[i]->size == 0) {
            return -1;
        }
        total = align_up_size(total, align);
        total += (size_t)refs[i]->size;
    }
    *out = total;
    return 0;
}

int model_get_max_layer_size(model_handle_t *m, size_t *out) {
    if (!m || !out) {
        return -1;
    }
    size_t max_size = 0;
    for (uint32_t i = 0; i < m->n_layers; ++i) {
        size_t sz = 0;
        if (model_get_layer_buffer_size(m, i, &sz) != 0) {
            return -1;
        }
        if (sz > max_size) {
            max_size = sz;
        }
    }
    *out = max_size;
    return 0;
}

int model_load_layer(model_handle_t *m, uint32_t layer_id, void *buffer, size_t buffer_size,
                     struct layer_view *out_view, size_t *out_used) {
    if (!m || !buffer || !out_view || layer_id >= m->n_layers) {
        return -1;
    }
    const struct layer_spec *ls = &m->layers[layer_id];
    size_t need = 0;
    if (model_get_layer_buffer_size(m, layer_id, &need) != 0) {
        fprintf(stderr, "model_load_layer: failed to get buffer size for layer %u\n", layer_id);
        return -1;
    }
    if (buffer_size < need) {
        fprintf(stderr, "model_load_layer: buffer too small for layer %u (need=%zu have=%zu)\n",
                layer_id, need, buffer_size);
        return -1;
    }
    if (need > m->stats.max_layer_size) {
        m->stats.max_layer_size = need;
    }
    if (buffer_size > m->stats.peak_buffer_usage) {
        m->stats.peak_buffer_usage = buffer_size;
    }
    memset(out_view, 0, sizeof(*out_view));
    out_view->layer_id = layer_id;

    const size_t align = 32;
    size_t off = 0;
    struct {
        const struct tensor_ref *ref;
        const void **dst;
        uint32_t *dtype;
        uint64_t *size;
    } fields[] = {
        { &ls->attn_norm, &out_view->attn_norm, &out_view->attn_norm_dtype, NULL },
        { &ls->attn_q, &out_view->attn_q, &out_view->attn_q_dtype, &out_view->attn_q_size },
        { &ls->attn_k, &out_view->attn_k, &out_view->attn_k_dtype, &out_view->attn_k_size },
        { &ls->attn_v, &out_view->attn_v, &out_view->attn_v_dtype, &out_view->attn_v_size },
        { &ls->attn_o, &out_view->attn_o, &out_view->attn_o_dtype, &out_view->attn_o_size },
        { &ls->ffn_norm, &out_view->ffn_norm, &out_view->ffn_norm_dtype, NULL },
        { &ls->ffn_gate, &out_view->ffn_gate, &out_view->ffn_gate_dtype, &out_view->ffn_gate_size },
        { &ls->ffn_up, &out_view->ffn_up, &out_view->ffn_up_dtype, &out_view->ffn_up_size },
        { &ls->ffn_down, &out_view->ffn_down, &out_view->ffn_down_dtype, &out_view->ffn_down_size },
    };

    uint64_t span_start = UINT64_MAX;
    uint64_t span_end = 0;
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        const struct tensor_ref *ref = fields[i].ref;
        if (ref->size == 0) {
            return -1;
        }
        if (ref->offset < span_start) {
            span_start = ref->offset;
        }
        uint64_t end = ref->offset + ref->size;
        if (end > span_end) {
            span_end = end;
        }
    }
    if (span_start == UINT64_MAX || span_end <= span_start) {
        return -1;
    }
    uint64_t span_size = span_end - span_start;
    if (ensure_io_buf(m, (size_t)span_size) != 0) {
        fprintf(stderr, "model_load_layer: io buf alloc failed (layer %u, size=%llu)\n",
                layer_id, (unsigned long long)span_size);
        return -1;
    }
    if (gguf_read_span(m->gguf, span_start, span_size, m->layer_io_buf) != 0) {
        fprintf(stderr, "model_load_layer: read span failed (layer %u, off=%llu size=%llu)\n",
                layer_id, (unsigned long long)span_start, (unsigned long long)span_size);
        return -1;
    }
    m->stats.layer_bytes_read += span_size;

    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        const struct tensor_ref *ref = fields[i].ref;
        off = align_up_size(off, align);
        uint8_t *dst = (uint8_t *)buffer + off;
        uint64_t rel = ref->offset - span_start;
        if (rel + ref->size > span_size) {
            fprintf(stderr, "model_load_layer: span bounds error (layer %u)\n", layer_id);
            return -1;
        }
        memcpy(dst, (const uint8_t *)m->layer_io_buf + rel, (size_t)ref->size);
        *fields[i].dst = dst;
        *fields[i].dtype = ref->dtype;
        if (fields[i].size) {
            *fields[i].size = ref->size;
        }
        off += (size_t)ref->size;
    }

    m->stats.layer_loads += 1;
    if (out_used) {
        *out_used = off;
    }
    return 0;
}

int model_get_resident(model_handle_t *m, struct resident_tensors *out) {
    if (!m || !out) {
        return -1;
    }
    *out = m->resident_loaded;
    return 0;
}

int model_get_info(model_handle_t *m, struct model_info *out) {
    if (!m || !out) {
        return -1;
    }
    struct gguf_kv_pair kv;
    memset(out, 0, sizeof(*out));
    out->n_layers = m->n_layers;
    if (gguf_find_kv(m->gguf, "llama.embedding_length", &kv) == 0 &&
        kv.type == GGUF_KV_UINT32) {
        out->n_embd = *(const uint32_t *)kv.value;
    }
    if (gguf_find_kv(m->gguf, "llama.attention.head_count", &kv) == 0 &&
        kv.type == GGUF_KV_UINT32) {
        out->n_heads = *(const uint32_t *)kv.value;
    }
    if (gguf_find_kv(m->gguf, "llama.attention.head_count_kv", &kv) == 0 &&
        kv.type == GGUF_KV_UINT32) {
        out->n_kv_heads = *(const uint32_t *)kv.value;
    }
    if (out->n_heads && out->n_embd) {
        out->head_dim = out->n_embd / out->n_heads;
    }
    return 0;
}

int model_get_layer_view(model_handle_t *m, uint32_t layer_id, const struct layer_view **out) {
    if (!m || !out) {
        return -1;
    }
    if (layer_id >= m->n_layers) {
        return -1;
    }
    size_t need = 0;
    if (model_get_layer_buffer_size(m, layer_id, &need) != 0 || need == 0) {
        return -1;
    }
    if (m->layer_buf_size < need) {
        void *nbuf = realloc(m->layer_buf, need);
        if (!nbuf) {
            return -1;
        }
        m->layer_buf = nbuf;
        m->layer_buf_size = need;
    }
    size_t used = 0;
    if (model_load_layer(m, layer_id, m->layer_buf, m->layer_buf_size, &m->layer_view, &used) != 0) {
        return -1;
    }
    *out = &m->layer_view;
    return 0;
}

uint32_t model_get_layer_count(model_handle_t *m) {
    if (!m) {
        return 0;
    }
    return m->n_layers;
}

int model_get_vocab_size(model_handle_t *m, uint32_t *out) {
    if (!m || !out) {
        return -1;
    }
    if (m->n_tokens > 0) {
        *out = m->n_tokens;
        return 0;
    }
    return -1;
}

int model_get_token_string(model_handle_t *m, uint32_t token_id, const char **out) {
    if (!m || !out) {
        return -1;
    }
    if (m->tokens && token_id < m->n_tokens) {
        *out = m->tokens[token_id];
        return 0;
    }
    return -1;
}

int model_get_streaming_stats(model_handle_t *m, struct streaming_stats *out) {
    if (!m || !out) {
        return -1;
    }
    *out = m->stats;
    return 0;
}

int model_update_peak_rss(model_handle_t *m, size_t rss_bytes) {
    if (!m) {
        return -1;
    }
    if (rss_bytes > m->stats.peak_rss) {
        m->stats.peak_rss = rss_bytes;
    }
    return 0;
}

static int match_token(const char *token, const char *text, size_t max_len) {
    if (!token || !text) {
        return 0;
    }
    size_t i = 0;
    for (; i < max_len && token[i] != '\0'; ++i) {
        if (token[i] != text[i]) {
            return 0;
        }
    }
    return token[i] == '\0' ? (int)i : 0;
}

int model_tokenize(model_handle_t *m, const char *text, uint32_t **out_tokens, uint32_t *out_len) {
    if (!m || !out_tokens || !out_len) {
        return -1;
    }
    *out_tokens = NULL;
    *out_len = 0;
    if (!text) {
        return 0;
    }
    size_t text_len = strlen(text);
    size_t norm_cap = text_len * 3 + 1;
    char *norm = (char *)malloc(norm_cap);
    if (!norm) {
        return -1;
    }
    size_t norm_len = 0;
    for (size_t i = 0; i < text_len; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (norm_len + 3 >= norm_cap) {
                break;
            }
            norm[norm_len++] = (char)0xE2;
            norm[norm_len++] = (char)0x96;
            norm[norm_len++] = (char)0x81;
        } else {
            norm[norm_len++] = (char)c;
        }
    }
    norm[norm_len] = '\0';

    size_t cap = norm_len + 8;
    uint32_t *tokens = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!tokens) {
        free(norm);
        return -1;
    }
    size_t count = 0;
    if (m->has_bos) {
        tokens[count++] = m->bos_token_id;
    }
    size_t i = 0;
    while (i < norm_len) {
        size_t best_len = 0;
        uint32_t best_id = 0;
        for (uint32_t t = 0; t < m->n_tokens; ++t) {
            const char *tok = m->tokens ? m->tokens[t] : NULL;
            if (!tok) {
                continue;
            }
            size_t max_len = norm_len - i;
            int mlen = match_token(tok, norm + i, max_len);
            if (mlen > 0 && (size_t)mlen > best_len) {
                best_len = (size_t)mlen;
                best_id = t;
            }
        }
        if (best_len == 0) {
            best_len = 1;
            best_id = 0;
        }
        if (count >= cap) {
            cap *= 2;
            uint32_t *nt = (uint32_t *)realloc(tokens, cap * sizeof(uint32_t));
            if (!nt) {
                free(tokens);
                return -1;
            }
            tokens = nt;
        }
        tokens[count++] = best_id;
        i += best_len;
    }
    free(norm);
    *out_tokens = tokens;
    *out_len = (uint32_t)count;
    return 0;
}

prefetch_handle_t *model_prefetch_layer_async(model_handle_t *m, uint32_t layer_id) {
    (void)m;
    (void)layer_id;
    return 0;
}

int model_is_ready(prefetch_handle_t *h) {
    (void)h;
    return 0;
}

layer_view_t *model_wait_layer(prefetch_handle_t *h) {
    (void)h;
    return 0;
}

void model_release_layer(layer_view_t *v) {
    (void)v;
}

void model_close(model_handle_t *m) {
    if (!m) {
        return;
    }
    gguf_close(m->gguf);
    resident_clear(&m->resident_loaded);
    free(m->layer_buf);
    free(m->layer_io_buf);
    free(m->layers);
    free(m);
}

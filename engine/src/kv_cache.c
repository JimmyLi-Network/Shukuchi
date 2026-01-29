#include "kv_cache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct q8_block {
    float scale;
    int8_t data[32];
};

struct kv_block {
    struct q8_block *k;
    struct q8_block *v;
    uint32_t seq_len;
};

struct kv_cache {
    struct kv_cache_config cfg;
    uint32_t n_blocks;
    uint32_t vec_dim;
    uint32_t q8_blocks_per_token;
    uint32_t q8_blocks_per_block;
    struct kv_block *blocks;
    uint32_t *layer_seq_len;
};

static void q8_quantize_vec(const float *x, uint32_t n, struct q8_block *out) {
    uint32_t n_blocks = (n + 31u) / 32u;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        uint32_t base = b * 32u;
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) {
            uint32_t idx = base + i;
            float v = (idx < n) ? x[idx] : 0.0f;
            float a = fabsf(v);
            if (a > max_abs) {
                max_abs = a;
            }
        }
        float scale = max_abs / 127.0f;
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        out[b].scale = scale;
        for (uint32_t i = 0; i < 32u; ++i) {
            uint32_t idx = base + i;
            float v = (idx < n) ? x[idx] : 0.0f;
            int q = (int)lrintf(v / scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out[b].data[i] = (int8_t)q;
        }
    }
}

static void q8_dequantize_vec(const struct q8_block *in, uint32_t n, float *out) {
    uint32_t n_blocks = (n + 31u) / 32u;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        uint32_t base = b * 32u;
        float scale = in[b].scale;
        for (uint32_t i = 0; i < 32u; ++i) {
            uint32_t idx = base + i;
            if (idx < n) {
                out[idx] = (float)in[b].data[i] * scale;
            }
        }
    }
}

kv_cache_t *kv_cache_create(const struct kv_cache_config *cfg) {
    if (!cfg || cfg->block_size == 0 || cfg->max_seq_len == 0) {
        return NULL;
    }
    kv_cache_t *c = (kv_cache_t *)calloc(1, sizeof(*c));
    if (!c) {
        return NULL;
    }
    c->cfg = *cfg;
    c->vec_dim = cfg->n_kv_heads * cfg->head_dim;
    c->n_blocks = (cfg->max_seq_len + cfg->block_size - 1) / cfg->block_size;
    c->q8_blocks_per_token = (c->vec_dim + 31u) / 32u;
    c->q8_blocks_per_block = c->q8_blocks_per_token * cfg->block_size;

    uint32_t total_blocks = cfg->n_layers * c->n_blocks;
    c->blocks = (struct kv_block *)calloc(total_blocks, sizeof(*c->blocks));
    if (!c->blocks) {
        free(c);
        return NULL;
    }
    c->layer_seq_len = (uint32_t *)calloc(cfg->n_layers, sizeof(uint32_t));
    if (!c->layer_seq_len) {
        free(c->blocks);
        free(c);
        return NULL;
    }

    for (uint32_t l = 0; l < cfg->n_layers; ++l) {
        for (uint32_t b = 0; b < c->n_blocks; ++b) {
            struct kv_block *blk = &c->blocks[l * c->n_blocks + b];
            blk->seq_len = 0;
            blk->k = (struct q8_block *)calloc(c->q8_blocks_per_block, sizeof(struct q8_block));
            blk->v = (struct q8_block *)calloc(c->q8_blocks_per_block, sizeof(struct q8_block));
            if (!blk->k || !blk->v) {
                kv_cache_destroy(c);
                return NULL;
            }
        }
    }

    return c;
}

void kv_cache_destroy(kv_cache_t *c) {
    if (!c) {
        return;
    }
    if (c->blocks) {
        uint32_t total_blocks = c->cfg.n_layers * c->n_blocks;
        for (uint32_t i = 0; i < total_blocks; ++i) {
            free(c->blocks[i].k);
            free(c->blocks[i].v);
        }
    }
    free(c->blocks);
    free(c->layer_seq_len);
    free(c);
}

int kv_cache_append(kv_cache_t *c, uint32_t layer, uint32_t pos,
                    const float *k, const float *v) {
    if (!c || !k || !v) {
        return -1;
    }
    if (layer >= c->cfg.n_layers || pos >= c->cfg.max_seq_len) {
        return -1;
    }
    uint32_t block_id = pos / c->cfg.block_size;
    uint32_t token_in_block = pos % c->cfg.block_size;
    struct kv_block *blk = &c->blocks[layer * c->n_blocks + block_id];

    struct q8_block *k_dst = blk->k + token_in_block * c->q8_blocks_per_token;
    struct q8_block *v_dst = blk->v + token_in_block * c->q8_blocks_per_token;
    q8_quantize_vec(k, c->vec_dim, k_dst);
    q8_quantize_vec(v, c->vec_dim, v_dst);

    uint32_t new_len = token_in_block + 1;
    if (new_len > blk->seq_len) {
        blk->seq_len = new_len;
    }
    if (pos + 1 > c->layer_seq_len[layer]) {
        c->layer_seq_len[layer] = pos + 1;
    }
    return 0;
}

int kv_cache_read_block(kv_cache_t *c, uint32_t layer, uint32_t block_id,
                        float *k_out, float *v_out) {
    if (!c || !k_out || !v_out) {
        return -1;
    }
    if (layer >= c->cfg.n_layers || block_id >= c->n_blocks) {
        return -1;
    }
    struct kv_block *blk = &c->blocks[layer * c->n_blocks + block_id];
    for (uint32_t t = 0; t < c->cfg.block_size; ++t) {
        struct q8_block *k_src = blk->k + t * c->q8_blocks_per_token;
        struct q8_block *v_src = blk->v + t * c->q8_blocks_per_token;
        q8_dequantize_vec(k_src, c->vec_dim, k_out + t * c->vec_dim);
        q8_dequantize_vec(v_src, c->vec_dim, v_out + t * c->vec_dim);
    }
    return 0;
}

int kv_cache_read_range(kv_cache_t *c, uint32_t layer,
                        uint32_t seq_start, uint32_t seq_end,
                        float *k_out, float *v_out) {
    if (!c || !k_out || !v_out || seq_end < seq_start) {
        return -1;
    }
    if (layer >= c->cfg.n_layers) {
        return -1;
    }
    if (seq_end > c->cfg.max_seq_len) {
        return -1;
    }
    uint32_t out_idx = 0;
    for (uint32_t pos = seq_start; pos < seq_end; ++pos) {
        uint32_t block_id = pos / c->cfg.block_size;
        uint32_t token_in_block = pos % c->cfg.block_size;
        struct kv_block *blk = &c->blocks[layer * c->n_blocks + block_id];
        struct q8_block *k_src = blk->k + token_in_block * c->q8_blocks_per_token;
        struct q8_block *v_src = blk->v + token_in_block * c->q8_blocks_per_token;
        q8_dequantize_vec(k_src, c->vec_dim, k_out + out_idx * c->vec_dim);
        q8_dequantize_vec(v_src, c->vec_dim, v_out + out_idx * c->vec_dim);
        out_idx++;
    }
    return 0;
}

int kv_cache_iterate(kv_cache_t *c, uint32_t layer,
                     uint32_t seq_start, uint32_t seq_end,
                     kv_block_cb cb, void *user) {
    if (!c || !cb || seq_end < seq_start) {
        return -1;
    }
    if (layer >= c->cfg.n_layers) {
        return -1;
    }
    uint32_t start_block = seq_start / c->cfg.block_size;
    uint32_t end_block = (seq_end + c->cfg.block_size - 1) / c->cfg.block_size;
    float *k_tmp = (float *)malloc((size_t)c->cfg.block_size * c->vec_dim * sizeof(float));
    float *v_tmp = (float *)malloc((size_t)c->cfg.block_size * c->vec_dim * sizeof(float));
    if (!k_tmp || !v_tmp) {
        free(k_tmp);
        free(v_tmp);
        return -1;
    }
    for (uint32_t b = start_block; b < end_block; ++b) {
        struct kv_block *blk = &c->blocks[layer * c->n_blocks + b];
        kv_cache_read_block(c, layer, b, k_tmp, v_tmp);
        uint32_t valid = blk->seq_len;
        cb(b, k_tmp, v_tmp, valid, user);
    }
    free(k_tmp);
    free(v_tmp);
    return 0;
}

void kv_cache_clear(kv_cache_t *c) {
    if (!c) {
        return;
    }
    uint32_t total_blocks = c->cfg.n_layers * c->n_blocks;
    for (uint32_t i = 0; i < total_blocks; ++i) {
        c->blocks[i].seq_len = 0;
    }
    for (uint32_t l = 0; l < c->cfg.n_layers; ++l) {
        c->layer_seq_len[l] = 0;
    }
}

uint32_t kv_cache_get_seq_len(kv_cache_t *c, uint32_t layer) {
    if (!c || layer >= c->cfg.n_layers) {
        return 0;
    }
    return c->layer_seq_len[layer];
}

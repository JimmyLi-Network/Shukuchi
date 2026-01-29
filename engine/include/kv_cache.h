#pragma once

#include <stdint.h>

enum kv_quant_type {
    KV_Q8_0 = 0,
    KV_Q4_0 = 1,
};

struct kv_cache_config {
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t block_size;
    uint32_t max_seq_len;
    enum kv_quant_type quant;
};

typedef struct kv_cache kv_cache_t;

kv_cache_t *kv_cache_create(const struct kv_cache_config *cfg);
void kv_cache_destroy(kv_cache_t *c);
int kv_cache_append(kv_cache_t *c, uint32_t layer, uint32_t pos,
                    const float *k, const float *v);
int kv_cache_read_block(kv_cache_t *c, uint32_t layer, uint32_t block_id,
                        float *k_out, float *v_out);
int kv_cache_read_range(kv_cache_t *c, uint32_t layer,
                        uint32_t seq_start, uint32_t seq_end,
                        float *k_out, float *v_out);
typedef void (*kv_block_cb)(uint32_t block_idx,
                            const float *k, const float *v,
                            uint32_t valid_tokens, void *user);
int kv_cache_iterate(kv_cache_t *c, uint32_t layer,
                     uint32_t seq_start, uint32_t seq_end,
                     kv_block_cb cb, void *user);
void kv_cache_clear(kv_cache_t *c);
uint32_t kv_cache_get_seq_len(kv_cache_t *c, uint32_t layer);

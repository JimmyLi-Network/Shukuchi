#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct model_handle model_handle_t;
typedef struct layer_view layer_view_t;
typedef struct prefetch_handle prefetch_handle_t;

struct tensor_ref {
    uint64_t offset;
    uint64_t size;
    uint32_t dtype;
};

struct layer_spec {
    struct tensor_ref attn_norm;
    struct tensor_ref attn_q;
    struct tensor_ref attn_k;
    struct tensor_ref attn_v;
    struct tensor_ref attn_o;
    struct tensor_ref ffn_norm;
    struct tensor_ref ffn_gate;
    struct tensor_ref ffn_up;
    struct tensor_ref ffn_down;
};

struct resident_spec {
    struct tensor_ref token_embd;
    struct tensor_ref output_norm;
    struct tensor_ref lm_head;
};

struct layer_view {
    uint32_t layer_id;
    const void *attn_norm;
    const void *attn_q;
    const void *attn_k;
    const void *attn_v;
    const void *attn_o;
    const void *ffn_norm;
    const void *ffn_gate;
    const void *ffn_up;
    const void *ffn_down;
    uint32_t attn_norm_dtype;
    uint32_t attn_q_dtype;
    uint32_t attn_k_dtype;
    uint32_t attn_v_dtype;
    uint32_t attn_o_dtype;
    uint32_t ffn_norm_dtype;
    uint32_t ffn_gate_dtype;
    uint32_t ffn_up_dtype;
    uint32_t ffn_down_dtype;
    uint64_t attn_q_size;
    uint64_t attn_k_size;
    uint64_t attn_v_size;
    uint64_t attn_o_size;
    uint64_t ffn_gate_size;
    uint64_t ffn_up_size;
    uint64_t ffn_down_size;
};

struct resident_tensors {
    const void *token_embd;
    const void *output_norm;
    const void *lm_head;
    uint32_t token_embd_dtype;
    uint32_t output_norm_dtype;
    uint32_t lm_head_dtype;
};

struct model_info {
    uint32_t n_layers;
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    float rope_theta;
};

struct streaming_stats {
    uint64_t layer_loads;
    uint64_t layer_bytes_read;
    size_t max_layer_size;
    size_t peak_buffer_usage;
    size_t peak_rss;
    uint32_t max_concurrent_buffers;
    uint32_t prefetch_hits;
    uint32_t prefetch_misses;
};

struct model_config {
    int prefer_gguf;
    int use_mmap;
};

model_handle_t *model_open(const char *path, const struct model_config *cfg);
int model_load_resident(model_handle_t *m);
int model_get_resident(model_handle_t *m, struct resident_tensors *out);
int model_get_info(model_handle_t *m, struct model_info *out);
int model_get_layer_view(model_handle_t *m, uint32_t layer_id, const struct layer_view **out);
int model_get_layer_buffer_size(model_handle_t *m, uint32_t layer_id, size_t *out);
int model_get_max_layer_size(model_handle_t *m, size_t *out);
int model_load_layer(model_handle_t *m, uint32_t layer_id, void *buffer, size_t buffer_size,
                     struct layer_view *out_view, size_t *out_used);
uint32_t model_get_layer_count(model_handle_t *m);
int model_get_vocab_size(model_handle_t *m, uint32_t *out);
int model_get_token_string(model_handle_t *m, uint32_t token_id, const char **out);
int model_get_streaming_stats(model_handle_t *m, struct streaming_stats *out);
int model_update_peak_rss(model_handle_t *m, size_t rss_bytes);
int model_tokenize(model_handle_t *m, const char *text, uint32_t **out_tokens, uint32_t *out_len);
prefetch_handle_t *model_prefetch_layer_async(model_handle_t *m, uint32_t layer_id);
int model_is_ready(prefetch_handle_t *h);
layer_view_t *model_wait_layer(prefetch_handle_t *h);
void model_release_layer(layer_view_t *v);
void model_close(model_handle_t *m);

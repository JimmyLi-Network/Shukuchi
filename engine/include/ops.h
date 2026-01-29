#pragma once

#include <stdint.h>

struct op_context {
    uint32_t n_threads;
};

int op_rmsnorm(const struct op_context *ctx, const float *x, const float *w, float *y,
               uint32_t n, uint32_t d);
int op_rope(const struct op_context *ctx, void *qk,
            uint32_t n_heads, uint32_t head_dim, uint32_t pos, float rope_theta);
int op_matmul_f32(const struct op_context *ctx,
                  const float *a, const float *b, float *c,
                  uint32_t m, uint32_t n, uint32_t k);
int op_matmul_q8_0(const struct op_context *ctx,
                   const void *a_q8, const float *b_f32, float *c,
                   uint32_t m, uint32_t n, uint32_t k);
int op_matmul_q4_k(const struct op_context *ctx,
                   const void *a_q4k, const float *b_f32, float *c,
                   uint32_t m, uint32_t k);
int op_matmul_q5_k(const struct op_context *ctx,
                   const void *a_q5k, const float *b_f32, float *c,
                   uint32_t m, uint32_t k);
int op_matmul_q6_k(const struct op_context *ctx,
                   const void *a_q6k, const float *b_f32, float *c,
                   uint32_t m, uint32_t k);
int op_attention(const struct op_context *ctx, const float *q,
                 const float *k, const float *v, float *out,
                 uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                 uint32_t seq_len, float scale, const float *mask);
int op_mlp_swiglu(const struct op_context *ctx,
                  const float *x, const void *w_gate, const void *w_up,
                  const void *w_down, float *y, uint32_t n,
                  uint32_t d_in, uint32_t d_ff);
int op_softmax(const struct op_context *ctx, float *x, uint32_t n);
int op_embed(const struct op_context *ctx, const void *table, uint32_t table_dtype,
             const uint32_t *tokens, float *out, uint32_t seq_len, uint32_t n_embd);

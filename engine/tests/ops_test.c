#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ops.h"

static int approx_eq(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

static uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) {
        return (uint16_t)(sign << 15);
    }
    if (exp >= 31) {
        return (uint16_t)((sign << 15) | (0x1F << 10));
    }
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant);
}

static void test_op_embed_f16(void) {
    const uint32_t n_vocab = 4;
    const uint32_t n_embd = 4;
    uint16_t table[n_vocab * n_embd];
    for (uint32_t i = 0; i < n_vocab * n_embd; ++i) {
        float v = (float)i * 0.1f;
        table[i] = float_to_half(v);
    }
    uint32_t tokens[2] = {1, 3};
    float out[2 * 4];
    struct op_context ctx = {0};
    assert(op_embed(&ctx, table, 1, tokens, out, 2, n_embd) == 0);
    for (uint32_t t = 0; t < 2; ++t) {
        for (uint32_t j = 0; j < n_embd; ++j) {
            float exp = (float)((tokens[t] * n_embd) + j) * 0.1f;
            assert(approx_eq(out[t * n_embd + j], exp, 0.01f));
        }
    }
}

static void test_op_rmsnorm(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4] = {0};
    struct op_context ctx = {0};
    assert(op_rmsnorm(&ctx, x, w, y, 1, 4) == 0);
    float mean_sq = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
    float inv_rms = 1.0f / sqrtf(mean_sq + 1e-5f);
    for (uint32_t i = 0; i < 4; ++i) {
        assert(approx_eq(y[i], x[i] * inv_rms, 1e-5f));
    }
}

static void test_op_rope(void) {
    float qk[2] = {1.0f, 0.0f};
    struct op_context ctx = {0};
    assert(op_rope(&ctx, qk, 1, 2, 1, 10000.0f) == 0);
    // For head_dim=2, freq=1, angle=1 rad
    assert(approx_eq(qk[0], cosf(1.0f), 1e-5f));
    assert(approx_eq(qk[1], sinf(1.0f), 1e-5f));
}

static void test_op_softmax(void) {
    float x[3] = {1.0f, 2.0f, 3.0f};
    struct op_context ctx = {0};
    assert(op_softmax(&ctx, x, 3) == 0);
    float sum = x[0] + x[1] + x[2];
    assert(approx_eq(sum, 1.0f, 1e-6f));
    assert(x[2] > x[1] && x[1] > x[0]);
}

struct block_q4_k {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

static void test_op_matmul_q4_k(void) {
    struct block_q4_k blk;
    memset(&blk, 0, sizeof(blk));
    blk.d = float_to_half(1.0f);
    blk.dmin = float_to_half(0.0f);
    // scales: sc=1, m=0 for all sub-blocks
    blk.scales[0] = 1;
    blk.scales[1] = 1;
    blk.scales[2] = 1;
    blk.scales[3] = 1;
    blk.scales[4] = 0;
    blk.scales[5] = 0;
    blk.scales[6] = 0;
    blk.scales[7] = 0;
    blk.scales[8] = 1;
    blk.scales[9] = 1;
    blk.scales[10] = 1;
    blk.scales[11] = 1;

    // set first byte: low=0, high=15 (-> values 0 and 15 with sc=1, m=0)
    blk.qs[0] = 0xF0;

    float b[256];
    memset(b, 0, sizeof(b));
    b[0] = 1.0f;
    b[32] = 1.0f;

    float c[1] = {0};
    struct op_context ctx = {0};
    assert(op_matmul_q4_k(&ctx, &blk, b, c, 1, 256) == 0);
    assert(approx_eq(c[0], 15.0f, 1e-3f));
}

static void test_op_attention(void) {
    float q[2] = {1.0f, 0.0f};
    float k[2 * 2] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    float v[2 * 2] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    float out[2] = {0};
    struct op_context ctx = {0};
    float scale = 1.0f / sqrtf(2.0f);
    assert(op_attention(&ctx, q, k, v, out, 1, 1, 2, 2, scale, NULL) == 0);
    float s0 = expf(1.0f * scale);
    float s1 = expf(0.0f);
    float w0 = s0 / (s0 + s1);
    float w1 = s1 / (s0 + s1);
    float exp0 = w0 * 1.0f + w1 * 3.0f;
    float exp1 = w0 * 2.0f + w1 * 4.0f;
    assert(approx_eq(out[0], exp0, 1e-5f));
    assert(approx_eq(out[1], exp1, 1e-5f));
}

static void test_op_mlp_swiglu(void) {
    const uint32_t d_in = 256;
    const uint32_t d_ff = 256;
    const uint32_t k = d_in;
    const uint32_t m = d_ff;
    const uint32_t blocks_per_row = k / 256;
    const uint32_t total_blocks = m * blocks_per_row;

    struct block_q4_k *w_gate = (struct block_q4_k *)calloc(total_blocks, sizeof(struct block_q4_k));
    struct block_q4_k *w_up   = (struct block_q4_k *)calloc(total_blocks, sizeof(struct block_q4_k));
    struct block_q4_k *w_down = (struct block_q4_k *)calloc(total_blocks, sizeof(struct block_q4_k));
    assert(w_gate && w_up && w_down);

    float x[256];
    for (uint32_t i = 0; i < d_in; ++i) {
        x[i] = (float)i * 0.01f;
    }
    float y[256];
    memset(y, 0, sizeof(y));

    struct op_context ctx = {0};
    assert(op_mlp_swiglu(&ctx, x, w_gate, w_up, w_down, y, 1, d_in, d_ff) == 0);
    for (uint32_t i = 0; i < d_in; ++i) {
        assert(approx_eq(y[i], 0.0f, 1e-6f));
    }

    free(w_gate);
    free(w_up);
    free(w_down);
}

int main(void) {
    test_op_embed_f16();
    test_op_rmsnorm();
    test_op_rope();
    test_op_softmax();
    test_op_matmul_q4_k();
    test_op_attention();
    test_op_mlp_swiglu();
    printf("PASS\n");
    return 0;
}

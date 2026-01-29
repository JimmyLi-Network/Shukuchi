#include "ops.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include "metal_ops.h"
static int metal_enabled_flag = -1;
static int metal_enabled(void) {
    if (metal_enabled_flag != -1) {
        return metal_enabled_flag;
    }
    const char *env = getenv("SHUKUCHI_METAL");
    if (env && (env[0] == '0' || env[0] == 'f' || env[0] == 'F')) {
        metal_enabled_flag = 0;
    } else {
        metal_enabled_flag = 1;
    }
    return metal_enabled_flag;
}
static void metal_note_cpu_fallback(void) {
    metal_matmul_q4k_vec(NULL, NULL, NULL, NULL, 0, 0);
}
#endif

struct q8_block {
    float scale;
    int8_t data[32];
};

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q4_k {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

struct block_q5_k {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};

struct block_q6_k {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};

static float half_to_float(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t f_sgn = ((uint32_t)h & 0x8000u) << 16;
    uint32_t f_exp;
    uint32_t f_sig;

    if (h_exp == 0) {
        if (h_sig == 0) {
            f_exp = 0;
            f_sig = 0;
        } else {
            int shift = 0;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                shift++;
            }
            h_sig &= 0x03FFu;
            f_exp = (uint32_t)(127 - 15 - shift) << 23;
            f_sig = (uint32_t)h_sig << 13;
        }
    } else if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = (uint32_t)h_sig << 13;
    } else {
        f_exp = (uint32_t)((h_exp >> 10) + (127 - 15)) << 23;
        f_sig = (uint32_t)h_sig << 13;
    }

    uint32_t f = f_sgn | f_exp | f_sig;
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static void dequant_q8_row(const struct q8_block *row, uint32_t n_embd, float *out) {
    uint32_t n_blocks = (n_embd + 31u) / 32u;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        float scale = row[b].scale;
        uint32_t base = b * 32u;
        for (uint32_t i = 0; i < 32u; ++i) {
            uint32_t idx = base + i;
            if (idx < n_embd) {
                out[idx] = (float)row[b].data[i] * scale;
            }
        }
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dequantize_row_q4_k(const struct block_q4_k *x, float *y, uint32_t k) {
    uint32_t nb = k / QK_K;
    for (uint32_t i = 0; i < nb; ++i) {
        const uint8_t *q = x[i].qs;
        const float d = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

static void dequantize_row_q5_k(const struct block_q5_k *x, float *y, uint32_t k) {
    uint32_t nb = k / QK_K;
    for (uint32_t i = 0; i < nb; ++i) {
        const float d = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        for (uint32_t sb = 0; sb < 8; ++sb) {
            uint8_t sc, m;
            get_scale_min_k4(sb, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            for (uint32_t l = 0; l < 32; ++l) {
                uint32_t idx = sb * 32 + l;
                uint8_t ql = (idx & 1u)
                    ? (uint8_t)((x[i].qs[idx / 2] >> 4) & 0xF)
                    : (uint8_t)(x[i].qs[idx / 2] & 0xF);
                uint8_t qh = (uint8_t)((x[i].qh[idx / 8] >> (idx & 7u)) & 0x1);
                uint8_t qv = (uint8_t)(ql | (qh << 4));
                y[i * QK_K + idx] = d1 * (float)qv - m1;
            }
        }
    }
}

static void dequantize_row_q6_k(const struct block_q6_k *x, float *y, uint32_t k) {
    uint32_t nb = k / QK_K;
    for (uint32_t i = 0; i < nb; ++i) {
        const float d = half_to_float(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;

        for (uint32_t n = 0; n < QK_K; n += 128) {
            for (uint32_t l = 0; l < 32; ++l) {
                uint32_t is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * (float)sc[is + 0] * q1;
                y[l + 32] = d * (float)sc[is + 2] * q2;
                y[l + 64] = d * (float)sc[is + 4] * q3;
                y[l + 96] = d * (float)sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

int op_rmsnorm(const struct op_context *ctx, const float *x, const float *w, float *y,
               uint32_t n, uint32_t d) {
    (void)ctx;
    const float eps = 1e-5f;
    for (uint32_t i = 0; i < n; ++i) {
        const float *xi = x + i * d;
        float *yi = y + i * d;
        float mean_sq = 0.0f;
        for (uint32_t j = 0; j < d; ++j) {
            float v = xi[j];
            mean_sq += v * v;
        }
        mean_sq /= (float)d;
        float inv_rms = 1.0f / sqrtf(mean_sq + eps);
        for (uint32_t j = 0; j < d; ++j) {
            yi[j] = xi[j] * inv_rms * w[j];
        }
    }
    return 0;
}

int op_rope(const struct op_context *ctx, void *qk,
            uint32_t n_heads, uint32_t head_dim, uint32_t pos, float rope_theta) {
    (void)ctx;
    if (!qk || head_dim == 0) {
        return -1;
    }
    float *q = (float *)qk;
    const float inv_theta = 1.0f / rope_theta;
    for (uint32_t h = 0; h < n_heads; ++h) {
        float *qh = q + (uint64_t)h * head_dim;
        for (uint32_t i = 0; i + 1 < head_dim; i += 2) {
            float freq = powf(inv_theta, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = qh[i];
            float x1 = qh[i + 1];
            qh[i]     = x0 * c - x1 * s;
            qh[i + 1] = x0 * s + x1 * c;
        }
    }
    return 0;
}

int op_matmul_f32(const struct op_context *ctx,
                  const float *a, const float *b, float *c,
                  uint32_t m, uint32_t n, uint32_t k) {
    (void)ctx;
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    return 0;
}

int op_matmul_q8_0(const struct op_context *ctx,
                   const void *a_q8, const float *b_f32, float *c,
                   uint32_t m, uint32_t n, uint32_t k) {
    (void)ctx;
    (void)a_q8;
    (void)b_f32;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    return 0;
}

int op_matmul_q4_k(const struct op_context *ctx,
                   const void *a_q4k, const float *b_f32, float *c,
                   uint32_t m, uint32_t k) {
    (void)ctx;
#if defined(__APPLE__)
    static void *metal_ctx = NULL;
    static int metal_ready = 0;
    static int metal_attempted = 0;
    if (metal_enabled()) {
        if (!metal_ctx && !metal_attempted) {
            metal_attempted = 1;
            metal_ctx = metal_ops_init("engine/metal/matmul_q4k.metal");
            metal_ready = metal_ctx ? 1 : 0;
        }
        if (metal_ready) {
            if (metal_matmul_q4k_vec(metal_ctx, a_q4k, b_f32, c, m, k) == 0) {
                return 0;
            }
        }
    } else {
        metal_note_cpu_fallback();
    }
#endif
    if (!a_q4k || !b_f32 || !c) {
        return -1;
    }
    if (k == 0 || (k % QK_K) != 0) {
        return -1;
    }
    uint32_t nb = k / QK_K;
#if defined(__APPLE__)
    metal_note_cpu_fallback();
#endif
    const struct block_q4_k *a = (const struct block_q4_k *)a_q4k;
    float tmp[QK_K];
    for (uint32_t row = 0; row < m; ++row) {
        float sum = 0.0f;
        const struct block_q4_k *row_blocks = a + (uint64_t)row * nb;
        for (uint32_t b = 0; b < nb; ++b) {
            dequantize_row_q4_k(&row_blocks[b], tmp, QK_K);
            const float *bv = b_f32 + (uint64_t)b * QK_K;
            for (uint32_t i = 0; i < QK_K; ++i) {
                sum += tmp[i] * bv[i];
            }
        }
        c[row] = sum;
    }
    return 0;
}

int op_matmul_q5_k(const struct op_context *ctx,
                   const void *a_q5k, const float *b_f32, float *c,
                   uint32_t m, uint32_t k) {
    (void)ctx;
#if defined(__APPLE__)
    static void *metal_ctx = NULL;
    static int metal_ready = 0;
    static int metal_attempted = 0;
    if (metal_enabled()) {
        if (!metal_ctx && !metal_attempted) {
            metal_attempted = 1;
            metal_ctx = metal_ops_init("engine/metal/matmul_q4k.metal");
            metal_ready = metal_ctx ? 1 : 0;
        }
        if (metal_ready) {
            if (metal_matmul_q5k_vec(metal_ctx, a_q5k, b_f32, c, m, k) == 0) {
                return 0;
            }
        }
    }
#endif
    if (!a_q5k || !b_f32 || !c) {
        return -1;
    }
    if (k == 0 || (k % QK_K) != 0) {
        return -1;
    }
    uint32_t nb = k / QK_K;
    const struct block_q5_k *a = (const struct block_q5_k *)a_q5k;
    float tmp[QK_K];
    for (uint32_t row = 0; row < m; ++row) {
        float sum = 0.0f;
        const struct block_q5_k *row_blocks = a + (uint64_t)row * nb;
        for (uint32_t b = 0; b < nb; ++b) {
            dequantize_row_q5_k(&row_blocks[b], tmp, QK_K);
            const float *bv = b_f32 + (uint64_t)b * QK_K;
            for (uint32_t i = 0; i < QK_K; ++i) {
                sum += tmp[i] * bv[i];
            }
        }
        c[row] = sum;
    }
    return 0;
}

int op_matmul_q6_k(const struct op_context *ctx,
                   const void *a_q6k, const float *b_f32, float *c,
                   uint32_t m, uint32_t k) {
    (void)ctx;
#if defined(__APPLE__)
    static void *metal_ctx = NULL;
    static int metal_ready = 0;
    static int metal_attempted = 0;
    if (metal_enabled()) {
        if (!metal_ctx && !metal_attempted) {
            metal_attempted = 1;
            metal_ctx = metal_ops_init("engine/metal/matmul_q4k.metal");
            metal_ready = metal_ctx ? 1 : 0;
        }
        if (metal_ready) {
            if (metal_matmul_q6k_vec(metal_ctx, a_q6k, b_f32, c, m, k) == 0) {
                return 0;
            }
        }
    }
#endif
    if (!a_q6k || !b_f32 || !c) {
        return -1;
    }
    if (k == 0 || (k % QK_K) != 0) {
        return -1;
    }
    uint32_t nb = k / QK_K;
    const struct block_q6_k *a = (const struct block_q6_k *)a_q6k;
    float tmp[QK_K];
    for (uint32_t row = 0; row < m; ++row) {
        float sum = 0.0f;
        const struct block_q6_k *row_blocks = a + (uint64_t)row * nb;
        for (uint32_t b = 0; b < nb; ++b) {
            dequantize_row_q6_k(&row_blocks[b], tmp, QK_K);
            const float *bv = b_f32 + (uint64_t)b * QK_K;
            for (uint32_t i = 0; i < QK_K; ++i) {
                sum += tmp[i] * bv[i];
            }
        }
        c[row] = sum;
    }
    return 0;
}

int op_attention(const struct op_context *ctx, const float *q,
                 const float *k, const float *v, float *out,
                 uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                 uint32_t seq_len, float scale, const float *mask) {
    (void)ctx;
    if (!q || !k || !v || !out || head_dim == 0 || n_heads == 0 || seq_len == 0) {
        return -1;
    }
    if (n_kv_heads == 0) {
        return -1;
    }
    float *scores = (float *)malloc((size_t)seq_len * sizeof(float));
    if (!scores) {
        return -1;
    }
    for (uint32_t h = 0; h < n_heads; ++h) {
        const float *qh = q + (uint64_t)h * head_dim;
        uint32_t kvh = h % n_kv_heads;
        for (uint32_t i = 0; i < seq_len; ++i) {
            const float *kh = k + ((uint64_t)i * n_kv_heads + kvh) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += qh[d] * kh[d];
            }
            float s = dot * scale;
            if (mask) {
                s += mask[i];
            }
            scores[i] = s;
        }
        float maxv = scores[0];
        for (uint32_t i = 1; i < seq_len; ++i) {
            if (scores[i] > maxv) {
                maxv = scores[i];
            }
        }
        float sum = 0.0f;
        for (uint32_t i = 0; i < seq_len; ++i) {
            scores[i] = expf(scores[i] - maxv);
            sum += scores[i];
        }
        float inv = (sum == 0.0f) ? 0.0f : (1.0f / sum);
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[h * head_dim + d] = 0.0f;
        }
        for (uint32_t i = 0; i < seq_len; ++i) {
            const float *vh = v + ((uint64_t)i * n_kv_heads + kvh) * head_dim;
            float w = scores[i] * inv;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out[h * head_dim + d] += w * vh[d];
            }
        }
    }
    free(scores);
    return 0;
}

int op_mlp_swiglu(const struct op_context *ctx,
                  const float *x, const void *w_gate, const void *w_up,
                  const void *w_down, float *y, uint32_t n,
                  uint32_t d_in, uint32_t d_ff) {
    (void)ctx;
    if (!x || !w_gate || !w_up || !w_down || !y) {
        return -1;
    }
    if (n != 1 || d_in == 0 || d_ff == 0) {
        return -1;
    }
    float *gate = (float *)malloc((size_t)d_ff * sizeof(float));
    float *up = (float *)malloc((size_t)d_ff * sizeof(float));
    float *hidden = (float *)malloc((size_t)d_ff * sizeof(float));
    if (!gate || !up || !hidden) {
        free(gate);
        free(up);
        free(hidden);
        return -1;
    }
    if (op_matmul_q4_k(ctx, w_gate, x, gate, d_ff, d_in) != 0) {
        free(gate); free(up); free(hidden);
        return -1;
    }
    if (op_matmul_q4_k(ctx, w_up, x, up, d_ff, d_in) != 0) {
        free(gate); free(up); free(hidden);
        return -1;
    }
    for (uint32_t i = 0; i < d_ff; ++i) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu = g * sig;
        hidden[i] = silu * up[i];
    }
    if (op_matmul_q4_k(ctx, w_down, hidden, y, d_in, d_ff) != 0) {
        free(gate); free(up); free(hidden);
        return -1;
    }
    free(gate);
    free(up);
    free(hidden);
    return 0;
}

int op_softmax(const struct op_context *ctx, float *x, uint32_t n) {
    (void)ctx;
    if (!x || n == 0) {
        return -1;
    }
    float maxv = x[0];
    for (uint32_t i = 1; i < n; ++i) {
        if (x[i] > maxv) {
            maxv = x[i];
        }
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    if (sum == 0.0f) {
        return -1;
    }
    float inv = 1.0f / sum;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] *= inv;
    }
    return 0;
}

int op_embed(const struct op_context *ctx, const void *table, uint32_t table_dtype,
             const uint32_t *tokens, float *out, uint32_t seq_len, uint32_t n_embd) {
    (void)ctx;
    if (!table || !tokens || !out) {
        return -1;
    }
    if (table_dtype == 1) { // F16
        const uint16_t *t = (const uint16_t *)table;
        for (uint32_t i = 0; i < seq_len; ++i) {
            uint32_t tok = tokens[i];
            const uint16_t *row = t + (uint64_t)tok * n_embd;
            float *dst = out + (uint64_t)i * n_embd;
            for (uint32_t j = 0; j < n_embd; ++j) {
                dst[j] = half_to_float(row[j]);
            }
        }
        return 0;
    }
    if (table_dtype == 10) { // Q8_0
        const struct q8_block *t = (const struct q8_block *)table;
        uint32_t blocks_per_row = (n_embd + 31u) / 32u;
        for (uint32_t i = 0; i < seq_len; ++i) {
            uint32_t tok = tokens[i];
            const struct q8_block *row = t + (uint64_t)tok * blocks_per_row;
            float *dst = out + (uint64_t)i * n_embd;
            dequant_q8_row(row, n_embd, dst);
        }
        return 0;
    }
    if (table_dtype == 12) { // Q4_K
        if ((n_embd % QK_K) != 0) {
            return -1;
        }
        const struct block_q4_k *t = (const struct block_q4_k *)table;
        uint32_t blocks_per_row = n_embd / QK_K;
        for (uint32_t i = 0; i < seq_len; ++i) {
            uint32_t tok = tokens[i];
            const struct block_q4_k *row = t + (uint64_t)tok * blocks_per_row;
            float *dst = out + (uint64_t)i * n_embd;
            dequantize_row_q4_k(row, dst, n_embd);
        }
        return 0;
    }
    return -1;
}

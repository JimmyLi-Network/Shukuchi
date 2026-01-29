#include <metal_stdlib>
using namespace metal;

struct Q4KBlock {
    ushort d;
    ushort dmin;
    uchar scales[12];
    uchar qs[128];
};

static inline float half_to_float(ushort h) {
    half v = as_type<half>(h);
    return (float)v;
}

static inline void get_scale_min_k4(uint j, thread const uchar *q, thread uchar &d, thread uchar &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

kernel void matmul_q4k_vec(
    device const Q4KBlock *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (k == 0 || (k % 256) != 0) {
        return;
    }
    uint nb = k / 256;
    float sum = 0.0f;
    device const Q4KBlock *row = weights + gid * nb;
    threadgroup float x_shared[256];
    for (uint b = 0; b < nb; ++b) {
        for (uint i = tid; i < 256; i += tg_size) {
            x_shared[i] = x[b * 256 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const Q4KBlock block = row[b];
        const float d = half_to_float(block.d);
        const float dmin = half_to_float(block.dmin);
        thread const uchar *q = block.qs;
        uint is = 0;
        for (uint j = 0; j < 256; j += 64) {
            uchar sc0, m0, sc1, m1;
            get_scale_min_k4(is + 0, block.scales, sc0, m0);
            get_scale_min_k4(is + 1, block.scales, sc1, m1);
            float d1 = d * (float)sc0;
            float m1f = dmin * (float)m0;
            float d2 = d * (float)sc1;
            float m2f = dmin * (float)m1;
            for (uint l = 0; l < 32; ++l) {
                float v0 = d1 * (float)(q[l] & 0xF) - m1f;
                float v1 = d2 * (float)(q[l] >> 4) - m2f;
                sum += v0 * x_shared[j + l];
                sum += v1 * x_shared[j + 32 + l];
            }
            q += 32;
            is += 2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y[gid] = sum;
}

struct Q5KBlock {
    ushort d;
    ushort dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

kernel void matmul_q5k_vec(
    device const Q5KBlock *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (k == 0 || (k % 256) != 0) {
        return;
    }
    uint nb = k / 256;
    float sum = 0.0f;
    device const Q5KBlock *row = weights + gid * nb;
    threadgroup float x_shared[256];
    for (uint b = 0; b < nb; ++b) {
        for (uint i = tid; i < 256; i += tg_size) {
            x_shared[i] = x[b * 256 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const Q5KBlock block = row[b];
        const float d = half_to_float(block.d);
        const float dmin = half_to_float(block.dmin);
        for (uint sb = 0; sb < 8; ++sb) {
            uchar sc, m;
            get_scale_min_k4(sb, block.scales, sc, m);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)m;
            for (uint l = 0; l < 32; ++l) {
                uint idx = sb * 32 + l;
                uchar ql = (idx & 1)
                    ? (uchar)((block.qs[idx / 2] >> 4) & 0xF)
                    : (uchar)(block.qs[idx / 2] & 0xF);
                uchar qh = (uchar)((block.qh[idx / 8] >> (idx & 7)) & 0x1);
                uchar qv = (uchar)(ql | (qh << 4));
                float v = d1 * (float)qv - m1;
                sum += v * x_shared[idx];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y[gid] = sum;
}

struct Q6KBlock {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    ushort d;
};

kernel void matmul_q6k_vec(
    device const Q6KBlock *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (k == 0 || (k % 256) != 0) {
        return;
    }
    uint nb = k / 256;
    float sum = 0.0f;
    device const Q6KBlock *row = weights + gid * nb;
    threadgroup float x_shared[256];
    for (uint b = 0; b < nb; ++b) {
        for (uint i = tid; i < 256; i += tg_size) {
            x_shared[i] = x[b * 256 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const Q6KBlock block = row[b];
        const float d = half_to_float(block.d);
        thread const uchar *ql = block.ql;
        thread const uchar *qh = block.qh;
        thread const char *sc = block.scales;
        for (uint n = 0; n < 256; n += 128) {
            for (uint l = 0; l < 32; ++l) {
                uint is = l / 16;
                int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                float s0 = d * (float)sc[is + 0];
                float s1 = d * (float)sc[is + 2];
                float s2 = d * (float)sc[is + 4];
                float s3 = d * (float)sc[is + 6];

                sum += (s0 * (float)q1) * x_shared[n + l + 0];
                sum += (s1 * (float)q2) * x_shared[n + l + 32];
                sum += (s2 * (float)q3) * x_shared[n + l + 64];
                sum += (s3 * (float)q4) * x_shared[n + l + 96];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y[gid] = sum;
}

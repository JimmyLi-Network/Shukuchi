#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "kv_cache.h"

static int approx_eq(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

int main(void) {
    struct kv_cache_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_layers = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.block_size = 4;
    cfg.max_seq_len = 8;
    cfg.quant = KV_Q8_0;

    kv_cache_t *c = kv_cache_create(&cfg);
    assert(c && "kv_cache_create failed");

    const uint32_t vec_dim = cfg.n_kv_heads * cfg.head_dim;
    float k[8];
    float v[8];
    for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t i = 0; i < vec_dim; ++i) {
            k[i] = (float)(t * 10 + (int)i) * 0.1f;
            v[i] = (float)(t * 10 + (int)i) * -0.1f;
        }
        assert(kv_cache_append(c, 0, t, k, v) == 0);
    }

    float k_out[4 * 8];
    float v_out[4 * 8];
    assert(kv_cache_read_block(c, 0, 0, k_out, v_out) == 0);

    for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t i = 0; i < vec_dim; ++i) {
            float k_exp = (float)(t * 10 + (int)i) * 0.1f;
            float v_exp = (float)(t * 10 + (int)i) * -0.1f;
            float k_got = k_out[t * vec_dim + i];
            float v_got = v_out[t * vec_dim + i];
            assert(approx_eq(k_got, k_exp, 0.05f));
            assert(approx_eq(v_got, v_exp, 0.05f));
        }
    }

    assert(kv_cache_get_seq_len(c, 0) == 4);
    kv_cache_clear(c);
    assert(kv_cache_get_seq_len(c, 0) == 0);

    kv_cache_destroy(c);
    printf("PASS\n");
    return 0;
}

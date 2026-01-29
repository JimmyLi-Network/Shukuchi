#pragma once

#include <stdint.h>

void *metal_ops_init(const char *shader_path);
void metal_ops_shutdown(void *ctx);
int metal_matmul_q4k_vec(void *ctx,
                         const void *weights,
                         const float *x,
                         float *y,
                         uint32_t m,
                         uint32_t k);
int metal_matmul_q5k_vec(void *ctx,
                         const void *weights,
                         const float *x,
                         float *y,
                         uint32_t m,
                         uint32_t k);
int metal_matmul_q6k_vec(void *ctx,
                         const void *weights,
                         const float *x,
                         float *y,
                         uint32_t m,
                         uint32_t k);
int metal_available(void);
void metal_ops_report(void);

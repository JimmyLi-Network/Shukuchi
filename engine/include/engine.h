#pragma once

#include <stddef.h>
#include "model_loader.h"
#include <stdint.h>

typedef struct engine_handle engine_handle_t;

typedef void (*token_callback)(uint32_t token_id, const char *text, void *user);

struct engine_config {
    uint32_t n_threads;
    uint32_t batch_size;
    uint32_t prefetch_depth;  // 2 or 3
    uint32_t kv_block_size;
    uint32_t kv_quant;        // 0=Q8_0, 1=Q4_0
    int use_mmap;
};

engine_handle_t *engine_open(const char *model_path, const struct engine_config *cfg);
int engine_set_prompt(engine_handle_t *h, const char *prompt);
int engine_generate(engine_handle_t *h, uint32_t max_tokens);
int engine_generate_stream(engine_handle_t *h, uint32_t max_tokens,
                           token_callback cb, void *user);
const char *engine_get_output(engine_handle_t *h);
const uint32_t *engine_get_tokens(engine_handle_t *h, size_t *n_tokens);
void engine_cancel(engine_handle_t *h);
void engine_close(engine_handle_t *h);

int engine_get_streaming_stats(engine_handle_t *h, struct streaming_stats *out);

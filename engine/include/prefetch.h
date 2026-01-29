#pragma once

#include <stdint.h>
#include <stddef.h>

#include "model_loader.h"

enum buffer_state {
    BUF_EMPTY = 0,
    BUF_LOADING = 1,
    BUF_READY = 2,
    BUF_IN_USE = 3,
    BUF_ERROR = 4,
};

struct layer_buffer {
    volatile uint32_t state;
    uint32_t layer_id;
    void *data;
    size_t size;
    size_t capacity;
    struct layer_view view;
};

struct prefetcher_config {
    uint32_t depth;
    model_handle_t *model;
    size_t buffer_size;
    struct streaming_stats *stats;
};

typedef struct prefetcher prefetcher_t;
typedef struct prefetch_request prefetch_request_t;

struct prefetch_metrics {
    uint64_t total_bytes_read;
    uint64_t total_read_time_us;
    uint64_t cache_hits;
    uint64_t cache_misses;
};

prefetcher_t *prefetcher_create(const struct prefetcher_config *cfg);
int prefetcher_start(prefetcher_t *p);
prefetch_request_t *prefetcher_request(prefetcher_t *p, uint32_t layer_id);
struct layer_buffer *prefetcher_wait(prefetch_request_t *req);
void prefetcher_release(prefetcher_t *p, struct layer_buffer *buf);
void prefetcher_cancel(prefetcher_t *p);
int prefetcher_get_metrics(prefetcher_t *p, struct prefetch_metrics *out);
void prefetcher_stop(prefetcher_t *p);

#include "prefetch.h"

#include "model_loader.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>

struct prefetch_request {
    prefetcher_t *p;
    uint32_t buf_index;
};

struct prefetcher {
    struct prefetcher_config cfg;
    struct layer_buffer *buffers;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int running;
    int cancel;
    struct prefetch_metrics metrics;
    struct streaming_stats *stats;
};

static uint32_t count_active_buffers(prefetcher_t *p) {
    uint32_t concurrent = 0;
    for (uint32_t i = 0; i < p->cfg.depth; ++i) {
        uint32_t st = p->buffers[i].state;
        if (st == BUF_LOADING || st == BUF_IN_USE) {
            concurrent++;
        }
    }
    return concurrent;
}

static void *prefetch_thread_main(void *arg) {
    prefetcher_t *p = (prefetcher_t *)arg;
    while (1) {
        pthread_mutex_lock(&p->mu);
        while (!p->cancel) {
            int found = 0;
            for (uint32_t i = 0; i < p->cfg.depth; ++i) {
                if (p->buffers[i].state == BUF_LOADING) {
                    found = 1;
                    break;
                }
            }
            if (found) {
                break;
            }
            pthread_cond_wait(&p->cv, &p->mu);
        }
        if (p->cancel) {
            pthread_mutex_unlock(&p->mu);
            break;
        }
        pthread_mutex_unlock(&p->mu);

        for (uint32_t i = 0; i < p->cfg.depth; ++i) {
            struct layer_buffer *buf = &p->buffers[i];
            pthread_mutex_lock(&p->mu);
            if (buf->state != BUF_LOADING) {
                pthread_mutex_unlock(&p->mu);
                continue;
            }
            uint32_t layer_id = buf->layer_id;
            pthread_mutex_unlock(&p->mu);

            int ok = model_load_layer(p->cfg.model, layer_id, buf->data, buf->capacity, &buf->view, &buf->size);

            pthread_mutex_lock(&p->mu);
            if (ok != 0) {
                fprintf(stderr, "prefetch: load failed for layer %u\n", layer_id);
                buf->state = BUF_ERROR;
            } else {
                buf->state = BUF_READY;
            }
            pthread_mutex_unlock(&p->mu);
            pthread_cond_broadcast(&p->cv);
        }
    }
    return NULL;
}

prefetcher_t *prefetcher_create(const struct prefetcher_config *cfg) {
    if (!cfg || cfg->depth == 0 || !cfg->model) {
        return NULL;
    }
    prefetcher_t *p = (prefetcher_t *)calloc(1, sizeof(*p));
    if (!p) {
        return NULL;
    }
    p->cfg = *cfg;
    p->buffers = (struct layer_buffer *)calloc(cfg->depth, sizeof(*p->buffers));
    if (!p->buffers) {
        free(p);
        return NULL;
    }
    size_t buf_size = cfg->buffer_size;
    if (buf_size == 0) {
        if (model_get_max_layer_size(cfg->model, &buf_size) != 0 || buf_size == 0) {
            free(p->buffers);
            free(p);
            return NULL;
        }
    }
    for (uint32_t i = 0; i < cfg->depth; ++i) {
        p->buffers[i].state = BUF_EMPTY;
        p->buffers[i].layer_id = 0;
        p->buffers[i].data = malloc(buf_size);
        p->buffers[i].size = 0;
        p->buffers[i].capacity = buf_size;
        memset(&p->buffers[i].view, 0, sizeof(p->buffers[i].view));
        if (!p->buffers[i].data) {
            for (uint32_t j = 0; j < i; ++j) {
                free(p->buffers[j].data);
            }
            free(p->buffers);
            free(p);
            return NULL;
        }
    }
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    p->running = 0;
    p->cancel = 0;
    memset(&p->metrics, 0, sizeof(p->metrics));
    p->stats = cfg->stats;
    if (p->stats) {
        size_t total_buf = (size_t)cfg->depth * buf_size;
        if (total_buf > p->stats->peak_buffer_usage) {
            p->stats->peak_buffer_usage = total_buf;
        }
        if (buf_size > p->stats->max_layer_size) {
            p->stats->max_layer_size = buf_size;
        }
    }
    return p;
}

int prefetcher_start(prefetcher_t *p) {
    if (!p || p->running) {
        return -1;
    }
    p->cancel = 0;
    if (pthread_create(&p->thread, NULL, prefetch_thread_main, p) != 0) {
        return -1;
    }
    p->running = 1;
    return 0;
}

prefetch_request_t *prefetcher_request(prefetcher_t *p, uint32_t layer_id) {
    if (!p || !p->buffers) {
        return NULL;
    }
    pthread_mutex_lock(&p->mu);
    uint32_t idx = p->cfg.depth;
    for (uint32_t i = 0; i < p->cfg.depth; ++i) {
        if (p->buffers[i].state == BUF_EMPTY) {
            idx = i;
            break;
        }
    }
    if (idx == p->cfg.depth) {
        pthread_mutex_unlock(&p->mu);
        return NULL;
    }
    p->buffers[idx].layer_id = layer_id;
    p->buffers[idx].state = BUF_LOADING;
    if (p->stats) {
        uint32_t concurrent = count_active_buffers(p);
        if (concurrent > p->stats->max_concurrent_buffers) {
            p->stats->max_concurrent_buffers = concurrent;
        }
    }
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);

    prefetch_request_t *req = (prefetch_request_t *)malloc(sizeof(*req));
    if (!req) {
        return NULL;
    }
    req->p = p;
    req->buf_index = idx;
    return req;
}

struct layer_buffer *prefetcher_wait(prefetch_request_t *req) {
    if (!req) {
        return NULL;
    }
    prefetcher_t *p = req->p;
    uint32_t idx = req->buf_index;
    free(req);
    if (!p || !p->buffers || idx >= p->cfg.depth) {
        return NULL;
    }
    int waited = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        uint32_t state = p->buffers[idx].state;
        if (state == BUF_READY) {
            p->buffers[idx].state = BUF_IN_USE;
            if (p->stats) {
                if (waited) {
                    p->stats->prefetch_misses += 1;
                } else {
                    p->stats->prefetch_hits += 1;
                }
                uint32_t concurrent = count_active_buffers(p);
                if (concurrent > p->stats->max_concurrent_buffers) {
                    p->stats->max_concurrent_buffers = concurrent;
                }
            }
            struct layer_buffer *buf = &p->buffers[idx];
            pthread_mutex_unlock(&p->mu);
            return buf;
        }
        if (state == BUF_ERROR || p->cancel) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        waited = 1;
        pthread_mutex_unlock(&p->mu);
        usleep(1000);
    }
}

void prefetcher_release(prefetcher_t *p, struct layer_buffer *buf) {
    if (!p || !buf) {
        return;
    }
    pthread_mutex_lock(&p->mu);
    buf->state = BUF_EMPTY;
    buf->layer_id = 0;
    buf->size = 0;
    memset(&buf->view, 0, sizeof(buf->view));
    pthread_mutex_unlock(&p->mu);
}

void prefetcher_cancel(prefetcher_t *p) {
    if (!p) {
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->cancel = 1;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}

int prefetcher_get_metrics(prefetcher_t *p, struct prefetch_metrics *out) {
    if (!p || !out) {
        return -1;
    }
    *out = p->metrics;
    return 0;
}

void prefetcher_stop(prefetcher_t *p) {
    if (!p) {
        return;
    }
    prefetcher_cancel(p);
    if (p->running) {
        pthread_join(p->thread, NULL);
    }
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    if (p->buffers) {
        for (uint32_t i = 0; i < p->cfg.depth; ++i) {
            free(p->buffers[i].data);
        }
        free(p->buffers);
    }
    free(p);
}

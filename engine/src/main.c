#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <stdint.h>
#include "engine.h"
#if defined(__APPLE__)
#include "metal_ops.h"
#endif

static void print_usage(const char *argv0) {
    fprintf(stderr, "Usage: %s <model.lstr> [--prompt \"...\"] [--max-tokens N]\n", argv0);
}

static void print_peak_rss(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return;
    }
#if defined(__APPLE__)
    double bytes = (double)ru.ru_maxrss;
#else
    double bytes = (double)ru.ru_maxrss * 1024.0;
#endif
    fprintf(stderr, "peak_rss_mb=%.2f\n", bytes / (1024.0 * 1024.0));
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int enable_rss = 0;
#if defined(__APPLE__)
    enable_rss = 1;
    const char *env = getenv("SHUKUCHI_METAL");
    int metal_disabled = (env && (env[0] == '0' || env[0] == 'f' || env[0] == 'F'));
    if (!metal_disabled && metal_available()) {
        fprintf(stderr, "Metal enabled\n");
    } else {
        fprintf(stderr, "Metal unavailable\n");
    }
#endif

    uint32_t max_tokens = 16;
    const char *prompt = NULL;
    const char *prefetch_env = getenv("SHUKUCHI_PREFETCH_DEPTH");
    uint32_t prefetch_depth = 3;
    if (prefetch_env && prefetch_env[0] != '\0') {
        prefetch_depth = (uint32_t)strtoul(prefetch_env, NULL, 10);
        if (prefetch_depth == 0) {
            prefetch_depth = 2;
        }
    }
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = (uint32_t)strtoul(argv[i + 1], NULL, 10);
            i++;
            continue;
        }
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[i + 1];
            i++;
            continue;
        }
    }

    {
        struct engine_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.n_threads = 4;
        cfg.batch_size = 1;
        cfg.prefetch_depth = prefetch_depth;
        cfg.kv_block_size = 32;
        cfg.kv_quant = 0;
        cfg.use_mmap = 0;

        engine_handle_t *h = engine_open(argv[1], &cfg);
        if (!h) {
            fprintf(stderr, "engine: failed to open model\n");
            return 1;
        }
        if (prompt) {
            engine_set_prompt(h, prompt);
        }
        engine_generate(h, max_tokens);
        struct streaming_stats stats;
        if (engine_get_streaming_stats(h, &stats) == 0) {
            fprintf(stderr, "streaming_stats: layer_loads=%llu layer_bytes_read=%llu max_layer_size=%zu peak_buffer_usage=%zu peak_rss=%zu max_concurrent_buffers=%u prefetch_hits=%u prefetch_misses=%u\n",
                    (unsigned long long)stats.layer_loads,
                    (unsigned long long)stats.layer_bytes_read,
                    stats.max_layer_size,
                    stats.peak_buffer_usage,
                    stats.peak_rss,
                    stats.max_concurrent_buffers,
                    stats.prefetch_hits,
                    stats.prefetch_misses);
        }
        engine_close(h);
    }
    if (enable_rss) {
        print_peak_rss();
    }
#if defined(__APPLE__)
    metal_ops_report();
#endif
    return 0;
}

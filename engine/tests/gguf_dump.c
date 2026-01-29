#include <stdio.h>
#include <stdint.h>

#include "gguf_reader.h"

static void print_kv_u32(gguf_file_t *f, const char *key) {
    struct gguf_kv_pair kv;
    if (gguf_find_kv(f, key, &kv) != 0) {
        printf("%s: (missing)\n", key);
        return;
    }
    if (kv.type != GGUF_KV_UINT32) {
        printf("%s: (type %u)\n", key, kv.type);
        return;
    }
    printf("%s: %u\n", key, *(const uint32_t *)kv.value);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    gguf_file_t *f = gguf_open(argv[1], 0);
    if (!f) {
        fprintf(stderr, "gguf_open failed\n");
        return 1;
    }
    if (gguf_read_header(f) != 0) {
        fprintf(stderr, "gguf_read_header failed\n");
        gguf_close(f);
        return 1;
    }

    print_kv_u32(f, "llama.context_length");
    print_kv_u32(f, "llama.embedding_length");
    print_kv_u32(f, "llama.block_count");
    print_kv_u32(f, "llama.attention.head_count");
    print_kv_u32(f, "llama.attention.head_count_kv");

    int64_t n_tensors = gguf_get_n_tensors(f);
    printf("tensors: %lld\n", (long long)n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        gguf_tensor_t t;
        if (gguf_get_tensor(f, i, &t) == 0) {
            printf("[%lld] %s dtype=%u offset=%llu size=%llu\n",
                   (long long)i, t.name, t.dtype,
                   (unsigned long long)t.offset,
                   (unsigned long long)t.size);
        }
    }

    gguf_close(f);
    return 0;
}

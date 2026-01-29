#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct gguf_file gguf_file_t;
typedef struct gguf_tensor gguf_tensor_t;

enum gguf_dtype {
    GGUF_F16 = 1,
    GGUF_F32 = 2,
    GGUF_Q8_0 = 10,
    GGUF_Q4_0 = 11,
};

enum gguf_kv_type {
    GGUF_KV_UINT8   = 0,
    GGUF_KV_INT8    = 1,
    GGUF_KV_UINT16  = 2,
    GGUF_KV_INT16   = 3,
    GGUF_KV_UINT32  = 4,
    GGUF_KV_INT32   = 5,
    GGUF_KV_FLOAT32 = 6,
    GGUF_KV_BOOL    = 7,
    GGUF_KV_STRING  = 8,
    GGUF_KV_ARRAY   = 9,
    GGUF_KV_UINT64  = 10,
    GGUF_KV_INT64   = 11,
    GGUF_KV_FLOAT64 = 12,
};

struct gguf_array {
    uint32_t type;         // gguf_type for array elements
    uint64_t n;            // number of elements
    const void *data;      // raw data for non-string arrays
    const char * const *strs; // string array (if type == GGUF_TYPE_STRING)
};

struct gguf_kv_pair {
    const char *key;
    const void *value;
    uint32_t type;
};

struct gguf_tensor {
    const char *name;
    uint32_t dtype;  // ggml_type numeric id
    uint64_t offset;
    uint64_t size;
};

gguf_file_t *gguf_open(const char *path, int use_mmap);
void gguf_close(gguf_file_t *f);
int gguf_read_header(gguf_file_t *f);
int gguf_find_kv(gguf_file_t *f, const char *key, struct gguf_kv_pair *out);
int gguf_find_tensor(gguf_file_t *f, const char *name, gguf_tensor_t *out);
const void *gguf_map_tensor(gguf_file_t *f, const gguf_tensor_t *t);
int gguf_read_tensor_data(gguf_file_t *f, const gguf_tensor_t *t, void *dst, size_t dst_size);
int gguf_read_span(gguf_file_t *f, uint64_t offset, uint64_t size, void *dst);

int64_t gguf_get_n_tensors(gguf_file_t *f);
int gguf_get_tensor(gguf_file_t *f, int64_t idx, gguf_tensor_t *out);

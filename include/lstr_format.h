#pragma once

#include <stdint.h>

// On-disk container format definitions (little-endian).

#define LSTR_MAGIC 0x5254534cU  // "LSTR"
#define LSTR_VERSION 1
#define LSTR_HEADER_SIZE 256

enum lstr_format_flags {
    LSTR_FLAG_HAS_CHECKSUMS = 1u << 0,
    LSTR_FLAG_HAS_TENSOR_TABLE = 1u << 1,
    LSTR_FLAG_LAYER_ALIGN_4K = 1u << 2,
    LSTR_FLAG_LAYER_ALIGN_2M = 1u << 3,
};

enum lstr_dtype {
    LSTR_DTYPE_F16 = 1,
    LSTR_DTYPE_F32 = 2,
    LSTR_DTYPE_Q8_0 = 10,
    LSTR_DTYPE_Q4_0 = 11,
};

struct lstr_header {
    uint32_t magic;              // LSTR_MAGIC
    uint32_t endian_tag;         // 0x01020304 for endianness check
    uint32_t version;            // format version
    uint32_t header_size;         // == LSTR_HEADER_SIZE
    uint32_t min_loader_version;  // minimum loader version

    uint32_t format_flags;        // lstr_format_flags bitmask
    uint32_t n_layers;
    uint32_t n_vocab;
    uint32_t n_embd;

    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t ctx_size;
    uint32_t rope_type;           // 0 = none, 1 = RoPE

    float rope_theta;
    float rope_scale;             // optional, 0 if unused
    uint32_t reserved0;
    uint32_t reserved1;

    uint64_t index_offset;
    uint64_t index_size;
    uint64_t resident_offset;
    uint64_t resident_size;

    uint64_t layers_offset;       // first layer offset
    uint64_t file_size;           // total file size

    char model_name[64];          // optional
    char build_tag[64];           // optional (git hash, etc.)

    uint8_t padding[LSTR_HEADER_SIZE - 4 * 11 - 8 * 6 - 64 - 64];
};

_Static_assert(sizeof(struct lstr_header) == LSTR_HEADER_SIZE,
               "lstr_header size mismatch");

struct lstr_layer_index_entry {
    uint32_t layer_id;
    uint32_t n_tensors;
    uint64_t offset;              // layer block offset (aligned)
    uint64_t size;                // total layer block size
    uint32_t dtype;               // lstr_dtype
    uint32_t tensor_table_offset; // bytes from layer base, 0 if none
    uint64_t checksum;            // optional xxhash64 (0 if unused)
};

struct lstr_tensor_entry {
    uint32_t tensor_id;            // model-specific id
    uint32_t dtype;                // lstr_dtype
    uint64_t offset;               // bytes from layer base
    uint64_t size;                 // bytes
};

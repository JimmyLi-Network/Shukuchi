#include "gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC "GGUF"
#define GGUF_DEFAULT_ALIGNMENT 32

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct gguf_kv_internal {
    char *key;
    uint32_t type;
    int is_array;
    struct gguf_array arr;
    void *value;
};

struct gguf_tensor_internal {
    char *name;
    uint32_t dtype;
    uint32_t n_dims;
    int64_t *dims;
    uint64_t offset;
    uint64_t size;
};

struct gguf_file {
    FILE *fp;
    int fd;
    int use_mmap;
    uint32_t version;
    int64_t n_tensors;
    int64_t n_kv;
    uint64_t data_start;
    uint64_t file_size;
    uint32_t alignment;

    struct gguf_kv_internal *kvs;
    struct gguf_tensor_internal *tensors;

    void *map_base;
    uint64_t map_size;
    void *map_buf;
    uint64_t map_buf_size;
};

static uint64_t file_size_bytes(FILE *fp) {
    long cur = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long end = ftell(fp);
    fseek(fp, cur, SEEK_SET);
    return (uint64_t)end;
}

static int read_u32(FILE *fp, uint32_t *out) {
    return fread(out, sizeof(*out), 1, fp) == 1 ? 0 : -1;
}

static int read_i64(FILE *fp, int64_t *out) {
    return fread(out, sizeof(*out), 1, fp) == 1 ? 0 : -1;
}

static int read_u64(FILE *fp, uint64_t *out) {
    return fread(out, sizeof(*out), 1, fp) == 1 ? 0 : -1;
}

static char *read_string(FILE *fp) {
    uint64_t len = 0;
    if (read_u64(fp, &len) != 0) {
        return NULL;
    }
    if (len > (uint64_t)(1024 * 1024)) {
        return NULL;
    }
    size_t nbytes = (size_t)len;
    char *s = (char *)malloc(nbytes + 1);
    if (!s) {
        return NULL;
    }
    if (nbytes > 0 && fread(s, 1, nbytes, fp) != nbytes) {
        free(s);
        return NULL;
    }
    s[nbytes] = '\0';
    return s;
}

static size_t gguf_type_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return sizeof(uint8_t);
        case GGUF_TYPE_INT8:    return sizeof(int8_t);
        case GGUF_TYPE_UINT16:  return sizeof(uint16_t);
        case GGUF_TYPE_INT16:   return sizeof(int16_t);
        case GGUF_TYPE_UINT32:  return sizeof(uint32_t);
        case GGUF_TYPE_INT32:   return sizeof(int32_t);
        case GGUF_TYPE_FLOAT32: return sizeof(float);
        case GGUF_TYPE_BOOL:    return sizeof(int8_t);
        case GGUF_TYPE_UINT64:  return sizeof(uint64_t);
        case GGUF_TYPE_INT64:   return sizeof(int64_t);
        case GGUF_TYPE_FLOAT64: return sizeof(double);
        default: return 0;
    }
}

static uint64_t align_up(uint64_t x, uint32_t a) {
    uint64_t mask = (uint64_t)a - 1;
    return (x + mask) & ~mask;
}

gguf_file_t *gguf_open(const char *path, int use_mmap) {
    gguf_file_t *f = (gguf_file_t *)calloc(1, sizeof(*f));
    if (!f) {
        return NULL;
    }
    f->fp = fopen(path, "rb");
    if (!f->fp) {
        free(f);
        return NULL;
    }
    f->fd = fileno(f->fp);
    f->use_mmap = use_mmap;
    f->file_size = file_size_bytes(f->fp);

    if (f->use_mmap && f->file_size > 0) {
        void *base = mmap(NULL, (size_t)f->file_size, PROT_READ, MAP_PRIVATE, f->fd, 0);
        if (base == MAP_FAILED) {
            f->use_mmap = 0;
        } else {
            f->map_base = base;
            f->map_size = f->file_size;
        }
    }
    return f;
}

void gguf_close(gguf_file_t *f) {
    if (!f) {
        return;
    }
    if (f->map_base) {
        munmap(f->map_base, (size_t)f->map_size);
    }
    if (f->fp) {
        fclose(f->fp);
    }
    if (f->kvs) {
        for (int64_t i = 0; i < f->n_kv; ++i) {
            free(f->kvs[i].key);
            if (f->kvs[i].is_array) {
                if (f->kvs[i].arr.type == GGUF_TYPE_STRING && f->kvs[i].arr.strs) {
                    for (uint64_t j = 0; j < f->kvs[i].arr.n; ++j) {
                        free((void *)f->kvs[i].arr.strs[j]);
                    }
                    free((void *)f->kvs[i].arr.strs);
                } else {
                    free((void *)f->kvs[i].arr.data);
                }
            } else if (f->kvs[i].type == GGUF_TYPE_STRING) {
                free(f->kvs[i].value);
            } else {
                free(f->kvs[i].value);
            }
        }
        free(f->kvs);
    }
    if (f->tensors) {
        for (int64_t i = 0; i < f->n_tensors; ++i) {
            free(f->tensors[i].name);
            free(f->tensors[i].dims);
        }
        free(f->tensors);
    }
    free(f->map_buf);
    free(f);
}

int gguf_read_header(gguf_file_t *f) {
    if (!f || !f->fp) {
        return -1;
    }

    char magic[4];
    if (fread(magic, 1, 4, f->fp) != 4) {
        return -1;
    }
    if (memcmp(magic, GGUF_MAGIC, 4) != 0) {
        return -1;
    }

    uint32_t version = 0;
    if (read_u32(f->fp, &version) != 0) {
        return -1;
    }
    f->version = version;

    if (read_i64(f->fp, &f->n_tensors) != 0) {
        return -1;
    }
    if (read_i64(f->fp, &f->n_kv) != 0) {
        return -1;
    }

    if (f->n_kv < 0 || f->n_tensors < 0) {
        return -1;
    }

    f->alignment = GGUF_DEFAULT_ALIGNMENT;
    f->kvs = (struct gguf_kv_internal *)calloc((size_t)f->n_kv, sizeof(*f->kvs));
    if (!f->kvs && f->n_kv > 0) {
        return -1;
    }

    for (int64_t i = 0; i < f->n_kv; ++i) {
        char *key = read_string(f->fp);
        if (!key) {
            return -1;
        }
        uint32_t type = 0;
        if (read_u32(f->fp, &type) != 0) {
            free(key);
            return -1;
        }

        f->kvs[i].key = key;
        f->kvs[i].type = type;
        f->kvs[i].is_array = (type == GGUF_TYPE_ARRAY);

        if (type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type = 0;
            uint64_t n = 0;
            if (read_u32(f->fp, &arr_type) != 0 || read_u64(f->fp, &n) != 0) {
                return -1;
            }
            f->kvs[i].arr.type = arr_type;
            f->kvs[i].arr.n = n;

            if (arr_type == GGUF_TYPE_STRING) {
                char **strs = (char **)calloc((size_t)n, sizeof(char *));
                if (!strs && n > 0) {
                    return -1;
                }
                for (uint64_t j = 0; j < n; ++j) {
                    strs[j] = read_string(f->fp);
                    if (!strs[j]) {
                        return -1;
                    }
                }
                f->kvs[i].arr.strs = (const char * const *)strs;
                f->kvs[i].arr.data = NULL;
            } else {
                size_t esz = gguf_type_size(arr_type);
                if (esz == 0) {
                    return -1;
                }
                size_t total = (size_t)n * esz;
                void *buf = NULL;
                if (total > 0) {
                    buf = malloc(total);
                    if (!buf) {
                        return -1;
                    }
                    if (fread(buf, 1, total, f->fp) != total) {
                        free(buf);
                        return -1;
                    }
                }
                f->kvs[i].arr.data = buf;
                f->kvs[i].arr.strs = NULL;
            }
        } else if (type == GGUF_TYPE_STRING) {
            char *val = read_string(f->fp);
            if (!val) {
                return -1;
            }
            f->kvs[i].value = val;
        } else {
            size_t esz = gguf_type_size(type);
            if (esz == 0) {
                return -1;
            }
            void *val = malloc(esz);
            if (!val) {
                return -1;
            }
            if (fread(val, 1, esz, f->fp) != esz) {
                free(val);
                return -1;
            }
            f->kvs[i].value = val;
        }
    }

    // Parse tensors
    f->tensors = (struct gguf_tensor_internal *)calloc((size_t)f->n_tensors, sizeof(*f->tensors));
    if (!f->tensors && f->n_tensors > 0) {
        return -1;
    }

    for (int64_t i = 0; i < f->n_tensors; ++i) {
        char *name = read_string(f->fp);
        if (!name) {
            return -1;
        }
        uint32_t n_dims = 0;
        if (read_u32(f->fp, &n_dims) != 0) {
            free(name);
            return -1;
        }
        int64_t *dims = NULL;
        if (n_dims > 0) {
            dims = (int64_t *)calloc(n_dims, sizeof(int64_t));
            if (!dims) {
                free(name);
                return -1;
            }
            for (uint32_t d = 0; d < n_dims; ++d) {
                if (read_i64(f->fp, &dims[d]) != 0) {
                    free(name);
                    free(dims);
                    return -1;
                }
            }
        }
        uint32_t ttype = 0;
        if (read_u32(f->fp, &ttype) != 0) {
            free(name);
            free(dims);
            return -1;
        }
        uint64_t offset = 0;
        if (read_u64(f->fp, &offset) != 0) {
            free(name);
            free(dims);
            return -1;
        }
        f->tensors[i].name = name;
        f->tensors[i].n_dims = n_dims;
        f->tensors[i].dims = dims;
        f->tensors[i].dtype = ttype;
        f->tensors[i].offset = offset;
    }

    // Resolve alignment from kv if present
    for (int64_t i = 0; i < f->n_kv; ++i) {
        if (strcmp(f->kvs[i].key, "general.alignment") == 0 &&
            f->kvs[i].type == GGUF_TYPE_UINT32) {
            f->alignment = *(const uint32_t *)f->kvs[i].value;
            break;
        }
    }

    uint64_t meta_end = (uint64_t)ftell(f->fp);
    f->data_start = align_up(meta_end, f->alignment);

    // Compute tensor sizes from offsets
    for (int64_t i = 0; i < f->n_tensors; ++i) {
        uint64_t cur = f->tensors[i].offset;
        uint64_t next = 0;
        if (i + 1 < f->n_tensors) {
            next = f->tensors[i + 1].offset;
        } else {
            uint64_t data_size = (f->file_size > f->data_start)
                ? (f->file_size - f->data_start) : 0;
            next = data_size;
        }
        if (next < cur) {
            return -1;
        }
        f->tensors[i].size = next - cur;
    }

    return 0;
}

int gguf_find_kv(gguf_file_t *f, const char *key, struct gguf_kv_pair *out) {
    if (!f || !key || !out) {
        return -1;
    }
    for (int64_t i = 0; i < f->n_kv; ++i) {
        if (strcmp(f->kvs[i].key, key) == 0) {
            out->key = f->kvs[i].key;
            if (f->kvs[i].is_array) {
                out->type = GGUF_TYPE_ARRAY;
                out->value = &f->kvs[i].arr;
            } else {
                out->type = f->kvs[i].type;
                out->value = f->kvs[i].value;
            }
            return 0;
        }
    }
    return -1;
}

int gguf_find_tensor(gguf_file_t *f, const char *name, gguf_tensor_t *out) {
    if (!f || !name || !out) {
        return -1;
    }
    for (int64_t i = 0; i < f->n_tensors; ++i) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            out->name = f->tensors[i].name;
            out->dtype = f->tensors[i].dtype;
            out->offset = f->tensors[i].offset;
            out->size = f->tensors[i].size;
            return 0;
        }
    }
    return -1;
}

const void *gguf_map_tensor(gguf_file_t *f, const gguf_tensor_t *t) {
    if (!f || !t || !f->fp) {
        return NULL;
    }
    if (t->size == 0) {
        return NULL;
    }
    uint64_t file_off = f->data_start + t->offset;
    if (f->use_mmap && f->map_base && file_off + t->size <= f->map_size) {
        return (const uint8_t *)f->map_base + file_off;
    }
    if (f->map_buf && f->map_buf_size < t->size) {
        free(f->map_buf);
        f->map_buf = NULL;
        f->map_buf_size = 0;
    }
    if (!f->map_buf && t->size > 0) {
        f->map_buf = malloc((size_t)t->size);
        if (!f->map_buf) {
            return NULL;
        }
        f->map_buf_size = t->size;
    }
    if (fseek(f->fp, (long)file_off, SEEK_SET) != 0) {
        return NULL;
    }
    if (fread(f->map_buf, 1, (size_t)t->size, f->fp) != t->size) {
        return NULL;
    }
    return f->map_buf;
}

int gguf_read_tensor_data(gguf_file_t *f, const gguf_tensor_t *t, void *dst, size_t dst_size) {
    if (!f || !t || !dst || !f->fp) {
        return -1;
    }
    if (t->size == 0 || t->size > dst_size) {
        return -1;
    }
    uint64_t file_off = f->data_start + t->offset;
    if (f->use_mmap && f->map_base && file_off + t->size <= f->map_size) {
        memcpy(dst, (const uint8_t *)f->map_base + file_off, (size_t)t->size);
        return 0;
    }
    size_t remaining = (size_t)t->size;
    uint8_t *out = (uint8_t *)dst;
    while (remaining > 0) {
        ssize_t n = pread(f->fd, out, remaining, (off_t)file_off);
        if (n <= 0) {
            return -1;
        }
        out += (size_t)n;
        file_off += (uint64_t)n;
        remaining -= (size_t)n;
    }
    return 0;
}

int gguf_read_span(gguf_file_t *f, uint64_t offset, uint64_t size, void *dst) {
    if (!f || !dst || size == 0 || !f->fp) {
        return -1;
    }
    uint64_t file_off = f->data_start + offset;
    if (f->use_mmap && f->map_base && file_off + size <= f->map_size) {
        memcpy(dst, (const uint8_t *)f->map_base + file_off, (size_t)size);
        return 0;
    }
    size_t remaining = (size_t)size;
    uint8_t *out = (uint8_t *)dst;
    while (remaining > 0) {
        ssize_t n = pread(f->fd, out, remaining, (off_t)file_off);
        if (n <= 0) {
            return -1;
        }
        out += (size_t)n;
        file_off += (uint64_t)n;
        remaining -= (size_t)n;
    }
    return 0;
}

int64_t gguf_get_n_tensors(gguf_file_t *f) {
    return f ? f->n_tensors : 0;
}

int gguf_get_tensor(gguf_file_t *f, int64_t idx, gguf_tensor_t *out) {
    if (!f || !out || idx < 0 || idx >= f->n_tensors) {
        return -1;
    }
    out->name = f->tensors[idx].name;
    out->dtype = f->tensors[idx].dtype;
    out->offset = f->tensors[idx].offset;
    out->size = f->tensors[idx].size;
    return 0;
}

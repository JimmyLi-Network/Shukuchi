#include "gguf_reader.h"
#include "model_loader.h"

#include <stdlib.h>
#include <string.h>

enum llama_tensor_field {
    LLAMA_TF_UNKNOWN = 0,
    LLAMA_TF_ATTN_NORM,
    LLAMA_TF_ATTN_Q,
    LLAMA_TF_ATTN_K,
    LLAMA_TF_ATTN_V,
    LLAMA_TF_ATTN_O,
    LLAMA_TF_FFN_NORM,
    LLAMA_TF_FFN_GATE,
    LLAMA_TF_FFN_UP,
    LLAMA_TF_FFN_DOWN,
};

static int get_kv_u32(gguf_file_t *f, const char *key, uint32_t *out) {
    struct gguf_kv_pair kv;
    if (gguf_find_kv(f, key, &kv) != 0) {
        return -1;
    }
    if (kv.type != GGUF_KV_UINT32) {
        return -1;
    }
    *out = *(const uint32_t *)kv.value;
    return 0;
}

int parse_layer_id(const char *tensor_name) {
    const char *p = tensor_name;
    if (!p || strncmp(p, "blk.", 4) != 0) {
        return -1;
    }
    p += 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int id = 0;
    while (*p >= '0' && *p <= '9') {
        id = id * 10 + (*p - '0');
        p++;
    }
    if (*p != '.') {
        return -1;
    }
    return id;
}

enum llama_tensor_field map_tensor_to_field(const char *tensor_name) {
    const char *p = tensor_name;
    if (!p || strncmp(p, "blk.", 4) != 0) {
        return LLAMA_TF_UNKNOWN;
    }
    p += 4;
    while (*p >= '0' && *p <= '9') {
        p++;
    }
    if (*p != '.') {
        return LLAMA_TF_UNKNOWN;
    }
    p++;

    if (strcmp(p, "attn_norm.weight") == 0) {
        return LLAMA_TF_ATTN_NORM;
    }
    if (strcmp(p, "attn_q.weight") == 0) {
        return LLAMA_TF_ATTN_Q;
    }
    if (strcmp(p, "attn_k.weight") == 0) {
        return LLAMA_TF_ATTN_K;
    }
    if (strcmp(p, "attn_v.weight") == 0) {
        return LLAMA_TF_ATTN_V;
    }
    if (strcmp(p, "attn_output.weight") == 0) {
        return LLAMA_TF_ATTN_O;
    }
    if (strcmp(p, "ffn_norm.weight") == 0) {
        return LLAMA_TF_FFN_NORM;
    }
    if (strcmp(p, "ffn_gate.weight") == 0) {
        return LLAMA_TF_FFN_GATE;
    }
    if (strcmp(p, "ffn_up.weight") == 0) {
        return LLAMA_TF_FFN_UP;
    }
    if (strcmp(p, "ffn_down.weight") == 0) {
        return LLAMA_TF_FFN_DOWN;
    }
    return LLAMA_TF_UNKNOWN;
}

static int map_tensor(gguf_file_t *f, const char *name, struct tensor_ref *out_ref) {
    gguf_tensor_t t;
    if (gguf_find_tensor(f, name, &t) != 0) {
        return -1;
    }
    if (out_ref) {
        out_ref->offset = t.offset;
        out_ref->size = t.size;
        out_ref->dtype = t.dtype;
    }
    return 0;
}

int build_layer_specs(gguf_file_t *f,
                      struct resident_spec *resident,
                      struct layer_spec **layers_out,
                      uint32_t *n_layers_out) {
    if (!f || !resident || !layers_out || !n_layers_out) {
        return -1;
    }

    uint32_t n_layers = 0;
    if (get_kv_u32(f, "llama.block_count", &n_layers) != 0 || n_layers == 0) {
        return -1;
    }

    struct layer_spec *layers = (struct layer_spec *)calloc(n_layers, sizeof(*layers));
    if (!layers) {
        return -1;
    }

    map_tensor(f, "token_embd.weight", &resident->token_embd);
    map_tensor(f, "output_norm.weight", &resident->output_norm);
    map_tensor(f, "output.weight", &resident->lm_head);

    uint32_t required_mask = (1u << LLAMA_TF_ATTN_NORM) |
                             (1u << LLAMA_TF_ATTN_Q) |
                             (1u << LLAMA_TF_ATTN_K) |
                             (1u << LLAMA_TF_ATTN_V) |
                             (1u << LLAMA_TF_ATTN_O) |
                             (1u << LLAMA_TF_FFN_NORM) |
                             (1u << LLAMA_TF_FFN_GATE) |
                             (1u << LLAMA_TF_FFN_UP) |
                             (1u << LLAMA_TF_FFN_DOWN);

    uint32_t *seen = (uint32_t *)calloc(n_layers, sizeof(uint32_t));
    if (!seen) {
        free(layers);
        return -1;
    }

    int64_t n_tensors = gguf_get_n_tensors(f);
    for (int64_t i = 0; i < n_tensors; ++i) {
        gguf_tensor_t t;
        if (gguf_get_tensor(f, i, &t) != 0 || !t.name) {
            continue;
        }
        int layer_id = parse_layer_id(t.name);
        if (layer_id < 0 || (uint32_t)layer_id >= n_layers) {
            continue;
        }
        enum llama_tensor_field field = map_tensor_to_field(t.name);
        if (field == LLAMA_TF_UNKNOWN) {
            continue;
        }
        struct layer_spec *lv = &layers[layer_id];
        switch (field) {
            case LLAMA_TF_ATTN_NORM:
                lv->attn_norm.offset = t.offset;
                lv->attn_norm.size = t.size;
                lv->attn_norm.dtype = t.dtype;
                break;
            case LLAMA_TF_ATTN_Q:
                lv->attn_q.offset = t.offset;
                lv->attn_q.size = t.size;
                lv->attn_q.dtype = t.dtype;
                break;
            case LLAMA_TF_ATTN_K:
                lv->attn_k.offset = t.offset;
                lv->attn_k.size = t.size;
                lv->attn_k.dtype = t.dtype;
                break;
            case LLAMA_TF_ATTN_V:
                lv->attn_v.offset = t.offset;
                lv->attn_v.size = t.size;
                lv->attn_v.dtype = t.dtype;
                break;
            case LLAMA_TF_ATTN_O:
                lv->attn_o.offset = t.offset;
                lv->attn_o.size = t.size;
                lv->attn_o.dtype = t.dtype;
                break;
            case LLAMA_TF_FFN_NORM:
                lv->ffn_norm.offset = t.offset;
                lv->ffn_norm.size = t.size;
                lv->ffn_norm.dtype = t.dtype;
                break;
            case LLAMA_TF_FFN_GATE:
                lv->ffn_gate.offset = t.offset;
                lv->ffn_gate.size = t.size;
                lv->ffn_gate.dtype = t.dtype;
                break;
            case LLAMA_TF_FFN_UP:
                lv->ffn_up.offset = t.offset;
                lv->ffn_up.size = t.size;
                lv->ffn_up.dtype = t.dtype;
                break;
            case LLAMA_TF_FFN_DOWN:
                lv->ffn_down.offset = t.offset;
                lv->ffn_down.size = t.size;
                lv->ffn_down.dtype = t.dtype;
                break;
            default: break;
        }
        seen[layer_id] |= (1u << field);
    }

    int ok = 0;
    for (uint32_t i = 0; i < n_layers; ++i) {
        if ((seen[i] & required_mask) != required_mask) {
            ok = -1;
            break;
        }
    }
    if (resident->token_embd.size == 0 || resident->output_norm.size == 0 || resident->lm_head.size == 0) {
        ok = -1;
    }

    free(seen);
    if (ok != 0) {
        free(layers);
        return -1;
    }

    *layers_out = layers;
    *n_layers_out = n_layers;
    return 0;
}

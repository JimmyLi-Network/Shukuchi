#pragma once

#include <stdint.h>

struct packer_config {
    int prefer_gguf;
    uint32_t alignment;
    int write_checksums;
};

int packer_run(const char *input_path, const char *output_path,
               const struct packer_config *cfg);

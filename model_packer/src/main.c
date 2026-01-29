#include <stdio.h>
#include <string.h>
#include "packer.h"

static void print_usage(const char *argv0) {
    fprintf(stderr, "Usage: %s <input> <output>\n", argv0);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    {
        struct packer_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.prefer_gguf = 1;
        cfg.alignment = 4096;
        cfg.write_checksums = 0;
        return packer_run(argv[1], argv[2], &cfg);
    }
}

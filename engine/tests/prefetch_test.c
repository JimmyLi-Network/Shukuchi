#include <stdio.h>
#include <assert.h>

#include "prefetch.h"
#include "model_loader.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    struct model_config cfg = {0};
    cfg.use_mmap = 0;
    model_handle_t *m = model_open(argv[1], &cfg);
    assert(m && "model_open failed");

    struct prefetcher_config pcfg = {0};
    pcfg.depth = 2;
    pcfg.model = m;
    prefetcher_t *p = prefetcher_create(&pcfg);
    assert(p && "prefetcher_create failed");

    prefetcher_start(p);

    uint32_t n_layers = model_get_layer_count(m);
    printf("n_layers = %u\n", n_layers);

    for (uint32_t i = 0; i < 3 && i < n_layers; i++) {
        printf("request layer %u\n", i);
        prefetch_request_t *req = prefetcher_request(p, i);

        printf("wait layer %u\n", i);
        struct layer_buffer *buf = prefetcher_wait(req);
        assert(buf && buf->state == BUF_IN_USE);
        assert(buf->data && buf->size > 0);
        assert(buf->view.attn_q && buf->view.attn_k && buf->view.attn_v);
        printf("got layer %u, data=%p size=%zu\n", i, buf->data, buf->size);

        prefetcher_release(p, buf);
        printf("released layer %u\n", i);
    }

    prefetcher_stop(p);
    model_close(m);
    printf("PASS\n");
    return 0;
}

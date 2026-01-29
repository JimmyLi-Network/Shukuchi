# engine

Skeleton for the streaming inference runtime.

Planned responsibilities:
- Open LSTR container, validate header/index.
- Optionally open GGUF and map tensors for compatibility.
- Load resident tensors and stream per-layer weights.
- Run per-token decode loop with prefetch scheduling.
- Maintain KV cache and report metrics.

Entry point: `engine/src/main.c`

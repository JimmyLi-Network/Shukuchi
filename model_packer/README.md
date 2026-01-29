# model_packer

Skeleton for the model container builder.

Planned responsibilities:
- Read source weights (GGUF first, other formats optional).
- Build LSTR container: header, index, resident section, layer blocks.
- Enforce alignment and write checksums when enabled.

Entry point: `model_packer/src/main.c`

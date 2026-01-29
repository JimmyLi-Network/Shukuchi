# Shukuchi: Streaming Giant Models on Small Memory

![shukuchi logo](logo.PNG)

## Abstract
Shukuchi is a streaming-first LLM inference engine built to run giant models on machines with small memory. It loads one layer at a time, pipelines I/O with compute through prefetching, and executes quantized weights on both CPU and Metal GPU backends. The goal is pragmatic: keep peak RSS low while still delivering end‑to‑end generation on 7B–70B models (e.g., Llama 3.x 70B under ~4 GB RSS).

## Motivation
Running 7B–70B models on commodity machines is often blocked by memory limits and inefficient I/O. shukuchi focuses on the core bottlenecks:
- **Peak RSS** from full-model residency.
- **I/O stalls** from naive layer loading.
- **Quantized compute** without bespoke runtimes.

## Key Ideas
- **Layer streaming**: per-layer reads (pread) instead of full mmap.
- **Prefetch pipeline**: double/triple buffering to hide I/O latency.
- **Quantized execution**: Q4_K/Q6_K weights, Q8_0 KV cache.
- **Metal acceleration (macOS)**: GPU matmul kernels for Q4_K/Q6_K.

## News
- **2026-01-29**: Metal Q4_K/Q6_K matmul enabled on macOS; streaming + prefetch stats validated.
- **2026-01-29**: True per-layer streaming with bounded buffers; peak RSS < 200 MB on TinyLlama 1.1B.
- **2026-01-29**: Simple GGUF tokenizer (greedy longest-match + BOS).

## Results (On-Going)

| Model | Quant | Peak RSS | Prefetch Hit Rate | Notes |
|---|---|---:|---:|---|
| TinyLlama 1.1B | Q4_K_M | ~190 MB | ~95% | Metal matmul ~4x vs CPU |
| Llama 3.1 8B | Q4_K_M | ~1.0 GB | low (Metal too fast vs I/O) | triple buffer, Metal enabled |
| Llama 3.3 70B | Q4_K_M | ~3.5 GB | 0% (early tokens) | Q4/Q5/Q6 mix, Metal enabled |

## In-Progress
- Raise prefetch hit rate on large models (ahead=2/3, I/O batching, larger buffers).
- Faster tokenizer (trie-based) and GGUF tokenizer metadata support.
- More Metal kernels (attention, RMSNorm, softmax).
- SIMD CPU fast paths (AVX2/NEON).

## Installation

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

## Usage

```bash
./build/shukuchi <model.gguf> --prompt "Hello world" --max-tokens 8
```

Optional environment variables:
- `SHUKUCHI_METAL=0` to force CPU (no Metal).
- `SHUKUCHI_PREFETCH_DEPTH=2|3` to adjust buffer depth.

## Streaming Stats
The runtime prints:
- `layer_loads`, `layer_bytes_read`
- `max_layer_size`, `peak_buffer_usage`, `peak_rss`
- `max_concurrent_buffers`, `prefetch_hits`, `prefetch_misses`

These are model- and hardware-dependent; use them to validate streaming behavior.

## Limitations
- Tokenizer is currently greedy longest-match; not a full SentencePiece implementation.
- Q4_0 and other GGUF dtypes are not supported yet.
- Metal requires a GUI-capable macOS session (headless sessions may not expose GPU).

## Citation
If you use this project, please cite the repository:

```bibtex
@misc{shukuchi2026,
  title = {shukuchi: Streaming-First LLM Inference},
  author = {shukuchi contributors},
  year = {2026},
  howpublished = {GitHub repository}
}
```

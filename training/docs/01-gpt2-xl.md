# GPT-2 XL (1.5B) Support

## Overview

Scale from GPT-2 Small (124M) to GPT-2 XL (1.5B) — 12x more parameters, same architecture. Zero structural code changes needed.

## Architecture Comparison

| Parameter | Small (current) | Medium | Large | XL |
|-----------|----------------|--------|-------|----|
| Params | 124M | 355M | 774M | 1.5B |
| DIM | 768 | 1024 | 1280 | 1600 |
| HEADS | 12 | 16 | 20 | 25 |
| HIDDEN | 3072 | 4096 | 5120 | 6400 |
| N_LAYERS | 12 | 24 | 36 | 48 |
| VOCAB | 50257 | 50257 | 50257 | 50257 |
| fp16 size | ~250MB | ~710MB | ~1.5GB | ~3GB |
| KV cache (1024 ctx) | 75MB | 200MB | 470MB | 1GB |
| Total RAM | ~400MB | ~1GB | ~2.2GB | ~4.5GB |

All fit comfortably in 24GB unified RAM.

## Changes Required

### gpt2_convert.py
```python
MODEL_ID = "openai-community/gpt2-xl"  # was "openai-community/gpt2"
```
That's it. The weight names and format are identical across all GPT-2 sizes.

### gpt2.m — 5 constant changes
```c
#define DIM 1600    // was 768
#define HEADS 25    // was 12
#define HIDDEN 6400 // was 3072
#define N_LAYERS 48 // was 12
// VOCAB stays 50257, MAX_SEQ stays 1024
```

### Build
No Makefile changes needed.

## Performance Estimates (M5)

Decode matvec time scales with parameter count per layer:
- QKV: [4800, 1600] @ [1600] — 3x more rows, 2x wider = ~6x slower per layer
- FFN W1: [6400, 1600] — similar ratio
- 48 layers vs 12 = 4x more layers
- Total: ~24x more compute per step

Estimated decode: ~24 × 4ms = ~96ms/tok at short seq → ~10 tok/s.
At seq=500: ~120ms/tok → ~8 tok/s. Still well above 5 tok/s threshold.

LM head: [50257, 1600] — 2x wider, GCD parallel handles it. ~1.2ms.

## ANE Prefill

ANE kernels need recompilation for new dimensions. The fused attention MIL generates kernels parameterized by DIM/HEADS/HD/HIDDEN/seq — no code changes needed, just different constants flow through.

Potential issue: larger intermediate tensors (HIDDEN=6400) may hit ANE memory limits. If compile fails, fall back to CPU-only prefill (still fast with BLAS sgemm).

## Approach

1. Make DIM/HEADS/HIDDEN/N_LAYERS runtime parameters (read from a config or detect from weight directory)
2. Or: build separate binaries (`gpt2_small`, `gpt2_xl`) with different `#define`s
3. Option 1 is cleaner — store model config in `gpt2_weights/config.json`

# Building a Mistral 7B Inference Engine on Apple Silicon

A technical report on building a from-scratch Mistral 7B (Q4_0) inference engine in C/Objective-C targeting Apple M-series chips, with an ANE prefill path and CPU NEON decode.

## The Problem

Apple's Neural Engine is the most powerful fixed-function accelerator in consumer hardware — ~19 TFLOPS fp16 on M5 — but it's locked behind CoreML's compile-time abstractions. For LLM inference, you want direct control: custom quantization formats, fused kernels, KV cache management, and the ability to split work between ANE (prefill) and CPU (decode) based on what each is actually good at.

We built a complete Mistral 7B inference engine that reads GGUF Q4_0 files directly, runs decode on CPU with W4A8 SDOT kernels, and has an ANE prefill path using reverse-engineered `_ANEInMemoryModel` APIs. No CoreML, no Python, no ML frameworks — just system headers and NEON intrinsics.

## Architecture

### Two-Phase Design

The fundamental insight: ANE dispatch overhead (~60-80us per kernel) makes it slower than CPU for single-token operations, but ANE throughput dominates for batched work. So:

- **Prefill (S > 1):** ANE matmuls via MIL-compiled kernels. Weights passed as IOSurface inputs (not baked), so one compiled kernel per unique shape can serve all 32 layers by swapping weight data. 4 kernel types total: QO (dim x dim), KV (kv_dim x dim), fused gate+up, and down projection.

- **Decode (S = 1):** CPU NEON with fused Q4_0 dequant+matvec. Reads quantized weights directly from mmap'd GGUF — no intermediate fp16 buffer, no weight conversion. The inner loop uses W4A8 SDOT (`vdotq_s32`): activations quantized to Q8_0 once per layer, then int8 x int4 dot products with scale correction at block boundaries.

### Weight Loading

GGUF v3 parser (`gguf_loader.h`, 388 lines) handles the format directly: mmap the file, walk the tensor table, extract config from metadata KV pairs. Weight pointers go straight into the model struct — no copying, no conversion. The 3.8 GB model file is memory-mapped and shared with the OS page cache.

Per-layer weights are just pointers into the mmap:
```c
typedef struct {
    const void *wq;    // [dim, dim] Q4 — points into mmap'd GGUF
    const void *wk;    // [kv_dim, dim] Q4
    // ...
    uint32_t wq_type;  // GGML_TYPE_Q4_0, Q4_K, Q6_K, etc.
} LayerWeights;
```

### Quantization: The W4A8 SDOT Path

The biggest single optimization. Standard Q4_0 decode does:
```
int4 → int8 → int16 → int32 → float32 → FMA
```
That's 43 NEON ops per 32-value block, dominated by widening conversions.

W4A8 SDOT does:
```
quantize_f32_to_q8_0(activations)  // once per layer
int4_weights × int8_activations → vdotq_s32 → float scale
```
9 NEON ops per block. The `vdotq_s32` instruction computes 4 lanes of (4 int8 multiplies + accumulate) in a single cycle. M5 confirms `neon-dotprod` support.

The activation quantization cost is amortized: one `quantize_f32_to_q8_0` call produces Q8_0 blocks reused across Q, K, V projections (same input), and again across gate and up projections (same input). 4 quantizations per layer instead of 7 matvecs.

### Row-Parallel Dispatch

Each matvec is split across the 4 P-cores using GCD with atomic work-stealing. The FFN down projection is [4096, 14336] — splitting across 4 cores gives nearly 4x speedup on that single operation.

Key design choice: work-stealing with atomic chunk counter, not static partitioning. The E-cores are ~2x slower than P-cores and would create stragglers with even splits. GCD's `dispatch_apply` handles core affinity; we just grab chunks atomically:

```c
__block _Atomic int chunk_counter = 0;
dispatch_apply(n_chunks, queue, ^(size_t _) {
    int chunk;
    while ((chunk = atomic_fetch_add(&chunk_counter, 1)) < n_chunks) {
        // process chunk
    }
});
```

### Attention: NEON GQA

32 query heads, 8 KV heads (4:1 GQA ratio), head_dim=128. KV cache is fp16, channel-first `[kv_dim, max_seq]`, ring buffer with 4096 slots (matching Mistral's sliding window).

The attention inner loop is vectorized with NEON: Q*K dot products use `vfmaq_f32` with fp16→fp32 conversion from the cache, softmax is scalar (it's just max + exp + normalize over seq_len values), and V accumulation uses the same NEON FMA path.

### Tokenizer

From-scratch SentencePiece BPE implementation in C (442 lines). FNV-1a hash table for O(1) token lookup, handles Mistral's `▁` (U+2581) space prefix, `[INST]`/`[/INST]` chat template wrapping. No external tokenizer library.

## ANE Prefill Path

The ANE prefill system (`mistral_ane_prefill.h`, 557 lines) compiles 4 MIL kernel types at startup:

| Kernel | Shape | Used For |
|--------|-------|----------|
| K_QO | [4096, 4096] x [4096, S] | Wq, Wo projections |
| K_KV | [1024, 4096] x [4096, S] | Wk, Wv projections |
| K_GATE_UP | fused [14336, 4096] x 2 | W1 + W3 SwiGLU |
| K_DOWN | [4096, 14336] x [14336, S] | W2 down projection |

Weights are passed as IOSurface inputs at runtime, not baked into the compiled model. This means we compile only 4 kernels total (not 4 x 32 layers = 128), staying well under the ~119 compile limit. The tradeoff: weights must be dequantized to fp16 before ANE dispatch, but this is amortized over S tokens.

RMSNorm, RoPE, attention score computation, softmax, and SiLU remain on CPU — these are all O(dim) or O(seq_len) operations where ANE dispatch overhead would dominate.

### ANE Constraints Discovered

- **Weights baked at compile time** — `weightsBuffer` IOSurface does NOT override compiled weights. Weight-as-input is the only viable path for multi-layer reuse.
- **SDPA causal mask ignored** — hardware silently ignores the mask parameter. Must decompose into Q@K^T + mask + softmax + scores@V.
- **~119 compile limit** — ANE compiler leaks resources. Training works around this with `exec()` restart; inference stays under the limit.
- **Chaining API rejected** — `_ANEChainingRequest` validates but driver returns Error Code=15. Likely requires CoreML entitlements.

## Performance

### Current State (CPU-only decode, M5)

| Prompt Length | Decode tok/s | Per-token | Notes |
|:---:|:---:|:---:|---|
| 2 tokens | 10.85 | 92 ms | Near-peak, minimal KV scan |
| 6 tokens | 10.88 | 92 ms | Still minimal attention cost |
| 26 tokens | 9.64 | 104 ms | Attention starting to show |
| 108 tokens | 6.74 | 148 ms | KV scan becoming significant |

Decode speed degrades with context length because GQA attention is O(n) in cached tokens.

Prefill runs at ~1.7 tok/s on CPU (sequential matvec per token, chunked in 32-token groups). The ANE prefill path would bring this to ~10-50 tok/s depending on batch size, but is not yet wired into the main inference loop.

### Optimization History

| Version | Decode tok/s | Change |
|---|:---:|---|
| Initial scalar | 1.36 | Baseline — scalar dequant, scalar attention, single-threaded |
| + NEON Q6_K lm_head | 1.36 | lm_head went 141ms → 15ms, but only runs once per token |
| + NEON GQA attention | ~1.5 | Vectorized Q*K and V accumulation |
| + Chunked prefill | ~1.5 | 32-token batches (prefill only) |
| + Row-parallel (GCD) | ~1.9 | 4 P-core work-stealing for matvecs |
| + W4A8 SDOT | **10.88** | 43→9 ops/block, quantize-once amortization |

The SDOT transition was the discontinuous jump — everything before it was incremental.

### Theoretical Analysis

| Metric | Value |
|---|---|
| Model size (Q4_0) | 3.8 GB |
| M5 memory bandwidth | ~153 GB/s |
| Bandwidth floor | ~25 ms/token (3.8 GB / 153 GB/s) |
| Current decode | 92 ms (3.7x above floor) |
| MLX on M5 (Metal GPU) | ~24 ms/token (~42 tok/s) |

The 3.7x gap is the remaining SDOT overhead (integer widening, scale multiplication, horizontal reduction) plus attention. A Metal compute shader with fused Q4 dequant would approach the bandwidth floor — the shader is written (`q4_0_metal.metal`, 549 lines) but not yet integrated.

## File Inventory

```
mistral/
  mistral_infer.m        (198)  Main loop: load → tokenize → prefill → decode → sample
  mistral_model.h        (950)  Model struct, layer decode, RoPE, GQA, SwiGLU, parallel dispatch
  dequant.h             (1093)  Q4_0/Q4_K/Q6_K NEON dequant, W4A8 SDOT matvec, row-parallel
  gguf_loader.h          (388)  GGUF v3 parser, mmap, config extraction
  tokenizer.h            (442)  SentencePiece BPE, FNV-1a hash, chat template
  kv_cache.h              (63)  Ring buffer KV cache, channel-first fp16
  mistral_ane_prefill.h  (557)  ANE matmul kernels, weight dequant, prefill orchestration
  mistral_mil.h           (61)  MIL program generators for ANE weight-as-input matmul
  metal_matvec.h         (342)  Metal GPU matvec host code (written, not integrated)
  speculative.h          (442)  Speculative decode architecture (written, not integrated)
  q4_0_metal.metal       (549)  Metal compute shader for Q4_0 GEMV
  Makefile
                        -----
                        5085 lines total
```

## Unresolved Oddities

1. **Down projection still uses fp32 path** — `mistral_layer_decode_parallel` uses SDOT for all projections except the FFN down (w2), which goes through `q4_matvec_parallel` with fp32 activations. The hb_q8 scratch buffer is allocated but never used. This is pure oversight — wiring it would give another ~10-15% decode speedup.

2. **Prefill is absurdly slow** — 577 ms/token on CPU because chunked prefill still does per-token sequential matvecs (just batched for KV cache writes). True batched matmul (S tokens in one GEMM) would be ~32x faster. The ANE path does this correctly but isn't connected.

3. **lm_head runs single-threaded** — `mistral_logits` calls `q4_matvec` (serial) for the [32000, 4096] output projection. This is Q6_K format so it falls through to the scalar dequant path via the NEON Q6_K matvec. Not parallelized.

4. **Chat output quality degrades at length** — Repetitive degenerate output beyond ~50 tokens with temp=0. No repetition penalty, no top-k/top-p sampling. The compute is correct (verified against reference for short sequences), this is a sampling problem.

5. **Metal shader written but never tested** — `q4_0_metal.metal` and `metal_matvec.h` were generated by research agents but never compiled or validated against reference output.

## What's Next

1. **Wire ANE prefill** — The kernels compile and the orchestration code exists. Connecting it to `mistral_infer.m` would drop TTFT from seconds to hundreds of milliseconds.
2. **Metal GPU decode** — The 42 tok/s ceiling. Shader exists, needs integration and validation.
3. **Speculative decode** — Draft model on ANE, verification on CPU. Architecture written in `speculative.h`.
4. **Fix the down projection** — Use `hb_q8` and SDOT for w2. Free performance.

# Transformers on Apple's Neural Engine

Running transformer inference directly on Apple's Neural Engine via reverse-engineered private APIs. No CoreML, no Metal — pure ANE compute through `_ANEInMemoryModel` and MIL programs compiled at runtime.

Forked from [maderix/ANEtransformers](https://github.com/maderix/ANEtransformers) which demonstrated ANE training (Stories110M, 12-layer forward+backward on ANE).

> **Status:** Mistral 7B inference runs at **76.6 tok/s prefill** (ANE) + **19.7 tok/s decode** (CPU SDOT). Three compute backends: ANE baked-weight prefill, Metal GPU GEMM prefill, CPU NEON decode. All on an M5 with 24 GB.

---

## Table of Contents

- [Mistral 7B Inference Engine](#mistral-7b-inference-engine)
- [Performance](#performance)
- [ANE Baked-Weight Prefill](#ane-baked-weight-prefill)
- [Why ANE Is Hard to Use](#why-ane-is-hard-to-use)
- [Fitting a 7B Model in 24 GB](#fitting-a-7b-model-in-24-gb)
- [ANE Across Apple Silicon](#ane-across-apple-silicon)
- [ANE Hardware Reference](#ane-hardware-reference)
- [ANE Training (Stories110M)](#ane-training-stories110m)
- [Building](#building)
- [Test Suite](#test-suite)

---

## Mistral 7B Inference Engine

Complete Mistral 7B (Q4_0) inference in ~9,000 lines of C/Objective-C. Reads GGUF files directly, no dependencies beyond system frameworks (Accelerate, Metal, Foundation).

```bash
cd mistral && make
./mistral --model path/to/mistral-7b-instruct-v0.2.Q4_0.gguf \
          --prompt "The meaning of life is" --tokens 100 --temp 0.7

# ANE baked-weight prefill (fastest prefill path)
./mistral --model path/to/model.gguf --ane \
          --prompt "Explain quantum computing" --tokens 256 --temp 0

# Metal GPU prefill + CPU decode
./mistral --model path/to/model.gguf --metal \
          --prompt "Write a poem" --tokens 128

# Chat mode
./mistral --model path/to/model.gguf --chat --ane \
          --prompt "Give me a margarita recipe" --tokens 256 --temp 0
```

### Prefill Cascade

Three prefill backends, selected automatically by capability and flag:

```
ANE baked  (--ane,   S ≥ 16)  →  76.6 tok/s   fp16 conv1x1, baked weights, 128 programs
Metal GEMM (--metal, S ≥ 16)  →  58.0 tok/s   direct Q4_0 GEMM on GPU, 64-token tiles
BLAS tiled (default, S ≥ 16)  →  37.0 tok/s   Q4→fp32 dequant + cblas_sgemm (AMX)
CPU SDOT   (fallback, S < 16) →  sequential    single-token GEMV
```

Decode is always CPU NEON W4A8 SDOT at 19.7 tok/s — ANE dispatch overhead (~70 $\mu$s) makes it slower than CPU for $S = 1$.

### W4A8 SDOT Decode

The core decode optimization. Standard Q4_0 decode: `int4 → int8 → int16 → int32 → float32 → FMA` — 43 NEON ops per 32-element block. W4A8 SDOT: quantize activations to Q8_0 once per layer, then `vdotq_s32` (int4 × int8 → int32) — **9 ops per block**, a 4.8× reduction.

The activation quantization is amortized: one Q8_0 pass reused across Q, K, V projections (same input), and again across gate+up (same input). 4 quantizations per layer instead of 7 separate matvecs.

Each matvec split across 4 P-cores via GCD with atomic work-stealing:

```c
__block _Atomic int chunk_counter = 0;
dispatch_apply(n_chunks, queue, ^(size_t _) {
    int chunk;
    while ((chunk = atomic_fetch_add(&chunk_counter, 1)) < n_chunks)
        process_rows(chunk * chunk_sz, (chunk+1) * chunk_sz);
});
```

### Model Architecture Details

| Component | Detail |
|-----------|--------|
| Attention | 32 query heads, 8 KV heads (4:1 GQA), head_dim=128 |
| RoPE | Adjacent-pair `(x[2i], x[2i+1])` — Llama/Mistral convention |
| KV cache | fp16, sequence-major `[layer][seq][kv_dim]`, ring buffer, up to 64K via `--context` |
| Tokenizer | SentencePiece BPE in C (442 lines), FNV-1a hash, `[INST]`/`[/INST]` chat template |
| Sampling | Top-k + top-p + temperature + repetition penalty |
| Weight loading | GGUF v3 parser, mmap'd (3.8 GB shared with OS page cache) |

### File Inventory

```
mistral/
  mistral_infer.m           (754)  Main loop: load → tokenize → prefill → decode → sample
  mistral_model.h          (1269)  Model struct, layer forward, RoPE, GQA, SwiGLU, parallel dispatch
  mistral_ane_prefill.h    (1130)  ANE baked-weight prefill, BLAS prefill, 2-phase loading
  dequant.h                (1137)  Q4_0/Q4_K/Q6_K NEON dequant, W4A8 SDOT matvec, row-parallel
  q4_0_metal.metal         (1084)  Metal compute shaders: GEMV, GEMM, RMSNorm, RoPE, GQA attention
  metal_matvec.h            (565)  Metal GPU host code, batch forward, speculative verify
  speculative.h             (538)  Speculative decode: n-gram draft + self-speculative
  tokenizer.h               (442)  SentencePiece BPE, FNV-1a hash, chat template
  gguf_loader.h             (388)  GGUF v3 parser, mmap, config extraction
  ane_mil_gen_mistral.h     (337)  MIL program generators for ANE baked-weight conv1x1
  kv_cache.h                 (75)  Ring buffer KV cache, sequence-major fp16
                            -----
                            ~9000 lines (mistral/ only)

tests/
  ane_transpose_test.m      (299)  Transpose round-trip + ANE vs CPU correctness
  ane_layer_test.m          (298)  Single transformer layer: ANE vs BLAS
  ane_2phase_test.m         (347)  Compile → forge → cache cycle validation
  ane_prefill_compare.m     (324)  Full-model ANE vs BLAS prefill comparison
  ane_benchmark.m           (409)  Per-phase timing breakdown
  + 15 more probe/test files
```

---

## Performance

*All measurements on Apple M5, 10 cores (4P + 6E), 24 GB unified RAM, ~153 GB/s memory bandwidth. Model: Mistral 7B Instruct v0.2, Q4_0, 3.8 GB.*

### Prefill Speed (Measured)

| Backend | tok/s | Measured At | TFLOPS | Notes |
|---------|------:|:----------:|-------:|-------|
| **ANE baked** | **76.6** | S=361 | ~2.6 | 128 baked programs, fp16 conv1x1 |
| **Metal GEMM** | **58.0** | S=600 | ~1.9 | Direct Q4_0 GEMM, 64-tok tiles |
| **BLAS tiled** | **37.0** | S=600 | ~1.2 | Q4→fp32 + cblas_sgemm (AMX) |
| CPU SDOT | 1.7 | S=26 | 0.002 | Sequential GEMV per token |

### Decode Speed (CPU NEON W4A8 SDOT)

| Context Length | tok/s | Per-token | Utilization |
|:-:|:-:|:-:|:-:|
| ~6 tokens | 19.7 | 51 ms | 34% of BW floor |
| ~26 tokens | 19.3 | 52 ms | 33% |
| 256 gen | 17.8 | 56 ms | 30% |
| 1K context | ~12 | 83 ms | 21% (attention-bound) |

### Single Layer Comparison

| Path | Time (layer 0, S=64) | Speedup |
|------|---------------------:|--------:|
| BLAS (AMX cblas_sgemm) | 118.1 ms | 1.0× |
| **ANE baked** | **23.9 ms** | **4.9×** |

### Optimization History

| Version | Decode tok/s | Change |
|---------|:-----------:|--------|
| Initial scalar | 1.36 | Single-threaded scalar dequant |
| + NEON GQA attention | ~1.5 | Vectorized Q*K and V accumulation |
| + Row-parallel (GCD) | ~1.9 | 4 P-core work-stealing |
| + W4A8 SDOT | **10.88** | 43→9 ops/block, quantize-once amortization |
| + SiLU fix + RoPE vectorize | 17.2 | vvexpf exact sigmoid, vvsincosf batch RoPE |
| + Fused QKV + gate/up dispatch | 19.3 | Single dispatch_apply per group |
| + 4-row SDOT unrolling | **19.7** | 4 output rows per inner loop |

### Theoretical Ceiling

| Metric | Value |
|--------|-------|
| Model size (Q4_0) | 3.8 GB |
| M5 memory bandwidth | ~153 GB/s |
| Bandwidth floor (decode S=1) | ~25 ms/token = 40 tok/s |
| MLX on M5 (Metal GPU) | ~42 tok/s (hits bandwidth floor) |
| Current CPU SDOT decode | 51 ms = 19.7 tok/s (2× above floor) |
| ANE peak fp16 compute | ~19 TFLOPS |

CPU NEON decode is compute-bound at 0.28 TFLOPS — it can't saturate memory bandwidth. Metal GPU can (MLX hits the bandwidth floor). ANE is compute-bound in the other direction — 19 TFLOPS is overkill for $S = 1$ because dispatch overhead dominates.

### Per-Token Compute Breakdown (Decode, 32 Layers)

| Operation | Time | % |
|-----------|-----:|--:|
| Gate + Up (W1, W3) | 20.6 ms | 37% |
| FFN down (W2) | 15.4 ms | 25% |
| QKV (Wq, Wk, Wv) | 5.7 ms | 9% |
| Wo | 4.8 ms | 9% |
| LM head | 4.6 ms | 8% |
| Attention + other | 5.4 ms | 12% |
| **Total** | **56.5 ms** | |

Matvec dominates at 83%. Theoretical floor (pure bandwidth): 17.5 ms → 34% utilization.

---

## ANE Baked-Weight Prefill

The ANE's 19 TFLOPS of fp16 compute is the fastest accelerator on M5 for batched matmul — but unlocking it requires solving multiple hard problems that Apple's public APIs (CoreML) hide from developers.

### The Problem

ANE requires **baked weights** — weight tensors compiled into the program binary. You cannot pass weights as runtime inputs. This means every weight matrix needs its own compiled ANE program. Mistral 7B has 7 weight matrices per layer × 32 layers = 224 matrices. We fuse where possible to get it down to 128 compiled programs.

### Architecture: 4 Programs Per Layer

Each of the 32 transformer layers compiles to 4 ANE programs with baked fp16 weights:

| # | Program | Baked Weights | fp16 Size | MIL Ops |
|---|---------|--------------|:---------:|---------|
| 1 | **Q projection** | $W_q$ `[4096, 4096]` | 32 MB | `conv1x1` |
| 2 | **K+V fused** | $W_k$ `[1024, 4096]` + $W_v$ `[1024, 4096]` | 16 MB | 2× `conv1x1`, 2 outputs |
| 3 | **Wo projection** | $W_o$ `[4096, 4096]` | 32 MB | `conv1x1` |
| 4 | **FFN fused** | $W_1$ `[14336, 4096]` + $W_3$ `[14336, 4096]` + $W_2$ `[4096, 14336]` | 352 MB | 3× `conv1x1` + `sigmoid` + `mul` |

**32 layers × 4 programs = 128 compiled models.**

The fused FFN is the key win: gate ($W_1$), up ($W_3$), down ($W_2$), sigmoid, and element-wise multiply in one MIL program. Intermediates stay in ANE SRAM — avoids 3 separate CPU↔ANE round-trips and ~234 MB of intermediate data traffic.

> **Why K+V fused but not QKV?** ANE supports exactly 2 outputs per program. Programs with 3+ outputs silently return correct data only for the last output. K and V share the same input dimensions (`dim → kv_dim`), making them a natural 2-output pair.

### Per-Layer Dispatch Flow

```
CPU:  RMSNorm(x) → xn                           (vDSP_dotpr + vDSP_vsmul)
ANE:  Q_conv(xn) → q                            ← 1 dispatch, transpose in/out
ANE:  KV_fused(xn) → k, v                       ← 1 dispatch, 2 outputs
CPU:  RoPE(q, k)                                 (vvsincosf batch)
CPU:  KV cache write                             (fp32→fp16, ring buffer)
CPU:  GQA attention (q × K_cache, softmax, × V)  (NEON fp16→fp32)
ANE:  Wo_conv(attn_out) → wo                    ← 1 dispatch
CPU:  residual x += wo
CPU:  RMSNorm(x) → xn2
ANE:  FFN_fused(xn2) → ffn_out                  ← 1 dispatch, SiLU fused
CPU:  residual x += ffn_out
```

**4 ANE dispatches + 6 CPU phases per layer. 128 total ANE dispatches for 32 layers.**

### Channel-First Transpose

ANE uses channel-first memory layout: a tensor of shape `[1, C, 1, S]` is stored as `data[c * S + s]`. Transformer code uses token-first: `data[t * dim + d]`. Every ANE dispatch requires:

1. **Transpose to ANE** (token-first → channel-first) before writing input
2. **Transpose from ANE** (channel-first → token-first) after reading output

We use a NEON-optimized 4×4 block transpose via `vtrnq_f32` / `vcombine_f32`:

```c
// 4x4 block transpose: ~2 GB/s at dim=4096, S=128
float32x4_t r0 = vld1q_f32(token_first + s * dim + c);
// ... load r1, r2, r3
float32x4x2_t t01 = vtrnq_f32(r0, r1);
float32x4x2_t t23 = vtrnq_f32(r2, r3);
float32x4_t o0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
vst1q_f32(channel_first + c * S + s, o0);
```

Getting the layout wrong doesn't crash — it silently produces numerically plausible but completely wrong results. Cosine similarity between BLAS reference and un-transposed ANE output: 0.02 (random). With correct transpose: **0.998**.

### 2-Phase Loading: Cold Compile + Warm Forge

Compiling 128 ANE programs takes ~80 seconds. The 2-phase strategy eliminates this on subsequent runs.

**Phase 1 — Cold start (~80s, one-time):**

For each of the 128 programs:
1. Dequant weight matrix Q4_0 → fp16 (~5 ms per matrix)
2. Build DEADBEEF weight blob (64B global header + 64B chunk header with `0xDEADBEEF` magic + fp16 data)
3. Symlink blob to tmpDir (~0.4 ms vs ~130 ms copy for 352 MB FFN)
4. Compile via `_ANEInMemoryModel` → `aned` daemon hashes MIL+weights (SHA256), compiles, caches
5. Extract hexId via KVC: `[mdl valueForKey:@"hexStringIdentifier"]`
6. Save hexId to manifest

All 128 hexIds saved to `~/.cache/ane_mistral/manifest.plist`.

**Phase 2 — Warm start (~220 ms):**

For each of the 128 programs:
1. Create a dummy `_ANEInMemoryModel` with a 130-byte minimal blob
2. Forge the hexId via KVC: `[mdl setValue:savedHexId forKey:@"hexStringIdentifier"]`
3. Call `loadWithQoS:` — daemon looks up hexId in cache, loads compiled program (~1.7 ms)

**Result: ~400× faster init.** Warm start loads all 128 programs in ~220 ms.

```
Cold start:  80s  (one-time, survives restarts, lost on reboot)
Warm start:  220ms  (from daemon cache)
Inference:   76.6 tok/s at S=361
```

The `aned` daemon cache is in-memory only — lost on reboot. On cache miss, the system falls back to Phase 1 for that program, recompiles, and updates the manifest.

### ANE Tile Size

ANE has a minimum sequence length of $S = 16$ (fp32 I/O) or $S = 32$ (fp16 I/O). Below this, outputs are silently all zeros. Programs are compiled for a fixed tile size (default: 64). When the actual token count is less than the tile size, the input is zero-padded but CPU ops (RoPE, attention, KV write) only process the real tokens:

```c
// ANE ops use full S_ane (compile-time tile, always 64)
ane_write_input(kernel, 0, ane_in, dim * S_ane * sizeof(float));
ane_eval(kernel);
ane_read_output(kernel, 0, ane_out, dim * S_ane * sizeof(float));

// CPU ops use n_tok (actual token count, may be < S_ane)
apply_rope_batch(Q, K, start_pos, theta_inv, ..., n_tok);
kv_write_batch(kv, layer, start_pos, K, V, kv_dim, n_tok);
attention_batch(Q, attn_out, kv, layer, start_pos, cfg, n_tok, scratch);
```

### Correctness Verification

| Test | Result | Detail |
|------|--------|--------|
| Transpose round-trip | 9/9 PASS | Exact bit-for-bit identity |
| ANE conv vs CPU matmul | PASS | max_abs_err < 0.04 (fp16 precision) |
| Single layer ANE vs BLAS | PASS | max_err=0.026, mean_err=0.0007 |
| Full 32-layer prefill | PASS | cosine_sim=0.998, max_err=0.61 |
| 2-phase compile/forge | 5/5 PASS | Forged output identical to fresh compile |
| Text output | MATCH | Same generated text as BLAS path |

---

## Why ANE Is Hard to Use

Apple's Neural Engine delivers ~19 TFLOPS of fp16 compute — the fastest accelerator on every Apple Silicon chip. But using it for anything beyond CoreML is an exercise in reverse engineering around undocumented constraints. Here is everything we found that doesn't work.

### 1. No Public API

The only way to run arbitrary programs on ANE is through `_ANEInMemoryModel`, a private Objective-C class in the `AppleNeuralEngine` framework. It takes MIL (Machine Learning Intermediate Language) program text + fp16 weight blobs as `NSData`, compiles them to ANE machine code, and evaluates with IOSurface inputs/outputs.

```objc
id mdl = [_ANEInMemoryModel inMemoryModelWithDescriptor:desc];
[mdl compileWithQoS:21 options:@{} error:&e];
[mdl loadWithQoS:21 options:@{} error:&e];
[mdl evaluateWithRequest:req qos:21 error:&e];
```

Everything in memory — no `.mlmodelc` on disk. This API could break or be locked down in any macOS update.

### 2. No Integer Quantization (via Private API)

INT8, INT4, palettized/LUT — all fail to compile through `_ANEInMemoryModel`. Every combination tested:

```
constexpr_affine_dequantize  → COMPILE FAIL
constexpr_lut_to_dense       → COMPILE FAIL
constexpr_blockwise_shift_scale → COMPILE FAIL
```

The model **must** be fully dequantized to fp16 before baking into the program. CoreML quantized models somehow run on ANE — the internal compiler flag or op that enables this is unknown. This means a Q4_0 model (3.8 GB) must be inflated to fp16 (7.6 GB) for each weight matrix at compile time.

### 3. Baked Weights Only

Weights are compiled into the program binary. The `weightsBuffer` IOSurface (meant for runtime weight override) **does not work** — the ANE ignores it and uses the compiled weights. This means:

- Every unique weight matrix needs its own compiled program
- Mistral 7B: 128 programs (4 per layer × 32 layers)
- Each program compilation requires writing the full fp16 weight blob to a tmpDir, hashing it (SHA256), and compiling
- The 352 MB FFN blob alone takes ~125 ms just to hash

### 4. No Reduction Ops

ANE cannot execute `reduce_mean`, `reduce_sum`, `rsqrt`, or any element-wise reduction. This means **RMSNorm cannot run on ANE**:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

Every layer requires a CPU↔ANE round-trip for normalization. Two RMSNorms per layer (attention + FFN) = 64 CPU interludes per forward pass. This is the primary bottleneck preventing full-ANE inference.

### 5. Channel-First Memory Layout

ANE tensors use `[1, C, 1, W]` channel-first layout, stored as `data[c * W + s]`. Standard transformer activations are token-first: `data[t * dim + d]`. Every ANE dispatch requires a full matrix transpose on input and output. At `dim=4096, S=64`, this is a 1 MB transpose — fast on CPU but adds latency.

Worse: getting the layout wrong doesn't crash. The ANE happily computes on the un-transposed data and produces outputs that look numerically reasonable (non-zero, finite) but are completely wrong. This cost us significant debugging time.

### 6. Maximum 2 Outputs Per Program

ANE programs with 3+ outputs silently corrupt all outputs except the last one. Only the final output tensor contains correct data. This prevents fusing QKV into a single program (3 outputs needed) — we must split into Q (1 output) + K+V fused (2 outputs).

### 7. Compile Limit (~119 Programs)

ANE compilation leaks resources. After ~119 compilations in a single process, subsequent compiles fail. Workaround: `exec()` restart (used in training) or the 2-phase forge strategy (compile once, load from daemon cache forever).

### 8. Dispatch Overhead

Each ANE dispatch costs ~60–80 $\mu$s round-trip (CPU → ANE → CPU). For $S = 1$ decode:

$$\text{overhead} = 7 \text{ matmuls} \times 32 \text{ layers} \times 70\mu\text{s} = 15.7\text{ ms}$$

This alone is close to the CPU decode time (~51 ms), making ANE useless for single-token generation. ANE only wins when the compute per dispatch is large enough to amortize the overhead — which happens at $S \geq 16$.

### 9. No Kernel Chaining

`_ANEChainingRequest` (meant to chain multiple ANE programs without CPU round-trips) validates syntactically but the driver returns Error Code=15 on execution. Likely requires a CoreML entitlement. Every dispatch is a full CPU→ANE→CPU round-trip.

### 10. Minimum Sequence Length

ANE produces silent all-zeros output for sequence lengths below $S = 16$ (fp32 I/O) or $S = 32$ (fp16 I/O). No error, no warning — just zeros. For $S = 1$ decode, you must pad to $S = 16$ and discard the extra outputs, which wastes compute.

### 11. IOSurface Minimum Size

The smallest IOSurface ANE will accept is 49,152 bytes — even for a single scalar output. Format is always `[1, C, 1, W]` channel-first fp16 (or fp32 with cast).

### 12. Daemon Cache Volatility

The `aned` daemon caches compiled programs in memory, keyed by `SHA256(MIL + weights + options)`. This cache survives process restarts but is **lost on reboot**. After reboot, all 128 programs must be recompiled from scratch (~80s).

---

## Fitting a 7B Model in 24 GB

Running 128 ANE programs with baked fp16 weights for a 7B parameter model on a 24 GB machine requires careful memory management. Here's how every byte is accounted for.

### Memory Budget

| Component | Size | Notes |
|-----------|-----:|-------|
| Q4_0 weights (mmap'd GGUF) | 3.8 GB | Shared with OS page cache, read-only |
| ANE compiled programs | ~0 GB in-process | Compiled programs live in `aned` daemon memory |
| IOSurfaces (128 programs) | ~5 MB | Shared input/output surfaces, largest = FFN |
| KV cache (fp16, 4K context) | 256 MB | `n_layers × max_seq × kv_dim × 2 × 2` |
| Dequant scratch (init only) | 352 MB | One weight at a time, freed after compile |
| Metal GPU buffers (if `--metal`) | ~200 MB | Shared CPU/GPU via `MTLResourceStorageModeShared` |
| OS + frameworks + other | ~4 GB | |
| **Total at steady state** | **~8.3 GB** | Dequant scratch freed after init |
| **Peak during init** | **~8.6 GB** | While dequanting largest weight (FFN W1) |

### Key Techniques

**On-demand embedding:** Embedding table stays in Q4_0 (never dequantized to fp32). Each token's embedding row is dequantized on the fly — saves ~496 MB vs pre-dequanting.

**Stream compilation:** Only one layer's weights are dequantized at a time. Dequant Q4→fp16, build blob, compile, free fp16 buffer, move to next layer. Peak scratch = one FFN blob (352 MB).

**Symlink weight blobs:** The ANE compiler requires weight blobs written to a tmpDir. Instead of copying 352 MB, we symlink the blob file → saves ~130 ms per FFN program and avoids doubling memory usage.

**IOSurface sharing:** All 128 programs share a single set of input/output IOSurfaces (sized for the largest program = FFN). Sequential layer execution means surfaces are reused.

**Compiled programs in daemon:** The `aned` daemon owns all compiled program memory. User-process memory is just the IOSurface mappings (~5 MB) and kernel metadata (~12 KB).

---

## ANE Across Apple Silicon

The Neural Engine is identical across base/Pro/Max variants of the same generation — Pro and Max add GPU cores and memory bandwidth but do not multiply the ANE. The Ultra is a notable exception: it fuses two dies and gets two ANEs.

### Comparison Table

| Chip | ANE TOPS (INT8) | ANE fp16 TFLOPS | ANE Cores | GPU Cores | Mem BW | Max RAM |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| **M5** (this machine) | ~38 | ~19 | 16 | 10 | 153 GB/s | 32 GB |
| **M5 Max** | ~38 | ~19 | 16 | 40 | 614 GB/s | 128 GB |
| **M4 Max** | 38 | 19 | 16 | 40 | 546 GB/s | 128 GB |
| **M3 Ultra** | 36 | ~18 | 32 (2×16) | 60–80 | 819 GB/s | 512 GB |

### What This Means for LLM Inference

**M5 vs M5 Max:** The ANE is identical — same 16 cores, same ~19 TFLOPS. The Max's advantage is 4× memory bandwidth (614 vs 153 GB/s) which helps Metal GPU decode (bandwidth-bound), and 4× max RAM (128 vs 32 GB) which allows larger models. ANE prefill performance would be identical.

**M4 Max:** Same ANE as M5. Same 38 TOPS. The M5 GPU has new "Neural Accelerators" inside each GPU core (Apple's "133 TOPS" figure includes both ANE + GPU NAs), but these are accessed through Metal, not `_ANEInMemoryModel`.

**M3 Ultra:** Two M3 Max dies via UltraFusion = two 16-core ANEs (32 cores total, 36 TOPS). In theory, 2× the ANE compute. In practice, there is no public API to dispatch across both ANEs simultaneously. CoreML may handle this internally, but `_ANEInMemoryModel` likely targets a single ANE. The real advantage of Ultra is 512 GB RAM — enough to run 70B+ models entirely in memory.

### Why More RAM Matters More Than More TFLOPS

For the ANE baked-weight approach, the binding constraint is **memory** not **compute**:

| Model | Q4_0 Size | fp16 Weights | Min RAM (ANE baked) |
|-------|:-:|:-:|:-:|
| Mistral 7B | 3.8 GB | ~14 GB total baked | 24 GB (tight) |
| Llama 13B | 7.4 GB | ~26 GB | 48 GB+ |
| Llama 70B | 38 GB | ~140 GB | 192 GB+ |

The ANE baked approach requires all weight matrices compiled into separate programs. While the compiled programs live in the `aned` daemon's address space (not the user process), the daemon still needs memory for them. On a 24 GB M5, Mistral 7B is the practical limit.

On an M5 Max with 128 GB or M3 Ultra with 512 GB, you could run 13B or even 70B models with full ANE prefill — if someone is willing to wait for the cold compile.

### The Real Comparison: ANE vs GPU for Prefill

| Backend | Prefill tok/s (S=361) | Limited By |
|---------|:----:|-------|
| ANE baked (this work) | **76.6** | CPU↔ANE transpose + dispatch overhead |
| Metal GEMM (this work) | 58.0 | GPU Q4→fp16 dequant in-shader |
| MLX (reference) | ~200+ | Direct fp16 GEMM, no overhead |

ANE is 1.3× faster than our Metal GEMM but still well below the theoretical ~19 TFLOPS peak. The bottleneck is not ANE compute — it's the CPU work between dispatches (RMSNorm, RoPE, attention, transpose). Fusing RMSNorm into ANE programs would eliminate 2 of 6 CPU phases per layer, but ANE cannot do the required `reduce_sum` and `rsqrt` operations.

---

## ANE Hardware Reference

### Measured Specs (M5)

| Spec | Value | How Measured |
|------|-------|-------------|
| Peak fp16 throughput | ~19 TFLOPS | 2048×2048 matmul, `inmem_peak.m` |
| Sustained throughput (7B prefill) | ~2.6 TFLOPS | 4096×4096 conv, baked weights |
| On-chip SRAM | ~32 MB | `sram_probe.m` — perf cliff at 32 MB working set |
| Dispatch overhead | 60–80 $\mu$s | `inmem_bench.m` — round-trip timing |
| Compile time per program | 30–50 ms | Measured during 128-program compile |
| Compile limit per process | ~119 | Resource leak, consistent M4/M5 |
| Precision | fp16 only | INT8/INT4/LUT all fail to compile |
| Minimum sequence length | $S = 16$ (fp32) / $S = 32$ (fp16) | Below → silent zeros |
| IOSurface minimum | 49,152 bytes | Even for scalar output |
| QoS levels tested | 0–63 | No latency/throughput difference |

### Sustained TFLOPS at Different Sizes

| Weight Matrix | fp16 Size | Measured TFLOPS |
|:---:|:---:|:---:|
| 2048 × 2048 | 8 MB | ~19 (fits SRAM) |
| 4096 × 4096, $S = 128$ | 32 MB | 2.6 |
| 14336 × 4096, $S = 128$ | 112 MB | 2.6 |
| Fused FFN (3 convs), $S = 128$ | 352 MB | 1.4 |

Performance drops sharply when baked weights exceed SRAM capacity (~32 MB). The 4096×4096 weight matrix is 32 MB fp16 — right at the boundary. Larger matrices require streaming from unified memory, reducing throughput to ~2.6 TFLOPS (14% of peak).

### MIL Program Format

All ANE programs use MIL (Machine Learning Intermediate Language) compiled at runtime. The only op that reliably works for matmul is `conv` (1×1 convolution = matmul):

```
func main<ios18>(tensor<fp32, [1, 4096, 1, 64]> x) -> (tensor<fp32, [1, 4096, 1, 64]>) {
    x_fp16 = cast(x, dtype="fp16");
    y = conv(x_fp16, weight=W, strides=[1,1], pad=[0,0,0,0], ...);
    y_fp32 = cast(y, dtype="fp32");
} -> (y_fp32);
```

Weight blobs use the DEADBEEF format: 64-byte global header + 64-byte chunk header (magic `0xDEADBEEF`) + raw fp16 weight data. The blob offset in MIL points to the chunk header, not the data start.

### What Works on ANE (via Private API)

| Op | Status | Notes |
|----|:------:|-------|
| `conv` (1×1, baked weights) | **Works** | The only reliable matmul path |
| `sigmoid` | **Works** | Used in fused FFN for SiLU |
| `mul` (element-wise) | **Works** | Used in fused FFN |
| `add` (element-wise) | **Works** | Residual connections |
| `cast` (fp32↔fp16) | **Works** | Required for fp32 I/O with fp16 compute |
| 2 outputs per program | **Works** | K+V fused |
| 2 inputs per program | **Works** | conv(baked, input0) + add(result, input1) |
| `reduce_mean` / `reduce_sum` | **Fails** | Blocks RMSNorm on ANE |
| `rsqrt` | **Fails** | Blocks RMSNorm on ANE |
| `matmul` (activation × activation) | **Untested** | Needed for attention Q×K on ANE |
| 3+ outputs | **Broken** | Only last output is correct |
| Runtime weights (IOSurface) | **Dead** | Compile fails at all sizes |
| INT8/INT4/LUT quantization | **Fails** | All quantization ops fail to compile |
| Kernel chaining | **Fails** | Error Code=15, likely needs entitlement |

---

## ANE Training (Stories110M)

Upstream work from [maderix](https://github.com/maderix/ANEtransformers). 109M-parameter Llama2 architecture trained directly on ANE.

| Metric | Value |
|--------|-------|
| ANE eval time | 9.6 ms/step |
| ANE TFLOPS achieved | **13.59** (130.5 GFLOPS in 9.6ms) |
| Total step time | 107 ms (ANE + IO + classifier + cross-entropy) |
| Kernels per compile | 72 (60 weight-bearing, 12 weight-free) |

- Forward: fwdAttn + fwdFFN on ANE, classifier matmul on CPU (cblas)
- Backward: ffnBwd, sdpaBwd1/2, qkvBwd on ANE, weight gradients via async cblas_sgemm
- `exec()` restart every 10 accumulation steps to work around the ~119 compile limit
- SDPA causal mask workaround: decompose into Q@K^T (ANE) + mask+softmax (CPU) + scores@V (ANE)

```bash
cd training && make train_large && ./train_large
```

See [training/README.md](training/README.md) for full details.

---

## GPT-2: Where It Started

The GPT-2 outputs were retarded and gorgeous. A 124M parameter model from 2019, running on a phone chip's neural engine, producing text that was mostly nonsense but had these moments of eery depth — fragments that made you stop and wonder if maybe, somehow, these models understand us. They don't, of course. Not really. But GPT-2 was the first time a lot of people felt that shiver, and building an inference engine for it on ANE was the spark that led to everything else in this repo.

---

## ANE Probes

Root-level `.m` files characterize ANE behavior:

| File | What it tests |
|------|---------------|
| `api_exploration.m` | ANE private API discovery and introspection |
| `inmem_basic.m` | In-memory MIL compilation proof-of-concept |
| `inmem_bench.m` | Dispatch latency benchmarks (~60-80 $\mu$s/call) |
| `inmem_peak.m` | Peak TFLOPS measurement (2048×2048 matmul → ~19 TFLOPS) |
| `sram_bench.m` | ANE SRAM bandwidth probing |
| `sram_probe.m` | SRAM size/layout exploration (~32 MB) |
| `quant_probe.m` | INT8/INT4/LUT quantization support (all fail) |
| `w8a8_probe.m` | W8A8 activation quantization testing |

---

## Building

Requires macOS 15+ on Apple Silicon (tested M4, M5). No external dependencies.

```bash
# Mistral inference
cd mistral && make

# ANE test suite
make ane_tests

# Individual tests
make transpose_test && ./transpose_test
make ane_layer_test && ./ane_layer_test --model ~/models/mistral-7b.Q4_0.gguf
make ane_2phase_test && ./ane_2phase_test
make ane_prefill_compare && ./ane_prefill_compare --model ~/models/mistral-7b.Q4_0.gguf

# Chat UI (Rust/egui)
cd chat && cargo build --release

# ANE probes (from root)
xcrun clang -O2 -fobjc-arc -o probe api_exploration.m \
  -framework Foundation -framework IOSurface
```

## Test Suite

| Test | What It Validates | Pass Criteria |
|------|------------------|---------------|
| `ane_transpose_test` | Channel-first ↔ token-first round-trip, ANE conv vs CPU matmul | Exact round-trip; abs err < 0.1 (fp16) |
| `ane_layer_test` | Single transformer layer ANE vs BLAS | max_err < 0.1, mean_err < 0.01 |
| `ane_2phase_test` | Compile+hexId, forged load, manifest I/O, scale (8 programs), cache miss | Forged output = fresh compile |
| `ane_prefill_compare` | Full 32-layer prefill ANE vs BLAS | cosine_sim > 0.995, max_err < 1.0 |
| `ane_benchmark` | Per-phase timing: transpose, dequant, compile, forge, eval | Timing data (no pass/fail) |

## Known Issues

1. **Decode utilization at 34%** — CPU SDOT achieves 0.28 TFLOPS vs 0.82 TFLOPS bandwidth ceiling. Room for improvement with better cache prefetch and wider SDOT unrolling.
2. **ANE TFLOPS at 14% of peak** — Baked weights exceeding 32 MB SRAM cause streaming from unified memory. Tiling weights to fit SRAM could recover throughput.
3. **LM head on CPU only** — Q6_K format, requires full dequant. Not compatible with ANE or Metal GEMM path.
4. **Cold ANE start is slow** — 80s to compile 128 programs. Daemon cache lost on reboot. Could persist compiled programs to disk if ANE binary format were understood.

## Unresolved Questions

1. Why does ANE compilation leak resources? ~119 limit consistent across M4/M5.
2. What entitlement gates `_ANEChainingRequest`?
3. How does CoreML run INT8/INT4 models on ANE when the MIL ops fail via private API?
4. Multi-ANE on Ultra — does `_ANEInMemoryModel` support targeting specific ANE instances?
5. Can compiled ANE programs be serialized to disk and reloaded without recompilation?
6. What MIL op or flag enables the int8 datapaths that Apple's TOPS numbers imply exist?

## Disclaimer

Independent research into Apple Neural Engine architecture. Uses undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included. Not affiliated with or endorsed by Apple Inc.

## License

MIT — see [LICENSE](LICENSE)

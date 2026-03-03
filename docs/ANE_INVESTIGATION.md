# Apple Neural Engine: What It Does, What It Doesn't, and What We Learned

A technical report on reverse-engineering ANE behavior on Apple M4/M5 silicon for transformer inference, based on systematic probing through private APIs.

## What the ANE Is

The Apple Neural Engine is a fixed-function matrix accelerator on Apple Silicon. On M5:

| Spec | Value |
|---|---|
| Peak throughput | ~19 TFLOPS fp16 |
| SRAM | ~32 MB |
| Precision | fp16 only (all int quantization ops fail to compile) |
| Dispatch overhead | ~60-80 us per kernel |
| Compile limit | ~119 kernels per process (resource leak) |

It's accessed through `_ANEInMemoryModel`, a private Objective-C class that takes MIL (Machine Intermediate Language) program text + fp16 weight blobs, compiles them into ANE machine code, and evaluates them with IOSurface inputs/outputs. No disk `.mlmodelc` needed — everything happens in memory.

## What ANE is Good For

**Batched matmul (prefill).** When you have S > 1 tokens to process, ANE's 19 TFLOPS fp16 dominates. A fused attention kernel (QKV + multi-head SDPA + output projection) running on ANE is 1.5-3x faster than the equivalent unfused kernels, because intermediates stay in ANE SRAM and avoid the round-trip to unified memory.

For Mistral 7B prefill, we compile 4 kernel shapes and reuse them across all 32 layers by passing weights as IOSurface inputs:

| Kernel | Shape | Reused across |
|---|---|---|
| K_QO | [4096, 4096] x [4096, S] | Wq, Wo (64 uses) |
| K_KV | [1024, 4096] x [4096, S] | Wk, Wv (64 uses) |
| K_GATE_UP | fused [14336, 4096] x 2 | W1+W3 (32 uses) |
| K_DOWN | [4096, 14336] x [14336, S] | W2 (32 uses) |

4 compiles, 192 evaluations. Well under the ~119 compile limit.

## What ANE is Bad For

**Single-token decode.** At S=1, the matmuls are tiny (GEMV, not GEMM). The 60-80us dispatch overhead per kernel call dominates the actual compute. For a Mistral layer with 7 matmuls, that's ~0.5ms in pure overhead — comparable to the entire CPU NEON compute time.

We measured this directly: CPU NEON decode at 92ms/token vs ANE dispatch overhead alone at ~0.5ms x 32 layers = ~16ms, but ANE also needs fp16 weight dequantization (Q4_0 → fp16 conversion for the weight IOSurfaces) which adds ~10-15ms per layer. Net result: ANE decode would be slower than CPU.

**Quantized inference.** ANE is fp16 only. Every quantization format we tested fails at compile time:

| Format | Result |
|---|---|
| INT8 (W8A8) | Compile error: "unsupported type" |
| INT4 (W4A16) | Compile error |
| Lookup table quantization | Compile error |
| Block-scaled int8 | Compile error |

This means any quantized model must be dequantized to fp16 before ANE can touch it. For a 3.8 GB Q4_0 model, that's ~7.6 GB of fp16 weights to materialize — you can do it per-layer and reuse the buffer, but it's a tax that CPU NEON doesn't pay (it reads Q4_0 directly).

## Throughput: With ANE vs Without

### Without ANE (current production path)

Everything on CPU NEON with W4A8 SDOT:

| Phase | Performance | Method |
|---|---|---|
| Prefill | ~1.7 tok/s | Sequential CPU matvec per token, chunked in 32s |
| Decode | 10.88 tok/s (92ms) | W4A8 SDOT, 4 P-core row-parallel, NEON GQA |
| TTFT (6 tokens) | ~3.9 seconds | CPU prefill dominates |
| TTFT (108 tokens) | ~62 seconds | Linear in prompt length |

### With ANE (projected, prefill only)

ANE handles the matmuls during prefill. CPU handles decode (unchanged):

| Phase | Performance | Method |
|---|---|---|
| Prefill | ~10-50 tok/s (est.) | ANE batched matmul, CPU RMSNorm/RoPE/softmax |
| Decode | 10.88 tok/s | Same CPU path (ANE dispatch overhead too high for S=1) |
| TTFT (6 tokens) | ~200-600 ms (est.) | ANE prefill + first decode step |
| TTFT (108 tokens) | ~2-10 seconds (est.) | Sublinear if ANE processes full S at once |

The ANE prefill kernels are compiled and the orchestration code exists (`mistral_ane_prefill.h`), but the path isn't wired into the main inference loop yet. The estimates are based on ANE's measured throughput (~19 TFLOPS fp16) derated for dequantization overhead and CPU-side ops.

### The Hybrid Architecture

The natural split:

```
[Prompt tokens] ──→ ANE prefill (batched matmul, high throughput)
                         │
                    [KV cache populated]
                         │
[Decode loop] ──→ CPU NEON (S=1 matvec, low latency)
                         │
                    [Token by token]
```

This is exactly what Apple does internally with CoreML-backed models. We just do it without CoreML.

## ANE Behavioral Findings

### Weights Are Immutable After Compile

Once `_ANEInMemoryModel` compiles a MIL program, the weights are baked into the ANE program. We tested every angle:

1. **File overwrite + reload** — ANE uses cached compiled version, ignores file changes
2. **`weightsBuffer` IOSurface** — Exists as API, does NOT override compiled weights. Output identical regardless of buffer contents
3. **`procedureIndex` parameter** — Attempted to select different weight sets. No effect

This is why weight-as-input (passing weights as IOSurface inputs alongside activations) is the only viable approach for multi-layer inference without compiling 128+ kernels.

### Chaining API Inaccessible

`_ANEChainingRequest` would let you chain multiple ANE kernels without CPU round-trips (keeping intermediates in SRAM). It validates — the API accepts the request object — but the driver rejects it with Error Code=15. Likely requires an entitlement that only CoreML holds.

This means every ANE kernel dispatch does a full CPU→ANE→CPU round-trip through unified memory. Fusion (combining multiple ops into one MIL program) is the only way to keep intermediates on-chip.

### QoS Has No Effect

Tested QoS levels 0 through 63. ANE runs at fixed frequency regardless. No latency difference, no throughput difference. The ANE is either busy or idle — there's no frequency scaling.

### IOSurface Constraints

- **Minimum size**: 49,152 bytes (even for a single scalar)
- **Spatial stride**: Padded to minimum 32 in the width dimension
- **Format**: `[1, C, 1, W]` channel-first fp16. The `C` dimension maps to output features, `W` to spatial (sequence length)
- **Memory**: IOSurfaces live in unified memory, zero-copy accessible by both CPU and ANE

### Performance Stats API

`_ANEPerformanceStats` exists with a `hwExecutionTime` property, but it requires factory construction through the model's `perfStatsMask`. We couldn't get it to return meaningful data without CoreML's initialization path. Hardware counters exist but are gated.

## SRAM Characterization

ANE has approximately 32 MB of on-chip SRAM. Intermediates from fused operations stay in SRAM — this is what makes fusion 1.5-3x faster than separate dispatches. Once a program exceeds SRAM capacity, intermediates spill to unified memory and you lose the fusion benefit.

For Mistral 7B:
- Attention intermediates (Q@K^T for S=128): `[32, 128, 128]` fp16 = 1 MB — fits in SRAM
- FFN gate intermediate: `[1, 14336, S]` fp16 — fits for S ≤ ~1024
- The full model doesn't fit — weights alone are 3.8 GB (7.6 GB fp16)

## Fused vs Unfused Benchmarks

Measured directly on M5 with `bench_fused.m`:

| Configuration | Time | Speedup |
|---|---|---|
| Separate QKV (3 dispatches) | ~240 us | 1.0x |
| Fused QKV (1 dispatch) | ~100 us | 2.4x |
| Separate attention (Q@K + softmax + @V) | ~180 us | 1.0x |
| Fused attention (single MIL) | ~80 us | 2.25x |
| Separate FFN (3 dispatches) | ~200 us | 1.0x |
| Fused FFN (W1+SiLU+W2 single MIL) | ~130 us | 1.54x |

Dispatch overhead is ~60-80us per call. With 3 unfused calls, you pay 180-240us in pure overhead. Fusion eliminates 2 of 3 dispatches AND keeps intermediates in SRAM.

**Conclusion**: If you're using ANE, you must fuse aggressively. The API overhead is the enemy, not the compute.

## Unresolved Questions

1. **ANE compiler resource leak** — Why does compilation leak? Is it an MPS-level cache, an IOKit allocation, or a driver bug? The ~119 limit is consistent across M4 and M5. Apple's own training loop (if it exists) must handle this somehow.

2. **Chaining entitlement** — What specific entitlement gates `_ANEChainingRequest`? Is it a com.apple.ane.* entitlement, or something broader? CoreML apps can presumably chain internally.

3. **Multi-engine** — M5 has one ANE instance. M5 Pro/Max/Ultra may have more. Does `_ANEInMemoryModel` support targeting specific engines? We haven't tested on multi-engine hardware.

4. **ANE frequency** — Is the ANE truly fixed-frequency, or does it have power states we can't observe? The QoS sweep showed no latency variation, but thermal throttling behavior is unknown.

5. **Hardware int8** — The ANE likely has int8 datapaths (common in neural accelerators), but the MIL compiler rejects all integer types. Is there a different MIL op or compilation flag that enables them? CoreML quantized models somehow run on ANE.

## Tools Used

All probing done through Objective-C runtime introspection (`objc_msgSend`, `class_copyMethodList`, `NSClassFromString`) targeting:

- `_ANEInMemoryModel` — model compilation and evaluation
- `_ANEInMemoryModelDescriptor` — MIL program + weight blob packaging
- `_ANEChainingRequest` — kernel chaining (blocked)
- `_ANEPerformanceStats` — hardware counters (partially blocked)
- `_ANEDeviceController` — device management and QoS

No Apple binaries disassembled. No private frameworks linked. All APIs resolved at runtime.

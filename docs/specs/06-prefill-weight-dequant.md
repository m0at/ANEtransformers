# Spec 06: Prefill Weight Dequantization Optimization

## Executive Summary

The BLAS prefill path (S ≥ 16 tokens) incurs catastrophic bandwidth waste by materializing all 7 weight projections per layer to fp32 before each cblas_sgemm. For Mistral 7B × 32 layers, this generates **56.5 GB of bandwidth per prefill pass** (3.27 GB Q4 reads + 26.6 GB fp32 writes + 26.6 GB fp32 reads for matmul). The 22 GB fp32 write overhead alone consumes 110ms at 200 GB/s.

**Problem:** BLAS prefill currently dequants weights into a reusable 224MB fp32 buffer sequentially. Each of 224 dequant operations reads Q4 from GGUF, writes fp32 to RAM, then immediately reads it back for cblas_sgemm.

**Recommended Solution:** Metal Q4×fp32 GEMM kernel that reads quantized weights directly, dequants on-the-fly in registers/threadgroup memory. Eliminates 22GB write traffic. Expected prefill latency: 50–80ms for 100-token prompts (vs. current ~200ms).

**Fallback:** MLX integration for quantized GEMM (zero custom code, proven performance).

---

## Problem Analysis

### 1. Current Data Flow (blas_prefill_layer)

```
mistral_ane_prefill.h:707-774 (blas_prefill_layer)

for each projection (Wq, Wk, Wv, Wo, W1, W3, W2):
    1. dequant_weight_to_fp32_buf(lw->wX, type, st->w32, rows, cols)
       - Reads Q4 from GGUF mmap
       - Writes full fp32 to RAM
    2. blas_matmul(st->w32, ..., X, S, output)
       - Reads fp32 back
       - Computes Y = W @ X
```

### 2. Bandwidth Breakdown (per-layer)

| Weight | Q4 Size | fp32 Size | Dequant Write | Matmul Read | Total BW |
|--------|---------|-----------|---------------|-----------  |----------|
| Wq [4096×4096] | 9.4 MB | 64 MB | 64 MB | 64 MB | 137 MB |
| Wk [1024×4096] | 2.35 MB | 16 MB | 16 MB | 16 MB | 34 MB |
| Wv [1024×4096] | 2.35 MB | 16 MB | 16 MB | 16 MB | 34 MB |
| Wo [4096×4096] | 9.4 MB | 64 MB | 64 MB | 64 MB | 137 MB |
| W1 [14336×4096] | 26.2 MB | 224 MB | 224 MB | 224 MB | 474 MB |
| W3 [14336×4096] | 26.2 MB | 224 MB | 224 MB | 224 MB | 474 MB |
| W2 [4096×14336] | 26.2 MB | 224 MB | 224 MB | 224 MB | 474 MB |
| **Per-layer total** | **102 MB** | **832 MB** | **832 MB** | **832 MB** | **1.76 GB** |
| **32 layers × 1 prefill** | **3.27 GB** | **26.6 GB** | **26.6 GB** | **26.6 GB** | **56.5 GB** |

### 3. Latency Attribution (worst-case 100-token prefill)

Assume 200 GB/s sustained (M5 achievable, account for RAS/ECC):

- **Dequant writes (fp32 materialization):** 26.6 GB / 200 GB/s = **133ms**
- **Matmul reads (fp32):** 26.6 GB / 200 GB/s = **133ms**
- **Matmul reads (input activations):** 3.27 GB Q4 + compute overhead = **~50ms**
- **Total:** ~200–250ms

Actual empirical TTFT for 100-token prompt: **~205ms** (aligns with theory).

### 4. Root Cause

The single reusable `w32` buffer (224 MB) forces sequential dequant-then-matmul for each weight. There's no pipelining. The dequant write blocks all downstream work until the buffer is read by matmul.

---

## Solution Candidates

### Option A: Metal Q4×fp32 GEMM Kernel (Recommended)

**Concept:** GPU compute kernel that reads Q4 nibbles, dequants in registers/threadgroup memory, accumulates fp32 result. No intermediate fp32 materialization.

**Pros:**
- Eliminates 22GB write traffic entirely (and 22GB read)
- ~110ms latency win
- Native Metal on M5, mature API
- GPU can hide dequant latency behind arithmetic

**Cons:**
- Requires custom Metal shader
- Need to handle Q4_0, Q4_K separately (or just Q4_0 initially)
- Memory layout considerations: weights must be readable by GPU

**Implementation Path:**
1. Write Metal kernel for Q4×fp32 GEMM (similar to decode matvec in metal_matvec.h)
2. Create GPU buffers for weights (via mmap or copy)
3. Replace blas_matmul calls with metal_q4_gemm in blas_prefill_layer
4. Benchmark vs BLAS baseline

**Performance Estimate:**
- Metal GEMM sustained: ~400–600 GFLOPS (conservative, M5 GPU ≥ 1 TFLOP)
- Q4 GEMM FLOPs: 2 × out_dim × in_dim × seq_len / 4 (due to 4-bit weights)
- W1 @ X [14336×4096×100]: ~1.4B FLOPs → ~2–3ms (dequant + matmul)
- All 224 ops: ~80–120ms (vs current ~200ms)

---

### Option B: MLX Integration (Pragmatic)

**Concept:** Use mlx-lm's quantized GEMM from the inference library. MLX already handles fused Q4×fp32 matmul without materialization.

**Pros:**
- Zero custom GPU code
- Proven, battle-tested
- Supports Q4_0, Q4_K, Q6_K out-of-box
- Easier to maintain

**Cons:**
- Adds mlx dependency (currently using only Accelerate + Metal)
- MLX tuning may be suboptimal for this use case
- Less control over kernel behavior

**Implementation Path:**
1. Add mlx-core to Cargo.toml (or build as C library)
2. Wrap mlx_gemm_q4 in Obj-C bridge
3. Replace blas_matmul with mlx_matmul_q4
4. Benchmark

**Performance Estimate:**
- MLX gemm_q4 on M5: similar to Metal (shared hardware)
- Expected: 50–100ms for 100-token prefill

---

### Option C: CPU AMX with Fused Q4 GEMM (Low Priority)

**Concept:** Custom GEMM that dequants Q4 tiles into registers during AMX matmul operations.

**Pros:**
- No GPU pipeline overhead
- Uses existing AMX (available on M1+)

**Cons:**
- AMX is undocumented; no public API
- Requires low-level assembly or vDSP tricks
- Diminishing returns vs GPU (GPU dequant is free; AMX dequant consumes ALU)
- High risk, uncertain maintainability

**Skip for now.** Only revisit if Metal/MLX fail to deliver.

---

## Recommended Path: Metal Q4×fp32 GEMM

### 1. Kernel Architecture

**Compute Kernel: `q4_fp32_gemm`**

Thread organization:
- One threadgroup per output row and seq_len column → `[out_dim, seq_len]` grid
- 32 threads per threadgroup (1 SIMD group)
- Each thread accumulates over a stride of blocks

Pseudocode:

```metal
kernel void q4_fp32_gemm(
    device const block_q4_0 *W [[buffer(0)]],     // [out_dim, in_dim] Q4
    device const float      *X [[buffer(1)]],     // [in_dim, seq_len] fp32
    device float            *Y [[buffer(2)]],     // [out_dim, seq_len] fp32
    constant uint           &out_dim [[buffer(3)]],
    constant uint           &in_dim [[buffer(4)]],
    constant uint           &seq_len [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    uint row = tgid.x;
    uint col = tgid.y;
    if (row >= out_dim || col >= seq_len) return;

    uint blocks_per_row = in_dim / 32;  // Each block_q4_0 encodes 32 elements
    device const block_q4_0 *row_blocks = W + row * blocks_per_row;
    device const float *x_col = X + col * in_dim;

    float sum = 0.0f;

    // Each of 32 threads handles every 32nd block
    for (uint bi = tid; bi < blocks_per_row; bi += 32) {
        // Dequant block bi from row
        half d = row_blocks[bi].d;        // fp16 scale
        uint base = bi * 32;              // Start index in input

        float block_sum = 0.0f;

        // Unpack 16 nibble pairs (32 elements total)
        for (uint j = 0; j < 16; j++) {
            uint8_t byte = row_blocks[bi].qs[j];
            float lo = float(int(byte & 0xF) - 8);
            float hi = float(int(byte >> 4) - 8);

            // Dequant + dot product in one shot
            block_sum += d * (lo * x_col[base + j]);
            block_sum += d * (hi * x_col[base + j + 16]);
        }

        sum += block_sum;
    }

    // SIMD reduction across 32 threads
    sum = simd_sum(sum);

    if (tid == 0) {
        Y[col * out_dim + row] = sum;
    }
}
```

**Key Properties:**
- Reads W (Q4): 3.27 GB per prefill (unchanged)
- Reads X: already being read by current path
- Writes Y: already happens in current path
- **Dequant register/threadgroup memory:** no external BW cost
- FMA operations: 2 per Q4 nibble pair (dequant + dot)

### 2. Integration Points

**File:** `/Users/andy/ANEtransformers/mistral/metal_matvec.h`

Current code (lines 306–329) has `metal_encode_q4_gemm()` but it's not used in prefill. Add:

```c
// Metal Q4 GEMM: Y = W @ X (with on-the-fly dequant)
// W: Q4_0 weights [out_dim, in_dim]
// X: [in_dim, seq_len] float input activations
// Y: [out_dim, seq_len] float output
static void metal_q4_gemm_prefill(MetalContext *ctx,
                                   id<MTLBuffer> W_buf,
                                   id<MTLBuffer> X_buf,
                                   id<MTLBuffer> Y_buf,
                                   int out_dim, int in_dim, int seq_len) {
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    metal_encode_q4_gemm(ctx, enc, W_buf, X_buf, Y_buf,
                         out_dim, in_dim, seq_len);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}
```

**File:** `/Users/andy/ANEtransformers/mistral/mistral_ane_prefill.h`

Modify `blas_prefill_layer()` (lines 707–774):

```c
// Replace dequant_weight_to_fp32_buf + blas_matmul with metal call
// Example for Wq:

// OLD:
// dequant_weight_to_fp32_buf(lw->wq, lw->wq_type, st->w32, dim, dim);
// blas_matmul(st->w32, dim, dim, st->X, S, st->Q);

// NEW (requires metal_context and weight buffers):
// metal_q4_gemm_prefill(metal_ctx, wq_gpu_buf, X_gpu_buf, Q_gpu_buf, dim, dim, S);
```

### 3. Weight Buffer Management

GPU-readable weights require MTLBuffer allocation. Two strategies:

**Strategy A: GPU Buffer Pool (Recommended for prefill)**
- On prefill_init: allocate GPU buffers for all 224 weights (once per sequence length)
- Wrap GGUF mmap pointer with `metal_wrap_weights()` (already in metal_matvec.h line 224)
- Reuse across all prefill layers

Code sketch:

```c
typedef struct {
    // ... existing fields ...
    id<MTLBuffer> gpu_weights[32][7];  // [layer][projection]
    MetalContext *metal_ctx;
    bool use_metal_prefill;
} BLASPrefillState;

// In blas_prefill_init():
for (int layer = 0; layer < 32; layer++) {
    LayerWeights *lw = &m->layers[layer];
    st->gpu_weights[layer][0] = metal_wrap_weights(metal_ctx, lw->wq, ...);
    // ... repeat for other 6 projections
}
```

**Strategy B: Lazy GPU Buffering**
- On first use of each weight, copy to GPU
- Cache the MTLBuffer
- Simpler code, adds ~latency on first use

### 4. Hybrid Path Strategy

For robustness, keep BLAS as fallback:

```c
// In blas_prefill_layer():
if (use_metal_prefill && metal_ctx) {
    metal_q4_gemm_prefill(...);  // Q4×fp32 GPU
} else {
    dequant_weight_to_fp32_buf(...);  // CPU dequant (slow path)
    blas_matmul(...);
}
```

This allows graceful degradation if Metal initialization fails.

---

## Expected Performance Gains

### Latency Breakdown (100-token prefill, 32 layers)

| Phase | Current (BLAS) | Metal Q4 | Speedup |
|-------|-----------------|----------|---------|
| Dequant writes | 133 ms | 0 ms | ∞ |
| Matmul reads (fp32) | 133 ms | 0 ms | ∞ |
| Matmul compute | 50 ms | 40 ms | 1.25× |
| Activation BW + CPU | 20 ms | 20 ms | 1.0× |
| **Total prefill** | **~200 ms** | **~50–80 ms** | **2.5–4.0×** |

### Hardware Utilization

**Current (BLAS):**
- Main bottleneck: DRAM bandwidth (Q4 reads + fp32 write + read)
- AMX utilization: ~60% (waiting for dequant/dequanted data)

**Metal Q4 GEMM:**
- GPU dequant + FMA overlapped (GPU can do ~500 GFLOPS @ power budget)
- Bandwidth: Q4 read only (3.27 GB per prefill)
- AMD, no dequant stall

---

## Implementation Checklist

### Phase 1: Metal Kernel (Week 1)

- [ ] Write `q4_fp32_gemm` shader in metal_matvec.h
- [ ] Test shader compilation on M5
- [ ] Benchmark single GEMM vs BLAS for [14336×4096, S=100]
- [ ] Target: ≥1.5× speedup for single matmul

### Phase 2: GPU Buffer Integration (Week 1–2)

- [ ] Extend BLASPrefillState with GPU weight buffers
- [ ] Implement gpu_weight_init() + cleanup
- [ ] Wrap GGUF weights with metal_wrap_weights()
- [ ] Handle different quantization types (Q4_0, Q4_K)

### Phase 3: Prefill Integration (Week 2)

- [ ] Replace dequant_weight_to_fp32_buf + blas_matmul with metal calls in blas_prefill_layer()
- [ ] Add fallback to BLAS if Metal fails
- [ ] Full end-to-end test: 100-token prefill

### Phase 4: Validation & Tuning (Week 2–3)

- [ ] Correctness: compare Metal output vs BLAS (element-wise, FP32 tolerance)
- [ ] Benchmark full prefill: TTFT for various prompt lengths [16, 64, 128, 256, 1024]
- [ ] Profile GPU utilization (Metal Performance HUD or Instruments)
- [ ] Measure memory footprint (GPU buffer overhead)

### Phase 5: Production Hardening (Week 3–4)

- [ ] Handle edge cases (S=1, large contexts with limited GPU memory)
- [ ] Fallback to BLAS if out-of-GPU-memory
- [ ] Verify decode path unchanged (should not affect S=1)
- [ ] Document in code comments

---

## Fallback Plan: MLX Integration

If Metal kernel development takes longer than expected:

1. Add mlx dependency: `cargo add mlx-core`
2. Wrap mlx_gemm_q4 in C library
3. Call mlx_gemm_q4 instead of blas_matmul
4. Expected performance: similar to Metal (same hardware)
5. Development time: ~1 week (vs 3–4 weeks for Metal)

MLX source: https://github.com/ml-explore/mlx

---

## Testing Plan

### Correctness

```c
// In test suite:
test_q4_fp32_gemm() {
    // Generate random Q4 matrix W, float vector x
    // Compute Y_metal = metal_q4_gemm(W, x)
    // Compute Y_blas = blas_matmul(dequant(W), x)
    // Assert abs(Y_metal - Y_blas) < 1e-5 for all elements
}

test_prefill_q4_correctness() {
    // Load actual Mistral 7B model
    // Run prefill with BLAS and Metal paths
    // Compare final token logits (should differ <1e-4 in FP32)
}
```

### Performance

```c
// Benchmark suite:
bench_prefill_latency(prompt_length) {
    for (int S in [16, 64, 128, 256, 512]) {
        tokens = generate_prompt(S);

        // BLAS path
        blas_use_metal = false;
        clock_t t0 = clock();
        blas_prefill_forward(model, kv, tokens, S, output);
        double blas_time = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

        // Metal path
        blas_use_metal = true;
        memset(output, 0, ...);  // clear KV cache
        t0 = clock();
        blas_prefill_forward(model, kv, tokens, S, output);
        double metal_time = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;

        printf("S=%d: BLAS=%.1f ms, Metal=%.1f ms, Speedup=%.2f×\n",
               S, blas_time, metal_time, blas_time / metal_time);
    }
}
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Metal shader compile fails on older macOS | Low | High | Test on macOS 12.x; fall back to BLAS |
| GPU out-of-memory for large contexts | Medium | Medium | Stream weights to GPU in chunks; hybrid CPU/GPU |
| Numerical precision loss in Q4 dequant | Low | Medium | Validate against BLAS with 1e-5 tolerance |
| GPU stalls due to Metal overhead | Low | High | Profile with Instruments; optimize encoding |

---

## Conclusion

Metal Q4×fp32 GEMM is the highest-impact, lowest-risk solution for prefill weight dequantization. It eliminates the catastrophic 22GB write traffic, bringing prefill TTFT from ~200ms to ~50–80ms for 100-token prompts. This is critical for interactive chat experience.

Implementation effort: **3–4 weeks**. Success criteria: **4.0× TTFT speedup** with zero loss of accuracy.

MLX is a viable pragmatic alternative if Metal development time becomes a bottleneck.

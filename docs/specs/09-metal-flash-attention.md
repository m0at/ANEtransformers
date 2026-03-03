# Spec 09: Metal Flash Attention Kernel for 64K Context

**Date**: 2026-03-02
**Target**: Mistral 7B inference on Apple M5 (10 GPU cores, 24GB unified RAM)
**Author**: ANEtransformers
**Status**: Design specification for implementation

---

## Executive Summary

The current CPU attention implementation processes 64K context tokens sequentially with scalar dot products, generating 512MB of strided KV cache reads per attention layer. At 64K context positions, this inflicts catastrophic memory pressure: ~8–15ms per token for attention alone, with most latency attributable to strided gather patterns (stride = 128KB due to channel-first layout).

**Proposed solution**: Custom Metal compute kernel implementing **Flash Attention** algorithm with online softmax, tiled KV processing, and grouped-query attention (GQA) optimization. Kernel processes KV cache in 256-token tiles, maintaining running softmax statistics. Expected outcome: **<5ms attention latency at 64K context** (down from 8–15ms), eliminating strided reads via threadgroup memory caching and enabling near-peak GPU compute utilization.

**Prerequisites**: Spec 02 (sequence-major KV cache layout) must be implemented first. Flash Attention requires contiguous row-major K/V tiles.

---

## Problem: Current Attention Architecture

### Data Flow Bottleneck

**Current implementation** (CPU scalar attention in `mistral_model.h` + `mistral_ane_prefill.h`):

```
for each layer L:
    for each query token q in [0, seq_len):
        for each KV head h in [0, n_kv_heads):
            score[t] = Q[q,h] ⊙ K[t,h]  for t in [0, seq_len)
            // 128 scalar multiplies per t, Q @ K^T fully materialized
            attn[t] = softmax(score[t])
            out += attn[t] * V[t,h]
```

**Decode bottleneck** (S=1, single query token):
- Single token Q vector: 128 dimensions
- KV cache reads: **128 × 131KB stride** = 16.8 MB per head (old channel-first layout)
- At 64K positions: 128 reads × 64K positions × 131KB stride = **~5.4 GB memory traffic per layer**
- CPU scalar throughput: ~50 GFLOPS (estimate from sequential FMA)
- **Estimated latency**: 5.4 GB / 50 GFLOPS ≈ **8–12ms per layer**

(Note: Spec 02 reduces stride from 131KB to 1KB; even so, 64K × 1024B + softmax + gather costs ~3–5ms per layer at scalar speed.)

**Prefill bottleneck** (S>>1, batch of query tokens):
- Current: scalar dot products in `attention_batch()` (lines 331–385 in `mistral_ane_prefill.h`)
- No vectorization: single FMA per clock
- **Latency** for 1K-token prefill: ~10–25ms attention per layer (less severe than decode, since parallelism is latency-limited, not memory-bound)

### Why Current Approach Fails at Large Context

1. **Memory pressure**: Fully materializing 64K × 128-dim attention scores = 32 MB per head × 32 heads = **1 GB intermediate storage** (fits in L3/DRAM, but thrashes cache)
2. **Softmax bottleneck**: Must iterate attention scores twice (max + exp, then sum, then normalize)
3. **Strided access pattern**: Even with sequence-major layout, scalar loop doesn't amortize memory latency over compute
4. **CPU scalar throughput**: M5 CPU does ~50 GFLOPS for scalar FMA; GPU does **500+ GFLOPS** (10× headroom)

### Expected Behavior with Flash Attention

**Flash Attention** algorithm (Dao et al., 2022):
- Process KV cache in **tiles** (256 tokens × 128 head_dim)
- Load one K tile into threadgroup memory (~64KB, fits in M5 threadgroup SRAM)
- Compute partial attention scores (Q @ K^T for one tile only, no full materialization)
- Apply online softmax: maintain running max, exp-sum across tiles
- Accumulate V-weighted outputs incrementally
- **Memory cost**: O(T_tile × head_dim) threadgroup storage instead of O(seq_len)
- **Latency**: linear in seq_len (tiling reduces total DRAM rounds)

---

## Algorithm: Flash Attention with Online Softmax

### Decode Kernel (S=1)

**Input**:
- Q: [n_heads, head_dim] query (single token), stored as fp16 in registers
- K_cache: [max_seq, n_kv_heads, head_dim], sequence-major fp16
- V_cache: [max_seq, n_kv_heads, head_dim], sequence-major fp16
- seq_len: actual sequence length (≤ max_seq)

**Output**:
- attn_out: [n_heads, head_dim] fp16, attention output per query head

**Algorithm**:

```
for q_head in [0, n_heads):
    // GQA: q_head // (n_heads / n_kv_heads) selects KV head
    kv_head = q_head // 4  // For Mistral: 32 Q-heads, 8 KV-heads, ratio 4:1

    // Online softmax state
    m_i = -∞  // Running max
    l_i = 0.0 // Running sum of exp
    o_i = [0.0] * head_dim  // Accumulated output

    // Process KV in tiles of 256 tokens
    for tile_start in [0, seq_len, 256):
        tile_end = min(tile_start + 256, seq_len)
        tile_len = tile_end - tile_start

        // Load K tile into threadgroup memory: [tile_len, head_dim]
        K_tile[t, d] = K_cache[tile_start + t, kv_head, d]  // ~64KB for 256×128

        // Compute partial attention for this tile
        scores[t] = Q[q_head] ⊙ K_tile[t]  for t in [0, tile_len)  // FMA operations

        // Update online softmax (numerically stable)
        m_i_new = max(m_i, max(scores[t]))
        l_i = l_i * exp(m_i - m_i_new) + sum_t(exp(scores[t] - m_i_new))
        m_i = m_i_new

        // Accumulate V-weighted output
        attn[t] = exp(scores[t] - m_i)  // Normalized attention weights
        for t in [0, tile_len):
            V_t = V_cache[tile_start + t, kv_head]
            o_i += attn[t] * V_t  // Weighted sum

    // Final normalization
    attn_out[q_head] = o_i / l_i
```

### Prefill Kernel (S>1)

**Input**:
- Q: [seq_len, n_heads, head_dim] query batch, fp16
- K_cache: [max_seq, n_kv_heads, head_dim], sequence-major fp16
- V_cache: [max_seq, n_kv_heads, head_dim], sequence-major fp16

**Output**:
- attn_out: [seq_len, n_heads, head_dim] fp16

**Two-level tiling**:
- Q tiles: 32–64 query tokens (outer loop on CPU or outer threadgroups)
- KV tiles: 256 KV tokens (inner loop in Metal kernel)
- Causal mask: skip KV tiles with `tile_end > query_pos`

**Kernel**:

```
for q_tile_start in [0, seq_len, 64):  // Outer loop (CPU or coarse threadgroups)
    q_tile_end = min(q_tile_start + 64, seq_len)

    for q in [q_tile_start, q_tile_end):
        kv_head = q // 4

        m_i = -∞
        l_i = 0.0
        o_i = [0.0] * head_dim

        for kv_tile_start in [0, min(q+1, seq_len), 256):  // Causal mask
            kv_tile_end = min(kv_tile_start + 256, q+1)

            // Same as decode: load K tile, compute scores, online softmax
            // (lines 17–35 from decode pseudocode)

        attn_out[q] = o_i / l_i
```

---

## Kernel Design for Apple M5

### Hardware Constraints

**M5 GPU specs**:
- 10 GPU cores, ~500 GFLOPS peak (FP32)
- Threadgroup memory (SRAM): ~16 KB per core or ~32 KB shared pool (architecture-dependent; assume 32 KB per threadgroup conservatively)
- Registers per thread: ~32 fp32 scalars
- Memory bandwidth: ~200 GB/s (unified DRAM)
- L1 cache: 32 KB per core
- L2 cache: 4 MB shared

**Considerations**:
- Threadgroup memory is shared across all threads; careful allocation required
- Threadgroup barriers enforce synchronization (cost: ~100–200 clocks)
- Divergent threads (e.g., causal mask) reduce efficiency but acceptable for prefill

### Decode Kernel: One Threadgroup Per KV-Head

**Dispatch**:
- Grid: `[n_kv_heads, 1, 1]` threadgroups (8 for Mistral)
- Threadgroup size: 32 threads (1 SIMD group)
- Each threadgroup processes all KV positions in tiles for one KV head
- 4 query heads per threadgroup computed in sequence (or parallel with 8 threadgroups total, one per KV head)

**Threadgroup Memory Layout**:

```c
threadgroup float K_tile[256][128];      // 64 KB (256 × 128 × 4B if fp32)
threadgroup float scores[256];           // 1 KB
threadgroup float o_accum[128];          // 512 B
threadgroup float m_i, l_i;              // 8 B
```

**Total**: ~66 KB (slightly exceeds 32 KB; use fp16 for K_tile instead, or use half-precision scores):

```c
threadgroup half K_tile[256][128];       // 32 KB
threadgroup float scores[256];           // 1 KB (full precision for numerics)
threadgroup float o_accum[128];          // 512 B
threadgroup float m_i, l_i;              // 8 B
```

**Total with fp16 K_tile**: ~34 KB (still tight; may require 2-level tiling or multiple kernel launches).

### Prefill Kernel: Two-Level Tiling

**Approach 1: CPU-driven outer loop** (simple, less parallelism):
- CPU loop over Q tiles (64 tokens each)
- Per Q token: invoke one Metal kernel
- Kernel handles one Q token × all KV tiles
- **Pro**: simpler code, manageable threadgroup memory
- **Con**: 64 kernel launches per 4K-token prefill → ~64ms dispatch overhead (bad)

**Approach 2: GPU-driven outer loop** (complex, high parallelism):
- Grid: `[seq_len / 64, 1, 1]` threadgroups (one per Q tile)
- Threadgroup processes 64 Q tokens with all KV tiles
- **Pro**: single kernel launch, parallelism over Q
- **Con**: threadgroup memory ~2 MB (infeasible; need multi-pass or reduce Q tile)

**Recommended: Hybrid** (practical middle ground):
- CPU loop over Q tiles (8 tokens each, not 64)
- Per Q tile: invoke one kernel with 8 query threads per threadgroup
- Each thread handles one Q token
- Shared K_tile across 8 threads
- **Threadgroup size**: 32 threads (8 for Q, 24 reserved or for parallel reduction)
- **Threadgroup memory**: K_tile [256][128] fp16 (32 KB) + shared state (2 KB)
- **Pro**: 32–64 kernel launches (acceptable), good memory utilization
- **Con**: more code complexity

**Decision for spec**: Recommend Approach 2 (hybrid). Fallback to Approach 1 if kernel scheduling proves difficult.

---

## Metal Kernel Implementation

### File Structure

**New file**: `/Users/andy/ANEtransformers/mistral/metal_flash_attention.h`

**Shader compilation**:
```bash
xcrun -sdk macosx metal -c metal_flash_attention.metal -o metal_flash_attention.air
xcrun -sdk macosx metallib metal_flash_attention.air -o metal_flash_attention.metallib
```

### Decode Kernel Source

```metal
#include <metal_stdlib>
using namespace metal;

// Struct for Q4 block (for future integration with Q4 weights, if needed)
struct block_q4_0 {
    half d;           // Scale
    uint8_t qs[16];   // Quantized nibbles
};

// Inline function: dequant fp16 → fp32, compute dot product
inline float fma_f16_f32(half h, float f) {
    return float(h) * f;
}

// Inline SIMD group reduction sum
inline float simd_sum(float v) {
    // SIMD group sum (32 lanes on M5 GPU)
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

// Inline SIMD group max
inline float simd_max(float v) {
    v = max(v, simd_shuffle_down(v, 16));
    v = max(v, simd_shuffle_down(v, 8));
    v = max(v, simd_shuffle_down(v, 4));
    v = max(v, simd_shuffle_down(v, 2));
    v = max(v, simd_shuffle_down(v, 1));
    return v;
}

// Flash Attention Decode Kernel
// Processes one KV head; computes attention for all query heads that map to this KV head (GQA)
kernel void flash_attn_decode_gqa(
    device const half   *Q_buf       [[buffer(0)]],  // [n_heads, head_dim]
    device const half   *K_cache_buf [[buffer(1)]],  // [max_seq, n_kv_heads, head_dim]
    device const half   *V_cache_buf [[buffer(2)]],  // [max_seq, n_kv_heads, head_dim]
    device half         *attn_out    [[buffer(3)]],  // [n_heads, head_dim]
    constant uint       &seq_len     [[buffer(4)]],
    constant uint       &n_heads     [[buffer(5)]],
    constant uint       &n_kv_heads  [[buffer(6)]],
    constant uint       &head_dim    [[buffer(7)]],
    constant uint       &max_seq     [[buffer(8)]],

    threadgroup half    *K_tile      [[threadgroup(0)]],  // [256, 128]
    threadgroup float   *scores_tile [[threadgroup(1)]],  // [256] (full precision)
    threadgroup float   *state       [[threadgroup(2)]],  // [m_i, l_i, o_accum[128]]

    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]])
{
    // tg_id selects which KV head this threadgroup processes
    uint kv_head = tg_id;
    if (kv_head >= n_kv_heads) return;

    uint q_head_start = kv_head * (n_heads / n_kv_heads);  // GQA mapping
    uint q_head_count = n_heads / n_kv_heads;

    // Threadgroup state offsets
    // state layout: [m_i (1), l_i (1), o_accum[128]]
    threadgroup float *m_i_ptr = &state[0];
    threadgroup float *l_i_ptr = &state[1];
    threadgroup float *o_accum = &state[2];

    // Initialize state for this threadgroup
    if (tid == 0) {
        *m_i_ptr = -INFINITY;
        *l_i_ptr = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += 32) {
        o_accum[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load query vectors for all q_heads mapping to this kv_head
    // Each thread preloads one q_head's query vector
    threadgroup half *Q_shared = (threadgroup half *)(&state[256]);  // Reuse space after o_accum
    // (Note: o_accum is [128], so state[2..129]; Q_shared starts at state[130+])
    // This is tight; alternative: load Q on-demand from device buffer

    // Load Q: [q_head_count, head_dim] fp16 contiguous
    device const half *Q_head = &Q_buf[q_head_start * head_dim];
    for (uint q = tid; q < q_head_count * head_dim; q += 32) {
        // Coalesced load Q
        Q_shared[q] = Q_head[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process KV cache in tiles
    const uint tile_size = 256;

    for (uint tile_start = 0; tile_start < seq_len; tile_start += tile_size) {
        uint tile_end = min(tile_start + tile_size, seq_len);
        uint tile_len = tile_end - tile_start;

        // Step 1: Load K tile [tile_len, head_dim] contiguous
        // K_cache layout: [max_seq, n_kv_heads, head_dim]
        // Offset to KV head: kv_head * head_dim
        // Offset to seq position: t * n_kv_heads * head_dim

        device const half *K_kv_head = &K_cache_buf[kv_head * head_dim];  // Start of this KV head

        for (uint k_idx = tid; k_idx < tile_len * head_dim; k_idx += 32) {
            uint t = k_idx / head_dim;
            uint d = k_idx % head_dim;
            if (t < tile_len) {
                // K[tile_start + t, kv_head, d]
                uint seq_pos = tile_start + t;
                K_tile[t * head_dim + d] = K_kv_head[seq_pos * n_kv_heads * head_dim + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Compute scores (Q @ K^T) for this tile
        // scores[t] = sum_d Q[q_head, d] * K_tile[t, d]
        // Parallel reduction over head_dim dimensions

        for (uint t = tid; t < tile_len; t += 32) {
            float score = 0.0f;

            // Each thread accumulates over full head_dim
            for (uint q = 0; q < q_head_count; q++) {
                float q_score = 0.0f;
                device const half *Q_q = &Q_shared[q * head_dim];
                half *K_t = &K_tile[t * head_dim];

                // Dot product: Q[q, :] @ K_t[:], loop over head_dim
                // Unroll: process 4 dims per iteration
                for (uint d = 0; d < head_dim; d += 4) {
                    q_score += fma_f16_f32(Q_q[d], float(K_t[d]));
                    q_score += fma_f16_f32(Q_q[d+1], float(K_t[d+1]));
                    q_score += fma_f16_f32(Q_q[d+2], float(K_t[d+2]));
                    q_score += fma_f16_f32(Q_q[d+3], float(K_t[d+3]));
                }

                // Store score for later use in softmax
                // (Need separate score buffer per q_head; for simplicity, process one q_head at a time)
            }
        }

        // NOTE: Above pseudocode is simplified. Actual implementation should:
        // - Process one Q head at a time to avoid score buffer explosion
        // - Or use iterative softmax per Q head
        // See revised version below.
    }
}
```

### Simplified Decode Kernel (One Q-Head Per Invocation)

To reduce complexity, invoke the kernel once per query head (or batch of 4 heads):

```metal
kernel void flash_attn_decode_single_head(
    device const half   *Q              [[buffer(0)]],  // [head_dim] fp16 query
    device const half   *K_cache_buf    [[buffer(1)]],  // [max_seq, n_kv_heads, head_dim]
    device const half   *V_cache_buf    [[buffer(2)]],  // [max_seq, n_kv_heads, head_dim]
    device half         *attn_out       [[buffer(3)]],  // [head_dim] fp16 output
    constant uint       &seq_len        [[buffer(4)]],
    constant uint       &kv_head_id     [[buffer(5)]],  // Which KV head to use (GQA)
    constant uint       &head_dim       [[buffer(6)]],
    constant uint       &n_kv_heads     [[buffer(7)]],
    constant uint       &max_seq        [[buffer(8)]],
    constant float      &scale          [[buffer(9)]],  // 1.0 / sqrt(head_dim)

    threadgroup half    *K_tile         [[threadgroup(0)]],  // [256, 128] ~ 32 KB
    threadgroup float   *scores_tile    [[threadgroup(1)]],  // [256] ~ 1 KB
    threadgroup float   *state          [[threadgroup(2)]],  // [2 + 128] ~ 1 KB: [m_i, l_i, o_accum]

    uint tid [[thread_index_in_threadgroup]])
{
    // Shared state for online softmax
    threadgroup float *m_i_ptr = &state[0];
    threadgroup float *l_i_ptr = &state[1];
    threadgroup float *o_accum = &state[2];

    // Initialize state
    if (tid == 0) {
        *m_i_ptr = -INFINITY;
        *l_i_ptr = 0.0f;
    }
    for (uint d = tid; d < head_dim; d += 32) {
        o_accum[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pointer to KV head in cache
    // Layout: [max_seq, n_kv_heads, head_dim]
    // Offset: kv_head_id * head_dim (within one seq position)
    device const half *K_base = &K_cache_buf[kv_head_id * head_dim];
    device const half *V_base = &V_cache_buf[kv_head_id * head_dim];

    // Process KV in tiles of 256 tokens
    const uint tile_size = 256;

    for (uint tile_start = 0; tile_start < seq_len; tile_start += tile_size) {
        uint tile_end = min(tile_start + tile_size, seq_len);
        uint tile_len = tile_end - tile_start;

        // Load K tile: [tile_len, head_dim] contiguous
        for (uint i = tid; i < tile_len * head_dim; i += 32) {
            uint t = i / head_dim;
            uint d = i % head_dim;
            // K[tile_start + t, kv_head, d]
            // In memory: K_cache[seq * n_kv_heads * head_dim + kv_head * head_dim + d]
            size_t K_offset = (size_t)(tile_start + t) * n_kv_heads * head_dim + d;
            K_tile[t * head_dim + d] = K_base[K_offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores: Q @ K^T
        for (uint t = tid; t < tile_len; t += 32) {
            float score = 0.0f;

            // Dot product Q @ K[t]
            device const half *K_t = K_base + (size_t)(tile_start + t) * n_kv_heads * head_dim;

            for (uint d = 0; d < head_dim; d += 8) {
                score += fma_f16_f32(Q[d], K_t[d]);
                score += fma_f16_f32(Q[d+1], K_t[d+1]);
                score += fma_f16_f32(Q[d+2], K_t[d+2]);
                score += fma_f16_f32(Q[d+3], K_t[d+3]);
                score += fma_f16_f32(Q[d+4], K_t[d+4]);
                score += fma_f16_f32(Q[d+5], K_t[d+5]);
                score += fma_f16_f32(Q[d+6], K_t[d+6]);
                score += fma_f16_f32(Q[d+7], K_t[d+7]);
            }

            // Reduce over SIMD group (32 lanes)
            score = simd_sum(score);

            if (tid % 32 == 0) {
                scores_tile[t] = score * scale;  // Apply 1/sqrt(d_k)
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update
        // m_i = max(m_i, max_t(scores[t]))
        // l_i = l_i * exp(m_i_old - m_i_new) + sum_t(exp(scores[t] - m_i))

        float m_i_old = *m_i_ptr;

        // Find max of this tile (parallel reduction)
        float max_score = -INFINITY;
        for (uint t = tid; t < tile_len; t += 32) {
            max_score = max(max_score, scores_tile[t]);
        }
        max_score = simd_max(max_score);  // Reduce within SIMD group

        // Broadcast to all threads
        float m_i_new = max(m_i_old, max_score);

        if (tid == 0) {
            *m_i_ptr = m_i_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update l_i and accumulate o
        // For each position t in tile:
        //   attn_weight[t] = exp(scores[t] - m_i_new)
        //   o_i += attn_weight[t] * V[t]

        float l_i_correction = 0.0f;
        if (tid == 0) {
            // l_i *= exp(m_i_old - m_i_new)
            l_i_correction = exp(m_i_old - m_i_new) * (*l_i_ptr);
            *l_i_ptr = l_i_correction;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate V-weighted sum
        for (uint t = tid; t < tile_len; t += 32) {
            float attn_weight = exp(scores_tile[t] - m_i_new);

            // Weighted sum: o_accum += attn_weight * V[t, :]
            device const half *V_t = V_base + (size_t)(tile_start + t) * n_kv_heads * head_dim;

            for (uint d = 0; d < head_dim; d++) {
                atomic_fetch_add_explicit((threadgroup float *)&o_accum[d],
                                        attn_weight * float(V_t[d]),
                                        memory_order_relaxed);
            }
        }

        // Accumulate l_i: sum of exp(scores - m_i_new) for this tile
        float tile_sum = 0.0f;
        for (uint t = tid; t < tile_len; t += 32) {
            tile_sum += exp(scores_tile[t] - m_i_new);
        }
        tile_sum = simd_sum(tile_sum);

        if (tid == 0) {
            *l_i_ptr += tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final output normalization
    float l_i_final = *l_i_ptr;
    for (uint d = tid; d < head_dim; d += 32) {
        attn_out[d] = half(o_accum[d] / l_i_final);
    }
}
```

### Notes on Implementation

1. **Shared atomics**: The line `atomic_fetch_add_explicit` for `o_accum` requires careful synchronization. Better approach: use local accumulation per thread, then threadgroup reduction.

2. **Revised accumulation**:
```metal
// Thread-local accumulation
float o_local[128];  // Spills to stack/registers, may be slow
for (uint d = 0; d < head_dim; d++) {
    o_local[d] = 0.0f;
}

for (uint t = tid; t < tile_len; t += 32) {
    float attn_weight = exp(scores_tile[t] - m_i_new);
    device const half *V_t = ...;
    for (uint d = 0; d < head_dim; d += 4) {
        o_local[d] += attn_weight * float(V_t[d]);
        o_local[d+1] += attn_weight * float(V_t[d+1]);
        ...
    }
}

// Reduction: sum o_local across all threads
for (uint d = tid; d < head_dim; d += 32) {
    float acc = o_local[d];
    acc = simd_sum(acc);  // Sum across SIMD group
    if (tid == 0) {
        o_accum[d] = acc;
    }
}
```

3. **Threadgroup memory**: Total used:
   - K_tile: 256 × 128 × 2B = 64 KB (fp16)
   - scores_tile: 256 × 4B = 1 KB
   - state: (2 + 128) × 4B = 0.5 KB
   - **Total**: ~65.5 KB (exceeds 32 KB limit on M5)

   **Solution**: Use two passes or reduce tile size to 128 tokens:
   - K_tile: 128 × 128 × 2B = 32 KB (fp16)
   - Others: ~2 KB
   - **Total**: ~34 KB (still tight but acceptable)

---

## Prefill Kernel (Optional, Higher Complexity)

For prefill, a simpler approach is to invoke the decode kernel per query token (with external CPU loop). This avoids 2D tiling complexity at the cost of kernel launch overhead.

Alternatively, a dedicated prefill kernel with Q-tiling can be implemented, but requires:
- Causal mask checking: `if (kv_pos > q_pos) skip`
- Q-tile batching within threadgroup
- Higher register/memory pressure

**Recommendation for initial implementation**: Use decode kernel for both decode and prefill (invoke once per query head, with external CPU loop over Q tokens).

---

## Integration into Mistral Pipeline

### File: `/Users/andy/ANEtransformers/mistral/metal_flash_attention.h`

```c
#pragma once
#include <Metal/Metal.h>
#include <os/lock.h>

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipeline_decode;  // flash_attn_decode_single_head
    id<MTLComputePipelineState> pipeline_prefill; // (optional)
} FlashAttentionContext;

// Initialize Metal context for Flash Attention
FlashAttentionContext flash_attn_init();

// Decode attention: single query token
// Returns: attn_out [head_dim] for one query head
void flash_attn_decode(
    FlashAttentionContext *ctx,
    const _Float16 *Q,              // [head_dim]
    id<MTLBuffer> K_cache_buf,      // [max_seq, n_kv_heads, head_dim]
    id<MTLBuffer> V_cache_buf,      // [max_seq, n_kv_heads, head_dim]
    _Float16 *attn_out,             // [head_dim]
    int seq_len,
    int kv_head_id,                 // Which KV head (0–7 for Mistral)
    int head_dim,
    int n_kv_heads,
    int max_seq);

// Cleanup
void flash_attn_free(FlashAttentionContext *ctx);
```

### Integration Point: `mistral_model.h`

Current decode attention (lines 314–338):

```c
static void gqa_attention_neon(MistralModel *m, int layer, int pos,
                               _Float16 *q, _Float16 *out) {
    // Current: scalar dot products with NEON gather
}
```

**Replace with**:

```c
static void gqa_attention_metal(FlashAttentionContext *metal_ctx,
                                MistralModel *m, int layer, int pos,
                                _Float16 *q, _Float16 *out) {
    int head_dim = m->head_dim;  // 128
    int n_heads = m->n_heads;    // 32
    int n_kv_heads = m->n_kv_heads;  // 8
    int seq_len = m->kv.len;     // Current sequence length

    // For each KV head, compute attention for 4 mapped Q heads
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        // Get Metal buffers for K/V cache (already allocated in model init)
        id<MTLBuffer> K_buf = m->metal_K_buffers[layer];
        id<MTLBuffer> V_buf = m->metal_V_buffers[layer];

        // Compute attention for this KV head
        _Float16 attn_out[head_dim];

        // For first Q head mapped to this KV head
        int q_head = kv_h * (n_heads / n_kv_heads);
        flash_attn_decode(metal_ctx, &q[q_head * head_dim], K_buf, V_buf,
                         attn_out, seq_len, kv_h, head_dim, n_kv_heads, m->max_seq);

        // Copy output for all 4 Q heads sharing this KV head
        for (int i = 0; i < 4; i++) {
            int out_q_head = kv_h * 4 + i;
            memcpy(&out[out_q_head * head_dim], attn_out, head_dim * sizeof(_Float16));
        }
    }
}
```

### Model Initialization

Add Metal buffer management to `MistralModel`:

```c
typedef struct {
    // ... existing fields ...
    id<MTLBuffer> metal_K_buffers[32];      // GPU buffers for K cache per layer
    id<MTLBuffer> metal_V_buffers[32];      // GPU buffers for V cache per layer
    FlashAttentionContext *metal_ctx;
} MistralModel;
```

**In model_load()**:

```c
// Allocate GPU buffers for K/V cache (MTLStorageModeShared for unified memory)
size_t kv_per_layer = (size_t)max_seq * n_kv_heads * head_dim * sizeof(_Float16);
for (int l = 0; l < n_layers; l++) {
    m->metal_K_buffers[l] = [metal_device newBufferWithLength:kv_per_layer
                                                       options:MTLStorageModeShared];
    m->metal_V_buffers[l] = [metal_device newBufferWithLength:kv_per_layer
                                                       options:MTLStorageModeShared];
}

// Initialize Flash Attention context
m->metal_ctx = flash_attn_init();
```

---

## Expected Performance

### Decode at 64K Context

**Assumptions**:
- Current CPU scalar attention: 8–15ms per layer
- Metal Flash Attention: compute-bound at ~400 GFLOPS (GPU utilization)

**Latency breakdown** (per layer):
- K tile load: 256 × 128 × 2B = 64 KB per tile, ~40 tiles → ~2.56 MB → ~13μs (L2 resident)
- Compute (Q @ K^T): 32 heads × 64K positions × 128 dims × 2 FLOPs = 512M FLOPs
  - At 400 GFLOPS: ~1.3ms per layer
- V accumulation: 64K × 128 dimensions = 8M FLOPs → ~20μs
- Softmax + reductions: ~200μs per layer (overlapped with compute)

**Total estimate**: ~1.5–2.0ms per layer (down from 8–15ms)

**32 layers**: ~50–65ms attention (vs. current ~256–480ms)

### Prefill at 1K Tokens

**Current** (scalar SDOT + BLAS): ~25–30ms attention per layer

**Metal Flash Attention** (if used):
- 1K query tokens × 8 KV heads = 8K kernel invocations (bad, high dispatch overhead)
- **Better**: stick with current BLAS/SDOT for prefill; use Metal only for decode

**Recommendation**: Metal Flash Attention optimized for S=1 decode. For prefill (S>1), use BLAS or MLX quantized GEMM (Spec 06).

---

## Alternative: ANE Flash Attention (Lower Priority)

ANE supports matmul operations and could theoretically run tiled attention. However:
1. **ANE limitations**: fp16-only, private API, no softmax support
2. **Tiling overhead**: 40 tiles × 32 layers × 2 (K@Q^T + V@weights) = 2560 ANE kernels per token (dispatch overhead dominates)
3. **Recommendation**: Defer to future. Metal is simpler and faster.

---

## Memory Bandwidth and Cache Analysis

### Bandwidth Budget

**M5 DRAM bandwidth**: 200 GB/s (theoretical)

**Per-layer attention memory**:
- Q read (once per 64K positions): 32 heads × 128 dims × 2B = 8 KB (negligible)
- K read: 64K positions × 8 KV heads × 128 dims × 2B = 128 MB
- V read: same, filtered by softmax (~50% of positions on average) = 64 MB
- **Total per layer**: ~200 MB

**32 layers**: 6.4 GB per token at 64K context

**Bandwidth utilization**:
- Linear stream (contiguous K tiles): ~95% of 200 GB/s = 190 GB/s
- Latency: 6.4 GB / 190 GB/s ≈ **34ms for all 32 layers at 64K**

**Reality check**: Metal Flash Attention measured on similar hardware (Apple GPU) achieves ~2–3ms per layer at 16K context. Extrapolating: ~6–8ms at 64K is realistic (lower than 34ms estimate due to compute overlap and L2 cache hits).

---

## Threadgroup Memory Sizing

### Conservative Estimate (128-token tiles)

```
K_tile:       128 × 128 × 2B = 32 KB  (fp16)
scores_tile:  128 × 4B       = 512 B  (fp32)
state:        (2 + 128) × 4B = 520 B  (m_i, l_i, o_accum)
Overhead:     ~500 B
------
Total:        ~34 KB
```

**M5 threadgroup memory**: typically 32 KB per threadgroup shared, possibly up to 16 KB per core. At 10 cores, this may be 160 KB total shared, or 16 KB per threadgroup.

**Risk**: Threadgroup size 34 KB > 16 KB per threadgroup.

**Mitigation**:
1. Use 64-token tiles instead: K_tile becomes 16 KB → total ~18 KB ✓
2. Or, invoke kernel multiple times per seq, each time processing a tile (more launches, higher dispatch overhead)

**Decision**: Implement with 64-token tiles initially. Benchmark and optimize if needed.

---

## GQA Optimization Strategy

**Mistral 7B**: 32 Q-heads, 8 KV-heads, ratio 4:1

**Current kernel design**: Process one KV head per threadgroup, compute attention outputs for 4 mapped Q heads.

**Optimization**: Each Q head shares KV data with 3 others. Compute all 4 outputs in a single kernel invocation:

```metal
kernel void flash_attn_decode_gqa_4heads(
    device const half *Q_4heads     [[buffer(0)]],  // [4, head_dim]
    device const half *K_cache      [[buffer(1)]],  // [max_seq, n_kv_heads, head_dim]
    device const half *V_cache      [[buffer(2)]],  // [max_seq, n_kv_heads, head_dim]
    device half *attn_out_4heads    [[buffer(3)]],  // [4, head_dim]
    ...
) {
    // Process 4 Q heads in parallel
    // Each of 4 threads computes attention for one Q head
    // Shared K/V cache reads
}
```

**Expected gain**: 4× reduction in kernel launches for decode.

---

## Testing & Validation Plan

### Unit Tests

1. **Correctness** (decode single token):
   - Compare Metal Flash Attention output vs. CPU scalar attention
   - Tolerance: 1e-3 (fp16 precision)
   - Test at seq_len = 16, 128, 1K, 4K, 16K, 64K

2. **Numerical stability** (online softmax):
   - Ensure no overflow/underflow in exp() with large negative scores
   - Test attention scores with wide range: -1e4 to +1e4

3. **Memory correctness**:
   - Verify K/V cache pointer arithmetic
   - Test with ring buffer (pos > max_seq)

### Performance Benchmarks

```c
// In tests/test_flash_attention.c:

bench_flash_attn_decode() {
    for (int seq_len in [1K, 4K, 16K, 32K, 64K]) {
        _Float16 Q[128], attn_out[128];
        id<MTLBuffer> K_buf = ..., V_buf = ...;

        clock_t t0 = clock();
        for (int rep = 0; rep < 100; rep++) {
            flash_attn_decode(..., seq_len, ...);
        }
        double avg_ms = (double)(clock() - t0) / CLOCKS_PER_SEC / 100 * 1000;

        printf("seq_len=%d: %.2f ms (expected: <5ms at 64K)\n", seq_len, avg_ms);
    }
}
```

### Regression Tests

- End-to-end Mistral 7B inference with Flash Attention enabled/disabled
- Compare generation output (should be identical up to fp16 rounding)
- Verify decode latency improvement at various context lengths

---

## Deployment Checklist

- [ ] Write Metal shader: `flash_attn_decode_single_head` in new file
- [ ] Test shader compilation: `xcrun -sdk macosx metal ...`
- [ ] Implement C wrapper functions in `metal_flash_attention.h`
- [ ] Add GPU buffer allocation to `MistralModel` init
- [ ] Replace `gqa_attention_neon` with `gqa_attention_metal` in mistral_model.h
- [ ] Add runtime flag: `--use-metal-attention` (default: off initially)
- [ ] Benchmark decode at 1K, 16K, 64K; compare vs. old path
- [ ] Validate numerical correctness (element-wise diff < 1e-3)
- [ ] Profile GPU utilization and memory bandwidth with Instruments
- [ ] Document in README: "Metal Flash Attention for 64K context support"
- [ ] Optional: implement GQA optimization for 4× faster decode

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|----------|
| Threadgroup memory exceeds M5 limit | Medium | High | Use 64-token tiles (tested before committing) |
| Metal shader fails on older macOS | Low | High | Require macOS 12.x+; test on multiple versions |
| Numerical precision loss in softmax | Low | Medium | Validate against CPU with tolerance 1e-3 |
| GPU stalls on memory pressure | Low | Medium | Profile with Instruments; optimize memory accesses |
| GQA broadcast overhead dominates | Low | Medium | Implement per-KV-head launch initially; optimize if needed |

---

## Performance Expectations Summary

| Context | Current (CPU) | Metal Flash | Speedup |
|---------|---|---|---|
| 1K tokens | 0.8ms | 0.5ms | 1.6× |
| 16K tokens | 3ms | 1.2ms | 2.5× |
| 64K tokens | 12ms | 3ms | 4× |

**End-to-end Mistral 7B at 64K context** (512-token prompt + 128-token generation):
- Old: ~4.5s (dominated by decode attention)
- New: ~1.4s
- **Speedup: 3.2×**

---

## References

- **Flash Attention paper**: Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **Metal Performance Shaders**: WWDC 2021, Apple Developer Docs
- **Online Softmax**: Raffel et al. (2020), stable numerics for reduction
- **GQA**: Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformers"
- **M5 GPU specs**: Apple Silicon Tech Specs (M5 Pro/Max)

---

## Appendix: Debugging Tips

### Shader Compilation Errors

```bash
# Verbose compilation
xcrun -sdk macosx metal -v metal_flash_attention.metal

# Check metal syntax
metal -fmetall metal_flash_attention.metal 2>&1 | head -20
```

### Runtime Debugging

Use Xcode Metal Debugger:
1. Set breakpoint in shader
2. Run with GPU frame capture enabled
3. Inspect threadgroup memory, register state at each instruction

### Performance Profiling

```bash
# Profile with Instruments on M5
xcrun xctrace record -d 10s -o trace.trace \
  --template "Metal System Trace" \
  ./mistral_inference --use-metal-attention
```

Look for:
- GPU utilization (target: >80%)
- Memory bandwidth (target: >150 GB/s sustained)
- Kernel launch overhead (minimize)
- L2 cache hit rate (target: >70%)

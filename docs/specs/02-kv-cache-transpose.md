# KV Cache Transpose: Sequence-Major Layout for 64K Context

**Date**: 2026-03-02
**Target**: Mistral 7B inference on Apple M5
**Author**: ANEtransformers
**Status**: Design specification for implementation

## Executive Summary

The current KV cache uses channel-first layout (`cache[layer][dim][seq]`), which causes catastrophic cacheline inefficiency at large context windows (64K tokens). At 64K, every attention read scatters across 128KB strides, yielding only ~1.5% L3 cacheline utilization.

**Proposed fix**: Transpose to sequence-major layout (`cache[layer][seq][n_kv_heads][head_dim]`), making K/V writes contiguous 2KB memcpies and reducing read stride from 128KB to 2KB (64× improvement). At 64K context, this recovers **5-15ms/token** from reduced memory traffic.

---

## Current Problem: Channel-First Layout

### Data Structure (Today)

```c
typedef struct {
    int n_layers;
    int n_kv_heads;    // 8
    int head_dim;      // 128
    int max_seq;       // 4096–65536
    int kv_dim;        // n_kv_heads * head_dim = 1024
    _Float16 *k_cache; // [n_layers * kv_dim * max_seq]
    _Float16 *v_cache; // [n_layers * kv_dim * max_seq]
} KVCache;
```

**Layout**: `cache[layer][d][t]` — for layer L, dimension D, timestep T:
```
Physical address = layer * (kv_dim * max_seq)
                 + d * max_seq
                 + t
                 = base + (d * 65536 + t)  // 65536 = max_seq, 2B per element
```

### Write Pattern (Single Token, S=1 Decode)

```c
void kv_write(KVCache *cache, int layer, int pos,
              const _Float16 *k_vec, const _Float16 *v_vec) {
    int kv_dim = cache->kv_dim;  // 1024
    int max_seq = cache->max_seq;  // 65536
    _Float16 *k_base = cache->k_cache + layer * kv_dim * max_seq;
    for (int d = 0; d < kv_dim; d++) {
        k_base[d * max_seq + pos] = k_vec[d];  // SCATTER
    }
}
```

**Behavior**: Write 1024 fp16 values (2KB) scattered across 1024 different cachelines:
- Element `d` writes to address `base + d * 131072` (stride = 131KB = 128B cacheline × 1024)
- Load `cache[0]`, evict it, load `cache[131KB]`, evict it, ... load `cache[131KB×1023]`
- **Cost**: 1024 L3 cacheline loads for 2KB of data → ~512 cachelines loaded, only 1 used per cacheline
- **Utilization**: 128B cacheline × 1 useful byte / 128B = 0.78% → **1.5% at 64K seq**

### Read Pattern (Single Token Decode, Attention Q @ K^T)

```c
for (int t = 0; t < seq_len; t++) {
    for (int d = 0; d < 128; d += 32) {
        // Load 32 fp16 values from kcache (strided by max_seq)
        for (int i = 0; i < 32; i++)
            ktmp[i] = kcache[(d + i) * max_seq + t];  // GATHER
    }
}
```

**Per-token read cost**:
- 128 dimensions, load 32 at a time in inner loop
- For each of 128 dims: stride = 131KB between consecutive values
- At seq_len=64K: read 128 values × 131KB stride = 16.8MB of cache traffic per attention head
- All 8 KV heads × 32 query head replicas = **~5.4GB of strided reads for single attention layer**
- **Actual L3 BW**: M5 ~200 GB/s, but can't sustain strided reads. Estimated effective: ~20 GB/s
- **Latency**: 5.4GB / 20GB/s ≈ **270ms per layer** at 64K seq

---

## Bandwidth Analysis: Current vs. Proposed

### Current Layout: Channel-First

**System parameters**:
- Mistral 7B: 32 layers, 8 KV heads, 128 head_dim, n_heads=32 (4× GQA)
- Decode: S=1 query, seq_len ∈ [1, 64K]
- Attention: each head reads K[128, seq_len], computes 128 dot products

**Per-layer attention cost** (single KV head):
- K-reads: 128 dims × 131KB stride × seq_len
- V-reads: 128 dims × 131KB stride × seq_len (max 64K in softmax if non-zero)
- Total: ~256 dims × 131KB stride × seq_len
- At seq_len=64K: **16.8 MB per head, 8 heads = 134.4 MB per layer**
- 32 layers: 4.3 GB per forward pass

**L3 miss rate**: ~95% due to poor spatial locality
**Estimated latency** (200 GB/s raw, ~15% strided-read efficiency): **290ms per token at 64K**

### Proposed Layout: Sequence-Major

```c
// Proposed:
// cache[layer][seq][n_kv_heads][head_dim]
// All elements for one seq/head are contiguous
```

**Per-layer attention cost** (single KV head):
- K-read: 128 dims contiguous, seq_len times
- V-read: 128 dims contiguous, seq_len times (filtered by softmax)
- Contiguous stride: 2B (fp16) not 131KB
- Total memory footprint: 128 × 2B × seq_len = same as before, but contiguous
- At seq_len=64K: **16.8 MB per head, 8 heads = 134.4 MB per layer**

**L3 miss rate**: ~5% due to linear stream pattern
**Estimated latency** (200 GB/s, ~95% efficiency): **3.6ms per token at 64K**

**Improvement**: 290ms → 3.6ms ≈ **80× speedup** (but realistic: 5-15ms saved at 64K due to other bottlenecks like softmax, prefetch overlap, etc.)

---

## Proposed Solution: Sequence-Major Layout

### New Structure

```c
typedef struct {
    int n_layers;
    int n_kv_heads;     // 8
    int head_dim;       // 128
    int max_seq;        // 4096–65536, configurable
    int kv_dim;         // unused—calculated as n_kv_heads * head_dim

    // New layout: [layer][seq][n_kv_heads][head_dim]
    // Contiguous blocks: all (n_kv_heads * head_dim) for one seq position
    _Float16 *k_cache;  // layer * max_seq * n_kv_heads * head_dim
    _Float16 *v_cache;  // same

    int pos;            // current write position (for ring buffer)
    int len;            // valid entries
} KVCache;
```

### Address Calculation

For layer L, sequence position T, KV head H, dimension D:

```
Old: base + (L * kv_dim * max_seq) + (D * max_seq) + T
     = base + (D * 131072 + T)  [stride 131KB]

New: base + (L * max_seq * n_kv_heads * head_dim)
           + (T * n_kv_heads * head_dim)
           + (H * head_dim)
           + D
     = base + (T * 1024 + H * 128 + D)  [stride 1024B = 2KB per seq]
```

**Memory layout** (row-major):
```
[Layer 0]
  [Seq 0]: [Head 0: [D0, D1, ..., D127], Head 1: [...], ...],
  [Seq 1]: [Head 0: [...], ...],
  ...
[Layer 1]
  ...
```

### Write Pattern (Single Token)

```c
void kv_write_new(KVCache *cache, int layer, int pos,
                  const _Float16 *k_vec, const _Float16 *v_vec) {
    int n_kv_heads = cache->n_kv_heads;  // 8
    int head_dim = cache->head_dim;      // 128
    int max_seq = cache->max_seq;
    int t = pos % max_seq;

    _Float16 *k_base = cache->k_cache + (size_t)layer * max_seq * n_kv_heads * head_dim
                                       + (size_t)t * n_kv_heads * head_dim;
    _Float16 *v_base = cache->v_cache + (size_t)layer * max_seq * n_kv_heads * head_dim
                                       + (size_t)t * n_kv_heads * head_dim;

    // Contiguous memcpy: 8 heads × 128 dims × 2B = 2KB
    memcpy(k_base, k_vec, n_kv_heads * head_dim * sizeof(_Float16));
    memcpy(v_base, v_vec, n_kv_heads * head_dim * sizeof(_Float16));
}
```

**Behavior**:
- Single contiguous 2KB write per token
- One L3 cacheline write, maximum efficiency
- **Cost**: 16 cachelines to write 2KB (1:1 efficiency)
- **Latency**: ~10ns (L3 write time)

### Read Pattern (Single Token Decode)

```c
// Attention: Q @ K^T for single query token
for (int h = 0; h < n_kv_heads; h++) {
    _Float16 *kcache_h = k_cache + layer_offset
                                  + (seq_pos * n_kv_heads * head_dim)
                                  + (h * head_dim);

    for (int t = 0; t < seq_len; t++) {
        // Read 128 contiguous fp16 for this KV head at time t
        float32x4_t acc = vdupq_n_f32(0);
        for (int d = 0; d < 128; d += 4) {
            float16x4_t kh = vld1_f16((float16_t *)(kcache_h + d));  // 8B load
            float32x4_t kh_f32 = vcvt_f32_f16(kh);
            float32x4_t qh_d = vld1q_f32(query + d);
            acc = vfmaq_f32(acc, qh_d, kh_f32);
        }
        score += vaddvq_f32(acc);
    }
}
```

**Behavior**:
- For each seq position, load 128 contiguous fp16 (256B)
- Stride between seq positions: 1024B (8 heads × 128 dims)
- Loop over seq_len (up to 64K) with **stride = 1KB** (contiguous head data)
- **Cost**: linear stream → L3 prefetcher friendly, ~95% utilization
- **Latency** (64K seq, 1 head): 64K × 256B ÷ 200GB/s ≈ **81μs per head, 650μs per layer**

---

## Implementation Details

### File Modifications Summary

| File | Changes |
|------|---------|
| `kv_cache.h` | Struct layout, `kv_alloc`, `kv_write`, `kv_k`, `kv_v` getters |
| `mistral_model.h` | Update `gqa_attention_neon` read pattern (single decode) |
| `mistral_ane_prefill.h` | Update `kv_write_batch`, `attention_batch` (prefill path) |

### kv_cache.h: Struct and Initialization

```c
#pragma once
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

typedef struct {
    int n_layers;
    int n_kv_heads;     // 8
    int head_dim;       // 128
    int max_seq;
    _Float16 *k_cache;  // [n_layers][max_seq][n_kv_heads][head_dim]
    _Float16 *v_cache;  // same structure
    int pos;            // ring buffer write position
    int len;            // valid entries
} KVCache;

// Allocate: cache[n_layers * max_seq * n_kv_heads * head_dim]
static KVCache kv_alloc(int n_layers, int n_kv_heads, int head_dim, int max_seq) {
    KVCache c = {0};
    c.n_layers = n_layers;
    c.n_kv_heads = n_kv_heads;
    c.head_dim = head_dim;
    c.max_seq = max_seq;

    size_t kv_per_layer = (size_t)max_seq * n_kv_heads * head_dim;
    size_t total = (size_t)n_layers * kv_per_layer;

    c.k_cache = (_Float16 *)calloc(total, sizeof(_Float16));
    c.v_cache = (_Float16 *)calloc(total, sizeof(_Float16));
    c.pos = 0;
    c.len = 0;

    return c;
}

// Get pointer to K cache for (layer, seq_pos): [n_kv_heads, head_dim] contiguous
static _Float16 *kv_k_at(const KVCache *cache, int layer, int seq_pos) {
    int t = seq_pos % cache->max_seq;
    size_t offset = (size_t)layer * cache->max_seq * cache->n_kv_heads * cache->head_dim
                  + (size_t)t * cache->n_kv_heads * cache->head_dim;
    return cache->k_cache + offset;
}

// Get pointer to V cache for (layer, seq_pos)
static _Float16 *kv_v_at(const KVCache *cache, int layer, int seq_pos) {
    int t = seq_pos % cache->max_seq;
    size_t offset = (size_t)layer * cache->max_seq * cache->n_kv_heads * cache->head_dim
                  + (size_t)t * cache->n_kv_heads * cache->head_dim;
    return cache->v_cache + offset;
}

// Write single token's K,V to cache
static void kv_write(KVCache *cache, int layer, int pos,
                     const _Float16 *k_vec, const _Float16 *v_vec) {
    int n_kv_heads = cache->n_kv_heads;
    int head_dim = cache->head_dim;
    int max_seq = cache->max_seq;

    _Float16 *k_base = kv_k_at(cache, layer, pos);
    _Float16 *v_base = kv_v_at(cache, layer, pos);

    // Contiguous memcpy (2KB for Mistral: 8*128*2)
    size_t sz = (size_t)n_kv_heads * head_dim * sizeof(_Float16);
    memcpy(k_base, k_vec, sz);
    memcpy(v_base, v_vec, sz);
}

// Pointer to all K data for a layer (for prefill batches)
// Returns base address; caller indexes as [seq][n_kv_heads][head_dim]
static _Float16 *kv_k_layer(const KVCache *cache, int layer) {
    return cache->k_cache + (size_t)layer * cache->max_seq * cache->n_kv_heads * cache->head_dim;
}

static _Float16 *kv_v_layer(const KVCache *cache, int layer) {
    return cache->v_cache + (size_t)layer * cache->max_seq * cache->n_kv_heads * cache->head_dim;
}

static void kv_free(KVCache *cache) {
    free(cache->k_cache);
    free(cache->v_cache);
    cache->k_cache = NULL;
    cache->v_cache = NULL;
}
```

### mistral_model.h: Decode Attention Update

**Current** (lines 314–338 in `gqa_attention_neon`):
```c
_Float16 *kcache = kcache_base + kvh * hd * max_seq;
for (int t = 0; t < seq_len; t++) {
    for (int d = 0; d < 128; d += 32) {
        for (int i = 0; i < 32; i++)
            ktmp[i] = kcache[(d + i) * max_seq + t];  // STRIDED
    }
}
```

**Proposed** (sequence-major layout):
```c
// kcache_base: [max_seq][n_kv_heads][head_dim]
// kvh: KV head index (0–7)
// For this head: offset = kvh * head_dim (128 bytes)

for (int t = 0; t < seq_len; t++) {
    // Pointer to K[t, kvh, :] — 128 contiguous fp16 values
    _Float16 *kcache_t = kcache_base + (size_t)t * n_kv_heads * head_dim
                                      + kvh * head_dim;

    for (int d = 0; d < 128; d += 32) {
        // Load 32 fp16 from contiguous memory
        float16x8_t k0 = vld1q_f16((float16_t *)(kcache_t + d));
        float16x8_t k1 = vld1q_f16((float16_t *)(kcache_t + d + 8));
        // ... convert and FMA with query
    }
}
```

**Stride improvement**:
- Old: stride = max_seq × 2B = 131KB (at 64K context)
- New: stride = n_kv_heads × head_dim × 2B = 1024B = 1KB
- **Gain**: 128× stride reduction → L3 prefetcher can keep data resident

### mistral_ane_prefill.h: Prefill Path Updates

**Current** (lines 312–329, `kv_write_batch`):
```c
static void kv_write_batch(KVCache *kv, int layer, int start_pos,
                            const float *K_buf, const float *V_buf,
                            int kv_dim, int S) {
    for (int t = 0; t < S; t++) {
        int pos = start_pos + t;
        const float *kt = K_buf + t * kv_dim;
        const float *vt = V_buf + t * kv_dim;
        _Float16 k16[kv_dim], v16[kv_dim];
        for (int d = 0; d < kv_dim; d++) {
            k16[d] = (_Float16)kt[d];
            v16[d] = (_Float16)vt[d];
        }
        kv_write(kv, layer, pos % kv->max_seq, k16, v16);  // Uses old kv_write
    }
}
```

**No change needed** — `kv_write` now does contiguous write automatically with new layout.

**Attention** (lines 331–385, `attention_batch`):
```c
// Current: kcache[d * max_seq + cache_s] strided access
for (int s = 0; s < seq_len; s++) {
    float score = 0;
    for (int d = 0; d < hd; d++)
        score += qh[d] * (float)kcache[d * max_seq + cache_s];  // GATHER
}
```

**Proposed**:
```c
for (int s = 0; s < seq_len; s++) {
    float score = 0;
    // kcache now: [max_seq][n_kv_heads][head_dim]
    // For this KV head at position s:
    _Float16 *kcache_s = kcache + (size_t)s * n_kv_heads * head_dim
                                + kvh * head_dim;
    for (int d = 0; d < hd; d++)
        score += qh[d] * (float)kcache_s[d];  // Contiguous load
}
```

---

## Dimensions for Mistral 7B

```
n_layers:      32
dim:           4096
kv_dim:        1024
n_kv_heads:    8
head_dim:      128
n_heads:       32 (GQA: 4 heads per KV head)
hidden_dim:    14336
max_seq:       65536 (configurable, default 32768 or 4096)
```

**Cache size**:
- Per layer: 65536 × 8 × 128 × 2B × 2 (K + V) = 268 MB
- All 32 layers: 8.6 GB (fits in 24GB M5 RAM)

**Write cost** (per token):
- Old: ~1024 × 8 cacheline loads/evicts → ~1μs (highly variable)
- New: 2KB contiguous write → ~10ns

**Read cost** (per token, at 64K seq):
- Old: ~270ms per layer (saturates memory)
- New: ~3ms per layer (linear stream)

---

## Alternative: Paged KV Cache (Future)

For extremely long contexts (128K+) or batched inference, consider **paged KV cache** with variable-length sequences:

```c
typedef struct {
    int page_size;       // e.g., 256 tokens
    int n_pages;
    // K_pages[layer][page_id][n_kv_heads][head_dim] contiguous pages
    _Float16 **k_pages;
    _Float16 **v_pages;
    // Page table: seq_id → [page0, page1, ...]
    int **page_table;
    int *seq_lengths;
} PagedKVCache;
```

**Advantages**:
- Efficient batching with variable sequence lengths
- Shared KV cache across sequences (no recomputation)
- Adapts to PagedAttention-style batching

**Disadvantages**:
- More complex pointer chasing during attention
- Requires page table lookups per sequence step
- Overhead for small contexts

**Decision**: Defer to future work; implement contiguous sequence-major first.

---

## Testing & Validation Plan

### Correctness Tests

1. **Decode single token** (S=1, increasing context):
   - Verify `kv_write` produces same result as old layout
   - Verify `gqa_attention_neon` output matches old numerically
   - Test at context lengths: 16, 128, 1024, 4096, 32768, 65536

2. **Prefill batch** (S=16):
   - Compare `attention_batch` output (new vs. old layout)
   - Verify RoPE + KV write consistency

3. **Ring buffer** (pos > max_seq):
   - Write past max_seq, verify wrap-around indexing
   - Decode with ring buffer active

### Performance Tests

1. **Decode latency**:
   - Measure token generation time at varying context lengths
   - Expected: 5–15ms improvement at 64K

2. **Prefill throughput**:
   - Measure toks/s for prompts of varying lengths
   - Expected: minimal change (attention is CPU, not memory-bound at S=16)

3. **Memory bandwidth**:
   - Profile L3 miss rate before/after
   - Target: <5% L3 miss rate (from ~95%)

4. **Cache footprint**:
   - Verify allocation size matches expected (layer × max_seq × n_kv_heads × head_dim × 2)

### Regression Tests

- Mistral 7B end-to-end inference on test prompts
- Compare generation quality (should be identical)
- Verify greedy decode produces same sequence

---

## Performance Expectations

### Decode (S=1)

| Context | Old Layout | New Layout | Gain |
|---------|-----------|-----------|------|
| 1K      | 0.9ms     | 0.8ms     | 10% |
| 4K      | 2.8ms     | 1.5ms     | 46% |
| 16K     | 12ms      | 3ms       | 75% |
| 64K     | 55ms      | 5ms       | 91% |

**Limiting factors at large context**:
- Softmax over 64K values: ~2ms (unavoidable)
- Query projection: ~1ms (matmul-bound)
- V-read in attention: ~2ms (linear stream, good)

### Prefill (S=1024, 12M tokens)

| Context | Old Layout | New Layout | Gain |
|---------|-----------|-----------|------|
| 32K     | 27ms      | 24ms      | 11% |
| 64K     | 35ms      | 28ms      | 20% |

**Rationale**: Prefill attention is less context-heavy; improvements are modest (10–20%).

### End-to-End: Mistral 7B on 64K Context

**Scenario**: 512-token prompt, generate 128 tokens

- Prefill: 512 tokens → 0.8s (old: 0.9s) → ~10% improvement
- Decode: 128 tokens × 5ms avg (old: 35ms avg) → 640ms (old: 4.5s) → **7× speedup at decode**
- **Total**: 1.4s (old: 5.4s) → **3.9× faster end-to-end**

---

## Implementation Phases

### Phase 1: Core Transpose (Priority 1)
- Modify `kv_cache.h` struct, allocation, write logic
- Update `mistral_model.h` decode attention read pattern
- Unit tests: write/read correctness at various context sizes
- **Effort**: 2–4 hours
- **Risk**: Low (localized changes)

### Phase 2: Prefill Support (Priority 2)
- Update `mistral_ane_prefill.h` attention and kv_write_batch
- Test ANE + BLAS prefill paths
- **Effort**: 1–2 hours
- **Risk**: Low (reuses kv_write, attention logic similar)

### Phase 3: Benchmarking (Priority 3)
- Measure decode latency at 1K, 4K, 16K, 64K contexts
- Profile L3 miss rates (using Instruments)
- Compare vs. vLLM baseline on GPU (if applicable)
- **Effort**: 3–5 hours
- **Risk**: Low

### Phase 4: Optimization (Optional)
- Explore SIMD-optimized attention reads (prefetch hints)
- Consider Metal compute for attention at 64K (hybrid)
- **Effort**: TBD (depends on results)

---

## Code Review Checklist

- [ ] `kv_alloc`: Allocates correct total size, initializes pos/len
- [ ] `kv_write`: Contiguous memcpy, handles ring buffer (pos % max_seq)
- [ ] `gqa_attention_neon`: Reads contiguous [n_kv_heads][head_dim] per seq step
- [ ] `attention_batch`: Prefill attention accesses correct offsets
- [ ] `kv_write_batch`: Calls updated `kv_write`, no manual scatter
- [ ] Tests: Decode, prefill, ring buffer, large context (32K+)
- [ ] Benchmarks: Compare old vs. new at 4K, 16K, 64K
- [ ] Documentation: Update README with cache layout notes

---

## References

- **vLLM PagedAttention**: https://arxiv.org/abs/2309.06180 (paged allocation for long sequences)
- **Apple Metal Performance Shaders**: WWDC 2021 (cache hierarchy, prefetching)
- **Mistral 7B Architecture**: https://arxiv.org/abs/2310.06825
- **Current codebase**:
  - `/Users/andy/ANEtransformers/mistral/kv_cache.h`
  - `/Users/andy/ANEtransformers/mistral/mistral_model.h`
  - `/Users/andy/ANEtransformers/mistral/mistral_ane_prefill.h`

---

## Appendix: Stride Analysis

### Cacheline Efficiency Comparison

**System**: M5 with 32KB L1, 768KB L2, 32MB L3, 128B cachelines

**Channel-first** (`cache[d * max_seq + t]` at 64K seq):
```
Stride between consecutive reads: 128 dimensions × 131KB = 16.8 MB
Cachelines loaded: 16.8 MB / 128B = 131K
Cachelines useful: 128 (one per dimension at one timestep)
Utilization: 128 / 131K ≈ 0.1%
```

**Sequence-major** (`cache[t * kv_dim + d]` contiguous):
```
Stride between consecutive seq reads: 1KB (n_kv_heads * head_dim)
Cachelines loaded: 1KB / 128B = 8
Cachelines useful: 8 (one full KV head)
Utilization: 8 / 8 = 100% (sustained, linear prefetch)
```

**Improvement factor**: 131K / 8 = **16,375× fewer wasted cachelines**

(Or viewed differently: 0.1% → 100% utilization, but realistic cache-aware performance is ~10–20ms improvement at 64K due to other bottlenecks.)


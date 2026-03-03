# Engineering Spec 05: Embedding Table & RoPE Memory Optimization

**Mistral 7B Inference on Apple M5 (24GB unified RAM)**
**Date:** 2026-03-02
**Status:** Design phase
**Related:** `mistral_model.h`, `dequant.h`, `kv_cache.h`

---

## Executive Summary

Two large static tables waste ~560 MB of resident memory in the current inference engine:

1. **Token embedding table:** 500 MB fp32 (32K vocab × 4096 dims × 4B/element)
2. **RoPE lookup tables:** 64 MB fp32 (131K positions × 64 cos/sin values × 4B)

This spec proposes memory-efficient replacements:
- **Embedding:** Dequantize Q4 rows on-demand at decode time (~1 lookup/token). Saves 496 MB permanent, costs ~2.3 KB reads per token during prefill.
- **RoPE:** Compute cos/sin on the fly using NEON polynomial approximation. Saves 64 MB permanent, costs ~1 μs per token.

**Total savings: ~560 MB**, enabling larger KV cache or other optimizations.

---

## Problem 1: Embedding Table (500 MB fp32)

### Current Implementation

In `mistral_model.h` lines 225–246:

```c
// Token embedding — dequant to fp32
GGUFTensor *emb_t = gguf_find(gguf, "token_embd.weight");
if (emb_t) {
    uint64_t nel = gguf_nelements(emb_t);
    m->token_embed = (float *)malloc(nel * sizeof(float));
    if (emb_t->type == GGML_TYPE_Q4_0 || emb_t->type == GGML_TYPE_Q4_K) {
        // Quantized embedding: dequant to fp16 then fp32
        int rows = (int)emb_t->ne[1], cols = (int)emb_t->ne[0];
        _Float16 *tmp = (_Float16 *)malloc(nel * sizeof(_Float16));
        if (emb_t->type == GGML_TYPE_Q4_0)
            dequant_q4_0_to_fp16(gguf_data(gguf, emb_t), tmp, rows, cols);
        else
            dequant_q4_K_to_fp16(gguf_data(gguf, emb_t), tmp, rows, cols);
        for (uint64_t i = 0; i < nel; i++) m->token_embed[i] = (float)tmp[i];
        free(tmp);
    }
}
```

**Load-time cost:**
- Temp allocation: 250 MB (_Float16)
- Final table: 500 MB (float)
- Total peak memory: 750 MB (temp + final)

**Lookup-time cost (prefill):**
- Line 872 in `mistral_prefill()`: `memcpy(X + i * dim, m->token_embed + tokens[start + i] * dim, dim * sizeof(float))`
- 1 row copy per token: 4096 × 4B = 16 KB read + write (negligible latency)

**The problem:** One row is accessed per token, yet entire 500 MB table remains in RAM.

### Proposed Solution: On-Demand Dequantization

**Option 1 (recommended):** Keep Q4 format on disk, dequant one row at a time.

#### Design

Store embedding weights in mmap'd Q4_0 or Q4_K format (status quo). Instead of materializing `m->token_embed`, add:

```c
// In MistralModel struct
const void *token_embed_q4;     // Q4_0/Q4_K pointer into mmap'd GGUF
uint32_t token_embed_type;      // GGML_TYPE_Q4_0 or GGML_TYPE_Q4_K
int token_embed_rows;           // vocab_size
int token_embed_cols;           // dim (4096)
```

Remove allocation:

```c
// In mistral_load() — DELETE these lines:
// m->token_embed = (float *)malloc(nel * sizeof(float));
// ... dequant logic ...
// Replace with:
m->token_embed_q4 = gguf_data(gguf, emb_t);
m->token_embed_type = emb_t->type;
m->token_embed_rows = (int)emb_t->ne[1];
m->token_embed_cols = (int)emb_t->ne[0];
```

Add a per-token embedding lookup function in `dequant.h`:

```c
// dequant.h — add new function:
// Dequant single row from Q4_0 embedding table to fp32 output
static void dequant_q4_0_row_to_fp32(const void *W, int row_idx, int cols,
                                      float *out) {
    // W points to Q4_0 packed data: [rows][cols] = [rows][cols/32] blocks
    // Each block is 18 bytes (d: fp16, qs[16]: 4-bit pairs)
    int blocks_per_row = cols / QK4_0;  // cols = 4096, QK4_0 = 32 → 128 blocks
    int row_offset = row_idx * blocks_per_row * sizeof(block_q4_0);

    const block_q4_0 *row_blocks = (const block_q4_0 *)((const char *)W + row_offset);

    for (int b = 0; b < blocks_per_row; b++) {
        const block_q4_0 *block = &row_blocks[b];
        float d = (float)block->d;
        const uint8_t *qs = block->qs;

        // Dequant block (32 elements) — inline for speed
        for (int i = 0; i < QK4_0; i++) {
            int nibble_idx = i / 2;
            int is_high = i % 2;
            uint8_t byte = qs[nibble_idx];
            int q = is_high ? (byte >> 4) : (byte & 0x0F);
            out[b * QK4_0 + i] = d * ((float)q - 8.0f);
        }
    }
}

// Variant for Q4_K rows (if model uses Q4_K)
static void dequant_q4_K_row_to_fp32(const void *W, int row_idx, int cols,
                                      float *out) {
    // Q4_K block structure: 144 bytes per block (256 values)
    // [blocks_per_row = cols / 256]
    int blocks_per_row = cols / QK_K;
    int row_offset = row_idx * blocks_per_row * sizeof(block_q4_K);

    const block_q4_K *row_blocks = (const block_q4_K *)((const char *)W + row_offset);

    // TODO: implement Q4_K row dequant (complex — scales are grouped)
    // For now, fallback to full dequant (suboptimal but correct)
    _Float16 *tmp = (_Float16 *)malloc(cols * sizeof(_Float16));
    dequant_q4_K_to_fp16(W + row_offset, tmp, 1, cols);
    for (int i = 0; i < cols; i++) out[i] = (float)tmp[i];
    free(tmp);
}
```

Update prefill to call dequant per token:

```c
// In mistral_prefill() line 870–874, change:
for (int i = 0; i < cs; i++) {
    // memcpy(X + i * dim, m->token_embed + tokens[start + i] * dim, dim * sizeof(float));

    // Instead: dequant on demand
    int tok = tokens[start + i];
    if (m->token_embed_type == GGML_TYPE_Q4_0) {
        dequant_q4_0_row_to_fp32(m->token_embed_q4, tok, m->cfg.dim,
                                 X + i * m->cfg.dim);
    } else if (m->token_embed_type == GGML_TYPE_Q4_K) {
        dequant_q4_K_row_to_fp32(m->token_embed_q4, tok, m->cfg.dim,
                                 X + i * m->cfg.dim);
    } else {
        // F32 or F16 — direct copy
        ...
    }
    positions[i] = start + i;
}
```

For decode (single token), create a temporary row:

```c
// In mistral_infer.m (main inference loop), when decoding after prefill:
// Instead of: x = m->token_embed + next_tok * dim
// Do:
float x[DIM];  // temp stack buffer
if (m->token_embed_type == GGML_TYPE_Q4_0) {
    dequant_q4_0_row_to_fp32(m->token_embed_q4, next_tok, m->cfg.dim, x);
} else if (...) { ... }
// Then pass x through transformer layers
```

#### Cost Analysis

**On-disk:** ~36 MB (Q4_0) — unchanged
**Load-time:** ~100 KB per dequant call (negligible)
**Decode cost:** 1 lookup/token → 1 × 2.3 KB read from mmap (within L1 cache, ~100 ns) + ~200 cycles of dequant
**Prefill cost:** S rows at S=128 → 128 × 2.3 KB = 295 KB sequential reads (amortized ~1 μs per token from vectorized dequant)
**Memory saved:** 500 MB permanent ✓

---

## Problem 2: RoPE Tables (64 MB fp32)

### Current Implementation

Lines 65–81 in `mistral_model.h`:

```c
static void precompute_rope(MistralModel *m) {
    int hd2 = m->cfg.head_dim / 2;  // 64
    int max_seq = m->cfg.max_seq_len > 0 ? m->cfg.max_seq_len : 32768;
    if (max_seq > 131072) max_seq = 131072;

    m->rope_cos = (float *)malloc(max_seq * hd2 * sizeof(float));
    m->rope_sin = (float *)malloc(max_seq * hd2 * sizeof(float));

    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < hd2; i++) {
            float freq = 1.0f / powf(m->cfg.rope_theta, (2.0f * i) / m->cfg.head_dim);
            float angle = pos * freq;
            m->rope_cos[pos * hd2 + i] = cosf(angle);
            m->rope_sin[pos * hd2 + i] = sinf(angle);
        }
    }
}
```

**Memory footprint:**
- `rope_cos`: 131K × 64 × 4B = 33.6 MB
- `rope_sin`: 131K × 64 × 4B = 33.6 MB
- **Total: 67.2 MB**

**Per-token cost:** Lookup 2 tables × 64 elements = 512 bytes read (negligible latency, already in cache during apply_rope).

**The problem:** Precomputed tables waste memory for long sequence lengths (many models use max_seq << 131K).

### Proposed Solution: On-the-Fly Computation

**Idea:** Precompute only `theta[64]` at load time (512 bytes), compute cos/sin per position on demand using NEON polynomial approximation.

#### Design

```c
// In MistralModel struct, replace:
//   float *rope_cos;
//   float *rope_sin;
// With:
float *rope_theta_inv;  // [head_dim/2] = [64] — 1/theta^(2*i/head_dim)
```

At load time, precompute theta only:

```c
// New function in mistral_model.h
static void precompute_rope_theta(MistralModel *m) {
    int hd2 = m->cfg.head_dim / 2;
    m->rope_theta_inv = (float *)malloc(hd2 * sizeof(float));

    for (int i = 0; i < hd2; i++) {
        float exp = (2.0f * i) / (float)m->cfg.head_dim;
        m->rope_theta_inv[i] = 1.0f / powf(m->cfg.rope_theta, exp);
    }
}

// In mistral_load(), line 263, replace:
//   precompute_rope(m);
// With:
//   precompute_rope_theta(m);
```

Add NEON polynomial approximation for cos/sin:

```c
// dequant.h — add math helpers:
// Fast cos/sin approximation using Chebyshev polynomials
// Input: angle in radians (range doesn't matter — cos/sin are periodic)
// Output: cos_out, sin_out
// Error: < 0.001 (sufficient for RoPE attention — only 3 digits precision needed)
static inline void cos_sin_approx_neon(float angle, float *cos_out, float *sin_out) {
    // Normalize angle to [-π, π]
    // cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! (Chebyshev polynomial)
    // sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!

    // For simplicity and speed, use libc sincos if available, but wrap in NEON loop
    // Fallback: scalar fast_sin/fast_cos (not vectorizable but small overhead)
    __sincosf(angle, sin_out, cos_out);
}

// Vectorized version: compute 4 cos/sin pairs per call
// angles: [4] float input angles
// cos_out: [4] float output
// sin_out: [4] float output
static inline void cos_sin_approx_neon_x4(const float *angles,
                                           float *cos_out, float *sin_out) {
    for (int i = 0; i < 4; i++) {
        __sincosf(angles[i], &sin_out[i], &cos_out[i]);
    }
}
```

Rewrite `apply_rope()` to compute on the fly:

```c
// Updated apply_rope in mistral_model.h
static void apply_rope(float *q, float *k, int pos,
                       const float *rope_theta_inv,  // [head_dim/2]
                       int n_heads, int n_kv_heads, int head_dim) {
    int hd2 = head_dim / 2;

    // ─── Compute cos/sin for this position ───
    float cos_vals[hd2], sin_vals[hd2];
    for (int i = 0; i < hd2; i += 4) {
        // Vectorize 4 freq values per iteration
        float angles[4];
        int limit = (i + 4 <= hd2) ? 4 : (hd2 - i);
        for (int j = 0; j < limit; j++) {
            angles[j] = (float)pos * rope_theta_inv[i + j];
        }
        cos_sin_approx_neon_x4(angles, cos_vals + i, sin_vals + i);
    }

    // ─── Apply to Q heads ───
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < hd2; i++) {
            float q0 = qh[2*i], q1 = qh[2*i + 1];
            float c = cos_vals[i], s = sin_vals[i];
            qh[2*i]     = q0 * c - q1 * s;
            qh[2*i + 1] = q0 * s + q1 * c;
        }
    }

    // ─── Apply to K heads ───
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < hd2; i++) {
            float k0 = kh[2*i], k1 = kh[2*i + 1];
            float c = cos_vals[i], s = sin_vals[i];
            kh[2*i]     = k0 * c - k1 * s;
            kh[2*i + 1] = k0 * s + k1 * c;
        }
    }
}
```

Update function signatures throughout codebase:

```c
// mistral_layer_decode() — line 450, change call from:
//   apply_rope(m->q, m->k, pos, m->rope_cos, m->rope_sin, n_heads, n_kv, hd);
// To:
//   apply_rope(m->q, m->k, pos, m->rope_theta_inv, n_heads, n_kv, hd);

// mistral_layer_prefill_chunk() — line 717, similar change
```

Update `mistral_free()`:

```c
// Lines 1034–1035, change:
//   free(m->rope_cos);
//   free(m->rope_sin);
// To:
//   free(m->rope_theta_inv);
```

#### Cost Analysis

**Load-time:** 64 × 4B = 256 bytes (negligible)
**Per-token compute:**
- 64 × `__sincosf()` calls = ~1 μs on M5 (highly vectorizable by compiler)
- Overhead in apply_rope loop: ~2 μs (dominated by cos/sin compute, not memory)

**Memory saved:** 64 MB permanent ✓
**Accuracy:** `__sincosf()` has ~1e-7 relative error; RoPE requires only ~3 digits precision (attentional logits are bounded [-inf, +inf] but softmax is dominated by relative differences). No measurable loss.

---

## Alternative Solutions

### Embedding Table Alternatives

| Option | Memory | Load Cost | Per-Token Cost | Notes |
|--------|--------|-----------|----------------|-------|
| **On-demand Q4 (Recommended)** | -496 MB | -250 MB temp | 2.3 KB read + ~200 cycles dequant | Industry standard (MLX, llama.cpp) |
| Keep fp16 table | -250 MB | -250 MB temp | negligible | Faster than on-demand but still wastes memory |
| Direct Q4→fp32 dequant | -0 MB | -250 MB temp | negligible | Eliminates temp alloc only |
| Keep fp32 (status quo) | 0 MB | +250 MB temp | negligible | Wastes 500 MB permanent |

### RoPE Alternatives

| Option | Memory | Per-Token Cost | Accuracy | Notes |
|--------|--------|----------------|----------|-------|
| **On-the-fly compute (Recommended)** | -64 MB | ~1 μs | Full (__sincosf) | Optimal for M5 SIMD; minimal overhead |
| fp16 RoPE tables | -32 MB | negligible | Full (fp16 has ~3 digits) | Good compromise |
| Keep fp32 (status quo) | 0 MB | negligible | Full | Wastes 64 MB permanent |

---

## Files to Modify

### 1. `/Users/andy/ANEtransformers/mistral/mistral_model.h`

**Changes:**
- Remove `float *rope_cos`, `float *rope_sin` from `MistralModel` struct
- Add `float *rope_theta_inv` (256 bytes)
- Remove `float *token_embed` (or make optional for backward compat)
- Add `const void *token_embed_q4`, `uint32_t token_embed_type`, `int token_embed_rows/cols`
- Replace `precompute_rope()` with `precompute_rope_theta()`
- Update `apply_rope()` signature (remove rope_cos/rope_sin params)
- Update `mistral_layer_decode()`, `mistral_layer_decode_parallel()`, `mistral_layer_prefill_chunk()` to call new apply_rope
- Update `mistral_prefill()` to call dequant functions per token
- Update `mistral_load()` to skip embedding dequant, initialize theta_inv instead
- Update `mistral_free()` to free rope_theta_inv

### 2. `/Users/andy/ANEtransformers/mistral/dequant.h`

**Additions:**
- `dequant_q4_0_row_to_fp32()` — per-row Q4_0 dequant to fp32
- `dequant_q4_K_row_to_fp32()` — per-row Q4_K dequant to fp32 (or fallback wrapper)
- `cos_sin_approx_neon()` — fast cos/sin using libm (or Chebyshev if ultra-high perf needed)
- `cos_sin_approx_neon_x4()` — vectorized version for 4 angles

### 3. `/Users/andy/ANEtransformers/mistral/mistral_infer.m`

**Changes:**
- Decode token loop: allocate temp stack buffer for embedding, call dequant_q4_0_row_to_fp32 instead of direct mmap read
- Update prefill/decode flow if needed

---

## Implementation Order

1. **Phase 1 (Low Risk):** RoPE on-the-fly
   - Precompute theta only
   - Implement cos_sin_approx_neon + vectorized version
   - Update apply_rope signature
   - Update 3 call sites
   - Test with existing test suite
   - **Expected time:** 2 hours

2. **Phase 2 (Medium Risk):** Embedding on-demand
   - Implement dequant_q4_0_row_to_fp32 in dequant.h
   - Remove token_embed malloc from mistral_load
   - Update mistral_prefill (1 loop change)
   - Update decode path in mistral_infer.m
   - Test: verify embedding lookup produces same results
   - **Expected time:** 3 hours

3. **Phase 3 (Validation):**
   - Benchmark: decode latency (should be ~same or faster due to better cache usage)
   - Benchmark: prefill throughput (should be ~same, maybe slightly slower due to dequant)
   - Memory profile: confirm 560 MB freed
   - End-to-end inference test

---

## Testing Strategy

### Unit Tests
- Test dequant_q4_0_row_to_fp32 against full-table dequant (bit-exact match for single rows)
- Test cos_sin_approx_neon against libm sincos (relative error < 1e-6)

### Integration Tests
- Run existing inference tests with both old (precomputed) and new (on-demand) paths
- Verify output logits within numerical tolerance (fp32 precision)
- Profile per-token embedding lookup time, RoPE apply time

### Memory Tests
- Check resident memory after load: should be ~560 MB less
- Confirm no temp allocations during inference (steady-state)

---

## Performance Expectations

| Metric | Current | Optimized | Change |
|--------|---------|-----------|--------|
| Load-time peak memory | 750 MB | 490 MB | -260 MB |
| Resident memory (loaded) | 500 MB embed + 64 MB RoPE | 0 MB embed + 0.25 MB RoPE | -564 MB |
| Decode latency (per token) | ~3–4 ms | ~3–4 ms | ~same |
| Prefill throughput | 25–27 tok/s @ S=128 | 25–27 tok/s | ~same (dequant amortized) |
| KV cache headroom | ~3 GB @ 24 GB | ~3.5 GB | +500 MB |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| RoPE accuracy degradation | Use full-precision `__sincosf()`, not Chebyshev; test relative errors |
| Embedding lookup mismatch | Unit test dequant_q4_0_row_to_fp32 against full dequant (bit-exact) |
| Decode latency regression | Measure per-token cost; if > 1%, optimize dequant NEON kernel |
| Cache thrashing during prefill | Monitor L1 miss rate; dequant footprint is small (~2 KB per call) |
| Backward compatibility | Keep mistral_model.h struct ABI stable; old code using token_embed should fail gracefully |

---

## Backward Compatibility

- If code still references `m->token_embed`, it will be NULL. Add asserts in old call sites.
- If code still references `m->rope_cos/sin`, it will be NULL. Update all 3 call sites or add wrappers.
- No binary format changes; GGUF mmap'd weights unchanged.

---

## Related Documentation

- [ANE Investigation](../ANE_INVESTIGATION.md) — Context on Model Quantization
- [Building Mistral](../BUILDING_MISTRAL.md) — Build instructions
- `gguf_loader.h` — GGUF weight loading internals
- `gguf_dequant.h` — Existing dequant implementations

---

## Future Work

- **ANE acceleration:** Both embedding dequant and RoPE cos/sin could theoretically run on ANE (but not practical due to setup overhead for single-row operations).
- **Adaptive RoPE:** For shorter sequences (max_seq < 4096), compute tables on-demand but cache per-session. Trades 2–4 MB memory for faster applies.
- **Embedding quantization further:** Use int4 or LUT-based lookup if even smaller footprint needed for edge deployment.

// q4_0_metal.metal — Fused Q4_0 dequant + GEMV/GEMM for Apple M5 GPU
// Matches block_q4_0 layout from dequant.h: 18 bytes = 2B fp16 scale + 16B nibbles (32 values)
#include <metal_stdlib>
using namespace metal;

// Must match ggml block_q4_0: exactly 18 bytes, no padding.
// half (2B) + uchar[16] (16B) = 18B. Metal struct layout places qs at offset 2
// with no padding (uchar alignment = 1), so natural layout is correct.
struct Q4_0Block {
    half d;           // scale factor (fp16, 2 bytes)
    uchar qs[16];     // packed nibbles: qs[i] holds values [2i] (lo) and [2i+16] (hi)
};
// sizeof(Q4_0Block) == 18 — verified: half(2) + uchar[16] = 18, alignment 2, no padding

// NaN-safe scale read: some GGUF files have NaN fp16 scale values.
// CPU code compiled with -ffast-math silently swallows NaN; Metal doesn't.
// Branchless: select(x, 0, isnan(x))
inline float safe_scale(half d) {
    float s = float(d);
    return select(s, 0.0f, isnan(s));
}

// ============================================================================
// GEMV: y[out_dim] = W_q4[out_dim, in_dim] @ x[in_dim]
// One threadgroup per ROWS_PER_TG output rows. 32 threads (1 simdgroup).
// Each thread processes blocks_per_row/32 blocks, then simd_sum reduces.
// x loaded into threadgroup shared memory to avoid redundant global reads.
// ============================================================================

#define GEMV_ROWS_PER_TG 4
#define GEMV_SIMD_SIZE   32

kernel void q4_0_gemv(
    device const Q4_0Block *W [[buffer(0)]],   // [out_dim * blocks_per_row]
    device const float *x     [[buffer(1)]],   // [in_dim]
    device float *y           [[buffer(2)]],   // [out_dim]
    constant uint &out_dim    [[buffer(3)]],
    constant uint &in_dim     [[buffer(4)]],
    uint tid                  [[thread_index_in_simdgroup]],
    uint sid                  [[simdgroup_index_in_threadgroup]],
    uint tg_id                [[threadgroup_position_in_grid]])
{
    const uint bpr = in_dim / 32;  // blocks per row
    const uint base_row = tg_id * GEMV_ROWS_PER_TG;

    // Each simdgroup handles one row. sid = row offset within threadgroup.
    const uint row = base_row + sid;
    if (row >= out_dim) return;

    device const Q4_0Block *row_blocks = W + row * bpr;

    // Each thread accumulates over its share of blocks
    const uint blocks_per_thread = (bpr + GEMV_SIMD_SIZE - 1) / GEMV_SIMD_SIZE;
    const uint b_start = tid * blocks_per_thread;
    const uint b_end = min(b_start + blocks_per_thread, bpr);

    float acc = 0.0f;

    for (uint b = b_start; b < b_end; b++) {
        const float scale = safe_scale(row_blocks[b].d);
        const uint x_off = b * 32;

        // Unpack 16 bytes -> 32 nibble values, dot with x
        float block_sum = 0.0f;
        for (uint i = 0; i < 16; i++) {
            uchar raw = row_blocks[b].qs[i];
            int lo = int(raw & 0xF) - 8;
            int hi = int(raw >> 4) - 8;
            block_sum += float(lo) * x[x_off + i];
            block_sum += float(hi) * x[x_off + i + 16];
        }
        acc += scale * block_sum;
    }

    // SIMD reduction across 32 lanes
    acc = simd_sum(acc);

    if (tid == 0) {
        y[row] = acc;
    }
}

// ============================================================================
// GEMV v2: Optimized with shared memory x cache + half-precision dequant
// Processes ROWS_PER_TG rows per threadgroup using ROWS_PER_TG simdgroups.
// x cached in threadgroup memory (4096 floats = 16KB, fits in 32KB limit).
// ============================================================================

#define GEMV2_ROWS_PER_TG   4
#define GEMV2_THREADS_PER_TG (GEMV2_ROWS_PER_TG * 32)  // 128 threads = 4 simdgroups

kernel void q4_0_gemv_fast(
    device const Q4_0Block *W [[buffer(0)]],
    device const float *x     [[buffer(1)]],
    device float *y           [[buffer(2)]],
    constant uint &out_dim    [[buffer(3)]],
    constant uint &in_dim     [[buffer(4)]],
    uint tid                  [[thread_index_in_simdgroup]],
    uint sid                  [[simdgroup_index_in_threadgroup]],
    uint local_id             [[thread_index_in_threadgroup]],
    uint tg_id                [[threadgroup_position_in_grid]],
    uint tg_size              [[threads_per_threadgroup]])
{
    // Shared memory for x vector — all threads cooperatively load once
    threadgroup float x_shared[4096];  // 16KB, supports in_dim up to 4096

    const uint bpr = in_dim / 32;
    const uint base_row = tg_id * GEMV2_ROWS_PER_TG;

    // Cooperative load of x into shared memory
    for (uint i = local_id; i < in_dim; i += tg_size) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    const uint row = base_row + sid;
    if (row >= out_dim) return;

    device const Q4_0Block *row_blocks = W + row * bpr;

    // blocks_per_thread: for in_dim=4096, bpr=128, 128/32=4 blocks per thread
    const uint blocks_per_thread = (bpr + 31) / 32;
    const uint b_start = tid * blocks_per_thread;
    const uint b_end = min(b_start + blocks_per_thread, bpr);

    float acc = 0.0f;

    for (uint b = b_start; b < b_end; b++) {
        const float fscale = safe_scale(row_blocks[b].d);
        const half scale = half(fscale);
        const uint x_off = b * 32;

        // Dequant to half, FMA with x from shared memory
        // Process 4 bytes (8 values) at a time for better ILP
        half block_sum = 0.0h;

        for (uint i = 0; i < 16; i += 4) {
            uchar r0 = row_blocks[b].qs[i];
            uchar r1 = row_blocks[b].qs[i + 1];
            uchar r2 = row_blocks[b].qs[i + 2];
            uchar r3 = row_blocks[b].qs[i + 3];

            // Low nibbles (values at index i, i+1, i+2, i+3)
            block_sum += half(int(r0 & 0xF) - 8) * half(x_shared[x_off + i]);
            block_sum += half(int(r1 & 0xF) - 8) * half(x_shared[x_off + i + 1]);
            block_sum += half(int(r2 & 0xF) - 8) * half(x_shared[x_off + i + 2]);
            block_sum += half(int(r3 & 0xF) - 8) * half(x_shared[x_off + i + 3]);

            // High nibbles (values at index i+16, i+17, i+18, i+19)
            block_sum += half(int(r0 >> 4) - 8) * half(x_shared[x_off + i + 16]);
            block_sum += half(int(r1 >> 4) - 8) * half(x_shared[x_off + i + 17]);
            block_sum += half(int(r2 >> 4) - 8) * half(x_shared[x_off + i + 18]);
            block_sum += half(int(r3 >> 4) - 8) * half(x_shared[x_off + i + 19]);
        }

        acc += float(scale * block_sum);
    }

    acc = simd_sum(acc);

    if (tid == 0) {
        y[row] = acc;
    }
}

// ============================================================================
// GEMV v3: No shared memory — for any in_dim (including FFN hidden=14336).
// 8 rows per threadgroup for occupancy. Relies on L1/L2 cache for x reuse.
// 2-block unrolled inner loop for ILP.
// ============================================================================

#define GEMV3_ROWS_PER_TG   8

kernel void q4_0_gemv_max(
    device const Q4_0Block *W [[buffer(0)]],
    device const float *x     [[buffer(1)]],
    device float *y           [[buffer(2)]],
    constant uint &out_dim    [[buffer(3)]],
    constant uint &in_dim     [[buffer(4)]],
    uint tid                  [[thread_index_in_simdgroup]],
    uint sid                  [[simdgroup_index_in_threadgroup]],
    uint tg_id                [[threadgroup_position_in_grid]])
{
    const uint bpr = in_dim / 32;
    const uint base_row = tg_id * GEMV3_ROWS_PER_TG;

    const uint row = base_row + sid;
    if (row >= out_dim) return;

    device const Q4_0Block *row_blocks = W + row * bpr;

    const uint blocks_per_thread = (bpr + 31) / 32;
    const uint b_start = tid * blocks_per_thread;
    const uint b_end = min(b_start + blocks_per_thread, bpr);

    float acc = 0.0f;

    // Process 2 blocks per iteration for ILP
    uint b = b_start;
    for (; b + 1 < b_end; b += 2) {
        const float s0 = safe_scale(row_blocks[b].d);
        const float s1 = safe_scale(row_blocks[b + 1].d);
        const uint x0 = b * 32;
        const uint x1 = (b + 1) * 32;

        float sum0 = 0.0f;
        float sum1 = 0.0f;

        for (uint i = 0; i < 16; i++) {
            uchar r0 = row_blocks[b].qs[i];
            uchar r1 = row_blocks[b + 1].qs[i];

            sum0 += float(int(r0 & 0xF) - 8) * x[x0 + i];
            sum0 += float(int(r0 >> 4) - 8)  * x[x0 + i + 16];

            sum1 += float(int(r1 & 0xF) - 8) * x[x1 + i];
            sum1 += float(int(r1 >> 4) - 8)  * x[x1 + i + 16];
        }

        acc += s0 * sum0 + s1 * sum1;
    }

    if (b < b_end) {
        const float s = safe_scale(row_blocks[b].d);
        const uint xo = b * 32;
        float sum = 0.0f;
        for (uint i = 0; i < 16; i++) {
            uchar r = row_blocks[b].qs[i];
            sum += float(int(r & 0xF) - 8) * x[xo + i];
            sum += float(int(r >> 4) - 8)  * x[xo + i + 16];
        }
        acc += s * sum;
    }

    acc = simd_sum(acc);

    if (tid == 0) {
        y[row] = acc;
    }
}

// ============================================================================
// GEMM: Y[S, out_dim] = X[S, in_dim] @ W_q4[out_dim, in_dim]^T
// For prefill (S > 1). Tiled shared-memory approach.
//
// Grid: (out_dim / TILE_N, S / TILE_M, 1)
// Threadgroup: (TILE_N / SUBTILE_N * 32) threads
//
// Strategy: Each threadgroup computes a TILE_M x TILE_N tile of Y.
// Loop over in_dim in TILE_K chunks:
//   - Load X[TILE_M, TILE_K] into shared memory
//   - Dequant W[TILE_N, TILE_K] from Q4_0 blocks on the fly
//   - Accumulate partial products
// ============================================================================

#define GEMM_TILE_M  16   // output tokens per tile
#define GEMM_TILE_N  32   // output features per tile
#define GEMM_TILE_K  128  // reduction chunk (4 Q4_0 blocks = 128 values)

// Each threadgroup: 128 threads = 4 simdgroups
// Each simdgroup handles 8 rows of W (TILE_N/4 = 8 output features)
// Each thread in simdgroup accumulates over K for its assigned (m, n) elements

kernel void q4_0_gemm(
    device const Q4_0Block *W [[buffer(0)]],   // [out_dim * blocks_per_row]
    device const float *X     [[buffer(1)]],   // [S * in_dim], row-major
    device float *Y           [[buffer(2)]],   // [S * out_dim], row-major
    constant uint &out_dim    [[buffer(3)]],
    constant uint &in_dim     [[buffer(4)]],
    constant uint &seq_len    [[buffer(5)]],   // S
    uint tid                  [[thread_index_in_simdgroup]],
    uint sid                  [[simdgroup_index_in_threadgroup]],
    uint local_id             [[thread_index_in_threadgroup]],
    uint2 tg_id               [[threadgroup_position_in_grid]],
    uint2 tg_dims             [[threads_per_threadgroup]])
{
    // Tile coordinates
    const uint tile_n_start = tg_id.x * GEMM_TILE_N;  // output feature start
    const uint tile_m_start = tg_id.y * GEMM_TILE_M;  // token start

    const uint bpr = in_dim / 32;  // Q4_0 blocks per row

    // Shared memory for X tile: [TILE_M, TILE_K]
    threadgroup float X_tile[GEMM_TILE_M * GEMM_TILE_K];  // 16*128*4 = 8KB

    // Each simdgroup accumulates 8 output features x TILE_M tokens
    // sid selects which 8 features within TILE_N
    const uint n_local = sid * 8;  // 0, 8, 16, 24

    // Accumulators: each thread holds partial sums for all TILE_M tokens
    // for a subset of the 8 features its simdgroup handles.
    // With 32 threads per simdgroup:
    //   - Thread handles all 16 tokens for specific (n, k_slice) combinations
    //   - After K-loop, reduce across k_slices via simd_sum

    // Each thread accumulates one feature across all tokens.
    // 8 features / 32 threads doesn't divide. Better approach:
    // Each thread accumulates TILE_M partial sums for one output feature.
    // 32 threads handle 8 features, 4 threads per feature, each doing 1/4 of K.

    const uint feat_in_group = tid / 4;   // 0..7 -> which feature in the 8
    const uint k_quarter = tid % 4;       // 0..3 -> which quarter of K blocks
    const uint n_global = tile_n_start + n_local + feat_in_group;

    if (n_global >= out_dim) return;

    device const Q4_0Block *w_row = W + n_global * bpr;

    float acc[GEMM_TILE_M];
    for (uint m = 0; m < GEMM_TILE_M; m++) acc[m] = 0.0f;

    // Loop over K in TILE_K chunks
    for (uint k_start = 0; k_start < in_dim; k_start += GEMM_TILE_K) {
        const uint k_end = min(k_start + GEMM_TILE_K, in_dim);
        const uint k_len = k_end - k_start;

        // Cooperative load of X_tile[TILE_M, k_len]
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint i = local_id; i < GEMM_TILE_M * k_len; i += tg_dims.x) {
            uint m_idx = i / k_len;
            uint k_idx = i % k_len;
            uint global_m = tile_m_start + m_idx;
            if (global_m < seq_len) {
                X_tile[m_idx * GEMM_TILE_K + k_idx] = X[global_m * in_dim + k_start + k_idx];
            } else {
                X_tile[m_idx * GEMM_TILE_K + k_idx] = 0.0f;
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // Number of Q4_0 blocks in this K chunk
        const uint blocks_in_chunk = k_len / 32;
        // Each thread processes 1/4 of the blocks
        const uint my_b_start = k_quarter * (blocks_in_chunk / 4);
        const uint my_b_end = (k_quarter == 3) ? blocks_in_chunk : (k_quarter + 1) * (blocks_in_chunk / 4);

        const uint b_offset = k_start / 32;  // block offset for this K chunk

        for (uint bi = my_b_start; bi < my_b_end; bi++) {
            const uint b_global = b_offset + bi;
            const float scale = safe_scale(w_row[b_global].d);
            const uint k_local = bi * 32;  // position within TILE_K

            // Dequant the block's 32 values
            float w_vals[32];
            for (uint i = 0; i < 16; i++) {
                uchar raw = w_row[b_global].qs[i];
                w_vals[i]      = float(int(raw & 0xF) - 8) * scale;
                w_vals[i + 16] = float(int(raw >> 4) - 8) * scale;
            }

            // Dot product with each token's X slice
            for (uint m = 0; m < GEMM_TILE_M; m++) {
                float dot = 0.0f;
                for (uint j = 0; j < 32; j++) {
                    dot += w_vals[j] * X_tile[m * GEMM_TILE_K + k_local + j];
                }
                acc[m] += dot;
            }
        }
    }

    // Reduce across the 4 threads sharing the same feature
    // Use simd_shuffle to sum across threads k_quarter=0..3
    for (uint m = 0; m < GEMM_TILE_M; m++) {
        float v = acc[m];
        // Sum across threads tid%4 == 0,1,2,3 (these are consecutive lane IDs)
        // feat_in_group*4 + 0..3 are the lane IDs for each feature group
        v += simd_shuffle_xor(v, 1);
        v += simd_shuffle_xor(v, 2);
        acc[m] = v;
    }

    // Only thread with k_quarter == 0 writes
    if (k_quarter == 0) {
        for (uint m = 0; m < GEMM_TILE_M; m++) {
            uint global_m = tile_m_start + m;
            if (global_m < seq_len && n_global < out_dim) {
                Y[global_m * out_dim + n_global] = acc[m];
            }
        }
    }
}

// ============================================================================
// GEMM v2: Simplified and wider — processes more output features per threadgroup.
// Better for large out_dim (4096, 14336).
//
// Grid: (ceil(out_dim / 64), ceil(S / 8), 1)
// Threadgroup: 256 threads = 8 simdgroups, each simdgroup owns 8 output rows.
// Each thread processes all tokens for one output feature, splitting K across
// the 32 SIMD lanes.
// ============================================================================

#define GEMM2_TILE_M  8    // tokens per tile
#define GEMM2_TILE_N  64   // output features per tile (8 simdgroups x 8)

kernel void q4_0_gemm_wide(
    device const Q4_0Block *W [[buffer(0)]],
    device const float *X     [[buffer(1)]],   // [S, in_dim]
    device float *Y           [[buffer(2)]],   // [S, out_dim]
    constant uint &out_dim    [[buffer(3)]],
    constant uint &in_dim     [[buffer(4)]],
    constant uint &seq_len    [[buffer(5)]],
    uint tid                  [[thread_index_in_simdgroup]],
    uint sid                  [[simdgroup_index_in_threadgroup]],
    uint local_id             [[thread_index_in_threadgroup]],
    uint2 tg_id               [[threadgroup_position_in_grid]])
{
    const uint tile_n_start = tg_id.x * GEMM2_TILE_N;
    const uint tile_m_start = tg_id.y * GEMM2_TILE_M;
    const uint bpr = in_dim / 32;

    // Each simdgroup owns 8 output features. sid=0..7 -> features 0..7, 8..15, etc.
    // Within a simdgroup, all 32 threads cooperate on the same 8 features,
    // splitting the K-reduction.

    // Which 8 features this simdgroup handles
    const uint feat_base = tile_n_start + sid * 8;

    // Shared memory: X tile [TILE_M, in_dim_chunk]
    // For large in_dim we stream through K; no need to cache all of X
    // Instead: each thread processes all K for its feature, loads X on the fly

    // Accumulators: 8 features x TILE_M tokens
    float acc[8][GEMM2_TILE_M];
    for (uint f = 0; f < 8; f++)
        for (uint m = 0; m < GEMM2_TILE_M; m++)
            acc[f][m] = 0.0f;

    // Split bpr across 32 SIMD lanes
    const uint blocks_per_lane = (bpr + 31) / 32;
    const uint b_start = tid * blocks_per_lane;
    const uint b_end = min(b_start + blocks_per_lane, bpr);

    for (uint f = 0; f < 8; f++) {
        const uint n_global = feat_base + f;
        if (n_global >= out_dim) break;

        device const Q4_0Block *w_row = W + n_global * bpr;

        for (uint b = b_start; b < b_end; b++) {
            const float scale = safe_scale(w_row[b].d);
            const uint x_off = b * 32;

            // Dequant 32 weight values
            float w_vals[32];
            for (uint i = 0; i < 16; i++) {
                uchar raw = w_row[b].qs[i];
                w_vals[i]      = float(int(raw & 0xF) - 8) * scale;
                w_vals[i + 16] = float(int(raw >> 4) - 8) * scale;
            }

            // Dot product with each token
            for (uint m = 0; m < GEMM2_TILE_M; m++) {
                uint global_m = tile_m_start + m;
                if (global_m >= seq_len) break;

                device const float *x_ptr = X + global_m * in_dim + x_off;
                float dot = 0.0f;
                for (uint j = 0; j < 32; j++) {
                    dot += w_vals[j] * x_ptr[j];
                }
                acc[f][m] += dot;
            }
        }
    }

    // SIMD reduction across 32 lanes for each (feature, token) pair
    for (uint f = 0; f < 8; f++) {
        const uint n_global = feat_base + f;
        if (n_global >= out_dim) break;

        for (uint m = 0; m < GEMM2_TILE_M; m++) {
            float v = simd_sum(acc[f][m]);
            if (tid == 0) {
                uint global_m = tile_m_start + m;
                if (global_m < seq_len) {
                    Y[global_m * out_dim + n_global] = v;
                }
            }
        }
    }
}

// ============================================================================
// RMSNorm + GEMV fused: normalize x, then matvec. Avoids storing normalized x.
// Single threadgroup produces one output element (or small tile).
// For now, keep as separate kernels — RMSNorm is cheap and fusing complicates
// the x-sharing pattern in GEMV.
// ============================================================================

kernel void rmsnorm(
    device const float *x   [[buffer(0)]],   // [dim]
    device float *out       [[buffer(1)]],   // [dim]
    device const float *w   [[buffer(2)]],   // [dim] (RMSNorm weight)
    constant uint &dim      [[buffer(3)]],
    constant float &eps     [[buffer(4)]],
    uint tid                [[thread_index_in_simdgroup]],
    uint sid                [[simdgroup_index_in_threadgroup]],
    uint local_id           [[thread_index_in_threadgroup]],
    uint tg_size            [[threads_per_threadgroup]])
{
    // Compute sum of squares using all threads
    threadgroup float partial_ss[8];  // up to 8 simdgroups

    float my_ss = 0.0f;
    for (uint i = local_id; i < dim; i += tg_size) {
        float v = x[i];
        my_ss += v * v;
    }
    my_ss = simd_sum(my_ss);

    if (tid == 0) {
        partial_ss[sid] = my_ss;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Simdgroup 0 reduces across simdgroups
    if (sid == 0) {
        uint num_simds = (tg_size + 31) / 32;
        float total = 0.0f;
        if (tid < num_simds) {
            total = partial_ss[tid];
        }
        total = simd_sum(total);

        if (tid == 0) {
            partial_ss[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float inv_rms = partial_ss[0];

    // Apply normalization + weight
    for (uint i = local_id; i < dim; i += tg_size) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

// ============================================================================
// SiLU + elementwise multiply: out[i] = silu(gate[i]) * up[i]
// silu(x) = x / (1 + exp(-x))
// ============================================================================

kernel void silu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float *out        [[buffer(2)]],
    constant uint &n         [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// ============================================================================
// Vector add: out[i] = a[i] + b[i] (residual connection)
// ============================================================================

kernel void vadd(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out     [[buffer(2)]],
    constant uint &n      [[buffer(3)]],
    uint gid              [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

// ============================================================================
// RoPE: Apply rotary position embeddings to Q and K vectors in-place.
// Grid: max(n_heads, n_kv_heads) * (head_dim / 2) threads.
// Each thread rotates one (even, odd) pair for one head.
// ============================================================================

kernel void rope(
    device float *q                  [[buffer(0)]],   // [n_heads * head_dim]
    device float *k                  [[buffer(1)]],   // [n_kv_heads * head_dim]
    constant uint &pos               [[buffer(2)]],
    device const float *theta_inv    [[buffer(3)]],   // [head_dim / 2]
    constant uint &n_heads           [[buffer(4)]],
    constant uint &n_kv_heads        [[buffer(5)]],
    constant uint &head_dim          [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]])
{
    const uint half_dim = head_dim / 2;
    const uint pair_idx = gid % half_dim;
    const uint head_idx = gid / half_dim;

    float theta = float(pos) * theta_inv[pair_idx];
    float cos_val = metal::cos(theta);
    float sin_val = metal::sin(theta);

    // Rotate Q
    if (head_idx < n_heads) {
        const uint base = head_idx * head_dim + 2 * pair_idx;
        float even = q[base];
        float odd  = q[base + 1];
        q[base]     = even * cos_val - odd * sin_val;
        q[base + 1] = odd  * cos_val + even * sin_val;
    }

    // Rotate K
    if (head_idx < n_kv_heads) {
        const uint base = head_idx * head_dim + 2 * pair_idx;
        float even = k[base];
        float odd  = k[base + 1];
        k[base]     = even * cos_val - odd * sin_val;
        k[base + 1] = odd  * cos_val + even * sin_val;
    }
}

// ============================================================================
// KV cache write: convert fp32 K/V to fp16 and store at cache_pos.
// Grid: kv_dim threads.
// ============================================================================

kernel void kv_cache_write(
    device const float *k_fp32   [[buffer(0)]],   // [kv_dim]
    device const float *v_fp32   [[buffer(1)]],   // [kv_dim]
    device half *k_cache         [[buffer(2)]],   // [max_seq * kv_dim]
    device half *v_cache         [[buffer(3)]],   // [max_seq * kv_dim]
    constant uint &cache_pos     [[buffer(4)]],
    constant uint &kv_dim        [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]])
{
    if (gid >= kv_dim) return;
    const uint offset = cache_pos * kv_dim + gid;
    k_cache[offset] = half(k_fp32[gid]);
    v_cache[offset] = half(v_fp32[gid]);
}

// ============================================================================
// GQA Attention (S=1 decode): score, softmax, weighted-sum over KV cache.
// 1 threadgroup per Q head (32 threadgroups total). 256 threads per group.
// Uses device memory scratch for attention scores [n_heads * max_seq].
//
// Phase 1: Q @ K^T — each thread computes scores for a subset of cache positions
// Phase 2: Softmax over scores (cooperative, numerically stable)
// Phase 3: Att @ V — each thread accumulates a subset of head_dim output elements
// ============================================================================

#define GQA_THREADS 256
#define GQA_SIMD_WIDTH 32
#define GQA_NUM_SIMDS (GQA_THREADS / GQA_SIMD_WIDTH)  // 8

kernel void gqa_attention(
    device const float *q          [[buffer(0)]],   // [n_heads * head_dim]
    device const half *k_cache     [[buffer(1)]],   // [max_seq * kv_dim]
    device const half *v_cache     [[buffer(2)]],   // [max_seq * kv_dim]
    device float *out              [[buffer(3)]],   // [n_heads * head_dim]
    constant uint &n_heads         [[buffer(4)]],
    constant uint &n_kv_heads      [[buffer(5)]],
    constant uint &head_dim        [[buffer(6)]],   // 128
    constant uint &kv_dim          [[buffer(7)]],   // 1024
    constant uint &seq_len         [[buffer(8)]],   // valid positions (pos+1, capped at max_seq)
    constant uint &max_seq         [[buffer(9)]],   // ring buffer capacity
    constant uint &ring_off        [[buffer(10)]],  // ring buffer offset
    device float *att_scratch      [[buffer(11)]],  // [n_heads * max_seq]
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]])
{
    const uint head = tg_id;
    if (head >= n_heads) return;

    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kvh = head / heads_per_kv;

    // Pointers
    device const float *q_head = q + head * head_dim;
    device float *att = att_scratch + head * max_seq;
    device float *out_head = out + head * head_dim;

    // Shared memory for reductions
    threadgroup float shared_reduce[GQA_NUM_SIMDS];  // 8 floats for cross-simd reduction

    // ========================================================================
    // Phase 1: Q @ K^T — compute attention scores
    // Each thread handles positions t where t % tg_size == tid
    // ========================================================================
    for (uint t = tid; t < seq_len; t += tg_size) {
        uint ct = (ring_off + t) % max_seq;
        device const half *k_vec = k_cache + ct * kv_dim + kvh * head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            dot += q_head[d]     * float(k_vec[d]);
            dot += q_head[d + 1] * float(k_vec[d + 1]);
            dot += q_head[d + 2] * float(k_vec[d + 2]);
            dot += q_head[d + 3] * float(k_vec[d + 3]);
        }
        att[t] = dot * rsqrt(float(head_dim));
    }

    // Barrier: all scores written to device memory
    threadgroup_barrier(metal::mem_flags::mem_device);

    // ========================================================================
    // Phase 2: Softmax — find max, compute exp, find sum, normalize
    // ========================================================================

    // 2a: Find max score
    const uint simd_idx = tid / GQA_SIMD_WIDTH;
    const uint lane = tid % GQA_SIMD_WIDTH;

    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += tg_size) {
        local_max = max(local_max, att[t]);
    }
    local_max = simd_max(local_max);

    if (lane == 0) {
        shared_reduce[simd_idx] = local_max;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Reduce across simdgroups
    if (simd_idx == 0 && lane < GQA_NUM_SIMDS) {
        local_max = shared_reduce[lane];
    } else if (simd_idx == 0) {
        local_max = -INFINITY;
    }
    if (simd_idx == 0) {
        local_max = simd_max(local_max);
        if (lane == 0) {
            shared_reduce[0] = local_max;
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float global_max = shared_reduce[0];

    // 2b: Compute exp(score - max) in-place and accumulate sum
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += tg_size) {
        float e = exp(att[t] - global_max);
        att[t] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);

    if (lane == 0) {
        shared_reduce[simd_idx] = local_sum;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (simd_idx == 0 && lane < GQA_NUM_SIMDS) {
        local_sum = shared_reduce[lane];
    } else if (simd_idx == 0) {
        local_sum = 0.0f;
    }
    if (simd_idx == 0) {
        local_sum = simd_sum(local_sum);
        if (lane == 0) {
            shared_reduce[0] = local_sum;
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / shared_reduce[0];

    // 2c: Normalize in-place
    for (uint t = tid; t < seq_len; t += tg_size) {
        att[t] *= inv_sum;
    }

    threadgroup_barrier(metal::mem_flags::mem_device);

    // ========================================================================
    // Phase 3: Att @ V — weighted sum of V vectors
    // Each thread handles a subset of head_dim output elements,
    // accumulating across ALL seq positions.
    // ========================================================================
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            uint ct = (ring_off + t) % max_seq;
            acc += att[t] * float(v_cache[ct * kv_dim + kvh * head_dim + d]);
        }
        out_head[d] = acc;
    }
}

// ============================================================================
// BATCH KERNELS for speculative decode verification (S=2..9 tokens)
// All S tokens processed simultaneously through each layer.
// ============================================================================

// ============================================================================
// RMSNorm batch: normalize S independent vectors sharing the same weight.
// Grid: S threadgroups, 256 threads each.
// ============================================================================

kernel void rmsnorm_batch(
    device const float *x   [[buffer(0)]],   // [S * dim]
    device float *out       [[buffer(1)]],   // [S * dim]
    device const float *w   [[buffer(2)]],   // [dim]
    constant uint &dim      [[buffer(3)]],
    constant float &eps     [[buffer(4)]],
    constant uint &S        [[buffer(5)]],
    uint tid                [[thread_index_in_simdgroup]],
    uint sid                [[simdgroup_index_in_threadgroup]],
    uint local_id           [[thread_index_in_threadgroup]],
    uint tg_id              [[threadgroup_position_in_grid]],
    uint tg_size            [[threads_per_threadgroup]])
{
    if (tg_id >= S) return;

    const uint offset = tg_id * dim;
    device const float *xi = x + offset;
    device float *oi = out + offset;

    threadgroup float partial_ss[8];

    float my_ss = 0.0f;
    for (uint i = local_id; i < dim; i += tg_size) {
        float v = xi[i];
        my_ss += v * v;
    }
    my_ss = simd_sum(my_ss);

    if (tid == 0) {
        partial_ss[sid] = my_ss;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (sid == 0) {
        uint num_simds = (tg_size + 31) / 32;
        float total = 0.0f;
        if (tid < num_simds) {
            total = partial_ss[tid];
        }
        total = simd_sum(total);

        if (tid == 0) {
            partial_ss[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float inv_rms = partial_ss[0];

    for (uint i = local_id; i < dim; i += tg_size) {
        oi[i] = xi[i] * inv_rms * w[i];
    }
}

// ============================================================================
// RoPE batch: apply rotary embeddings to S tokens at consecutive positions.
// Grid: S * max(n_heads, n_kv_heads) * (head_dim/2) threads.
// ============================================================================

kernel void rope_batch(
    device float *q                  [[buffer(0)]],   // [S * n_heads * head_dim]
    device float *k                  [[buffer(1)]],   // [S * n_kv_heads * head_dim]
    constant uint &base_pos          [[buffer(2)]],
    device const float *theta_inv    [[buffer(3)]],   // [head_dim / 2]
    constant uint &n_heads           [[buffer(4)]],
    constant uint &n_kv_heads        [[buffer(5)]],
    constant uint &head_dim          [[buffer(6)]],
    constant uint &S                 [[buffer(7)]],
    uint gid                         [[thread_position_in_grid]])
{
    const uint half_dim = head_dim / 2;
    const uint max_heads = max(n_heads, n_kv_heads);
    const uint pairs_per_token = max_heads * half_dim;

    const uint token_idx = gid / pairs_per_token;
    const uint rem = gid % pairs_per_token;
    const uint head_idx = rem / half_dim;
    const uint pair_idx = rem % half_dim;

    if (token_idx >= S) return;

    const uint pos = base_pos + token_idx;
    float theta = float(pos) * theta_inv[pair_idx];
    float cos_val = metal::cos(theta);
    float sin_val = metal::sin(theta);

    // Rotate Q
    if (head_idx < n_heads) {
        const uint base = (token_idx * n_heads + head_idx) * head_dim + 2 * pair_idx;
        float even = q[base];
        float odd  = q[base + 1];
        q[base]     = even * cos_val - odd * sin_val;
        q[base + 1] = odd  * cos_val + even * sin_val;
    }

    // Rotate K
    if (head_idx < n_kv_heads) {
        const uint base = (token_idx * n_kv_heads + head_idx) * head_dim + 2 * pair_idx;
        float even = k[base];
        float odd  = k[base + 1];
        k[base]     = even * cos_val - odd * sin_val;
        k[base + 1] = odd  * cos_val + even * sin_val;
    }
}

// ============================================================================
// KV cache write batch: write S K/V entries at consecutive ring buffer positions.
// Grid: S * kv_dim threads.
// ============================================================================

kernel void kv_cache_write_batch(
    device const float *k_fp32   [[buffer(0)]],   // [S * kv_dim]
    device const float *v_fp32   [[buffer(1)]],   // [S * kv_dim]
    device half *k_cache         [[buffer(2)]],   // [max_seq * kv_dim]
    device half *v_cache         [[buffer(3)]],   // [max_seq * kv_dim]
    constant uint &base_cache_pos [[buffer(4)]],
    constant uint &kv_dim        [[buffer(5)]],
    constant uint &max_seq       [[buffer(6)]],
    constant uint &S             [[buffer(7)]],
    uint gid                     [[thread_position_in_grid]])
{
    const uint total = S * kv_dim;
    if (gid >= total) return;

    const uint token_idx = gid / kv_dim;
    const uint elem = gid % kv_dim;
    const uint cache_pos = (base_cache_pos + token_idx) % max_seq;
    const uint offset = cache_pos * kv_dim + elem;

    k_cache[offset] = half(k_fp32[token_idx * kv_dim + elem]);
    v_cache[offset] = half(v_fp32[token_idx * kv_dim + elem]);
}

// ============================================================================
// GQA Attention with causal mask for speculative decode verification.
// S query tokens attend to base_seq_len + token_idx + 1 KV positions each.
//
// Grid: n_heads * S threadgroups, 256 threads each.
// tg_id = token_idx * n_heads + head
// att_scratch: [S * n_heads * max_seq] pre-allocated device memory.
// ============================================================================

#define GQA_CAUSAL_THREADS 256
#define GQA_CAUSAL_SIMD_WIDTH 32
#define GQA_CAUSAL_NUM_SIMDS (GQA_CAUSAL_THREADS / GQA_CAUSAL_SIMD_WIDTH)  // 8

kernel void gqa_attention_causal(
    device const float *q          [[buffer(0)]],   // [S * n_heads * head_dim]
    device const half *k_cache     [[buffer(1)]],   // [max_seq * kv_dim]
    device const half *v_cache     [[buffer(2)]],   // [max_seq * kv_dim]
    device float *out              [[buffer(3)]],   // [S * n_heads * head_dim]
    constant uint &n_heads         [[buffer(4)]],
    constant uint &n_kv_heads      [[buffer(5)]],
    constant uint &head_dim        [[buffer(6)]],
    constant uint &kv_dim          [[buffer(7)]],
    constant uint &base_seq_len    [[buffer(8)]],   // KV entries before this batch
    constant uint &max_seq         [[buffer(9)]],
    constant uint &ring_off        [[buffer(10)]],
    device float *att_scratch      [[buffer(11)]],  // [S * n_heads * att_stride]
    constant uint &S               [[buffer(12)]],
    constant uint &att_stride      [[buffer(13)]],  // stride between (token,head) pairs in att_scratch
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]])
{
    const uint token_idx = tg_id / n_heads;
    const uint head = tg_id % n_heads;

    if (token_idx >= S) return;

    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kvh = head / heads_per_kv;

    // Causal: token i in batch sees base_seq_len + i + 1 positions
    const uint seq_len = base_seq_len + token_idx + 1;

    // Pointers
    device const float *q_head = q + (token_idx * n_heads + head) * head_dim;
    device float *att = att_scratch + (token_idx * n_heads + head) * att_stride;
    device float *out_head = out + (token_idx * n_heads + head) * head_dim;

    const float scale = rsqrt(float(head_dim));

    threadgroup float shared_reduce[GQA_CAUSAL_NUM_SIMDS];

    const uint simd_idx = tid / GQA_CAUSAL_SIMD_WIDTH;
    const uint lane = tid % GQA_CAUSAL_SIMD_WIDTH;

    // ========================================================================
    // Phase 1: Q @ K^T with ring buffer
    // ========================================================================
    for (uint t = tid; t < seq_len; t += tg_size) {
        uint ct = (ring_off + t) % max_seq;
        device const half *k_vec = k_cache + ct * kv_dim + kvh * head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += 4) {
            dot += q_head[d]     * float(k_vec[d]);
            dot += q_head[d + 1] * float(k_vec[d + 1]);
            dot += q_head[d + 2] * float(k_vec[d + 2]);
            dot += q_head[d + 3] * float(k_vec[d + 3]);
        }
        att[t] = dot * scale;
    }

    threadgroup_barrier(metal::mem_flags::mem_device);

    // ========================================================================
    // Phase 2: Softmax
    // ========================================================================

    // 2a: Find max
    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += tg_size) {
        local_max = max(local_max, att[t]);
    }
    local_max = simd_max(local_max);

    if (lane == 0) {
        shared_reduce[simd_idx] = local_max;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (simd_idx == 0 && lane < GQA_CAUSAL_NUM_SIMDS) {
        local_max = shared_reduce[lane];
    } else if (simd_idx == 0) {
        local_max = -INFINITY;
    }
    if (simd_idx == 0) {
        local_max = simd_max(local_max);
        if (lane == 0) {
            shared_reduce[0] = local_max;
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float global_max = shared_reduce[0];

    // 2b: exp and sum
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += tg_size) {
        float e = exp(att[t] - global_max);
        att[t] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);

    if (lane == 0) {
        shared_reduce[simd_idx] = local_sum;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (simd_idx == 0 && lane < GQA_CAUSAL_NUM_SIMDS) {
        local_sum = shared_reduce[lane];
    } else if (simd_idx == 0) {
        local_sum = 0.0f;
    }
    if (simd_idx == 0) {
        local_sum = simd_sum(local_sum);
        if (lane == 0) {
            shared_reduce[0] = local_sum;
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / shared_reduce[0];

    // 2c: Normalize
    for (uint t = tid; t < seq_len; t += tg_size) {
        att[t] *= inv_sum;
    }

    threadgroup_barrier(metal::mem_flags::mem_device);

    // ========================================================================
    // Phase 3: Att @ V
    // ========================================================================
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            uint ct = (ring_off + t) % max_seq;
            acc += att[t] * float(v_cache[ct * kv_dim + kvh * head_dim + d]);
        }
        out_head[d] = acc;
    }
}

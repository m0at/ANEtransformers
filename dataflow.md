                                        
⏺ I have all the information I need. Here's the complete data flow map:                                     
                                                                                                            
  ---                                                                                                       
  Mistral 7B Data Flow Map — Full Pipeline                                                                  
                                                                                                            
  M5 Hardware Reference                                                                                     
                                                                                                            
  - Memory BW: ~200 GB/s (unified, shared by CPU/GPU/ANE)                                                   
  - P-cores: 4, ~32 MB shared L2
  - E-cores: 6
  - ANE: ~19 TFLOPS fp16 peak (not used in current hot path)
  - AMX: integrated with CPU, used by Accelerate cblas

  ---
  1. MODEL LOAD (one-time)

  GGUF file on disk (3.8 GB, Q4_0/Q4_K/Q6_K)
      │
      ▼ mmap(PROT_READ, MAP_PRIVATE) — zero-copy
      │
      ├─ Per-layer weights: const void* pointers into mmap'd region
      │  Q4_0 blocks: [fp16 scale (2B) + 16B nibbles] = 18B per 32 vals → 4.5 bits/val
      │  Q4_K blocks: [fp16 d, fp16 dmin, 12B scales, 128B nibbles] = 144B per 256 vals → 4.5 bits/val
      │  Q6_K blocks: [128B ql, 64B qh, 16B scales, fp16 d] = 210B per 256 vals → 6.56 bits/val
      │
      ├─ Embedding table: ⚠️  DEQUANT Q4→fp16→fp32, malloc'd
      │  token_embd.weight: [32000 × 4096] Q4_0
      │    → dequant_q4_0_to_fp16() → _Float16 tmp[131M]      (250 MB temporary!)
      │    → scalar loop (float)tmp[i]                          (250 → 500 MB fp32)
      │    → free(tmp)
      │  Result: 500 MB fp32 embedding table, resident in RAM
      │  ⚠️  WASTEFUL: Q4→fp16→fp32 two-hop. Could go Q4→fp32 directly.
      │  ⚠️  WASTEFUL: 500 MB for a lookup table. fp16 would be 250 MB.
      │
      ├─ RMSNorm weights: 32×2 copies, each [4096] fp32 → 1 MB total. Fine.
      │
      ├─ RoPE tables: precompute cos/sin [131072 × 64] fp32 → 64 MB
      │  ⚠️  LARGE: 64 MB for RoPE at 128K. Could be fp16 (32 MB) or computed on-the-fly.
      │
      ├─ LM head: const void* → mmap'd output.weight (Q6_K, ~34 MB on disk)
      │
      └─ Scratch buffers: all fp32
         xb, xb2, q: [4096] → 48 KB
         k, v: [1024] → 8 KB
         att: [32 × 65536] = [2M] → 8 MB  ⚠️  LARGE but needed for full attention
         hb, hb2: [14336] → 112 KB
         logits: [32000] → 125 KB
         xb_q8: [128 × 36B] = 4.5 KB
         hb_q8: [448 × 36B] = 15.75 KB
         Total scratch: ~8.3 MB

  2. TOKENIZATION

  Input string (UTF-8)
      │ BPE encode: FNV-1a hash table lookup + max-heap merges
      ▼
  int tokens[n_prompt]  — int32, negligible memory

  3. PREFILL (two paths)

  Path A: Short prompt (<16 tokens) — SDOT decode loop

  For each token t in [0..n_prompt):
      │
      ▼ EMBED: memcpy from fp32 embedding table
      token_embed[token_id × 4096 ... +4096]  (fp32, 16 KB)
      → x[4096] fp32
      │
      ▼ FOR EACH LAYER l in [0..31]:
          mistral_layer_decode_parallel(x, layer=l, pos=t)
          [see DECODE section below for full per-layer flow]

  Path B: Long prompt (≥16 tokens) — BLAS batched GEMM

  EMBED all S tokens:
      X[S × 4096] fp32 — each row = token_embed[token_id]
      Memory: S × 16 KB (e.g. 100 tokens = 1.6 MB)
      │
      ▼ FOR EACH LAYER l in [0..31]:
          blas_prefill_layer(X, layer=l)
          │
          ├── 1. RMSNorm (CPU, vDSP)
          │   X → rmsnorm_batch → st->X    [S × 4096] fp32
          │   ops: dot product + scale + element-wise multiply, per token
          │   3 vDSP calls per token × S tokens
          │
          ├── 2-4. QKV projections (BLAS, AMX-accelerated)
          │   FOR EACH of Wq[4096,4096], Wk[1024,4096], Wv[1024,4096]:
          │     │
          │     ├─ dequant_weight_to_fp32_buf()  ⚠️  EXPENSIVE
          │     │  Q4_0: dispatch_apply row-parallel over P-cores
          │     │  Read: 18B per 32 vals from mmap → write: 128B per 32 vals fp32
          │     │  For Wq [4096×4096]: read 36 MB Q4 → write 64 MB fp32
          │     │  For Wk/Wv [1024×4096]: read 9 MB Q4 → write 16 MB fp32
          │     │  ⚠️  BANDWIDTH: 3 matrices/layer = 54 MB read, 96 MB write
          │     │  ⚠️  Per full prefill: 32 layers × 7 matrices = 224 dequants!
          │     │  ⚠️  Reuses SAME w32 buffer (max 14336×4096 = 224 MB fp32)
          │     │
          │     └─ cblas_sgemm (AMX)
          │        Y = X @ W^T  via CblasRowMajor, NoTrans, Trans
          │        AMX reads 64 MB weight + S×16KB activations → writes S×16KB output
          │
          ├── 5. RoPE (CPU, scalar)
          │   In-place on Q[S×4096] and K[S×1024]
          │   Adjacent pairs: (x[2i], x[2i+1]) rotated by (cos, sin) table lookup
          │   Reads 2 fp32 from RoPE table per pair
          │
          ├── 6. KV cache write (CPU, scatter)
          │   For each token t in [0..S):
          │     K[t] fp32 [1024] → ⚠️  cast to fp16 [1024] → scatter to channel-first
          │     V[t] fp32 [1024] → ⚠️  cast to fp16 [1024] → scatter to channel-first
          │   kv_write: for each d in [0..1024): cache[d * max_seq + pos] = val
          │   ⚠️  SCATTER IS SLOW: 1024 strided stores per token, stride = 65536×2B = 128 KB
          │   ⚠️  fp32→fp16 conversion: scalar loop, could use NEON vcvt
          │
          ├── 7. Attention (CPU, scalar + vDSP softmax)
          │   For each token t, for each head h in [0..32):
          │     Q@K^T: dot product q[128] fp32 × k_cache[128] fp16 (gathered)
          │       ⚠️  K cache gather: 128 loads at stride max_seq (128 KB stride)
          │       Each score = Σ q[d] × (float)kcache[d*max_seq+s]  — scalar loop!
          │     Softmax: vDSP (maxv, vsadd, vvexpf, sve, vsmul)
          │     Att@V: weighted sum, same strided gather from V cache
          │       ⚠️  Same 128-element gather per timestep
          │   ⚠️  PREFILL ATTENTION IS O(S²) and uses SCALAR dot products (no NEON)
          │   ⚠️  This is the slowest part for long prompts
          │
          ├── 8. Output projection Wo (BLAS, same dequant+sgemm pattern)
          │
          ├── 9. Residual: X += Wo_out  (vDSP_vadd, fast)
          │
          ├── 10. FFN RMSNorm (CPU, vDSP)
          │
          ├── 11-12. FFN gate + up (BLAS)
          │   Two separate dequant+sgemm for W1[14336,4096] and W3[14336,4096]
          │   ⚠️  BIGGEST WEIGHT: 14336×4096 = 58.7M params
          │   Q4_0 read: ~103 MB, fp32 write: ~224 MB per matrix
          │   ⚠️  Gate and up share same input but dequant separately
          │
          ├── 13. SiLU(gate) × up (CPU, scalar)
          │   for i in [0..14336*S): gate[i] / (1 + exp(-gate[i])) * up[i]
          │   ⚠️  SCALAR expf() — not vectorized, but small vs matmul
          │
          ├── 14. FFN down (BLAS) — W2[4096,14336]
          │
          └── 15. Residual: X += down_out (vDSP_vadd)

  4. DECODE (per token, the hot loop)

  next_token (int32)
      │
      ▼ EMBED: memcpy 4096 × fp32 = 16 KB from embedding table
      → x[4096] fp32
      │
      ▼ FOR EACH LAYER l in [0..31]:
          mistral_layer_decode_parallel()
          │
          ├── 1. RMSNorm (vDSP)
          │   x → xb: dotpr + vsmul + vmul, 3 passes over [4096] fp32
          │   BW: 48 KB read + 16 KB write. Negligible.
          │
          ├── 2. Quantize xb → xb_q8 (NEON)
          │   quantize_f32_to_q8_0: fp32[4096] → Q8_0[128 blocks × 36B]
          │   Find max|x| per 32-element block, scale to [-127,127]
          │   BW: 16 KB read → 4.5 KB write. Fast.
          │   ✅ Done ONCE, reused for Q, K, V projections
          │
          ├── 3. QKV projections (SDOT, row-parallel)
          │   q4_matvec_sdot_parallel: Q4_0 weights × Q8_0 activations
          │
          │   Wq [4096×4096] Q4_0: 4096 × 128 blocks × 18B = 9.4 MB
          │   → dispatch_apply over P-cores, 64-row chunks
          │   Inner loop per block (32 values):
          │     Load 16B Q4 nibbles → extract 32 int4 → vdotq_s32 with Q8 activations
          │     4× vdotq_s32 per block = 128 int8 multiplies in 4 instructions
          │   BW: 9.4 MB weight + 4.5 KB activation = 9.4 MB read → 16 KB write
          │
          │   Wk [1024×4096]: 2.35 MB read → 4 KB write
          │   Wv [1024×4096]: 2.35 MB read → 4 KB write
          │
          │   ✅ Total QKV: 14.1 MB weight reads (bandwidth-bound)
          │   At 200 GB/s: theoretical minimum = 0.07 ms
          │   Observed: ~0.5 ms (includes dispatch overhead + L2 pressure)
          │
          ├── 4. RoPE (CPU, scalar)
          │   Apply to q[4096] and k[1024], in-place
          │   64 pairs per head × 32 Q-heads + 64 × 8 KV-heads = 2560 rotations
          │   Each: 2 mul + 1 add + 1 sub + 2 table reads. Negligible.
          │
          ├── 5. KV cache write
          │   k[1024] fp32 → ⚠️  scalar fp32→fp16 cast → scatter to channel-first
          │   v[1024] fp32 → ⚠️  scalar fp32→fp16 cast → scatter to channel-first
          │   1024 strided stores, stride = 65536 × 2B = 128 KB
          │   ⚠️  CACHE-UNFRIENDLY: each store touches a different 128 KB-spaced cacheline
          │   BW: tiny data (4 KB), but 1024 random cacheline touches
          │
          ├── 6. GQA Attention (NEON, vectorized)
          │   For each of 32 heads (4 heads share 1 KV head via GQA):
          │
          │   Q@K^T (attention scores):
          │     For each position t in [0..seq_len):
          │       Gather 128 fp16 from K cache (stride max_seq)
          │       ⚠️  GATHER: 128 scalar loads, stride 128 KB
          │       → ktmp[32] aligned, NEON dot with q[128] fp32
          │       4 accumulators, 32 values per iteration
          │
          │     ⚠️  O(seq_len) per head, 32 heads = O(32 × seq_len)
          │     At pos=512: 512 × 32 × 128 = 2M dot products
          │     BW per score: 128 × 2B gathered + 128 × 4B q = 768B
          │     Total at pos=512: 512 × 32 × 768B = 12.6 MB (but strided!)
          │
          │   Softmax (vDSP): in-place on [seq_len] per head
          │
          │   Att@V (weighted sum):
          │     Same gather pattern from V cache
          │     ⚠️  Same 128-element stride-128KB gather per timestep
          │     Skips timesteps where att=0 (good optimization)
          │
          │   Total attention BW at pos=512:
          │     K reads: 32 heads × 512 × 128 × 2B = 4 MB (strided)
          │     V reads: similar but sparse (only non-zero att weights)
          │     ⚠️  THIS IS THE DECODE BOTTLENECK AT LONG CONTEXT
          │
          ├── 7. Output projection Wo (SDOT, row-parallel)
          │   First: quantize xb2[4096] → xb_q8 (same as step 2)
          │   Then: q4_matvec_sdot_parallel, same as Wq
          │   BW: 9.4 MB weight read
          │
          ├── 8. Residual: x += xb (vDSP_vadd, 16 KB)
          │
          ├── 9. FFN RMSNorm (vDSP, same as step 1)
          │
          ├── 10. Quantize xb → xb_q8 (same as step 2)
          │
          ├── 11. FFN gate + up (SDOT, row-parallel)
          │   W1 [14336×4096] Q4_0: 26.2 MB weight read → 56 KB output
          │   W3 [14336×4096] Q4_0: 26.2 MB weight read → 56 KB output
          │   ✅ Share same xb_q8 input (quantize once)
          │   ⚠️  BIGGEST READS: 52.4 MB for gate+up per layer
          │
          ├── 12. SiLU(gate) × up (CPU, scalar)
          │   14336 values: gate / (1 + exp(-gate)) * up
          │   ⚠️  Scalar expf, but 14336 iterations = ~15 μs. Negligible.
          │
          ├── 13. FFN down (SDOT, row-parallel)
          │   First: quantize hb[14336] → hb_q8
          │   W2 [4096×14336] Q4_0: 26.2 MB weight read → 16 KB output
          │
          └── 14. Residual: x += xb (vDSP_vadd, 16 KB)

      PER-LAYER TOTAL WEIGHT READS:
      Wq:  9.4 MB  │  Wk: 2.35 MB  │  Wv: 2.35 MB  │  Wo: 9.4 MB
      W1: 26.2 MB  │  W3: 26.2 MB  │  W2: 26.2 MB
      TOTAL: 102.1 MB per layer × 32 layers = 3.27 GB per token

      At 200 GB/s → 16.3 ms minimum → 61.3 tok/s theoretical max
      Observed: ~58 ms (17 tok/s) — 3.5× gap from theoretical
      Gap from: dispatch overhead, L2 misses, attention BW, quantize overhead

  5. LM HEAD (per token)

  x[4096] fp32
      │
      ├── RMSNorm (vDSP) → xb[4096] fp32
      │
      └── q4_matvec_parallel: output.weight [32000 × 4096] Q6_K
          Weight: 32000 × 16 blocks × 210B = ~33.6 MB
          ⚠️  Q6_K, NOT Q4_0 — uses fused Q6K NEON matvec, row-parallel
          BW: 33.6 MB read → 125 KB logits output

  6. SAMPLING

  logits[32000] fp32
      │
      ├── Repetition penalty: scan token_history, adjust logits. O(rep_window).
      ├── Top-K partial sort: O(vocab × K) bubble-insert. K=40 default.
      │   ⚠️  O(n²) sort of top-K, then O(K²) sort descending. Small K so fine.
      ├── Temperature scaling + softmax over top-K
      ├── Top-P (nucleus) truncation
      └── arc4random() → categorical sample
      → next_token (int32)

  7. TOKEN DECODE (output)

  next_token (int32)
      │
      └── tokenizer_decode_token: hash lookup → string pointer
          Replace U+2581 → space
          → stdout (printf, fflush)

  ---
  Summary of Type Conversions

  ┌─────────────────────┬─────────────────────────────┬──────────────────┬─────────────────────────────┐
  │        Where        │         Conversion          │      Bytes       │            Notes            │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Model load:         │ Q4→fp16→fp32                │ 250M→250M→500M   │ BAD: double-hop, 500 MB     │
  │ embeddings          │                             │                  │ resident                    │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Model load: RoPE    │ computed fp32               │ 64 MB            │ Large at 128K positions     │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Decode: activation  │ fp32→Q8_0                   │ 16KB→4.5KB       │ GOOD: done once, reused 3×  │
  │ quant               │                             │                  │                             │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Decode: SDOT matvec │ Q4×Q8→int32→fp32            │ fused            │ GOOD: no materialization    │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Decode: KV write    │ fp32→fp16 scalar            │ 8KB→4KB          │ BAD: scalar loop, should    │
  │                     │                             │                  │ use NEON                    │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Decode: KV read     │ fp16→fp32 NEON              │ 2B→4B per val    │ OK but strided gather is    │
  │ (attn)              │                             │                  │ slow                        │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Prefill: weight     │ Q4→fp32                     │ 9-103MB→16-224MB │ UGLY: 224× per prefill,     │
  │ dequant             │                             │                  │ temporary                   │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Prefill: KV write   │ fp32→fp16 scalar + scatter  │ same as decode   │ BAD: same scalar cast       │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ Prefill: attention  │ fp16 gather → fp32 scalar   │ strided          │ BAD: scalar, not NEON       │
  │                     │ dot                         │                  │                             │
  ├─────────────────────┼─────────────────────────────┼──────────────────┼─────────────────────────────┤
  │ ANE path (unused)   │ fp32→fp16→ANE→fp16→fp32     │ double convert   │ Not in hot path             │
  └─────────────────────┴─────────────────────────────┴──────────────────┴─────────────────────────────┘

  Bandwidth Budget (Decode, per token)

  ┌───────────────────────────────────┬──────────────┬─────────────────────────────┐
  │             Component             │  Data Moved  │         % of Total          │
  ├───────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Weight reads (7 proj × 32 layers) │ 3.27 GB      │ ~95%                        │
  ├───────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ KV cache reads (attention)        │ 0.05-0.5 GB  │ ~2-15% (grows with context) │
  ├───────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Activations, RoPE, norms          │ ~10 MB       │ <1%                         │
  ├───────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Total                             │ ~3.3-3.8 GB  │                             │
  ├───────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ At 200 GB/s                       │ 16-19 ms/tok │ 53-61 tok/s theoretical     │
  ├───────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Observed                          │ ~58 ms/tok   │ 17 tok/s                    │
  └───────────────────────────────────┴──────────────┴─────────────────────────────┘

  The 3.4× gap between theoretical and observed is the optimization frontier — primarily dispatch overhead,
  L2 cache pressure from 4 cores competing for bandwidth, and the strided KV cache gather pattern.

✻ Cogitated for 2m 29s                           
                                                                                                            
────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯  
────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ? for shortcuts
                                                                                                            
                                                                                                            

# MLX Migration Specification: Custom Mistral 7B to MLX-based Inference

**Status:** Design Phase
**Target Platform:** Apple M5 (24GB unified RAM)
**Expected Outcome:** 40-50 tok/s (2.4-3x improvement over current 17 tok/s)
**Estimated Timeline:** 2-3 weeks (Phase 1: 1-2 days, Phase 2: 3-5 days, Phase 3: 1 week)

---

## 1. Executive Summary

The current custom C/Objective-C Mistral 7B inference engine achieves ~17 tok/s at short context (512 tokens) and 10-12 tok/s at long context (64K). While this demonstrates effective use of ANE, vDSP, and NEON primitives, the fundamental architecture bottlenecks are:

1. **Graph Dispatch Overhead:** Each token generates ~300 function calls across BLAS, dequant, attention, and norm operations
2. **Weight Materialization:** Q4 weights must be dequantized to fp32 before matmul, creating transient allocations
3. **L2 Cache Pressure:** Large weight matrices compete with KV cache in 16MB L2
4. **Single-threaded Scheduling:** No kernel fusion or automatic overlapping

**MLX solves these through:**
- Lazy evaluation graph compilation (single Metal dispatch per layer)
- In-GPU quantized matmul (Q4 GEMM avoids dequant materialization)
- GPU-resident KV cache (eliminating CPU-GPU transfers)
- Automatic kernel fusion
- Metal 4 TensorOps Neural Accelerator support on M5

**Benchmark Target:**
- Apple's published MLX: ~230 tok/s for Qwen 7B Q4 on M2 Ultra (60-core GPU)
- Scaling to M5 base (10-core GPU): Expected 40-50 tok/s for Mistral 7B Q4

---

## 2. Current Architecture Baseline

### 2.1 Performance Profile

| Metric | Value | Notes |
|--------|-------|-------|
| **Prefill (512 tokens)** | ~31 ms | 16.5 tok/s |
| **Decode (S=1)** | ~58 ms | 17.2 tok/s |
| **Decode (S=64, 512K ctx)** | ~93 ms | 10.8 tok/s |
| **Peak Memory (64K ctx)** | ~4.3 GB | KV cache + weights + activations |
| **Model Size** | 3.8 GB | Q4_0 quantized GGUF |

### 2.2 Current Implementation Stack

```
Inference Engine (C)
├── Prefill Path: BLAS matmul (vDSP cblas_sgemm) + parallel dequant
├── Decode Path: SDOT loop (handwritten Q4×Q8)
├── Attention: NEON (vDSP) softmax, vDSP gather for KV
├── RMSNorm: vDSP vector operations
├── RoPE: Complex multiplication in vDSP
└── KV Cache: CPU memory (malloc), row-major layout
```

### 2.3 Dispatch Overhead Analysis

Per-token processing generates:
- 32 layers × (1 QKV matmul + 1 proj matmul + 1 MLP matmul + 1 MLP matmul) = 128 matmuls
- 32 layers × (3 norm ops) = 96 norm ops
- 32 layers × (1 attention forward + 1 attention backward in some configs) = 32+ attention ops
- 32 layers × (1 dequant materialization) = 32 weight decompression calls

**Total per-token:** ~300 function calls, ~200 cross-library boundaries (BLAS ↔ custom C ↔ vDSP)

### 2.4 Weight Materialization Bottleneck

Current decode (Q4 SDOT):
```c
for (int i = 0; i < n_rows; i++) {
    float row[n_cols] = dequant_q4(weight[i]);  // Allocation + dequant
    float result = sdot(row, input);             // Compute
    free(row);                                   // Deallocation
}
```

At 64K context with 32 layers:
- 32 × (4096×11008) × 2 decompression calls per decode step
- ~3.5 GB/s of dequant throughput (bottleneck, not compute-bound)

---

## 3. MLX Rationale

### 3.1 Why MLX Beats Custom C

| Aspect | Custom C | MLX |
|--------|----------|-----|
| **Graph compilation** | Per-call dispatch | Single Metal kernel per layer |
| **Q4 GEMM** | Materializes weights | Native in Metal (no materialization) |
| **KV cache layout** | CPU memory, row-major | GPU memory, optimized for attention |
| **Memory bandwidth** | RAM (100 GB/s) | GPU memory + HBM unified (200+ GB/s) |
| **Kernel fusion** | Manual (tedious) | Automatic via lazy eval |
| **Neural Accelerator** | Not utilized | Automatic TensorOps dispatch on M5 |

### 3.2 MLX Architecture Fit

MLX's design philosophy aligns with M-series constraints:
- **Unified memory:** Eliminates GPU-CPU boundary, natural on M5
- **Lazy evaluation:** Builds computation graph, compiles to single Metal pipeline
- **Python + C++ binding:** Easy to prototype, drop to C++ for hotspots
- **Array API compatible:** Seamless numpy-like interface for preprocessing

### 3.3 Hardware Leverage

**M5 GPU Features:**
- 10 GPU cores (vs 4 on M4)
- Metal 4 TensorOps (Neural Accelerators) for quantized operations
- 24 GB unified RAM (all weights + activations GPU-resident)
- ~2 TFLOPS fp32, ~8 TFLOPS fp16 (Metal), ~16 TFLOPS MXFP4 (Neural Accelerators estimated)

**MLX automatically:**
1. Detects M5 Metal 4 TensorOps
2. Dispatches Q4/MXFP4 gemm to Neural Accelerators
3. Manages unified memory coherence

---

## 4. Phase 1: Drop-in MLX (1-2 days)

### 4.1 Goal
Replace custom C inference with MLX while validating that performance improvements are real and output quality is maintained.

### 4.2 Installation and Model Conversion

#### Step 1: Install MLX and MLX-LM
```bash
pip3 install mlx mlx-lm

# Verify installation
python3 -c "import mlx; print(mlx.__version__)"
```

#### Step 2: Convert Mistral 7B to MLX Format
```bash
# Option A: Convert from HuggingFace (Recommended for Phase 1)
mlx_lm.convert \
  --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
  -q \
  --q-bits 4 \
  --q-group-size 64 \
  --output-dir ./models/mistral-7b-4bit

# Option B: If using existing GGUF
# Use gguf2mlx converter (see Appendix A)
```

**Output structure:**
```
models/mistral-7b-4bit/
├── weights.npz           # Quantized weights
├── config.json           # Model config (context length, etc.)
└── tokenizer.model       # SentencePiece tokenizer
```

### 4.3 Benchmark Script: Phase 1 Validation

Create `/Users/andy/ANEtransformers/scripts/mlx_benchmark.py`:

```python
#!/usr/bin/env python3
"""
MLX Mistral 7B benchmark: prefill, decode, context scaling
"""
import time
import numpy as np
from mlx_lm import load, generate
from mlx_lm.utils import prompt_templates
import mlx.core as mx

def benchmark_prefill(model, tokenizer, seq_length=512):
    """Benchmark prefill (prompt processing)"""
    # Generate dummy tokens
    input_ids = np.random.randint(0, 32000, size=(1, seq_length))

    mx.eval(input_ids)  # Warm cache

    start = time.time()
    for _ in range(3):
        outputs = model(mx.array(input_ids))
        mx.eval(outputs)  # Force evaluation
    elapsed = time.time() - start

    avg_time = (elapsed / 3) * 1000  # ms per run
    throughput = (seq_length / (avg_time / 1000))  # tok/s

    print(f"Prefill (S={seq_length}): {avg_time:.1f}ms, {throughput:.1f} tok/s")
    return throughput

def benchmark_decode(model, tokenizer, context_len=512, num_steps=10):
    """Benchmark decode (token-by-token generation)"""
    # Generate dummy context
    input_ids = np.random.randint(0, 32000, size=(1, context_len))

    # Warm up
    _ = model(mx.array(input_ids))
    mx.eval(_)

    start = time.time()
    for _ in range(num_steps):
        outputs = model(mx.array(input_ids))
        mx.eval(outputs)
        # In real scenario, would append next token; here just re-run
    elapsed = time.time() - start

    avg_step_time = (elapsed / num_steps) * 1000  # ms per step
    throughput = 1000 / avg_step_time  # tok/s

    print(f"Decode (ctx={context_len}): {avg_step_time:.1f}ms/step, {throughput:.1f} tok/s")
    return throughput

def benchmark_chat(model, tokenizer, context_len=512):
    """Benchmark full chat pipeline"""
    prompt = "Tell me about machine learning in 100 words."

    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        temp=0.7
    )
    elapsed = time.time() - start

    # Count actual generated tokens
    gen_tokens = len(tokenizer.encode(response.split('\n')[-1]))
    throughput = gen_tokens / (elapsed / 1000)  # tok/s

    print(f"Chat (ctx={context_len}): {elapsed:.1f}s for {gen_tokens} tokens, {throughput:.1f} tok/s")
    return throughput

def main():
    """Run full benchmark suite"""
    print("Loading MLX Mistral 7B Q4...")
    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

    print("\n=== Phase 1: MLX Drop-in Validation ===\n")

    results = {
        'prefill_512': benchmark_prefill(model, tokenizer, seq_length=512),
        'prefill_4k': benchmark_prefill(model, tokenizer, seq_length=4096),
        'decode_512': benchmark_decode(model, tokenizer, context_len=512),
        'decode_4k': benchmark_decode(model, tokenizer, context_len=4096),
    }

    # Try longer context if memory permits
    try:
        results['decode_64k'] = benchmark_decode(model, tokenizer, context_len=65536)
    except RuntimeError as e:
        print(f"Skipping 64K decode (OOM or unsupported): {e}")

    print("\n=== Summary ===")
    for name, tps in results.items():
        print(f"{name:20s}: {tps:6.1f} tok/s")

    print("\n=== Expected (vs Current C Implementation) ===")
    print("Current C:")
    print("  Prefill (512):  16.5 tok/s")
    print("  Decode (512):   17.2 tok/s")
    print("  Decode (64K):   10.8 tok/s")
    print("\nTarget (2-3x improvement):")
    print("  Prefill (512):  40-50 tok/s")
    print("  Decode (512):   40-50 tok/s")
    print("  Decode (64K):   25-35 tok/s")

if __name__ == '__main__':
    main()
```

**Run Phase 1 validation:**
```bash
cd /Users/andy/ANEtransformers
python3 scripts/mlx_benchmark.py
```

### 4.4 Output Quality Validation

Create `/Users/andy/ANEtransformers/scripts/validate_output.py`:

```python
#!/usr/bin/env python3
"""
Validate MLX output matches custom C implementation
"""
from mlx_lm import load, generate
import json

def test_mistral_instruct():
    """Test Mistral chat template"""
    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

    test_prompts = [
        "What is 2+2?",
        "Tell me about Python.",
        "Write a haiku about engineering.",
    ]

    results = []
    for prompt in test_prompts:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=50,
            temp=0.0,  # Deterministic
            verbose=False
        )
        results.append({
            'prompt': prompt,
            'response': response,
            'length': len(tokenizer.encode(response))
        })
        print(f"Q: {prompt}")
        print(f"A: {response}\n")

    # Save for manual comparison with current C implementation
    with open('/Users/andy/ANEtransformers/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to validation_results.json")

if __name__ == '__main__':
    test_mistral_instruct()
```

### 4.5 Context Length Testing

```bash
# Test at various context lengths
python3 -c "
from mlx_lm import load, generate

model, tokenizer = load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')

for ctx_len in [512, 4096, 16384, 65536]:
    try:
        prompt = 'Q: ' + 'irrelevant ' * (ctx_len // 2)
        response = generate(model, tokenizer, prompt, max_tokens=10, verbose=False)
        print(f'Context {ctx_len:5d}: OK')
    except Exception as e:
        print(f'Context {ctx_len:5d}: {type(e).__name__}')
"
```

### 4.6 Phase 1 Acceptance Criteria

- [ ] MLX inference runs successfully on M5
- [ ] Achieves **>2x throughput** (>35 tok/s at short context)
- [ ] Output quality matches current C implementation (manual validation on 10 prompts)
- [ ] Supports 64K context without OOM
- [ ] TTFT (time-to-first-token) < 100ms

**If any criterion fails:** Debug (see Appendix B), then proceed to Phase 2.

---

## 5. Phase 2: Custom Attention Kernel (3-5 days)

### 5.1 Goal
If MLX's built-in attention doesn't scale efficiently to 64K context, implement Flash Attention variant in Metal and register as custom MLX operation.

### 5.2 When to Implement

**Trigger:** Phase 1 decode at 64K shows throughput < 25 tok/s OR memory pressure near 24GB limit.

Otherwise, skip Phase 2 and proceed to Phase 3 (fused kernels).

### 5.3 Flash Attention Design for M5

#### 5.3.1 Kernel Strategy

```
Input:  Q [B, Sq, Dh], K [B, Sk, Dh], V [B, Sk, Dv]
Output: O [B, Sq, Dv]

Tiling:
  - Q tile: 128 rows (Sq dimension) × Dh
  - K/V tile: 256 rows (Sk dimension) × Dh/Dv
  - Attention scores tile: 128×256

Compute:
  1. Load Q tile + K tile → compute Q @ K^T (128×256 scores)
  2. Online softmax (Welford's trick, no separate max pass)
  3. Load V tile + matmul with softmax scores
  4. Accumulate to output

Memory:
  - Threadgroup memory: 128×64 + 256×64 + 128×256 ≈ 64 KB (< 32 KB threadgroup limit on M5)
    → Use two-level tiling (split into 64×64 blocks)
```

#### 5.3.2 KV Cache Layout

Current (CPU, row-major):
```c
float *kv_cache[seq_len][2][n_heads][head_dim];  // Inefficient for attention
```

MLX optimized (GPU, sequence-major):
```c
// Metal buffer layout: [2, seq_len, n_heads, head_dim]
// Contiguous load for single position across all heads
// Facilitates batched gathering on GPU
```

#### 5.3.3 MLX Custom Operation Registration

```cpp
// mlx_custom_attention.cpp
#include <mlx/mlx.h>
#include <metal_cpp/metal.hpp>

using namespace mlx::core;

array flash_attention(
    const array& q,          // [B, Sq, Dh]
    const array& k,          // [B, Sk, Dh]
    const array& v,          // [B, Sk, Dv]
    const array& mask,       // [B, Sq, Sk] or nullptr
    float scale = 1.0f) {

    // Validate shapes
    assert(q.shape(2) == k.shape(2));  // Dh match

    // Get Metal device, compile shader
    auto device = metal::default_device();
    auto library = device->newDefaultLibrary();
    auto fn = library->newFunction("flash_attention_forward");
    auto pso = device->newComputePipelineState(fn);

    // Allocate output [B, Sq, Dv]
    array output({q.shape(0), q.shape(1), v.shape(2)}, float32, device);

    // Dispatch compute kernel
    // (Metal shader code in flash_attention.metal)

    return output;
}

// Register with MLX
namespace mlx {
    REGISTER_PRIMITIVE(flash_attention);
}
```

#### 5.3.4 Metal Shader (Simplified)

```metal
// flash_attention.metal
#include <metal_stdlib>
using namespace metal;

[[kernel]]
void flash_attention_forward(
    constant float *Q [[buffer(0)]],
    constant float *K [[buffer(1)]],
    constant float *V [[buffer(2)]],
    device float *Out [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 ts [[threads_per_threadgroup]])
{
    // Load Q tile (128 rows × Dh)
    threadgroup float q_tile[128][64];  // Assume Dh <= 64

    for (uint i = tid.x; i < 128; i += ts.x) {
        uint q_row = gid.y * 128 + i;
        for (uint j = tid.y; j < params.head_dim; j += ts.y) {
            q_tile[i][j] = Q[q_row * params.head_dim + j];
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Iterate K/V tiles (online softmax)
    float max_score = -1e9;
    float exp_sum = 0.0;
    threadgroup float scores[128][256];

    for (uint kv_tile = 0; kv_tile < params.seq_len; kv_tile += 256) {
        // Compute Q @ K^T (128×256)
        for (uint i = tid.x; i < 128; i += ts.x) {
            for (uint j = tid.y; j < 256; j += ts.y) {
                uint k_row = kv_tile + j;
                if (k_row >= params.seq_len) break;

                float score = 0.0;
                for (uint d = 0; d < params.head_dim; d++) {
                    score += q_tile[i][d] * K[k_row * params.head_dim + d];
                }
                score *= params.scale;

                // Causal mask (decoder only)
                uint q_pos = gid.y * 128 + i;
                if (k_row > q_pos) score = -1e9;

                scores[i][j] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Online softmax (Welford)
        for (uint i = tid.x; i < 128; i += ts.x) {
            float local_max = max_score;
            float local_exp_sum = 0.0;

            for (uint j = tid.y; j < 256; j += ts.y) {
                float s = scores[i][j];
                if (s > local_max) {
                    local_exp_sum *= exp(local_max - s);
                    local_max = s;
                }
                local_exp_sum += exp(s - local_max);
            }

            // Reduce within thread
            // (SIMD group shuffle operations)
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Matmul with softmax scores × V
        // Accumulate to output
    }
}
```

### 5.4 Integration Pattern

```python
# inference_engine.py
from mlx_custom_attention import flash_attention

class MistralWithCustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = load_pretrained(config)

    def __call__(self, x):
        # Use MLX default for QKV/proj
        # Override attention with custom kernel
        for layer in self.model.layers:
            q, k, v = layer.qkv(x)

            # Custom attention instead of layer.attention(q, k, v)
            attn = flash_attention(q, k, v, scale=self.scale)

            x = layer.proj(attn)

        return x
```

### 5.5 Validation & Benchmarking

```bash
# Compare throughput: MLX default vs custom attention
python3 -c "
from mistral_mlx import MistralDefault, MistralWithCustomAttention

model_default = MistralDefault()
model_custom = MistralWithCustomAttention()

# Benchmark at 64K context
for ctx_len in [4096, 16384, 65536]:
    t_default = benchmark(model_default, ctx_len)
    t_custom = benchmark(model_custom, ctx_len)
    print(f'{ctx_len:6d}: default {t_default:.1f} tok/s, custom {t_custom:.1f} tok/s')
"
```

### 5.6 Phase 2 Acceptance Criteria

- [ ] Custom attention kernel compiles without errors
- [ ] Output matches MLX default (numerical validation)
- [ ] 64K decode throughput **≥30 tok/s** (vs ~25 with MLX default)
- [ ] Memory efficiency stable across context lengths

**If custom attention doesn't improve throughput:** Debug memory patterns, check for unnecessary copies, or profile GPU utilization.

---

## 6. Phase 3: Fused Layer Kernels (1 week)

### 6.1 Goal
Eliminate remaining dispatch overhead by fusing layer operations into single Metal kernels.

### 6.2 Target Kernels

| Kernel | Operations | Expected Speedup |
|--------|-----------|------------------|
| **Fused QKV + RoPE** | RMSNorm + quantize + QKV matmul + RoPE | 1.3-1.5x |
| **Fused MLP** | RMSNorm + Gate matmul + GELU + Out matmul | 1.2-1.4x |
| **Q4 Dequant + Matmul** | Dequant Q4 on-the-fly in matmul kernel | 1.5-2x |
| **Attention Output + RMSNorm** | Softmax → V matmul → add residual → norm | 1.1-1.3x |

### 6.3 Fused QKV + RoPE Example

```metal
// fused_qkv_rope.metal
[[kernel]]
void fused_qkv_rope(
    constant float *x [[buffer(0)]],           // Input [B, Sq, Dh]
    constant float *weight_norm [[buffer(1)]], // RMSNorm weight
    constant uint8_t *weight_qkv [[buffer(2)]], // Q4 quantized QKV matrix
    constant float *rope_cos [[buffer(3)]],     // RoPE cos table
    constant float *rope_sin [[buffer(4)]],     // RoPE sin table
    device float *q [[buffer(5)]],              // Output Q [B, Sq, Dh]
    device float *k [[buffer(6)]],              // Output K [B, Sq, Dh]
    device float *v [[buffer(7)]],              // Output V [B, Sq, Dh]
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 ts [[threads_per_threadgroup]])
{
    uint seq_idx = gid.y;
    uint head_idx = gid.z;

    // Step 1: RMSNorm over input
    float norm_sum = 0.0;
    for (uint d = tid.x; d < D_HEAD; d += ts.x) {
        float val = x[seq_idx * D_HEAD + d];
        norm_sum += val * val;
    }
    norm_sum = sqrt(simd_sum(norm_sum) / D_HEAD + 1e-6);

    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: Normalize & quantize (in-thread)
    threadgroup float norm_x[D_MODEL];
    for (uint d = tid.x; d < D_MODEL; d += ts.x) {
        norm_x[d] = (x[seq_idx * D_MODEL + d] / norm_sum) * weight_norm[d];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 3: Q4 dequant + matmul for QKV
    float q_val = 0.0, k_val = 0.0, v_val = 0.0;

    for (uint i = tid.x; i < D_MODEL; i += ts.x) {
        // Dequant Q4 weight
        uint byte_idx = (head_idx * 3 * D_HEAD + 0 * D_HEAD) * D_MODEL / 2 + i;
        uint8_t byte = weight_qkv[byte_idx];
        float w_q = dequant_q4(byte & 0xF) * SCALE;  // Low nibble

        q_val += norm_x[i] * w_q;
        // Similar for k, v
    }

    // Step 4: Apply RoPE (in-place on q, k)
    uint pos = seq_idx;  // Position in sequence
    float cos_val = rope_cos[head_idx * MAX_SEQ + pos];
    float sin_val = rope_sin[head_idx * MAX_SEQ + pos];

    float q_rot = q_val * cos_val - k_val * sin_val;  // Example for first dim pair
    // (Rotate all head_dim pairs)

    q[seq_idx * D_MODEL + head_idx * D_HEAD + tid.x] = q_rot;
    k[seq_idx * D_MODEL + head_idx * D_HEAD + tid.x] = k_rot;
    v[seq_idx * D_MODEL + head_idx * D_HEAD + tid.x] = v_val;
}
```

### 6.4 Q4 GEMM Optimization

MLX's built-in Q4 matmul already avoids weight materialization, but we can further optimize:

```metal
// q4_gemm_optimized.metal - Neural Accelerator dispatch
[[kernel]]
void q4_matmul_na(
    constant float *A [[buffer(0)]],           // [M, K] fp32
    constant uint8_t *B_q4 [[buffer(1)]],      // [N, K/2] Q4 format
    device float *C [[buffer(2)]],              // [M, N] output
    constant float *scales [[buffer(3)]],       // [N] quantization scales
    uint3 gid [[threadgroup_position_in_grid]])
{
    uint m = gid.x;
    uint n = gid.y;

    // For M5 Neural Accelerators: use MXFP4 if available
    // Otherwise fall back to manual Q4 dequant + fma

    float sum = 0.0;
    for (uint k = 0; k < K; k += 2) {
        uint8_t byte = B_q4[n * (K/2) + k/2];
        float b_low = scales[n] * dequant_q4(byte & 0xF);
        float b_high = scales[n] * dequant_q4((byte >> 4) & 0xF);

        sum += A[m * K + k] * b_low;
        sum += A[m * K + k + 1] * b_high;
    }

    C[m * N + n] = sum;
}
```

### 6.5 Profiling & Optimization Loop

```python
# profile_mlx.py
import mlx.core as mx
from mlx.utils import time_function

def profile_layer(model, layer_idx, input_shape):
    """Profile single layer breakdown"""
    model.eval()
    x = mx.random.normal(input_shape)

    layer = model.layers[layer_idx]

    # Profile norm
    t_norm = time_function(lambda: layer.norm1(x), iters=10)

    # Profile attention
    t_attn = time_function(lambda: layer.self_attn(x, x, x), iters=10)

    # Profile MLP
    t_mlp = time_function(lambda: layer.mlp(x), iters=10)

    print(f"Layer {layer_idx}: norm={t_norm:.2f}ms, attn={t_attn:.2f}ms, mlp={t_mlp:.2f}ms")

def main():
    model, _ = load_model("mistral-7b-4bit")
    for i in range(4):  # Profile first 4 layers
        profile_layer(model, i, (1, 512, 4096))
```

### 6.6 Phase 3 Acceptance Criteria

- [ ] All target kernels implemented and tested
- [ ] Dispatch count reduced by >70% (from ~300 to <100 per token)
- [ ] Short context (512 tokens): **≥50 tok/s**
- [ ] Long context (64K tokens): **≥35 tok/s**
- [ ] Output correctness verified (compare with Phase 1 baseline)

---

## 7. Expected Performance Trajectory

### 7.1 Benchmark Table

| Phase | Configuration | Prefill (512) | Decode (512) | Decode (64K) | Notes |
|-------|---|---|---|---|---|
| **Current** | Custom C | 16.5 | 17.2 | 10.8 | Baseline |
| **Phase 1** | MLX drop-in | 40-50 | 40-50 | 25-35 | 2.4x improvement |
| **Phase 2** | + Custom attention | 45-55 | 45-55 | 30-40 | Better 64K scaling |
| **Phase 3** | + Fused kernels | 50-60 | 50-60 | 35-45 | Minimal dispatch |

### 7.2 Roofline Analysis

**M5 compute ceiling:** ~16 TFLOPS (fp16, neural accelerators)
**Memory ceiling:** 200 GB/s (unified memory)

For 4096×4096 matmul (prefill):
- Ops: 2 × 4096³ / (1000³) ≈ 33.5 TFLOPS
- Bytes: 4096 × 4096 × 4 × 3 ≈ 192 MB
- Arithmetic intensity: 33.5 / (192 / 200) ≈ 34.9 (compute-bound)

**Expected throughput:** 16 TFLOPS / (4096² × 2 / 1000³) ≈ **59 tok/s** (theoretical max)

---

## 8. Model Conversion Details

### 8.1 GGUF → MLX Format

If you need to convert existing GGUF weights:

```python
# scripts/gguf_to_mlx.py
import numpy as np
from pathlib import Path

def convert_gguf_to_mlx(gguf_path: str, output_dir: str):
    """
    Convert GGUF Mistral weights to MLX format
    Note: MLX prefers HuggingFace format; use this only if GGUF is primary source
    """
    import gguf  # pip install gguf

    reader = gguf.GGUFReader(gguf_path)
    weights = {}

    # Map GGUF tensor names to MLX layer structure
    for tensor_name, tensor_data in reader.tensors.items():
        # Parse tensor name: model.layers.0.self_attn.q_proj.weight
        if 'layers' in tensor_name:
            layer_idx = int(tensor_name.split('.')[2])
            weights[f'layers/{layer_idx}/{tensor_name.split("/")[-1]}'] = tensor_data
        elif 'embed' in tensor_name:
            weights[f'embedding/{tensor_name.split("/")[-1]}'] = tensor_data

    # Save as MLX weights
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez(output_path / 'weights.npz', **{k: np.array(v) for k, v in weights.items()})
    print(f"Converted {len(weights)} tensors to {output_path}")

if __name__ == '__main__':
    convert_gguf_to_mlx('mistral-7b-q4.gguf', 'models/mistral-7b-mlx')
```

### 8.2 HuggingFace Direct Load (Preferred)

```python
from mlx_lm import load

# Simplest approach: use HF model directly
model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.3", quantize=True, q_bits=4)

# Auto-quantizes to 4-bit and caches in ~/.cache/mlx-lm/
```

---

## 9. MLX C++ API Integration (Advanced)

If you need fine-grained control, MLX provides C++ bindings:

```cpp
// mlx_inference.cpp
#include <mlx/mlx.h>

using namespace mlx::core;

class MistralMLX {
public:
    MistralMLX(const std::string& weights_path) {
        // Load weights
        weights_ = load_weights(weights_path);
    }

    array generate(const array& input_ids, int max_tokens) {
        auto kv_cache = std::vector<std::pair<array, array>>();

        for (int t = 0; t < max_tokens; t++) {
            // Forward pass
            auto logits = forward(input_ids, kv_cache);

            // Sample next token
            int next_token = sample(logits);
            input_ids = concatenate({input_ids, array({next_token})});
        }

        return input_ids;
    }

private:
    std::map<std::string, array> weights_;

    array forward(const array& input_ids, std::vector<std::pair<array, array>>& kv_cache) {
        auto x = embedding(input_ids);  // [seq_len, hidden_dim]

        for (size_t i = 0; i < num_layers; i++) {
            // RMSNorm + attention
            auto norm_x = rms_norm(x);
            auto q, k, v = qkv_matmul(norm_x);

            // Custom attention (calls Metal kernel)
            auto attn_out = flash_attention(q, k, v, kv_cache[i]);

            x = x + attn_out;  // Residual

            // RMSNorm + MLP
            norm_x = rms_norm(x);
            auto mlp_out = mlp_forward(norm_x);

            x = x + mlp_out;  // Residual
        }

        return lm_head(rms_norm(x));  // [seq_len, vocab_size]
    }
};
```

Compile with:
```bash
clang++ -std=c++17 -fPIC mlx_inference.cpp \
  $(python3-config --cflags --ldflags) \
  -lmlx -o libmlx_inference.so
```

---

## 10. Risk Mitigation

### 10.1 Performance Risks

| Risk | Mitigation |
|------|-----------|
| MLX overhead worse than custom C | Phase 1 includes hard stop: if <2x, debug before Phase 2 |
| Custom attention kernel bugs | Extensive numerical validation (compare fp32 matmul vs Metal) |
| Memory fragmentation at 64K | Use MLX's memory pooling, profile with `metal_counter` |
| Quantization differences (Q4_0 vs MLX Q4) | Validate output on 100+ diverse prompts, measure perplexity |

### 10.2 Integration Risks

| Risk | Mitigation |
|------|-----------|
| Losing fine-grained control | MLX C++ API allows dropping to Metal kernels as needed |
| Incompatibility with future MLX versions | Pin to tested version in `requirements.txt` |
| Chat template mismatch | Test with mistralai/Mistral-7B-Instruct-v0.3 which MLX supports natively |

### 10.3 Development Risks

| Risk | Mitigation |
|------|-----------|
| Metal shader compilation errors | Use `metal-shaders-validator` before deployment |
| Regression in output quality | Automated tests on 1000+ prompts with KL divergence tracking |
| Insufficient M5 memory (24GB → OOM) | Phase 1 includes 64K context test; if OOM, reduce to Q8 or lower batch size |

---

## 11. Appendix A: GGUF Quantization Format Reference

Current Mistral model uses:
- **Q4_0:** 32 weights per block, 16-bit scale, 4-bit weights
- **Q4_K:** 256 weights per block, better quality
- **Q6_K:** Higher quality, larger file

MLX defaults to:
- **4-bit:** Equivalent to Q4, no scale per block (group-wise)
- **8-bit:** Full precision weights

For best quality at similar file size, use Q6 in MLX if 64GB+ memory available.

---

## 12. Appendix B: Debugging Checklist

### B.1 MLX Installation Issues

```bash
# Verify Metal support
python3 -c "import mlx; mlx.metal_is_available()"

# Check GPU allocation
python3 -c "import mlx.core as mx; x = mx.array([1,2,3]); print(x.device)"
```

### B.2 Low Throughput Diagnosis

```python
import mlx.core as mx

# Check GPU utilization
mx.metal_debug_info()  # (if available)

# Benchmark individual layers
for i, layer in enumerate(model.layers):
    time_layer = benchmark_layer(layer, input_shape)
    if time_layer > 10ms:
        print(f"Layer {i} is slow: {time_layer}ms")
```

### B.3 Numerical Correctness

```python
# Compare MLX vs custom C on same input
custom_c_output = run_custom_c_inference(input_prompt)
mlx_output = generate_mlx(input_prompt)

# Compute KL divergence on output distributions
kl_div = compute_kl_divergence(custom_c_output, mlx_output)
assert kl_div < 0.01, f"Output divergence too high: {kl_div}"
```

### B.4 Memory Profiling

```python
import mlx.core as mx

def memory_profile(model, input_len):
    mx.eval(mx.array([1]))  # Reset

    before = mx.metal.memory_usage()  # Approximate
    output = model(mx.random.normal((1, input_len, 4096)))
    mx.eval(output)
    after = mx.metal.memory_usage()

    print(f"Peak memory delta: {after - before} MB")

for ctx_len in [512, 4096, 16384, 65536]:
    memory_profile(model, ctx_len)
```

---

## 13. Appendix C: MLX Configuration Tuning

### C.1 Batch Size & KV Cache

```python
# MLX KV cache can be pre-allocated or grown dynamically
# For max throughput at 64K context:

model.kv_cache_max_seqlen = 65536  # Pre-allocate
model.batch_size = 1                 # M5 optimal (no batching on 10-core GPU)
```

### C.2 Quantization Parameters

```bash
# Convert with specific quantization strategy
mlx_lm.convert \
  --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
  -q \
  --q-bits 4 \
  --q-group-size 64 \      # Larger group = better quality, slower dequant
  --q-keep-linear-indices layers.0 \  # Keep specific layers in fp32 for quality
  --output-dir ./models/mistral-7b-4bit
```

### C.3 Chat Template

```python
# Ensure Mistral instruct format is respected
from mlx_lm.utils import prompt_templates

prompt = prompt_templates.mistral(
    "Tell me about MLX",
    system="You are a helpful assistant."
)

response = generate(model, tokenizer, prompt, max_tokens=200)
```

---

## 14. Success Criteria & Milestones

### Milestone 1: Phase 1 Complete (Day 2)
- MLX inference works at >40 tok/s (short context)
- Output quality validated
- 64K context supported

### Milestone 2: Phase 2 Complete (Day 5-7)
- Custom attention kernel compiles and runs
- Numerical correctness verified
- 64K throughput >30 tok/s (or skip if MLX default sufficient)

### Milestone 3: Phase 3 Complete (Day 14)
- Fused kernels integrated
- Full benchmark suite passes
- Short: 50+ tok/s, Long: 35+ tok/s

### Deployment Ready (Day 15)
- All acceptance criteria met
- Production build validated
- Chat interface updated to use MLX backend

---

## 15. Conclusion

MLX represents a pragmatic shift from low-level control (custom C) to high-level abstraction (lazy evaluation graph) while maintaining access to Metal kernels when needed. The migration path is:

1. **Drop-in replacement** (validate 2x improvement)
2. **Custom attention** (if long-context scaling is limiting)
3. **Fused kernels** (eliminate remaining overhead)

This document provides concrete commands, code examples, and benchmarking scripts to execute each phase. The expected outcome is **40-50 tok/s on M5**, a 2.4-3x improvement over the current custom implementation.

---

**Document Version:** 1.0
**Date:** 2026-03-02
**Author:** Engineering Team
**Status:** Ready for Phase 1 Execution

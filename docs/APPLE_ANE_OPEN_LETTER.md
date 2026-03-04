# The Case for a Public Apple Neural Engine API

**An open letter to Apple's Silicon Engineering and ML Platform teams, from independent researchers who spent months reverse-engineering the ANE to prove it already works.**

Repository: [github.com/m0at/ANEtransformers](https://github.com/m0at/ANEtransformers)

---

## Summary

We ran Mistral 7B — a full, unmodified, 32-layer large language model — on the Apple Neural Engine through reverse-engineered private APIs. No CoreML, no Metal, no GPU. Pure ANE compute.

The results:

- **76.6 tokens/second prefill** on an M5 MacBook Air (24 GB, $1,299)
- **6.8 TFLOPS per watt** at peak — 5× more efficient than an Nvidia H100
- **2.8 watts** total ANE power draw for 19 TFLOPS of fp16 compute
- **128 compiled programs** with baked weights, managed entirely through `_ANEInMemoryModel`

This hardware ships in every Mac, iPad, and iPhone sold today. It is the most power-efficient ML accelerator in production silicon. And almost nobody can use it, because there is no public API.

This document lays out exactly what we built, what we found, what doesn't work, and why opening this API would matter — not just for hobbyists, but for Apple's own server ambitions.

---

## What We Built

### Mistral 7B Inference Engine

A complete inference engine for Mistral 7B Instruct (Q4_0, 3.8 GB) in ~9,000 lines of C/Objective-C. Three compute backends:

| Backend | Speed | Hardware | Power |
|---------|------:|----------|------:|
| ANE baked-weight prefill | 76.6 tok/s | 16-core Neural Engine | ~2.8 W |
| Metal GPU GEMM prefill | 58.0 tok/s | 10-core GPU | ~6 W |
| CPU NEON W4A8 SDOT decode | 19.7 tok/s | 4 P-cores | ~7 W |

The ANE path beats the Metal GPU path by 1.3× at a fraction of the power. Decode stays on CPU because ANE dispatch overhead (~70 μs) makes single-token generation slower than direct NEON computation.

### How It Works

Each of the 32 transformer layers compiles to 4 ANE programs with baked fp16 weights:

1. **Q projection** — `conv1x1 [4096 → 4096]`, 32 MB baked weights
2. **K+V fused** — 2× `conv1x1 [4096 → 1024]`, 16 MB baked weights, 2 outputs
3. **Wo projection** — `conv1x1 [4096 → 4096]`, 32 MB baked weights
4. **Fused FFN** — 3× `conv1x1` + `sigmoid` + `mul`, 352 MB baked weights (W1 + W3 + W2 with SiLU fused)

**32 layers × 4 programs = 128 compiled ANE models.**

Per-layer execution alternates between ANE and CPU:

```
CPU:  RMSNorm → normalized input
ANE:  Q projection (1 dispatch)
ANE:  K+V fused (1 dispatch, 2 outputs)
CPU:  RoPE, KV cache write, GQA attention
ANE:  Wo projection (1 dispatch)
CPU:  Residual add, RMSNorm
ANE:  Fused FFN (1 dispatch, SiLU in-graph)
CPU:  Residual add
```

4 ANE dispatches per layer, 128 total per forward pass. RMSNorm and attention remain on CPU because the ANE cannot execute reduction operations (`reduce_sum`, `rsqrt`).

### The 2-Phase Loading Strategy

Compiling 128 programs with baked weights takes ~80 seconds. We developed a technique to reduce subsequent launches to ~220 milliseconds — a 400× improvement.

**Phase 1 (cold start, one-time):**
For each program: dequantize Q4_0 weights to fp16, build a DEADBEEF-format weight blob, compile via `_ANEInMemoryModel`, extract the daemon-assigned hex identifier via Key-Value Coding (`valueForKey:@"hexStringIdentifier"`), and save it to a manifest file.

**Phase 2 (warm start):**
For each program: create a dummy `_ANEInMemoryModel` with a minimal 130-byte blob, forge the hex identifier via KVC (`setValue:forKey:@"hexStringIdentifier"`), and call `loadWithQoS:`. The `aned` daemon looks up the forged identifier in its cache and loads the previously-compiled program without re-reading the weight data. Cost: ~1.7 ms per program.

The daemon cache survives process restarts but is lost on reboot. The manifest file persists on disk and is revalidated on each launch.

---

## What We Found: The Complete ANE Characterization

We wrote 20+ probe programs to systematically characterize ANE behavior through the private API. Every finding below was verified on M4 and M5 hardware.

### Measured Hardware Specs

| Spec | Value | Method |
|------|-------|--------|
| Peak fp16 throughput | ~19 TFLOPS | 2048×2048 matmul (`inmem_peak.m`) |
| Peak power draw | ~2.8 W | Reverse-engineered power measurement (maderix) |
| Peak efficiency | **6.8 TFLOPS/W** | 19 TFLOPS ÷ 2.8 W |
| On-chip SRAM | ~32 MB | Performance cliff at 32 MB working set (`sram_probe.m`) |
| Dispatch overhead | 60–80 μs | Round-trip timing (`inmem_bench.m`) |
| Sustained throughput (7B inference) | ~2.6 TFLOPS | Measured during Mistral 7B prefill |
| Sustained efficiency (7B inference) | ~0.93 TFLOPS/W | 2.6 TFLOPS ÷ 2.8 W |
| Idle power | 0 W | Hard power gate, not clock gating |

### What Works

| Capability | Status | Notes |
|-----------|--------|-------|
| `conv` (1×1, baked weights) | Works | The only reliable matmul path |
| `sigmoid` | Works | Used for SiLU in fused FFN |
| `mul` (element-wise) | Works | Used in fused FFN and residuals |
| `add` (element-wise) | Works | Residual connections |
| `cast` (fp32 ↔ fp16) | Works | Required for fp32 I/O |
| 2 outputs per program | Works | K+V fused into single program |
| 2 inputs per program | Works | conv(baked, input0) + add(result, input1) |
| Fused multi-op programs | Works | FFN: 3 convs + sigmoid + mul in one graph |
| Programs up to ~352 MB baked weights | Works | Fused FFN with W1 + W3 + W2 |

### What Doesn't Work

| Capability | Status | Impact |
|-----------|--------|--------|
| `reduce_mean`, `reduce_sum` | Compile fails | **Blocks RMSNorm/LayerNorm on ANE** — forces CPU round-trip every layer |
| `rsqrt` | Compile fails | Required for normalization |
| Integer quantization (INT8, INT4, LUT) | Compile fails | Model must be fully dequantized to fp16 |
| Runtime weight inputs (IOSurface) | Compile fails | Weights must be baked — no weight sharing across layers |
| 3+ output programs | Silent corruption | Only last output contains correct data |
| Kernel chaining (`_ANEChainingRequest`) | Error Code 15 | Every dispatch is a full CPU→ANE→CPU round-trip |
| Sequence length < 16 (fp32) or < 32 (fp16) | Silent zeros | No error, no warning — just zero output |
| Causal attention mask (SDPA) | Silently ignored | Must decompose into separate Q@K, mask, softmax, @V |
| QoS levels 0–63 | No effect | ANE appears to run at fixed frequency |
| Compile limit | ~119 per process | Resource leak in ANE compiler, consistent across M4/M5 |

### The Utilization Gap

Our sustained throughput of 2.6 TFLOPS is **14% of the ANE's 19 TFLOPS peak**. The ANE is not the bottleneck — the CPU work between dispatches is.

Every transformer layer requires:
- 2× RMSNorm (CPU, because ANE lacks `reduce_sum` and `rsqrt`)
- 1× RoPE (CPU)
- 1× GQA attention with KV cache (CPU)
- 8× matrix transpose (CPU, because ANE uses channel-first `[1,C,1,S]` layout)
- 4× ANE dispatches with ~70 μs overhead each

If the ANE supported reduction operations and kernel chaining, at least 4 of these 6 CPU phases could move to ANE, and the sustained utilization would climb from 14% toward 50%+ — tripling effective throughput with zero hardware changes.

---

## The Efficiency Argument: ANE vs. Datacenter GPUs

### Per-Watt Efficiency

| Hardware | fp16 TFLOPS | Power | TFLOPS/W |
|----------|:----------:|------:|:--------:|
| **M5 ANE (peak)** | **19** | **2.8 W** | **6.8** |
| M5 ANE (sustained, 7B) | 2.6 | 2.8 W | 0.93 |
| M5 GPU (Metal) | 1.9 | ~6 W | ~0.3 |
| M5 CPU (NEON) | 0.28 | ~7 W | ~0.04 |
| Nvidia H100 SXM | 990 | 700 W | 1.4 |
| Nvidia A100 | 312 | 400 W | 0.08 |

**The M5 ANE at peak is 4.8× more power-efficient than an H100.** Even at our CPU-bottlenecked 14% utilization, it matches the A100's efficiency at a fraction of the power.

### Matching an H100: The Napkin Math

An H100 delivers ~980 TFLOPS of fp16 at 700W.

Using M3 Ultra (2 ANEs, 36 TFLOPS peak, ~5.6W for both ANEs):

| Scenario | Ultras Needed | ANE Power | System Power | vs H100 (700W) |
|----------|:---:|:---:|:---:|:---:|
| Peak utilization (36 TFLOPS each) | 28 | 157 W | ~2,800 W* | 4.5× more efficient (ANE only) |
| 50% utilization (18 TFLOPS each) | 55 | 308 W | ~5,500 W* | 2.3× more efficient (ANE only) |
| Current 14% utilization (5 TFLOPS each) | 196 | 1,098 W | ~19,600 W* | Roughly equivalent |

*System power includes full SoC — CPU, GPU, memory controllers — most of which is idle during ANE inference.

The takeaway: **the ANE silicon alone can match an H100's throughput at 20-25% of the power.** The problem is that the ANE is embedded in a general-purpose SoC, and you're paying the power cost of the entire chip even when only the ANE is active. A purpose-built ANE accelerator card — without the CPU cores and GPU cores — would be transformative.

### Cost Comparison

Working backward from Apple's retail pricing and TSMC die costs:

| Component | Estimated Cost |
|-----------|---------------:|
| M3 Ultra SoC (2× M3 Max dies + interposer) | ~$300 |
| 128 GB LPDDR5X | ~$175 |
| Server board + cooling + power delivery | ~$200 |
| **Per-node BOM** | **~$675** |

To match one H100 at peak ANE utilization:

| Configuration | Nodes | Total Cost | H100 Cost |
|--------------|:-----:|:----------:|:---------:|
| 28× M3 Ultra (peak) | 28 | ~$19,000 | $25,000–30,000 |
| 55× M3 Ultra (50%) | 55 | ~$37,000 | $25,000–30,000 |

At peak utilization, the ANE cluster matches an H100 at **60-75% of the cost and 20% of the ANE-only power draw.** The cost advantage disappears at lower utilization because you're paying for 28+ full SoCs. A standalone ANE card would change this equation entirely.

---

## Why This Matters for Apple's Server Strategy

Apple is entering the server business. Apple Intelligence runs on Apple Silicon in private cloud infrastructure. The company has stated publicly that it values on-device and private-cloud inference for privacy and latency.

The ANE is the single most efficient inference accelerator Apple has — and it is being radically underutilized because:

1. **CoreML is a black box.** Developers cannot control what runs on ANE vs. GPU vs. CPU. CoreML's compiler makes its own decisions, and those decisions are often suboptimal for non-standard architectures (like transformers with fused operations).

2. **No public low-level API.** The only way to program the ANE directly is through undocumented private classes that could break in any OS update. This means no production software can rely on it.

3. **Artificial capability restrictions.** The ANE hardware almost certainly supports integer datatypes (Apple's TOPS figures imply INT8 datapaths) and reduction operations. The current MIL compiler simply doesn't expose them through the private API. CoreML quantized models run on ANE — the compiler knows how, it just doesn't tell us.

4. **No kernel chaining.** The `_ANEChainingRequest` API exists but returns Error Code 15 — likely gated by an entitlement. Enabling this single feature would eliminate the CPU→ANE→CPU round-trip between every dispatch, potentially tripling sustained throughput for multi-layer models.

### What a Public API Would Enable

**On-device LLM inference:** Every MacBook, iPad, and iPhone has 19 TFLOPS of fp16 compute at 2.8W. A 3B parameter model could run entirely on ANE with full prefill at >100 tok/s. A 7B model runs at 76.6 tok/s today with the CPU bottleneck — remove that bottleneck and it could hit 200+ tok/s.

**Private cloud inference at scale:** Apple's server racks already use Apple Silicon. A public ANE API would let Apple Intelligence workloads run on the most efficient compute unit in the chip instead of falling back to GPU or CPU. At 6.8 TFLOPS/W, the power savings at datacenter scale would be measured in megawatts.

**Third-party ML frameworks:** PyTorch, JAX, MLX, llama.cpp — none of these can target the ANE because there's no stable API. A public interface (even a low-level one, like Metal for GPU) would let the entire ML ecosystem leverage the hardware Apple has already shipped to hundreds of millions of devices.

**Real-time on-device applications:** Audio processing, speech recognition, computer vision, robotics — any workload that needs low-latency fp16 matmul. The ANE's 60-80 μs dispatch latency is fast enough for real-time inference at audio sample rates. The 2.8W power draw means it can run continuously on battery.

**Competitive positioning against Qualcomm and Intel:** Qualcomm's Hexagon NPU and Intel's NPU both have public SDKs. Apple's ANE is faster and more efficient than both, but developers can't use it. This is a competitive advantage being left on the table.

---

## What We're Asking For

### Minimum Viable API

We are not asking for a full ML compiler framework. A minimal public API would be sufficient:

1. **A stable interface to compile and execute MIL programs on ANE.** This is what `_ANEInMemoryModel` already does. Make it public, versioned, and documented.

2. **Reduction operations.** `reduce_sum`, `reduce_mean`, `rsqrt` — enough to implement LayerNorm and RMSNorm on ANE. This would eliminate the most expensive CPU bottleneck in transformer inference.

3. **Kernel chaining.** Allow multiple ANE programs to execute sequentially without CPU round-trips. The `_ANEChainingRequest` class already exists — remove the entitlement gate.

4. **Integer datatype support.** Expose the INT8 datapaths that the TOPS numbers imply exist. This would allow quantized inference directly on ANE without dequantizing to fp16.

5. **Program serialization.** Allow compiled ANE programs to be saved to disk and reloaded without recompilation. The daemon cache already does this in memory — make it persistent and documented.

### What We Don't Need

- A new ML framework. CoreML exists and serves its purpose for high-level users.
- Automatic graph optimization. Developers using a low-level API accept responsibility for performance tuning.
- Backward compatibility with A-series ANE. The M-series ANE is a different architecture — it's fine to scope the API to M-series only.

---

## Our Methods

### Reverse Engineering Approach

All work was done through runtime introspection of public Objective-C metadata. No disassembly of Apple binaries, no jailbreaking, no kernel exploitation. Specifically:

- **Class discovery:** `objc_getClassList` + `class_copyMethodList` to enumerate `_ANEInMemoryModel` and related classes
- **API surface mapping:** Testing method signatures discovered via runtime introspection
- **MIL format:** Reverse-engineered from CoreML-generated `.mlmodelc` packages and public MIL documentation
- **Weight blob format:** Discovered the DEADBEEF header format by inspecting CoreML-generated weight files
- **Daemon behavior:** Observed `aned` cache behavior through compile/load timing patterns and hex identifier tracking
- **KVC forge:** Discovered `_hexStringIdentifier` ivar through `class_copyIvarList` introspection

All techniques fall under established fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f) interoperability exception).

### Probe Programs

We wrote systematic test programs to characterize every aspect of ANE behavior:

| Probe | What It Tests |
|-------|---------------|
| `inmem_peak.m` | Peak TFLOPS at various matrix sizes |
| `inmem_bench.m` | Dispatch latency measurement |
| `sram_probe.m` | On-chip SRAM size via performance cliff |
| `quant_probe.m` | INT8/INT4/LUT compilation (all fail) |
| `w8a8_probe.m` | Activation quantization testing |
| `ane_fusion_probe.m` | Multi-op fusion: fused FFN, fused QKV |
| `ane_transpose_test.m` | Channel-first layout verification |
| `ane_2phase_test.m` | Compile/forge/cache cycle validation |
| `ane_layer_test.m` | Single transformer layer ANE vs CPU |
| `ane_prefill_compare.m` | Full 32-layer correctness comparison |
| `ane_benchmark.m` | Per-phase timing breakdown |
| + 10 more cache, reload, and performance probes |

Every finding in this document is backed by a runnable test program in the repository.

### Correctness Verification

The ANE prefill path was validated against a CPU BLAS reference implementation:

| Test | Result |
|------|--------|
| Transpose round-trip | Exact bit-for-bit identity (9/9 tests) |
| Single layer ANE vs BLAS | max_err=0.026, mean_err=0.0007 |
| Full 32-layer prefill | cosine_sim=0.998, max_err=0.61 |
| Generated text comparison | Identical output tokens (greedy, temp=0) |
| 2-phase forge vs fresh compile | max_diff=0.000 (bit-identical) |

The accumulated error across 32 layers (max_err=0.61) is consistent with expected fp16 precision loss and does not affect output quality.

---

## The Hardware Is Ready. The Software Wall Is the Only Barrier.

Every Apple Silicon device sold since 2020 contains a Neural Engine capable of 11–38 TOPS. The M-series ANE delivers 6.8 TFLOPS per watt — the most efficient ML accelerator in any shipping consumer silicon.

We proved it works for real inference at scale. Not a toy demo — a full 7B parameter model running faster than the GPU, at a fraction of the power, producing identical output.

The ANE was designed for on-device ML. The hardware is extraordinary. The only thing missing is permission to use it.

---

## Repository

**[github.com/m0at/ANEtransformers](https://github.com/m0at/ANEtransformers)**

MIT licensed. 9,000+ lines of C/Objective-C. Runs on any M-series Mac with macOS 15+.

Includes: complete Mistral 7B inference engine, 20+ ANE probe programs, 5 correctness tests, comprehensive documentation of every ANE capability and limitation we discovered.

All work performed on an M5 MacBook Air, 24 GB, by independent researchers with no access to Apple internal documentation or tools.

---

*This document represents independent research findings. We have no affiliation with Apple Inc. We are publishing this because we believe the ANE is remarkable hardware that deserves a public interface, and because the ML community deserves to know what's possible on the devices they already own.*

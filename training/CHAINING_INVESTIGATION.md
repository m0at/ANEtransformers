# ANE Multi-Kernel Dispatch: Chaining Investigation

**Machine**: Apple M5, macOS 26.3 (Darwin 25.3.0)
**Date**: 2026-03-02
**Test file**: `test_chaining.m`

---

## Motivation

Each ANE kernel dispatch incurs a CPU round-trip: the host sends a request to the ANE daemon, waits for completion, reads back outputs, then dispatches the next kernel. For a multi-layer network, this per-layer overhead dominates small kernels. We investigated every available mechanism to eliminate these round-trips.

---

## Approaches Tested

### 1. Sequential Dispatch (Baseline)

Two separately compiled 64x64 conv kernels. Conv1 scales by 2x, conv2 by 3x. Between dispatches, the CPU copies conv1's output IOSurface into conv2's input IOSurface via `memcpy`.

```
eval(k1) → IOSurfaceLock → memcpy(k2.in, k1.out) → IOSurfaceUnlock → eval(k2)
```

**Result**: 0.15-0.19 ms/iter. Correct output (6.0). This is the baseline.

### 2. Shared IOSurface (Zero-Copy Between Kernels)

Eliminated the `memcpy` by rebuilding k2's `_ANERequest` to use k1's output IOSurface directly as k2's input. No lock/unlock/copy between dispatches — just `eval(k1)` then `eval(k2)`.

**Result**: **Slower** — 0.20-0.25 ms/iter (0.6-0.95x of baseline). The ANE driver appears to insert a synchronization barrier when the same IOSurface appears as output of one eval and input of the next. The fence overhead exceeds the memcpy savings. Output still correct (6.0).

### 3. `_ANEChainingRequest` (Hardware Chaining API)

The private framework exposes an explicit chaining mechanism. Getting the object types right required significant reverse engineering.

#### Type Discovery Process

The `_ANEChainingRequest` factory method is:
```objc
+chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:
    procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:
```

**Attempt 1**: Passed `_ANEIOSurfaceObject` arrays directly.
Crash: `-[_ANEIOSurfaceObject symbolIndex]: unrecognized selector`
The `validate` method expects objects with `symbolIndex`.

**Attempt 2**: Wrapped IOSurfaces in `_ANEBuffer` objects.
`_ANEBuffer` wraps `_ANEIOSurfaceObject` and adds `symbolIndex` (NSNumber) and `source` (int64: 0=input, 1=output, shown in description as `ANEBufferProducerAgent`).

```objc
// source=0 for input buffers, source=1 for output buffers
id inBuf = [_ANEBuffer bufferWithIOSurfaceObject:aio symbolIndex:@0 source:0];
```

Passed `@[inBuf]` as inputs and `@[@[outBuf1], @[outBuf2]]` (nested arrays) as outputSets.
Object created successfully. But `validate` crashed: `-[__NSSingleObjectArrayI outputBuffer]: unrecognized selector`
The inner arrays get checked for `outputBuffer`, which means outputSets wants `_ANEIOSurfaceOutputSets` objects, not nested arrays.

**Attempt 3**: Used `_ANEIOSurfaceOutputSets` with NULL stats surface.
`objectWithstatsSurRef:outputBuffer:` returned **nil** when statsSurRef is NULL.

**Attempt 4**: Used `_ANEIOSurfaceOutputSets` with a real stats IOSurface.
**Success.** `validate` returned **YES**.

#### Correct Types (Final)

```
inputBuffer:  NSArray<_ANEBuffer>
outputSets:   NSArray<_ANEIOSurfaceOutputSets>
signalEvents: @[] (empty NSArray, NOT nil — nil causes internal crash)
```

Where `_ANEIOSurfaceOutputSets` requires:
```objc
// statsSurRef MUST be non-NULL (returns nil otherwise)
// outputBuffer is NSArray<_ANEBuffer>
id oset = [_ANEIOSurfaceOutputSets objectWithstatsSurRef:realSurface outputBuffer:@[outBuf]];
```

#### Chaining Flow

With validation passing, the intended execution flow is:

1. `[client prepareChainingWithModel:options:chainingReq:qos:error:]` — set up the chain
2. `[client enqueueSetsWithModel:outputSet:options:qos:error:]` — enqueue output buffers (takes `_ANEOutputSetEnqueue` objects)
3. `[client buffersReadyWithModel:inputBuffers:options:qos:error:]` — signal inputs are ready (takes `_ANEInputBuffersReady`)

#### Driver Rejection

`prepareChainingWithModel` **always fails** with:

```
Error Domain=com.apple.appleneuralengine Code=15
"ANEProgramChainingPrepare() Failed: Program chaining prepare error"
```

This is the ANE kernel driver (`ANEProgramChainingPrepare`) rejecting the request. We never reach steps 2 or 3.

#### What Was Tried to Fix the Driver Error

| Variation | Result |
|-----------|--------|
| With loopback (`lbInputSymbolId:@[@0]`, `lbOutputSymbolId:@[@0]`) | Code 15 |
| Without loopback (`@[]`, `@[]`) | Code 15 |
| Single output set | Code 15 |
| Two output sets | Code 15 |
| `setQueueDepth:2` on model before prepare | Code 15 |
| Unload, setQueueDepth:2, reload, then prepare | Code 15 |
| Compile with `@{@"ANEChainingEnabled": @YES}` | Code 15 |
| Compile with `@{@"enableChaining": @YES}` | Code 15 |
| Compile with `@{@"ANEProgramChainingEnabled": @YES}` | Code 15 |
| Compile with `@{@"queueDepth": @2}` | Code 15 |
| Compile with `@{@"OutputSetCount": @2}` | Code 15 |
| Compile with `@{@"ANEFOutputSetCount": @2}` | Code 15 |
| Pass compile options through to prepareChaining options | Code 15 |

None of the compile or load options produced a different result. The model's default `queueDepth` is 127 (maximum). `programHandle` and `intermediateBufferHandle` are set correctly. `procedureInfoForProcedureIndex:0` returns the expected input/output symbol index arrays.

### 4. Real-Time Evaluation Path

`_ANEClient` has a dedicated real-time path:

```objc
[client beginRealTimeTask];
[client mapIOSurfacesWithModel:request:cacheInference:error:];
[client evaluateRealTimeWithModel:options:request:error:];
[client unmapIOSurfacesWithModel:request:];
[client endRealTimeTask];
```

**Results**:
- `beginRealTimeTask` → returns NO (likely requires entitlement)
- `mapIOSurfacesWithModel` → fails with error 0x12 ("Program IOSurfaces map failure")
- `evaluateRealTimeWithModel` → **works anyway** despite the above failures, correct output (2.0)
- **No speedup**: 0.13 ms/eval vs 0.12 ms/eval for normal path (0.87x)

The RT path appears to be functionally equivalent to the normal path when called without proper setup. With entitlements and successful `mapIOSurfaces`, it might provide deterministic latency guarantees rather than throughput improvement.

Also tested: `doEvaluateDirectWithModel:options:request:qos:error:` — works, appears to be the unwrapped internal eval path. Same performance as standard eval.

### 5. Fused MIL Program (Two Convolutions in One Model)

Instead of two separate models, wrote a single MIL program containing both convolutions:

```
x → cast(fp16) → conv(W1) → conv(W2) → cast(fp32) → y
```

Two weight blobs (`w1.bin`, `w2.bin`) referenced from the same MIL text. Compiled, loaded, and evaluated as a single model.

**Results**:

| Metric | Value |
|--------|-------|
| Compile time | 12-32 ms |
| Load time | ~1.7 ms |
| Eval latency | **0.09-0.11 ms/iter** |
| Output | 6.0 (correct) |
| vs Sequential | **1.4-2.1x faster** |
| vs Shared surface | **2.2x faster** |

The ANE compiler sees both convolutions in a single program and can schedule them on the hardware pipeline without any CPU intermediation. Intermediate activations stay in ANE SRAM. This is the practical winner.

---

## Summary Table

| Approach | ms/iter | Speedup | Status |
|----------|---------|---------|--------|
| Sequential (2 dispatches + memcpy) | 0.15-0.19 | 1.0x (baseline) | Works |
| Shared IOSurface (zero-copy) | 0.20-0.25 | 0.6-0.95x (slower) | Works but slower |
| `_ANEChainingRequest` | N/A | N/A | validate=YES, driver rejects |
| Real-time eval | 0.13 | 0.87x | Works, no benefit |
| **Fused MIL** | **0.09-0.11** | **1.4-2.1x** | **Works, clear winner** |

---

## Remaining Suspicions About `_ANEChainingRequest`

The chaining API is fully constructed and validates, but the driver rejects it. Several hypotheses remain:

1. **`.hwx` binary flag**: The compiled ANE program (the `.hwx` blob inside the model) may need a specific internal flag or structure to declare chaining support. The MIL compiler (`coremlc`) may only emit this for certain model topologies or when invoked through a specific CoreML path (e.g., `MLNeuralNetworkEngine` pipeline rather than MIL-direct). We only control the MIL text and compile options dict — neither produced a different `.hwx`.

2. **Multi-procedure models**: The chaining API has `procedureIndex` parameters everywhere. A single MIL program with one `func main` compiles to one procedure (index 0). Chaining may be designed for models with **multiple procedures** (multiple `func` blocks in MIL, or models compiled from `.espresso.net` / NeuralNetwork proto that decompose into subgraphs). We never tested multi-procedure MIL because the syntax for declaring multiple functions with separate entry points is undocumented.

3. **`loadModelNewInstance`**: `_ANEClient` has `loadModelNewInstance:options:modelInstParams:qos:error:` which takes `_ANEModelInstanceParameters`. This might create a model instance specifically configured for chaining (the `procedureArray` parameter). Our attempt crashed because `procedureArray` expects objects with a `weightArray` method, not plain `NSNumber`s. The correct objects are likely `_ANEProcedure` or similar internal types.

4. **`_ANEProgramForEvaluation`**: Has `processInputBuffers:model:options:error:` and `processOutputSet:model:options:error:` — which mirror the chaining flow (`buffersReady` and `enqueueSets`). This might be the correct low-level object for chaining, created from `programWithHandle:intermediateBufferHandle:queueDepth:` rather than going through `_ANEClient`.

5. **Entitlements**: The RT path failures (`beginRealTimeTask`, `mapIOSurfaces`) suggest some APIs require specific entitlements (`com.apple.ane.iokit-user-access` or similar). Chaining may have the same restriction. System processes like `mediaserverd` or `coreml` likely have these entitlements.

6. **Firmware version**: The `fwEnqueueDelay` parameter on `_ANEChainingRequest` references firmware-level timing. Chaining may only work with specific ANE firmware versions or only on certain chip steppings. The M5's ANE is family H16 (same as M4).

7. **SharedEvents for synchronization**: The chaining request has a `signalEvents` array. Multi-model chaining might require `_ANESharedSignalEvent` / `_ANESharedWaitEvent` objects to coordinate execution between chained programs, and the driver rejects without them. These require `IOSurfaceSharedEvent` (Metal shared event) objects we haven't tried constructing.

---

## Practical Conclusion

**Use fused MIL programs.** For any multi-layer ANE workload, concatenate all operations into a single MIL program. The ANE compiler handles internal scheduling, intermediate activations stay in SRAM, and you get 1.4-2.1x over sequential dispatch with zero API complexity.

The `_ANEChainingRequest` API is likely used internally by CoreML for specific model configurations (possibly streaming/real-time audio/video models where the same model runs repeatedly with loopback). Without access to the specific `.hwx` flags or entitlements it requires, fused MIL is the only viable path for third-party code.

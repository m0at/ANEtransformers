# Streaming Token Output

## Overview

Print each token immediately as it's generated instead of accumulating in a buffer. At 250 tok/s, text appears to flow instantaneously.

## Current Behavior

Tokens are generated one at a time but only printed in batches every 10 steps:
```
[step 10/200, seq=16, 4.0 ms/tok, decode/CPU]
```

The generated text is interleaved with status lines, making the output hard to read.

## Proposed Behavior

Two output modes:

### Default: Streaming
```
$ ./gpt2 --prompt "The meaning of life"
The meaning of life is not the same as the meaning of death.█
```
- Each token printed immediately via `fwrite(stdout)` + `fflush(stdout)`
- No status lines during generation
- Final summary after generation completes:
```
---
210 tokens, 4.8 ms/tok avg, 208 tok/s
Prefill: 6 tok in 19.7ms (ANE), Decode: 204 tok in 979ms (CPU)
```

### Verbose: `--verbose` flag
```
$ ./gpt2 --prompt "The meaning of life" --verbose
[prefill] 6 tok, 19.7ms, ANE
[decode]  not → 4.1ms
[decode]  the → 3.9ms
[decode]  same → 4.0ms
...
```
Per-token timing for profiling.

## Implementation

### Token-to-string decoding
The tokenizer already has `gpt2_decode()` which converts token IDs back to text. Call it per-token:

```c
// In decode loop, after getting next_token:
char buf[256];
int len = gpt2_decode(&tokenizer, next_token, buf, sizeof(buf));
fwrite(buf, 1, len, stdout);
fflush(stdout);
```

### Timing accumulation
Track total decode time and print summary at end:
```c
uint64_t decode_start = mach_absolute_time();
// ... decode loop ...
double total_decode_ms = tb_ms(mach_absolute_time() - decode_start);
printf("\n---\n%d tokens, %.1f ms/tok avg, %.0f tok/s\n",
       n_generated, total_decode_ms / n_generated,
       1000.0 * n_generated / total_decode_ms);
```

### Handle whitespace
GPT-2 BPE tokens encode leading spaces as part of the token (e.g., " the" not "the"). The existing `gpt2_decode()` already handles this via the byte-to-unicode mapping, so tokens print with correct spacing naturally.

## Changes
- `gpt2.m`: ~20 lines changed in the generation loop
- Add `--verbose` flag parsing alongside existing `--greedy`, `--tokens`, etc.

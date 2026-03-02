# Top-k / Top-p (Nucleus) Sampling

## Overview

The current sampler is either greedy (argmax) or pure temperature sampling. Both produce poor text quality — greedy is repetitive, pure temperature is incoherent. Top-k and top-p (nucleus) sampling fix this.

## Current State

```c
if (temperature <= 0) {
    // Greedy: always pick highest logit
} else {
    // Temperature: softmax over ALL 50257 tokens, sample proportionally
}
```

Greedy output is a loop:
> "He was the one who had the power to destroy the world. He was the one who had the power to destroy the world. He was the one..."

Temperature=1.0 samples from the full distribution including garbage tokens with tiny probabilities.

## Proposed Sampling Pipeline

```
logits[50257]
  → temperature scaling: logits[i] /= temperature
  → top-k filter: keep only the k highest logits, set rest to -inf
  → top-p filter: sort remaining, keep smallest set summing to >= p
  → softmax over surviving tokens
  → sample from distribution
```

### Parameters
- `--temperature T` (default 0.8): Higher = more random, lower = more deterministic
- `--top-k K` (default 40): Only consider top K tokens. 0 = disabled
- `--top-p P` (default 0.95): Only consider tokens in the top P cumulative probability mass
- `--greedy`: Shortcut for temperature=0 (override all sampling)

### Implementation

```c
static int sample_token(float *logits, int vocab, float temp, int top_k, float top_p) {
    // 1. Temperature
    if (temp <= 0) { /* argmax and return */ }
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < vocab; i++) logits[i] *= inv_temp;

    // 2. Top-k: partial sort to find k-th largest, mask rest
    if (top_k > 0 && top_k < vocab) {
        // Use nth_element-style partial sort or a simple linear scan
        // For k=40 out of 50257, linear scan with a min-heap of size k is fastest
        float threshold = kth_largest(logits, vocab, top_k);
        for (int i = 0; i < vocab; i++)
            if (logits[i] < threshold) logits[i] = -INFINITY;
    }

    // 3. Softmax
    float max_l = -INFINITY;
    for (int i = 0; i < vocab; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0;
    for (int i = 0; i < vocab; i++) {
        logits[i] = (logits[i] == -INFINITY) ? 0 : expf(logits[i] - max_l);
        sum += logits[i];
    }
    for (int i = 0; i < vocab; i++) logits[i] /= sum;

    // 4. Top-p: sort by probability descending, accumulate until >= p
    if (top_p < 1.0f) {
        // Sort indices by probability (or use partial approach)
        // Zero out tokens beyond the nucleus
    }

    // 5. Re-normalize and sample
    // ... standard categorical sampling ...
}
```

### Performance
The sampling function runs once per token. At 50257 vocab:
- Temperature scaling: 50K multiplies = ~0.01ms
- Top-k (k=40): One linear scan with heap = ~0.05ms
- Softmax: 50K exp + sum = ~0.05ms
- Top-p sort: Only over k=40 surviving tokens = negligible

Total: <0.1ms — invisible compared to the 4ms decode.

## Quality Impact

With good sampling parameters (temp=0.8, top_k=40, top_p=0.95), GPT-2 124M produces reasonable text:
> "The future of artificial intelligence is a topic that has been debated for decades. Some researchers believe that we are on the cusp of a breakthrough that could fundamentally change how we interact with technology..."

vs greedy:
> "The future of artificial intelligence is uncertain. We're not sure what the future will look like. We're not sure what the future will look like. We're not sure what the future will look like."

## Changes
- `gpt2.m`: Replace `lm_head_sample()` with `sample_token()`, add CLI flag parsing (~50 lines)

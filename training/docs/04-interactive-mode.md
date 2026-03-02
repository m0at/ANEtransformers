# Interactive Chat Mode

## Overview

Multi-turn interactive session with rolling context window. Type a message, get a response, type another — context accumulates across turns.

## Usage

```
$ ./gpt2 --interactive
GPT-2 124M on ANE | 1024 ctx | 250 tok/s
> Tell me about dragons

Dragons are mythical creatures that appear in folklore across nearly every
culture. In European tradition, they are typically depicted as large,
fire-breathing reptiles with wings...

[47 tokens, 4.2 ms/tok, 238 tok/s]

> What about in Chinese mythology?

In Chinese mythology, dragons are fundamentally different from their Western
counterparts. The Chinese dragon, or "long," is a benevolent creature
associated with water, rainfall, and imperial power...

[62 tokens, 4.8 ms/tok, 208 tok/s]

> ^D
```

## Context Management

GPT-2 has a 1024-token context window. The KV cache holds all previous tokens.

### Token budget
```
[system_prefix] [turn1_user] [turn1_response] [turn2_user] [turn2_response] ...
```

### When context fills up (approaching 1024 tokens):
- **Option A: Sliding window** — Drop oldest turns, keep most recent N tokens of context. Requires re-prefilling the KV cache from the retained tokens.
- **Option B: Hard stop** — Print warning, start fresh session.
- **Recommended: Option A** with re-prefill. When context exceeds 900 tokens, drop oldest turn(s) to get back to ~512 tokens, re-run ANE prefill on retained context.

### Turn format
Use a simple separator to delineate turns:
```
<user message>\n\n---\n\n<model response>\n\n---\n\n<user message>...
```
No special tokens needed — GPT-2 doesn't have chat tokens, but it handles conversational patterns reasonably with simple formatting.

## Implementation

### Main loop
```c
// Interactive mode
char input[4096];
int context_tokens[MAX_SEQ];
int context_len = 0;

while (1) {
    printf("> ");
    if (!fgets(input, sizeof(input), stdin)) break;

    // Tokenize user input
    int user_tokens[512];
    int user_len = gpt2_encode(&tokenizer, input, user_tokens, 512);

    // Append to context
    memcpy(context_tokens + context_len, user_tokens, user_len * sizeof(int));
    context_len += user_len;

    // Check context overflow → sliding window if needed
    if (context_len > 900) {
        // Drop oldest turns, re-prefill
    }

    // Generate until EOS-like pattern or max tokens
    // Print tokens as they stream
    // Append generated tokens to context
}
```

### Stop conditions
GPT-2 doesn't have a dedicated EOS token for chat, so use heuristics:
- Stop after `\n\n` (double newline — paragraph break)
- Stop after `>` at start of line (model generating a fake user prompt)
- Stop after `--max-reply N` tokens (default 200)
- User can press Ctrl+C to interrupt generation (signal handler)

### KV Cache Reuse
The key optimization: when the user types a new message, we don't need to re-run the full context through the model. The KV cache already has all previous tokens computed. We only need to:
1. Run the new user tokens through ANE prefill (fast)
2. Continue decoding from the updated cache

This makes multi-turn nearly free — only the new input tokens cost prefill time.

### Sliding window re-prefill
When we drop old context, we must invalidate the KV cache and re-prefill:
```c
// Clear KV cache
for (int l = 0; l < N_LAYERS; l++) {
    memset(kv[l].k, 0, DIM * MAX_SEQ * sizeof(float));
    memset(kv[l].v, 0, DIM * MAX_SEQ * sizeof(float));
}
// Re-prefill retained context through ANE
// ... same as initial prefill path ...
```

## Changes
- `gpt2.m`: Add `--interactive` flag, readline loop, context management (~100 lines)
- Optional: link `libreadline` for line editing / history (or keep it simple with `fgets`)

---
phase: 14-model-adapter-enhancements
plan: 08
title: Streaming Pattern Filtering
type: execute
subsystem: mlx-server/inference
tags: [streaming, pattern-filtering, tool-calls, reasoning]
dependency-graph:
  requires: ["14-07"]
  provides: ["StreamingProcessor class", "real-time pattern filtering"]
  affects: ["streaming chat completions", "user experience"]
tech-stack:
  added: []
  patterns: ["streaming processor", "partial marker detection", "feed/finalize pattern"]
key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/services/response_processor.py
    - backend/mlx_manager/mlx_server/services/inference.py
    - backend/tests/mlx_server/test_response_processor.py
decisions:
  - key: partial-marker-buffering
    choice: "Buffer incomplete markers until next token determines if pattern or content"
    rationale: "Prevents false positives when '<' appears alone"
  - key: python-tag-pattern
    choice: "Include <|python_tag|>...<|eom_id|> in pattern filtering"
    rationale: "Llama Python-style tool calls also need filtering"
  - key: recursive-after-pattern
    choice: "Recursively call feed() for content after pattern ends"
    rationale: "Handles cases where pattern end and new content in same token"
metrics:
  duration: 44m
  completed: 2026-02-02
---

# Phase 14 Plan 08: Streaming Pattern Filtering Summary

StreamingProcessor that filters tool call and thinking markers during token generation, ensuring users never see raw XML tags.

## What Was Built

### StreamingProcessor Class
Added to `response_processor.py`:

```python
class StreamingProcessor:
    """Streaming-aware processor that filters patterns during generation.

    Usage:
        processor = StreamingProcessor()
        for token in generation:
            output, should_yield = processor.feed(token)
            if should_yield and output:
                yield output
        result = processor.finalize()
    """
```

Key features:
- **Pattern detection**: Detects `<think>`, `<tool_call>`, `<function=`, `<|python_tag|>` starts
- **Buffering**: Buffers tokens until pattern end marker found
- **Partial marker handling**: Handles patterns split across tokens (e.g., `<tool` then `_call>`)
- **Clean yields**: Only yields non-pattern content to client
- **Finalize extraction**: Uses ResponseProcessor for final tool call and reasoning extraction

### Inference Integration
Updated `_stream_chat_generate()` in `inference.py`:

```python
stream_processor = StreamingProcessor()  # Created at start

# In token loop:
filtered_output, should_yield = stream_processor.feed(token_text)
if should_yield and filtered_output:
    yield {"delta": {"content": filtered_output}, ...}

# After generation:
result = stream_processor.finalize()  # Extracts tool_calls, reasoning
```

### Test Coverage
Added 20 new tests covering:
- Basic filtering (think tags, tool calls, function tags)
- Partial marker detection across multiple tokens
- Multiple patterns in single response
- finalize() extraction of reasoning and tool calls
- Edge cases (empty tokens, pattern split many ways)
- Helper methods (get_accumulated_text, get_pending_content)

## Key Behaviors

1. **Before (raw markers visible)**:
   ```
   User sees: "Hello <think>analyzing</think> world"
   ```

2. **After (clean stream)**:
   ```
   User sees: "Hello  world"
   (reasoning extracted: "analyzing")
   ```

3. **Partial marker handling**:
   ```python
   tokens = ["Hello ", "<tool", "_call>", "{...}", "</tool_call>", " Done"]
   # Yields: "Hello ", "Done"
   # Buffers: "<tool_call>{...}</tool_call>" (never yielded)
   ```

## Commits

| Hash | Message |
|------|---------|
| c849855 | feat(14-08): add StreamingProcessor for real-time pattern filtering |
| 5c04a2b | feat(14-08): integrate StreamingProcessor into streaming inference |
| 8eabb5d | test(14-08): add comprehensive StreamingProcessor tests |

## Files Modified

| File | Changes |
|------|---------|
| `response_processor.py` | Added StreamingProcessor class (186 lines) |
| `inference.py` | Integrated StreamingProcessor in _stream_chat_generate |
| `test_response_processor.py` | Added 20 streaming tests |

## Verification

- All 64 response processor tests pass (44 existing + 20 new)
- Linting clean (ruff check)
- Type checking clean (mypy)
- StreamingProcessor correctly filters patterns during generation

## Deviations from Plan

None - plan executed exactly as written.

## Success Criteria Met

1. Tool call markers never visible during streaming - ACHIEVED
2. Thinking tags never visible during streaming - ACHIEVED
3. Reasoning content extracted in streaming mode - ACHIEVED
4. Feature parity between streaming and non-streaming - ACHIEVED
5. All tests pass - ACHIEVED (64/64)

## Next Phase Readiness

Plan 14-09 (Generic OpenAI/Anthropic-compatible providers) can proceed.
The streaming infrastructure now provides clean output for all model types.

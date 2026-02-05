---
status: testing
phase: 15-code-cleanup-integration-tests
source: 15-04-SUMMARY.md, 15-05-SUMMARY.md, 15-06-SUMMARY.md, 15-07-SUMMARY.md
started: 2026-02-03T13:00:00Z
updated: 2026-02-03T13:00:00Z
---

## Current Test

number: 1 (RETEST)
name: Thinking content streams to UI
expected: |
  Using a model with enable_thinking=True (e.g., Qwen3):
  1. Send a message in Chat UI
  2. See thinking/reasoning bubble appear and fill during generation
  3. After thinking completes, see the actual response content OR tool call
  4. Response is NOT empty (was the main bug)
  5. No raw <think> tags visible in UI
awaiting: user response after restarting server

## Tests

### 1. Thinking content streams to UI
expected: Using a model with enable_thinking=True, the thinking bubble appears and fills during generation, then the response content appears. Response is NOT empty.
result: issue
reported: "Multiple issues: (1) <think> tags appear raw in UI twice before thinking content, (2) No message response after thinking - only thinking block shown, (3) Model sees tools and knows how to use them but never makes tool call, (4) mx.metal.device_info deprecation warning in logs"
severity: blocker
fixes_applied: |
  - Nested <think> tag filtering in StreamingProcessor (model outputs <think><think> with tools)
  - Unclosed tool call extraction (model doesn't output </tool_call>)
  - Vision test mock fix (added model_type)
  - GLM-4.7 prompt-embedded thinking (template ends with <think>)
  - GLM-4.7 compact tool call parser (<tool_call>func<param>val</param>)
  - UI ThinkingBubble duration persistence (stored message now includes thinkingDuration)
  - GLM4 tool format changed to Hermes/JSON style (more compatible)
  - Lower temperature (0.3) for tool calls (prevents verbose reasoning hitting token limits)
  - Copy Chat button for easy transcript sharing
  - MLX_MANAGER_LOG_LEVEL environment variable for debug logging
  - Tool message conversion in GLM4 adapter (tokenizer can't handle 'tool' role)
  - Note: mx.metal.device_info deprecation is in mlx-lm library, not our code
retest_round: 4

### 2. Per-model memory metrics
expected: In Servers panel, each loaded model shows its OWN memory usage (size_gb based), not the same divided value across all models.
result: [pending]

### 3. Stop button unloads model
expected: Clicking Stop on a running (non-preloaded) model actually unloads it from memory. The server tile should disappear or show unloaded state.
result: [pending]

### 4. Gemma 3 vision detection
expected: Gemma 3 multimodal models show Vision badge AND load correctly for vision inference (images work). Badge display matches actual model capability.
result: [pending]

### 5. Model downloads start immediately
expected: When initiating a model download, UI shows progress immediately (within 2-3 seconds). Download does NOT hang before showing first progress update.
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0

## Gaps

[none yet]

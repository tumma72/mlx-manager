---
status: blocked
phase: 14-model-adapter-enhancements
source: [14-01-SUMMARY.md, 14-02-SUMMARY.md, 14-03-SUMMARY.md, 14-04-SUMMARY.md, 14-05-SUMMARY.md, 14-06-SUMMARY.md, 14-07-SUMMARY.md, 14-08-SUMMARY.md, 14-09-SUMMARY.md]
started: 2026-02-02T15:30:00Z
updated: 2026-02-02T15:30:00Z
---

## Current Test

[UAT HALTED - Critical infrastructure bugs found]

Multiple blocker-level bugs discovered that prevent meaningful UAT:
1. Database migration missing for api_type/name columns
2. Vision models broken (processor attribute access)
3. Qwen thinking mode broken (enable_thinking parameter)
4. Excessive INFO logging for every token
5. Server gauges show system metrics, not per-model metrics

Need integration tests before resuming UAT.

## Tests

### 1. Tool Call Markers Removed from Content
expected: When a model generates tool calls, raw XML markers like `<tool_call>...</tool_call>` or `<function=name>...</function>` are removed from the content field. Tool calls appear in the separate tool_calls array, not inline in content.
result: pass

### 2. Thinking/Reasoning Tags Filtered from Stream
expected: During streaming, `<think>`, `<thinking>`, `<reasoning>`, or `<reflection>` tags are never visible to the user. The stream shows clean content only, with reasoning available in the final response's reasoning_content field.
result: issue
reported: "Thinking tags are filtered correctly (not visible), but the extracted thinking content is not displayed in the Chat UI thinking bubble. This worked previously with mlx-openai-server."
severity: major

### 3. Streaming Pattern Filtering Works
expected: When streaming a response from a reasoning model (like DeepSeek-R1 or Qwen3-thinking), the raw `<think>...</think>` tags never appear in the streamed output. User sees only the clean response.
result: pass

### 4. Multiple Cloud Providers Can Be Configured
expected: In Settings > Cloud Providers, you can configure multiple providers (e.g., Together AND Groq AND OpenAI). Each provider can have its own API key and optional base URL. All configured providers appear in the list.
result: pass

### 5. Provider Auto-Fill Works
expected: When selecting a provider type (e.g., Together, Groq, Fireworks), the base URL placeholder shows the default URL for that provider. You don't need to manually enter the base URL for known providers.
result: pass

### 6. Custom OpenAI-Compatible Provider
expected: You can configure a custom "OpenAI-compatible" provider with any base URL and API key. This allows connecting to self-hosted endpoints or lesser-known providers.
result: issue
reported: "Can only add 1 custom provider per type (openai_compatible, anthropic_compatible). Need UI to add multiple custom providers with: name, description, API compatibility type (OpenAI/Anthropic), base_url, and optional API key (blank for local providers)."
severity: major

### 7. Structured Output Validation
expected: When response_format specifies a JSON schema, the model's output is validated against that schema. Invalid output returns a 400 error with details about what failed validation.
result: skipped
reason: "UAT halted - critical infrastructure bugs found that block testing"

### 8. Tests Pass with Coverage
expected: Running `make test` passes all backend (1267 tests, 96% coverage) and frontend (981 tests, 99% line coverage) tests. Coverage thresholds are met.
result: skipped
reason: "UAT halted - need integration tests for core functionality first"

## Summary

total: 8
passed: 4
issues: 2
pending: 0
skipped: 2

## Gaps

- truth: "Thinking/reasoning content is extracted and displayed in Chat UI thinking bubble"
  status: failed
  reason: "User reported: Thinking tags are filtered correctly (not visible), but the extracted thinking content is not displayed in the Chat UI thinking bubble. This worked previously with mlx-openai-server."
  severity: major
  test: 2
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "Multiple custom providers can be configured with name, description, API type, base_url, and optional API key"
  status: failed
  reason: "User reported: Can only add 1 custom provider per type (openai_compatible, anthropic_compatible). Need UI to add multiple custom providers with: name, description, API compatibility type (OpenAI/Anthropic), base_url, and optional API key (blank for local providers)."
  severity: major
  test: 6
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

## Critical Infrastructure Bugs (found during UAT)

- truth: "Database has api_type and name columns for cloud_credentials"
  status: failed
  reason: "sqlite3.OperationalError: no such column: cloud_credentials.api_type - Migration was never created for new columns"
  severity: blocker
  test: N/A
  root_cause: "Added columns to SQLModel but no database migration"
  artifacts:
    - path: "backend/mlx_manager/models.py"
      issue: "CloudCredential has api_type, name fields but DB doesn't"
  missing:
    - "Database migration to add api_type and name columns"

- truth: "Vision/multimodal models process images correctly"
  status: failed
  reason: "Gemma vision fails: processor does not have 'chat_template' or 'tokenizer' attribute"
  severity: blocker
  test: N/A
  root_cause: "Vision processor handling broken in embedded server"
  artifacts:
    - path: "backend/mlx_manager/mlx_server/services/vision.py"
      issue: "Processor attribute access fails"

- truth: "Qwen models with thinking mode work correctly"
  status: failed
  reason: "Tokenizer doesn't support enable_thinking: Can only get item pairs from a mapping"
  severity: blocker
  test: N/A
  root_cause: "enable_thinking parameter passed incorrectly to tokenizer"
  artifacts:
    - path: "backend/mlx_manager/mlx_server/models/adapters/qwen.py"
      issue: "apply_chat_template with enable_thinking fails"

- truth: "Streaming logs at appropriate level"
  status: failed
  reason: "Every token logged at INFO level: 'First content starts with: ...' - should be DEBUG"
  severity: minor
  test: N/A
  root_cause: "Debug logging left at INFO level"
  artifacts:
    - path: "backend/mlx_manager/routers/chat.py"
      issue: "logger.info for token content"

- truth: "Server gauges show per-model metrics"
  status: failed
  reason: "All server cards show identical 65% memory, 0% CPU regardless of model - metrics not model-specific"
  severity: major
  test: N/A
  root_cause: "System-level metrics shown instead of per-model pool metrics"
  artifacts:
    - path: "frontend/src/lib/components/servers/ServerCard.svelte"
      issue: "Shows system memory, not model pool memory"

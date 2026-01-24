---
status: diagnosed
phase: 06-bug-fixes-stability
source: 06-01-SUMMARY.md, 06-02-SUMMARY.md, 06-03-SUMMARY.md, 06-04-SUMMARY.md, 06-05-SUMMARY.md, 06-06-SUMMARY.md, 06-07-SUMMARY.md
started: 2026-01-24T12:00:00Z
updated: 2026-01-24T12:30:00Z
---

## Current Test

[testing complete]

## Tests

### 1. No console.log debug output in browser
expected: Open the app in browser, navigate through models/servers/chat pages. Open DevTools Console — no debug log statements visible (no "polling", "startup", "state" type messages).
result: issue
reported: "When starting up a Profile/Server the console fills with Failed to load resource: Could not connect to the server errors for /models endpoint. These are health check polling attempts that should be caught and not appear as errors in the console."
severity: major

### 2. Server CPU gauge shows non-zero values during inference
expected: Start a server with a model loaded. Send a chat message. While the model is generating, check the server tile — CPU gauge should show non-zero percentage reflecting actual usage.
result: pass

### 3. Server memory gauge reflects actual model size
expected: Start a server with a model loaded. The memory gauge on the server tile should show GB-scale values reflecting the model's memory footprint (not just the small parent process).
result: issue
reported: "The memory gauge shows 17307 MB instead of converting to GB. Should auto-convert units: 1024 KB -> 1 MB, 1024 MB -> 1 GB."
severity: minor

### 4. Tool-use badge displayed on capable models
expected: Browse models that support function-calling (e.g., Qwen or models tagged "tool-use"). An amber "Tool Use" badge with wrench icon should appear on their model tile.
result: issue
reported: "No Tool Use badge visible on any models, even though many support tool use."
severity: major

### 5. Profile description uses textarea
expected: Go to profile create/edit form. The "Description" field should be a multi-line textarea (not a single-line input), allowing multiple lines of text.
result: pass

### 6. Profile system prompt field exists
expected: Go to profile create/edit form. There should be a "System Prompt" textarea field with a character counter. Entering text over 2000 chars shows a soft limit warning.
result: pass

### 7. System prompt displayed as pinned message in chat
expected: Set a system prompt on a profile, then open chat with that server. The system prompt should appear at the top of the chat as a grayed-out, italic pinned message.
result: pass

### 8. Hint shown when no system prompt is set
expected: Open chat with a server that has no system prompt configured. A dismissible hint should appear suggesting to set one, with a link to profile settings.
result: pass

### 9. Chat retry on model loading
expected: Start a server and immediately send a chat message before the model is fully loaded. The UI should show "Connecting to model... (attempt X/3)" and automatically retry. After retries exhaust, a "Retry" button should appear.
result: pass

### 10. MCP mock tools available
expected: With a server running, call GET /api/mcp/tools (authenticated). Should return two tools: get_weather and calculate, in OpenAI function-calling format.
result: issue
reported: "Returns 401 Not authenticated even when authenticated. Also the endpoint alone doesn't satisfy the requirement — needs full chat integration: tool toggle switch in chat, tools passed to model via OpenAI API, and visualization of tool call/response flow in chat UI (what LLM sent, what tool returned, LLM's final response)."
severity: major

### 11. Server polling doesn't cause UI flicker
expected: On the servers page with a running server, observe the server tile for 10+ seconds. The tile should remain stable without flickering, jumping, or unnecessary re-renders during polling updates.
result: pass

## Summary

total: 11
passed: 7
issues: 4
pending: 0
skipped: 0

## Gaps

- truth: "No debug/error output in browser console during normal operation"
  status: failed
  reason: "User reported: Health check polling generates browser-level 'Failed to load resource' errors in console when server is starting up"
  severity: major
  test: 1
  root_cause: "StartingTile.svelte:130 uses direct fetch() to http://host:port/v1/models for health checks. Browser logs all failed network requests as errors even though the catch block handles them. Need to suppress browser-level logging."
  artifacts:
    - path: "frontend/src/lib/components/servers/StartingTile.svelte"
      issue: "Direct fetch() to server generates browser console errors on connection refused"
      lines: "129-143"
  missing:
    - "Use AbortController with timeout signal to make failures explicit rather than browser-logged"

- truth: "Memory gauge shows human-readable units (auto-convert KB->MB->GB)"
  status: failed
  reason: "User reported: Shows 17307 MB instead of ~16.9 GB. Should auto-convert at 1024 boundaries."
  severity: minor
  test: 3
  root_cause: "ServerTile.svelte:115-118 and ServerCard.svelte:123-127 hardcode 'MB' unit. Existing formatBytes() utility in format.ts already handles auto-conversion but isn't used here."
  artifacts:
    - path: "frontend/src/lib/components/servers/ServerTile.svelte"
      issue: "Hardcoded MB display: server.memory_mb.toFixed(0) MB"
      lines: "115-118"
    - path: "frontend/src/lib/components/servers/ServerCard.svelte"
      issue: "Hardcoded MB display: server.memory_mb.toFixed(1) MB"
      lines: "123-127"
    - path: "frontend/src/lib/utils/format.ts"
      issue: "formatBytes() utility exists but not used for memory display"
      lines: "6-16"
  missing:
    - "Use formatBytes(server.memory_mb * 1024 * 1024) in both components"

- truth: "Tool-use badge displayed on capable models"
  status: failed
  reason: "User reported: No Tool Use badge visible on any models despite many supporting tool use"
  severity: major
  test: 4
  root_cause: "Backend /api/models/config/{model_id} endpoint never passes HuggingFace tags to detect_tool_use(). Tags are available in search results but the config endpoint doesn't accept or forward them. Detection falls back to config-only check which misses most tool-capable models."
  artifacts:
    - path: "backend/mlx_manager/routers/models.py"
      issue: "get_model_config() calls extract_characteristics without tags parameter"
      lines: "278-307"
    - path: "frontend/src/lib/api/client.ts"
      issue: "getConfig() doesn't accept or send tags"
      lines: "298-305"
    - path: "frontend/src/lib/stores/models.svelte.ts"
      issue: "fetchConfig() receives tags but doesn't pass to API client"
      lines: "148-184"
  missing:
    - "Add tags query parameter to /api/models/config/{model_id} endpoint"
    - "Pass tags from frontend store through API client to backend"
    - "Backend passes tags to detect_tool_use()"

- truth: "MCP mock tools accessible and integrated with chat"
  status: failed
  reason: "User reported: Auth fails (401 when authenticated). Endpoint alone insufficient — needs chat toggle, tools passed to model, and tool call/response visualization in chat UI."
  severity: major
  test: 10
  root_cause: "401 caused by missing frontend MCP API client (no auth headers sent on direct fetch). Full integration gap: chat sends to /api/chat/completions which proxies to mlx-server, but never includes tools array. No UI for tool toggle, tool call display, or tool result visualization."
  artifacts:
    - path: "frontend/src/lib/api/client.ts"
      issue: "No MCP API client methods with getAuthHeaders()"
    - path: "backend/mlx_manager/routers/chat.py"
      issue: "Proxy to mlx-server doesn't include tools in request JSON"
      lines: "67-71"
    - path: "frontend/src/routes/(protected)/chat/+page.svelte"
      issue: "No tool toggle, no tool call parsing, no tool result display"
  missing:
    - "Add MCP client methods to frontend API client with auth headers"
    - "Add tools toggle UI in chat (fetches from /api/mcp/tools)"
    - "Include tools array in chat completions request when enabled"
    - "Backend forwards tools to mlx-server"
    - "Parse tool_calls from streaming response"
    - "Execute tool calls via /api/mcp/execute"
    - "Display tool call/result in chat UI"

- truth: "GLM-4.7-Flash thinking content parsed and hidden from response"
  status: failed
  reason: "User reported: Thinking content mixed into response despite glm4_moe parser config. Full thinking/planning process visible in output instead of being collapsed into thinking panel."
  severity: major
  test: 9
  root_cause: "Parser only recognizes explicit <think>...</think> XML tags. GLM-4 outputs thinking as structured text with headers (Analyze the Request:, Brainstorming, etc.) WITHOUT wrapping in tags. The glm4_moe parser config tells server to look for <think> tags but model doesn't output them. Parser config != model output format."
  artifacts:
    - path: "backend/mlx_manager/routers/chat.py"
      issue: "Fallback parser only matches <think>/<thinking>/<reasoning> tags"
      lines: "119-167"
  missing:
    - "Either instruct model via system prompt to use <think> tags, or accept this is a model behavior limitation and document it"

- truth: "Text file attachments send content to model"
  status: failed
  reason: "User reported: Text file attachments via button/drag-drop don't actually send file content to the model. Models reply they didn't receive any file."
  severity: major
  test: 9
  root_cause: "buildMessageContent() in chat page treats ALL attachments as images: encodes as base64 data URL and wraps in image_url content part. Text files should be read as plain text (FileReader.readAsText) and included as text content parts instead."
  artifacts:
    - path: "frontend/src/routes/(protected)/chat/+page.svelte"
      issue: "buildMessageContent() uses encodeFileAsBase64 + type:'image_url' for all files including text"
      lines: "194-210"
  missing:
    - "Check attachment.type === 'text' and use FileReader.readAsText()"
    - "Include text content as {type:'text', text:'[File: name]\\ncontent'} part"
    - "Keep base64/image_url encoding only for image/video types"

- truth: "Chat input field grows vertically with content"
  status: failed
  reason: "User reported: Input field scrolls horizontally instead of wrapping text and growing one line at a time. Makes it impossible to review longer messages."
  severity: minor
  test: 9
  root_cause: "Chat uses <Input> component (wraps <input type='text'>) with fixed h-10 height. Single-line input can't wrap or grow. Needs <textarea> with auto-resize logic based on scrollHeight."
  artifacts:
    - path: "frontend/src/routes/(protected)/chat/+page.svelte"
      issue: "Uses <Input> component (single-line) instead of auto-growing textarea"
      lines: "708-713"
  missing:
    - "Replace <Input> with <textarea> element"
    - "Add auto-resize: set height=auto then height=scrollHeight on input events"
    - "Set min-height (1 line) and max-height (~150px) with overflow-y-auto"

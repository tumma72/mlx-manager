---
status: diagnosed
phase: 06-bug-fixes-stability
source: 06-08-SUMMARY.md, 06-09-SUMMARY.md, 06-10-SUMMARY.md, 06-11-SUMMARY.md, 06-12-SUMMARY.md, 06-13-SUMMARY.md
started: 2026-01-24T19:00:00Z
updated: 2026-01-24T19:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Memory display uses appropriate units (GB for large values)
expected: Start a server with a model loaded. The server tile's memory gauge should display values in GB when >= 1024 MB (e.g., "16.9 GB" instead of "17307 MB").
result: pass

### 2. Chat textarea auto-grows with content
expected: Open chat with a running server. Type a long message — the input field should wrap text and grow vertically (up to ~150px). Enter submits, Shift+Enter inserts a newline.
result: pass

### 3. Health check polling deferred (no console errors on startup)
expected: Start a server profile. Open DevTools Console. During model loading, you should NOT see "Failed to load resource" errors for /v1/models. Polling starts after a ~5s delay once PID is confirmed.
result: issue
reported: "Still shows Failed to load resource: Could not connect to the server errors in console for /v1/models during startup. Multiple errors appear despite the 5s delay fix."
severity: major

### 4. Tool-use badge appears on capable models
expected: Browse models in the models panel. Models with "tool-use" tag from HuggingFace (e.g., Qwen function-calling models) should show an amber "Tool Use" badge with wrench icon.
result: issue
reported: "No Tool Use badge visible on any models (local or searched). Backend logs show 404 Config not found errors rolling back DB sessions for each model. Browser console shows 404 errors for model config requests."
severity: major

### 5. Text file attachments readable by model
expected: In chat, attach a .txt or .py file via the attachment button. Send a message asking about the file. The model should be able to read and reference the file's contents (not receive gibberish base64).
result: issue
reported: "Works for .txt and .py but fails for .log, .md and other text formats. Filtering by file extension instead of mime-type. Error message only says 'Unsupported file format' without listing supported formats, leaving user with no way to recover except trial and error."
severity: major

### 6. MCP tools toggle in chat
expected: In chat with a running server, there should be a Wrench button in the input bar. Clicking it toggles MCP tools on/off. When enabled, tool definitions are sent to the model.
result: issue
reported: "Tools toggle works and model receives/calls tools correctly, but tool call/result display is broken. Shows as large bold markdown text ('Tool call:' and 'Result:') instead of a proper collapsible UI. Should use a collapsible panel like the Thinking block showing 'Tool Calls: N' with expandable code-formatted details inside."
severity: major

### 7. Tool-use execution loop works end-to-end
expected: With tools enabled in chat, ask the model something that requires a tool (e.g., "What's the weather in Paris?" or "Calculate 42 * 17"). The model should call the tool, results should appear inline, and the model should incorporate the result in its final response. Max 3 rounds enforced.
result: pass

## Summary

total: 7
passed: 3
issues: 4
pending: 0
skipped: 0

## Gaps

- truth: "No console errors during server startup health check polling"
  status: failed
  reason: "User reported: Still shows Failed to load resource: Could not connect to the server errors in console for /v1/models during startup. Multiple errors appear despite the 5s delay fix."
  severity: major
  test: 3
  root_cause: "Browser fetch() API logs network errors at native level before JS promise rejects — cannot be suppressed via try-catch. 5s delay reduces but cannot eliminate errors since model load takes 15-30s."
  artifacts:
    - path: "frontend/src/lib/components/servers/StartingTile.svelte"
      issue: "fetch() to /v1/models generates browser-level console errors on connection refused regardless of catch block"
      lines: "139-155"
  missing:
    - "Switch to backend-mediated health polling: frontend polls backend API which checks server health internally, eliminating browser console errors"
    - "Alternative: Use SSE push from backend when server becomes ready"

- truth: "Tool-use badge displayed on capable models"
  status: failed
  reason: "User reported: No Tool Use badge visible on any models (local or searched). Backend logs show 404 Config not found errors rolling back DB sessions for each model. Browser console shows 404 errors for model config requests."
  severity: major
  test: 4
  root_cause: "Frontend fallback parseCharacteristicsFromName() missing tool-use detection for when backend returns 404. Also HTTPException logged as DB error in database.py. Debug agent applied fix: added tool-use pattern matching to frontend fallback and fixed DB error logging."
  artifacts:
    - path: "frontend/src/lib/stores/models.svelte.ts"
      issue: "parseCharacteristicsFromName() had no tool-use detection from tags"
      lines: "95-130"
    - path: "backend/mlx_manager/database.py"
      issue: "HTTPException logged as database error during session rollback"
      lines: "130-156"
  missing:
    - "Fix already applied by debug agent — verify it works"

- truth: "Text file attachments work for all text mime-types"
  status: failed
  reason: "User reported: Works for .txt and .py but fails for .log, .md and other text formats. Uses extension filtering instead of mime-type detection. Error message unhelpful — says 'Unsupported file format' without listing what IS supported."
  severity: major
  test: 5
  root_cause: "Validation checks file.type.startsWith('text/') plus specific application/ types but misses many text mime types (text/markdown, text/x-log, etc.). Also macOS may report application/octet-stream for unknown extensions. acceptedFormats string includes .log .md etc but validation rejects them."
  artifacts:
    - path: "frontend/src/routes/(protected)/chat/+page.svelte"
      issue: "Incomplete mime-type validation at lines 127-132 doesn't cover all text formats"
      lines: "127-137"
  missing:
    - "Use extension-based detection matching the acceptedFormats list as single source of truth"
    - "Extract text file extensions into a constant"
    - "Improve error message to list supported formats"

- truth: "Tool calls displayed in collapsible panel with code formatting"
  status: failed
  reason: "User reported: Tool call/result rendered as large bold markdown text instead of collapsible UI. Should match Thinking block pattern — 'Tool Calls: N' header, expandable, code-formatted details inside."
  severity: major
  test: 6
  root_cause: "Tool calls concatenated as markdown strings into message.content and rendered by Markdown component with prose CSS, making bold text large/prominent. No dedicated ToolCallBubble component exists."
  artifacts:
    - path: "frontend/src/routes/(protected)/chat/+page.svelte"
      issue: "Lines 413-423 append tool calls as markdown strings into assistantContent"
      lines: "413-423"
    - path: "frontend/src/lib/components/ui/thinking-bubble.svelte"
      issue: "ThinkingBubble shows the desired collapsible pattern to follow"
      lines: "1-56"
  missing:
    - "Create ToolCallBubble component (collapsible panel, tool icon, 'Tool Calls: N' header, code-formatted details)"
    - "Create parseToolCalls() function similar to parseThinking()"
    - "Store tool call data separately from message content string"
    - "Render tool calls via ToolCallBubble instead of inline markdown"

---
phase: 06-bug-fixes-stability
plan: 11
subsystem: api
tags: [chat, mcp, tools, sse, streaming]

# Dependency graph
requires:
  - phase: 06-bug-fixes-stability
    provides: Chat completion streaming infrastructure, MCP tools endpoints
provides:
  - Backend chat proxy forwards tools to mlx-openai-server
  - Backend emits tool_call and tool_calls_done SSE events
  - Frontend ToolCall and ToolDefinition TypeScript interfaces
  - Frontend mcp.listTools() and mcp.executeTool() API methods
affects: [06-13-mcp-chat-integration, chat-ui, tool-use]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Chat proxy forwards tools array to mlx-openai-server when present"
    - "SSE events for tool_call chunks (type:tool_call) and completion (type:tool_calls_done)"
    - "MCP API client with typed tool definitions and execution"

key-files:
  created: []
  modified:
    - backend/mlx_manager/routers/chat.py
    - frontend/src/lib/api/types.ts
    - frontend/src/lib/api/client.ts

key-decisions:
  - "Tools array forwarded verbatim to mlx-openai-server (OpenAI-compatible format)"
  - "tool_choice defaults to 'auto' when tools are present"
  - "Tool call chunks emitted as individual SSE events for streaming UX"
  - "finish_reason 'tool_calls' triggers tool_calls_done event for frontend state management"

patterns-established:
  - "ChatRequest extended with optional tools/tool_choice fields matching OpenAI API"
  - "SSE event types: tool_call (chunks), tool_calls_done (completion signal)"
  - "MCP client follows existing API pattern (auth headers, handleResponse error handling)"

# Metrics
duration: 111s
completed: 2026-01-24
---

# Phase 06 Plan 11: MCP Tools Backend Summary

**Backend chat proxy forwards tools to mlx-openai-server and emits tool_call SSE events; frontend provides typed MCP client with listTools and executeTool methods**

## Performance

- **Duration:** 1 min 51s
- **Started:** 2026-01-24T14:47:58Z
- **Completed:** 2026-01-24T14:49:49Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Chat completions endpoint forwards tools array to mlx-openai-server when present
- Streaming response parser emits tool_call chunks and tool_calls_done events via SSE
- Frontend TypeScript interfaces for ToolCall and ToolDefinition
- Frontend MCP API client with listTools and executeTool methods

## Task Commits

Each task was committed atomically:

1. **Task 1: Backend chat proxy tool forwarding** - `5eaecb5` (feat)
2. **Task 2: Frontend types and MCP API client** - `c26c588` (feat)

## Files Created/Modified
- `backend/mlx_manager/routers/chat.py` - Extended ChatRequest with tools/tool_choice, added tool_call parsing and SSE emission
- `frontend/src/lib/api/types.ts` - Added ToolCall and ToolDefinition interfaces
- `frontend/src/lib/api/client.ts` - Added mcp.listTools() and mcp.executeTool() API methods

## Decisions Made

**1. Tools forwarding pattern:**
- ChatRequest accepts optional `tools` (list[dict]) and `tool_choice` (str) fields
- When tools are present, they are forwarded verbatim to mlx-openai-server
- tool_choice defaults to "auto" when tools are provided

**2. SSE event structure:**
- Tool call chunks emitted as `{"type": "tool_call", "tool_call": {index, id, function: {name, arguments}}}`
- Finish reason "tool_calls" triggers `{"type": "tool_calls_done"}` event
- Follows existing SSE pattern (thinking, response, done events)

**3. Frontend API design:**
- MCP client follows existing pattern (getAuthHeaders, handleResponse)
- ToolCall interface mirrors OpenAI's tool_calls structure (id, function.name, function.arguments)
- ToolDefinition uses OpenAI function calling schema (type, function.name, function.parameters)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for plan 06-13 (MCP chat integration UI):**
- Backend can forward tools and parse tool_calls from streaming responses
- Frontend has typed interfaces for tool definitions and tool calls
- Frontend has API client to fetch available tools and execute them
- SSE events provide hooks for UI to display tool calls and execution results

**Blockers:** None - infrastructure layer complete.

---
*Phase: 06-bug-fixes-stability*
*Completed: 2026-01-24*

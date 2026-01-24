---
phase: 06-bug-fixes-stability
plan: 16
subsystem: ui
tags: [svelte, chat, tool-calls, collapsible, mcp]

# Dependency graph
requires:
  - phase: 06-13
    provides: MCP tool execution infrastructure and SSE streaming
provides:
  - ToolCallBubble component for collapsible tool call display
  - Structured ToolCallData storage separate from message content
  - Collapsible UI for tool calls matching ThinkingBubble pattern
affects: [chat-ux, tool-use-workflows]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ToolCallBubble follows ThinkingBubble pattern (collapsible, icon, formatted content)
    - Tool calls stored as structured metadata on messages, not concatenated into content string

key-files:
  created:
    - frontend/src/lib/components/ui/tool-call-bubble.svelte
  modified:
    - frontend/src/lib/components/ui/index.ts
    - frontend/src/routes/(protected)/chat/+page.svelte
    - frontend/src/lib/components/servers/StartingTile.svelte

key-decisions:
  - "Tool calls rendered in collapsible panel with wrench icon, not inline markdown"
  - "ToolCallData interface includes id, name, arguments, result, error fields"
  - "Amber border for tool call panel (distinguishes from thinking's muted border)"
  - "Green background for successful results, red text for errors"
  - "JSON arguments auto-formatted with JSON.stringify pretty-print"

patterns-established:
  - "Tool execution data stored as ToolCallData[] metadata on Message objects"
  - "streamingToolCalls state for real-time display during execution"
  - "ToolCallBubble rendered after ThinkingBubble, before response Markdown"

# Metrics
duration: 4min
completed: 2026-01-24
---

# Phase 06 Plan 16: Tool Call Display UX Summary

**Collapsible tool call display with wrench icon, code-formatted arguments, and color-coded results replacing ugly inline markdown**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-24T16:48:52Z
- **Completed:** 2026-01-24T16:52:52Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- ToolCallBubble component with collapsible UI matching ThinkingBubble pattern
- Structured ToolCallData storage (id, name, arguments, result, error) separate from message content
- Tool calls rendered in clean panel with wrench icon, expandable code-formatted details
- Real-time streaming tool call display during execution
- Multiple tool calls grouped in single panel per response

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ToolCallBubble component** - `e4749ac` (feat)
   - Created tool-call-bubble.svelte with Collapsible UI
   - Export from ui/index.ts
   - Fixed missing INITIAL_HEALTH_DELAY_MS constant (Rule 3 - blocking)

2. **Task 2: Refactor chat to use structured tool call data** - `3ec8b4b` (refactor)
   - Added ToolCallData interface and Message.toolCalls field
   - Imported ToolCallBubble component
   - Replaced markdown concatenation with structured data building
   - Render ToolCallBubble in both stored messages and streaming state

## Files Created/Modified
- `frontend/src/lib/components/ui/tool-call-bubble.svelte` - Collapsible tool call display with wrench icon, code-formatted arguments, color-coded results
- `frontend/src/lib/components/ui/index.ts` - Export ToolCallBubble
- `frontend/src/routes/(protected)/chat/+page.svelte` - Structured tool call handling, ToolCallData storage, ToolCallBubble rendering
- `frontend/src/lib/components/servers/StartingTile.svelte` - Added missing INITIAL_HEALTH_DELAY_MS constant

## Decisions Made

1. **Amber border for tool call panel** - Distinguishes from thinking's muted border, green success badge, and blue info
2. **Green background for results** - `bg-green-50 dark:bg-green-950/30` with green text for successful tool results
3. **JSON argument formatting** - Try JSON.parse → JSON.stringify with pretty-print, fallback to raw string if invalid
4. **Panel collapsed by default** - Unlike ThinkingBubble which auto-expands when streaming, tool calls start collapsed for less visual noise
5. **Tool call limit warning unchanged** - Still concatenates warning text to assistantContent (only tool call display refactored, not limit logic)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added missing INITIAL_HEALTH_DELAY_MS constant**
- **Found during:** Task 1 (npm run check after creating ToolCallBubble)
- **Issue:** StartingTile.svelte referenced INITIAL_HEALTH_DELAY_MS constant that wasn't defined, blocking type checking
- **Fix:** Added `const INITIAL_HEALTH_DELAY_MS = 5_000;` based on STATE.md decision "Health check polling delayed 5s after PID confirmation"
- **Files modified:** frontend/src/lib/components/servers/StartingTile.svelte
- **Verification:** `npm run check` passes with 0 errors
- **Committed in:** e4749ac (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking)
**Impact on plan:** Blocking issue fix necessary to proceed with type checking. No scope creep.

## Issues Encountered
None - plan executed smoothly after auto-fixing blocking type error.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Tool call UX dramatically improved - clean collapsible panels instead of bold markdown
- Structured data storage enables future enhancements (e.g., copy tool result, re-run tool)
- Pattern established for rendering auxiliary content (thinking, tool calls) separate from main response

---
*Phase: 06-bug-fixes-stability*
*Completed: 2026-01-24*

# Phase 6: Bug Fixes & Stability - Context

**Gathered:** 2026-01-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Clean up technical debt (logging, validation, polling, debug statements) and integrate deferred features from Phase 5 (MCP mock, tool-use badge, profile enhancements). Fix runtime bugs with server gauges and failed model handling in chat.

</domain>

<decisions>
## Implementation Decisions

### Profile system prompt
- New textarea field in profile edit form for default system prompt
- Soft character limit with counter (warn at threshold, don't block)
- When server starts with a system prompt, it appears as a pinned/grayed-out first message in chat
- System prompt is locked from profile — to change it, user edits the profile (not editable per-session)
- If no system prompt is set, show a dismissible hint: "No system prompt — set one in profile settings"

### Failed model handling in chat
- The core issue is timing: /v1/models responds immediately but /v1/completions may not be ready until model fully loads in memory
- Use backoff retry approach when system prompt is sent or first user message is sent:
  - Try to send the message
  - On error, retry up to 3 times with linear wait (longer wait = more likely model is loaded)
  - Show progress to user: "Connecting to model... (attempt 2/3)"
  - If all retries fail, show error in chat area telling user to check Server panel for errors
- Error appears as an inline message in the chat area (consistent with existing error patterns)
- Include a "Retry" button in the error message that attempts to resend without leaving chat page
- Chat input remains functional — user can attempt to send messages (triggering the retry flow if model isn't ready)

### Profile model description
- Replace input field with textarea for model description (simple UI fix)

### Claude's Discretion
- Exact retry wait times for backoff (e.g., 1s, 2s, 3s or similar linear progression)
- Character limit threshold for system prompt counter
- Pinned message visual styling (grayed-out, italic, or similar treatment)
- Hint dismissal mechanism (X button, auto-dismiss, or session-persistent)
- MCP mock tool design and UI (not discussed — implement based on codebase patterns)
- Tool-use badge visual treatment (not discussed — follow existing badge patterns from Phase 4)

</decisions>

<specifics>
## Specific Ideas

- System prompt pinned message should feel "present but not intrusive" — the user knows it's there but it doesn't dominate the chat
- Retry progress gives confidence that the system is working, not broken — the wait is expected for large models
- The timing issue is specifically about model loading into memory: server process starts fast but completions endpoint needs the model fully loaded

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-bug-fixes-stability*
*Context gathered: 2026-01-24*

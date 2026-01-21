# Phase 5: Chat Multimodal Support - Context

**Gathered:** 2026-01-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Support image/video attachments for testing multimodal models and display thinking content for reasoning models. Users can attach media, see it sent to the model, and view model thinking in a collapsible format.

</domain>

<decisions>
## Implementation Decisions

### Media attachment UX
- Single media attachment button (unified for images and videos)
- Button positioned inside input field (iMessage style)
- Drag-and-drop enabled for full chat area
- Attached files show as thumbnail row above input
- Hover over thumbnail reveals X button to remove
- File picker filters based on model capabilities (image-only vs image+video)

### Video handling
- Send video directly to model (no frame extraction)
- Maximum video duration: 2 minutes
- If model doesn't support video, file picker filters to image formats only

### Thinking display
- Stream thinking content live as it generates
- Initially collapsed after streaming completes
- Collapsed indicator: subtle text "Thought for Xs" above response
- Expanded styling: muted/grayed text + bordered block + italic
- Click to toggle expanded/collapsed

### Error display
- Errors appear in chat window with red border and red text
- Error has summary line with collapsible details
- Copy icon to copy error to clipboard
- Default state: expanded when error occurs
- Auto-collapses when next message is sent

### Attachment constraints
- No file size limit (user's responsibility)
- Maximum 3 attachments per message

### Claude's Discretion
- Video thumbnail styling (first frame vs icon overlay)
- Exact thinking panel animations
- Drag-drop visual feedback styling
- Specific image/video format lists

</decisions>

<specifics>
## Specific Ideas

- "Thought for Xs" summary like Claude's web UI
- Single unified media button keeps UI clean
- Error messages should be actionable with copy functionality

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-chat-multimodal-support*
*Context gathered: 2026-01-21*

# Phase 4: Model Discovery & Badges - Context

**Gathered:** 2026-01-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Detect model characteristics from config.json and display them as visual badges and technical specs on model tiles. Refactor search UX to toggle between local/online with filter modal for characteristics.

</domain>

<decisions>
## Implementation Decisions

### Badge Design
- Three badge types: multimodal capability, architecture family, quantization level
- Visual style: Icon + text (icon paired with short label)
- Placement: Horizontal row below model name
- Color coding: Semantic colors per type — blue for architecture, green for multimodal, purple for quantization

### Specs Display
- Location: Expandable section within the model tile (click to reveal)
- Content: Show everything available from config.json
- Format: Grouped sections organized by category (Model, Architecture, Capabilities)
- Values: Raw values only — no interpretive context like "large" or "fast"

### Search UX Refactor
- Replace "Downloaded only" checkbox with toggle switch: "My Models" (default) / "HuggingFace"
- Add filter icon button that opens a modal dialog
- Filter modal: Grouped sections for characteristics (multimodal, architecture, quantization, etc.)
- Active filters: Display as removable badge chips below search field with X to clear
- Filter logic: AND (all selected filters must match)
- Filter scope: Same filters work on both local and online modes

### Local vs Remote Models
- Same tile display for both — no visual distinction except download button
- HuggingFace results: Fetch config.json via lazy loading (background fetch after results load, badges appear progressively)
- No deduplication needed — toggle ensures local and online never mixed in same view

### Claude's Discretion
- Exact badge icons per category
- Specific colors within the semantic palette
- Expandable section animation/transition
- Filter modal layout and sizing
- Loading state while configs fetch in background

</decisions>

<specifics>
## Specific Ideas

- Toggle switch should feel like an on/off button that changes its label when toggled
- Filter chips below search should immediately refresh the list when X is clicked
- "My Models" as default because users primarily work with downloaded models

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-model-discovery-badges*
*Context gathered: 2026-01-20*

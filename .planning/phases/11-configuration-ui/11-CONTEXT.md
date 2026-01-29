# Phase 11: Configuration UI - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Visual configuration for model pool, cloud providers, and routing rules. Users configure server behavior through the UI — API keys, memory limits, eviction policies, and model-to-backend routing patterns. Changes apply without server restart.

</domain>

<decisions>
## Implementation Decisions

### API Key Management UX
- Display entered keys with last 4 characters visible (e.g., `****...abc1`) — enough to identify which key
- Auto-test connection on save, plus manual re-test button after for subsequent verification
- Invalid/expired key errors shown as inline error text directly below the API key input field
- One key per provider type (simple: one OpenAI key, one Anthropic key)

### Model Routing Rules UI
- Card-based layout for rules — each rule displayed as a card with pattern, backend, and actions
- Pattern entry via dropdown (exact/prefix/regex) + text field for the pattern value
- Rule priority managed by drag-and-drop reordering — top card = highest priority
- Include "Test rule" feature: inline text input to enter model name, shows which rule matches and which backend it routes to
- Rules with warning badge (unconfigured provider) are disabled at runtime — server skips them

### Model Pool Settings
- Memory limit configurable via toggle for % or GB mode, then slider to set value
- Preload model list managed via searchable dropdown selector (pick from downloaded models)
- Memory usage indicator belongs on running server tiles (with existing CPU/GPU gauges), NOT on config page
- Add gauge to server tiles showing percentage of allocated pool memory in use
- Eviction policy user-selectable under "Advanced Options" section — dropdown with LRU (default), LFU, or TTL-based options

### Provider Configuration Flow
- Providers organized as expandable accordion sections — click to expand/configure
- Green/red status dot on section header indicates connection status at a glance
- Warning badge shown on routing rules that reference unconfigured providers
- Such rules with warnings are skipped by server at runtime (not applied)

### Claude's Discretion
- Whether custom base URL field is visible by default or under Advanced toggle
- Exact card design and spacing for routing rules
- Order of fields within provider sections
- Slider step increments for memory configuration

</decisions>

<specifics>
## Specific Ideas

- Runtime memory usage belongs on server tiles alongside existing CPU/GPU gauges — configuration page is for settings, not live metrics
- Eviction policy and other power-user options should be tucked under "Advanced Options" to keep the default view clean
- Clear separation: configuration time vs runtime information

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-configuration-ui*
*Context gathered: 2026-01-29*

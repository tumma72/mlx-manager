---
status: testing
phase: 04-model-discovery-badges
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md]
started: 2026-01-20T14:30:00Z
updated: 2026-01-20T15:10:00Z
---

## Current Test

number: 1
name: Model Badges Display
expected: |
  On the Models page, model tiles show colored badges: blue badge for architecture family (Llama, Qwen, etc.), green badge for multimodal/vision capability, purple badge for quantization (4-bit, 8-bit, etc.). Badges appear below the model name.
awaiting: re-test after fix

## Tests

### 1. Model Badges Display
expected: On the Models page, model tiles show colored badges: blue badge for architecture family (Llama, Qwen, etc.), green badge for multimodal/vision capability, purple badge for quantization (4-bit, 8-bit, etc.). Badges appear below the model name.
result: issue
reported: "models page doesn't even load at the moment (or is blank), there are tests errors and the coverage dropped again below threshold"
severity: blocker

### 2. Skeleton Loading for Badges
expected: When viewing HuggingFace search results, badges show gray skeleton placeholders while loading, then populate with actual badge data.
result: [pending]

### 3. Expandable Model Specs
expected: On a model tile, clicking "Show specs" reveals technical details: context window, layers, hidden size, vocab size, attention heads, KV cache. Panel slides open smoothly.
result: [pending]

### 4. Toggle Switch (My Models / HuggingFace)
expected: At the top of the Models page, there's a pill-shaped toggle to switch between "My Models" (local) and "HuggingFace" (online search). Clicking toggles the view mode.
result: [pending]

### 5. Filter Button and Modal
expected: A filter icon button shows next to search. Clicking opens a modal with three sections: Architecture (checkboxes), Capabilities (radio: Any/Text-only/Multimodal), Quantization (checkboxes for bit levels).
result: [pending]

### 6. Apply Filters
expected: In the filter modal, selecting some filters and clicking "Apply" filters the model list to only show matching models. "Clear All" resets the filters.
result: [pending]

### 7. Active Filter Chips
expected: When filters are applied, removable chips appear below the search bar showing active filters (e.g., "Llama", "4-bit", "Multimodal"). Clicking X on a chip removes that filter.
result: [pending]

### 8. Filter Count Badge
expected: When filters are active, the filter button shows a small badge with the count of active filters.
result: [pending]

## Summary

total: 8
passed: 0
issues: 1
pending: 7
skipped: 0

## Gaps

- truth: "Models page loads and displays model badges"
  status: fix-applied
  reason: "User reported: models page doesn't even load (or is blank), there are tests errors and the coverage dropped below threshold"
  severity: blocker
  test: 1
  root_cause: "Module exports from FilterModal.svelte not resolving correctly through barrel file. Also type errors in test files causing coverage issues."
  artifacts: [filter-types.ts]
  missing: []
  fix: "Moved FilterState type and related exports from FilterModal.svelte module script to separate filter-types.ts file. Fixed type errors in test files."
  debug_session: ""

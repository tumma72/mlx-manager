---
status: diagnosed
phase: 11-configuration-ui
source: [11-01-SUMMARY.md, 11-02-SUMMARY.md, 11-03-SUMMARY.md, 11-04-SUMMARY.md, 11-05-SUMMARY.md]
started: 2026-01-30T10:00:00Z
updated: 2026-01-30T10:35:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Settings Page Navigation
expected: Click "Settings" in navbar → /settings page loads with three sections visible (Cloud Providers, Model Pool, Routing Rules)
result: pass
note: "Model Pool shows 'Failed to load pool configuration' error - will test separately"

### 2. Provider Accordion Status Dots
expected: Cloud Providers section shows OpenAI and Anthropic as expandable accordion items. Each has a status dot: gray if not configured, green if connected, red if error.
result: pass

### 3. Save API Key with Masking
expected: Expand a provider (OpenAI or Anthropic). Enter an API key in the input field. Characters should display masked as `****...last4` while typing. Click "Save & Test" to save.
result: issue
reported: "No, it shows ****...saved (enter new key to update) instead"
severity: minor

### 4. Test Provider Connection
expected: After saving an API key, the "Test Connection" button should validate the saved credentials. Success shows green status dot and success message. Invalid key shows red dot and error message.
result: pass

### 5. Delete Provider Credentials
expected: When a provider has saved credentials, a Delete button is visible. Clicking it removes the credentials and status dot returns to gray (unconfigured).
result: pass

### 6. Model Pool Memory Slider
expected: Model Pool section has a memory limit slider. Above the slider is a toggle to switch between "%" and "GB" modes. Moving the slider updates the displayed value.
result: pass

### 7. Model Pool Mode Toggle
expected: Toggle between % and GB mode. The value should convert between modes (e.g., 50% of 64GB system = 32 GB). Slider range adjusts based on mode.
result: pass

### 8. Model Pool Eviction Policy
expected: Click "Advanced Options" to expand. Shows an eviction policy dropdown with options: LRU, LFU, TTL.
result: pass

### 9. Model Pool Save
expected: After changing memory limit or eviction policy, click Save. Changes persist when navigating away and returning to /settings.
result: issue
reported: "NO, an error appears with written 'Failed to save pool configuration' and the pulldown menu from where I should be able to choose one of the downloaded models is empty"
severity: major

### 10. Routing Rules List
expected: Routing Rules section shows a list of configured rules (may be empty initially). Each rule card shows: pattern, pattern type badge (exact/prefix/regex), and target backend.
result: issue
reported: "When adding a rule to redirect to OpenAI or Anthropic, after selecting the provider in the pulldown menu, a yellow warning 'Provider not configured' appears even when both providers are configured with green dots"
severity: major

### 11. Create Routing Rule
expected: Click "Add Rule" or similar. A form appears with: Pattern input, Pattern type selector, Backend dropdown (Local/OpenAI/Anthropic), optional fallback backend. Submit creates the rule.
result: pass

### 12. Drag-Drop Rule Reorder
expected: With 2+ rules, drag a rule card by its handle to reorder. The new order persists after refresh.
result: pass

### 13. Rule Test Input
expected: There's a test input to check which rule matches a model name. Enter a model name and see which rule (if any) matches and what backend it routes to.
result: pass

### 14. Delete Routing Rule
expected: Each rule card has a delete button. Clicking it removes the rule from the list.
result: issue
reported: "pass, but we had agreed to use custom modal dialogs with clear explanations for all blocking decisions, so all user confirmation requests need a Svelte modal dialog not the standard browser one"
severity: minor

## Summary

total: 14
passed: 10
issues: 4
pending: 0
skipped: 0

## Gaps

- truth: "API key input shows masked value as ****...last4 while typing"
  status: not_a_bug
  reason: "User reported: No, it shows ****...saved (enter new key to update) instead"
  severity: minor
  test: 3
  root_cause: "NOT A BUG - Implementation works as designed. The ****...saved is the placeholder when field is empty. When typing, password dots show in field and ****...last4 overlay appears on the right side."
  artifacts: []
  missing: []
  debug_session: ".planning/debug/api-key-masking-display.md"

- truth: "Model pool configuration saves successfully and preload dropdown shows downloaded models"
  status: failed
  reason: "User reported: NO, an error appears with written 'Failed to save pool configuration' and the pulldown menu from where I should be able to choose one of the downloaded models is empty"
  severity: major
  test: 9
  root_cause: "localStorage token key mismatch - ModelPoolSettings.svelte uses hardcoded 'mlx_manager_token' but authStore uses 'mlx_auth_token'. Causes 401 errors on API calls."
  artifacts:
    - path: "frontend/src/lib/components/settings/ModelPoolSettings.svelte"
      issue: "Lines 22-53: Local API helpers use wrong token key 'mlx_manager_token'"
  missing:
    - "Replace local API helpers with imports from shared settings client in $lib/api/client"
  debug_session: ".planning/debug/model-pool-save-failure.md"

- truth: "Routing rule form shows no warning when selecting a configured provider"
  status: failed
  reason: "User reported: When adding a rule to redirect to OpenAI or Anthropic, after selecting the provider in the pulldown menu, a yellow warning 'Provider not configured' appears even when both providers are configured with green dots"
  severity: major
  test: 10
  root_cause: "Svelte 5 reactivity loss - RuleForm.svelte line 12 destructures configuredProviders from $props(), which captures initial empty value and never updates when parent data changes."
  artifacts:
    - path: "frontend/src/lib/components/settings/RuleForm.svelte"
      issue: "Line 12: Destructuring from $props() loses reactivity"
  missing:
    - "Access props through props object or use $derived to maintain reactivity"
  debug_session: ".planning/debug/provider-not-configured-warning.md"

- truth: "Delete confirmations use custom Svelte modal dialogs, not browser native confirm()"
  status: failed
  reason: "User reported: we had agreed to use custom modal dialogs with clear explanations for all blocking decisions, so all user confirmation requests need a Svelte modal dialog not the standard browser one"
  severity: minor
  test: 14
  root_cause: "RoutingRulesSection uses native confirm() at line 86. ProviderForm has NO delete confirmation at all (lines 98-113 delete immediately). ConfirmDialog component exists but is not used."
  artifacts:
    - path: "frontend/src/lib/components/settings/RoutingRulesSection.svelte"
      issue: "Line 86: Uses native confirm() instead of ConfirmDialog"
    - path: "frontend/src/lib/components/settings/ProviderForm.svelte"
      issue: "Lines 98-113: Deletes immediately without any confirmation"
  missing:
    - "Import and use ConfirmDialog from $components/ui in both files"
    - "Add dialog state and show confirmation before delete operations"
  debug_session: ".planning/debug/browser-native-confirm-dialog.md"

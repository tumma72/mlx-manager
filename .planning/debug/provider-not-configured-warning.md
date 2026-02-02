---
status: diagnosed
trigger: "Provider Not Configured Warning shows even when provider is configured"
created: 2026-01-30T12:00:00Z
updated: 2026-01-30T12:00:00Z
symptoms_prefilled: true
goal: find_root_cause_only
---

## Current Focus

hypothesis: CONFIRMED - Svelte 5 props destructuring causes reactivity loss
test: Traced data flow from loadData() through derived to RuleForm prop
expecting: Destructured prop not updating when parent data changes
next_action: Return diagnosis

## Symptoms

expected: No warning when selected provider (OpenAI/Anthropic) is configured (shows green dot)
actual: Yellow "Provider not configured" warning appears even with configured providers
errors: None - UI logic issue
reproduction: Create routing rule, select OpenAI or Anthropic as backend
started: Unknown

## Eliminated

## Evidence

- timestamp: 2026-01-30T12:05:00Z
  checked: RuleForm.svelte warning logic (lines 22-24)
  found: showWarning derived uses `configuredProviders.includes(backendType)` - Array method
  implication: Expects configuredProviders to be an array

- timestamp: 2026-01-30T12:05:00Z
  checked: RuleForm.svelte Props interface (lines 7-10)
  found: `configuredProviders: BackendType[]` - typed as array
  implication: Component expects array of BackendType

- timestamp: 2026-01-30T12:06:00Z
  checked: settingsStore.svelte.ts configuredProviders (lines 22-24 and 37-39)
  found: Returns `Set<BackendType>` not array - `$derived.by(() => new Set(...))`
  implication: Store provides Set, but RuleForm expects array

- timestamp: 2026-01-30T12:06:00Z
  checked: RoutingRulesSection.svelte (line 28)
  found: Creates its OWN configuredProviders as array - `credentials.map((c) => c.backend_type)`
  implication: RoutingRulesSection passes array correctly to RuleForm

- timestamp: 2026-01-30T12:07:00Z
  checked: Where settingsStore.configuredProviders is used
  found: settingsStore.configuredProviders returns Set - code uses `configuredProviders.has()` method internally
  implication: If any component tries to use settingsStore.configuredProviders directly with .includes(), it would fail

- timestamp: 2026-01-30T12:10:00Z
  checked: RuleForm.svelte line 12 props destructuring
  found: `let { onSave, configuredProviders }: Props = $props();` - destructured from $props()
  implication: In Svelte 5, destructured props lose reactivity unless using $bindable or accessing via props object

- timestamp: 2026-01-30T12:11:00Z
  checked: RoutingRulesSection.svelte data loading pattern
  found: credentials loaded async in loadData() on mount, configuredProviders derived from credentials
  implication: When RuleForm mounts, credentials is [] so configuredProviders is []. Even after credentials loads, RuleForm's destructured configuredProviders stays as initial empty []

- timestamp: 2026-01-30T12:12:00Z
  checked: showWarning derivation in RuleForm (line 22-24)
  found: `backendType !== 'local' && !configuredProviders.includes(backendType)` - checks against non-reactive variable
  implication: configuredProviders stays [] forever, so includes() always returns false, warning always shows for non-local

## Resolution

root_cause: Svelte 5 reactivity loss through props destructuring. In RuleForm.svelte line 12, `configuredProviders` is destructured from `$props()`. In Svelte 5, destructured props capture the initial value and DO NOT update when the parent's data changes. When RoutingRulesSection mounts, it passes `configuredProviders` (derived from empty `credentials` array). The API loads providers, `credentials` updates, `configuredProviders` in parent updates to contain the providers, but RuleForm's destructured `configuredProviders` remains the initial empty array []. Thus `configuredProviders.includes(backendType)` always returns false for any backendType, triggering the warning.

fix:
verification:
files_changed: []

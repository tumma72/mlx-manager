---
status: diagnosed
trigger: "Model Pool Save Failure - Failed to save pool configuration error, empty preload models dropdown"
created: 2026-01-30T00:00:00Z
updated: 2026-01-30T00:01:00Z
symptoms_prefilled: true
goal: find_root_cause_only
---

## Current Focus

hypothesis: CONFIRMED - localStorage key mismatch causes auth failure
test: Compared component's token key vs authStore's token key
expecting: Keys to match
next_action: Return diagnosis

## Symptoms

expected: Save should persist pool configuration; dropdown should list downloaded models
actual: "Failed to save pool configuration" error on save; preload models dropdown is empty
errors: "Failed to save pool configuration"
reproduction: Click Save button in Model Pool Settings; open preload models dropdown
started: Unknown

## Eliminated

- hypothesis: API endpoint missing or misconfigured
  evidence: Backend router registered in main.py; endpoint responds with 401 (auth required) confirming it exists
  timestamp: 2026-01-30T00:00:30Z

- hypothesis: Database table not created
  evidence: SQLModel.metadata.create_all creates ServerConfig table; endpoint logic handles creating default config if missing
  timestamp: 2026-01-30T00:00:45Z

## Evidence

- timestamp: 2026-01-30T00:00:10Z
  checked: ModelPoolSettings.svelte lines 24-31 (getAuthHeaders function)
  found: Uses localStorage.getItem('mlx_manager_token')
  implication: Component has its own auth header logic instead of using shared client

- timestamp: 2026-01-30T00:00:20Z
  checked: auth.svelte.ts line 10
  found: TOKEN_KEY = "mlx_auth_token"
  implication: authStore stores token with key 'mlx_auth_token'

- timestamp: 2026-01-30T00:00:25Z
  checked: Comparison of token keys
  found: Component uses 'mlx_manager_token' but actual token is stored under 'mlx_auth_token'
  implication: getAuthHeaders() returns null token, API calls fail with 401

- timestamp: 2026-01-30T00:00:35Z
  checked: client.ts lines 51-52
  found: Shared API client correctly uses authStore.token
  implication: The models.listLocal() call should work correctly via shared client

- timestamp: 2026-01-30T00:00:40Z
  checked: ModelPoolSettings.svelte lines 33-53
  found: Component defines its own getPoolConfig() and updatePoolConfig() functions that bypass the shared settings client
  implication: Pool config API calls use wrong auth token

## Resolution

root_cause: |
  ModelPoolSettings.svelte defines its own local API helper functions (getPoolConfig, updatePoolConfig)
  at lines 33-53 that use a hardcoded localStorage key 'mlx_manager_token' (line 26).

  However, the authStore uses 'mlx_auth_token' as the token key (auth.svelte.ts line 10).

  This mismatch means:
  1. getAuthHeaders() in the component returns no auth token
  2. The PUT /api/settings/pool call fails with 401 (Not authenticated)
  3. The component catches this as "Failed to save pool configuration"

  The dropdown issue is SEPARATE - the component correctly uses models.listLocal() from the shared
  client, which uses authStore.token correctly. If dropdown is empty, it's either:
  - No models are actually downloaded
  - Or the initial getPoolConfig() fails first, causing the loading to fail entirely

  Most likely: The entire onMount fails at getPoolConfig() due to 401, setting error state
  before localModels can be populated.

fix: |
  Replace the component's local API helpers with the shared settings client:
  1. Import { settings } from '$lib/api/client'
  2. Replace getPoolConfig() call with settings.getPoolConfig()
  3. Replace updatePoolConfig() call with settings.updatePoolConfig()
  4. Remove the local getAuthHeaders, getPoolConfig, updatePoolConfig functions

verification:
files_changed: []

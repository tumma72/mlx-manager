---
status: diagnosed
trigger: "When restarting a server, the tile completely disappears for a couple of seconds, then reappears. Expected behavior: show smooth status transitions"
created: 2026-01-19T00:00:00Z
updated: 2026-01-19T00:00:00Z
symptoms_prefilled: true
goal: find_root_cause_only
---

## Current Focus

hypothesis: CONFIRMED - Tile disappears due to conditional rendering gap between ServerTile and StartingTile
test: Traced restart flow through code
expecting: Found evidence of state gap
next_action: Document root cause

## Symptoms

expected: Show smooth status transitions ("restarting" or "shutting down..." -> "starting..." -> "running") without destroying and recreating the tile
actual: Tile completely disappears for a couple of seconds, then reappears
errors: None specified
reproduction: Restart a running server via UI
started: Unknown - behavior observed during UAT

## Eliminated

## Evidence

- timestamp: 2026-01-19T00:01:00Z
  checked: +page.svelte rendering logic (lines 49-60, 126-132)
  found: Two separate lists with mutually exclusive conditions - startingOrFailedProfiles renders StartingTile, runningServers renders ServerTile
  implication: During restart, server moves between lists causing unmount/remount

- timestamp: 2026-01-19T00:02:00Z
  checked: runningServers derived (lines 56-60)
  found: Filters out servers where isStarting OR isFailed is true
  implication: When restart() adds profileId to startingProfiles, server immediately excluded from runningServers

- timestamp: 2026-01-19T00:03:00Z
  checked: startingOrFailedProfiles derived (lines 49-53)
  found: Filters profiles where isStarting OR isFailed is true
  implication: Profile only appears here when startingProfiles.has(profileId)=true AND profile is in profileStore.profiles

- timestamp: 2026-01-19T00:04:00Z
  checked: restart() in servers.svelte.ts (lines 193-199)
  found: Adds profileId to startingProfiles, then calls API (no immediate refresh)
  implication: Frontend state changes BEFORE backend API completes

- timestamp: 2026-01-19T00:05:00Z
  checked: The gap timing
  found: restart() adds to startingProfiles -> server removed from runningServers (ServerTile unmounts) -> BUT profile may still exist in servers list -> startingOrFailedProfiles checks profileStore.profiles NOT serverStore.servers -> Profile appears in StartingTile
  implication: The gap is due to polling/refresh timing - server state is stale

- timestamp: 2026-01-19T00:06:00Z
  checked: API restart flow
  found: Backend restart stops server, waits, then starts it. During stop phase, server is removed from backend list.
  implication: When refresh happens mid-restart, server isn't in either list

## Resolution

root_cause: The page renders two separate tile types (ServerTile for running, StartingTile for starting/failed) using mutually exclusive filter conditions. During restart, the backend API stops then starts the server, causing a window where: (1) Server is removed from backend list (not in runningServers), (2) Profile's server state is in flux (may not be in startingProfiles yet or refresh clears it). The architecture assumes servers transition instantly between states, but the restart API operation is not atomic.

fix: Need unified tile component OR preserve tile visibility during restart transition
verification:
files_changed: []

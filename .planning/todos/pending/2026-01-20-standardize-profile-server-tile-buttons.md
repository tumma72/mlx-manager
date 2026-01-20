---
created: 2026-01-20T12:15
title: Standardize buttons between profile and server tiles
area: frontend/ui
files:
  - frontend/src/lib/components/profiles/ProfileTile.svelte
  - frontend/src/lib/components/servers/ServerTile.svelte
---

## Problem

There are inconsistencies between the buttons displayed in the profiles tiles (on the profiles page) compared with those displayed in the server tiles (on the server page).

## Context

User reported: "there are inconsistencies between the buttons displayed in the profiles tiles in the profiles page compared with those displayed in the server page on a server tile. I think we should standardize the UI to always use the same buttons (I prefere the smaller versions on the server tiles also on the profiles)"

## Solution

1. Audit the button styles used in ProfileTile vs ServerTile
2. Standardize to use the smaller button style from ServerTile across both components
3. Ensure consistent iconography and hover states

---
created: 2026-01-20T17:26
title: Use custom confirmation dialog for model delete
area: frontend/ui
files:
  - frontend/src/lib/components/models/ModelTile.svelte
  - frontend/src/lib/components/ui/ConfirmDialog.svelte
---

## Problem

The model Delete button uses the native browser `confirm()` dialog instead of the custom `ConfirmDialog` component used elsewhere in the app. This creates a visual inconsistency — other destructive actions use the styled modal confirmation dialog while model deletion shows the browser's native alert.

## Solution

Replace the native `confirm()` call in the model delete handler with the custom `ConfirmDialog` component to match the rest of the application's UX pattern.

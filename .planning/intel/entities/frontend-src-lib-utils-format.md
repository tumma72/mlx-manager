---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/utils/format.ts
type: util
updated: 2026-01-21
status: active
---

# format.ts

## Purpose

Formatting utilities for displaying human-readable values in the UI. Provides functions for formatting bytes, durations, numbers with K/M suffixes, relative timestamps, and text truncation.

## Exports

- `formatBytes(bytes: number, decimals?: number) -> string` - Format bytes to human readable (KB, MB, GB)
- `formatDuration(seconds: number) -> string` - Format seconds to duration (e.g., "2h 15m")
- `formatNumber(num: number) -> string` - Format with K/M suffix
- `formatRelativeTime(date: string | Date) -> string` - Format to relative time (e.g., "2h ago")
- `truncate(text: string, length: number) -> string` - Truncate with ellipsis

## Dependencies

None

## Used By

TBD

---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/utils/index.ts
type: module
updated: 2026-01-21
status: active
---

# index.ts (utils)

## Purpose

Barrel export for utility functions. Re-exports formatting and reconciliation utilities, plus provides the cn() helper for combining Tailwind CSS classes with conditional logic.

## Exports

- All exports from `./format` (formatBytes, formatDuration, etc.)
- All exports from `./reconcile` (reconcileArray, reconcileSet, etc.)
- `cn(...inputs: ClassValue[]) -> string` - Tailwind class name combiner using clsx and tailwind-merge

## Dependencies

- [[frontend-src-lib-utils-format]] - Formatting utilities
- [[frontend-src-lib-utils-reconcile]] - Reconciliation utilities
- clsx - Conditional class names
- tailwind-merge - Intelligent Tailwind class merging

## Used By

TBD

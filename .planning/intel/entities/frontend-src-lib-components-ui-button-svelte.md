---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/components/ui/button.svelte
type: component
updated: 2026-01-21
status: active
---

# button.svelte

## Purpose

Reusable button component with variant and size styling. Supports multiple visual variants (default, destructive, outline, secondary, ghost, link) and sizes (default, sm, lg, icon). Renders as either a button or anchor tag depending on whether href is provided.

## Exports

- `ButtonVariant` - Type for button variant options
- `ButtonSize` - Type for button size options
- `buttonVariants` - Object mapping variants to Tailwind classes
- `buttonSizes` - Object mapping sizes to Tailwind classes
- Default export: Button component

## Dependencies

- [[frontend-src-lib-utils-index]] - cn() for class name merging

## Used By

TBD

## Notes

Uses Svelte 5 $props() and $derived() runes. Props include variant, size, class, disabled, type, href, title, onclick, and children snippet.

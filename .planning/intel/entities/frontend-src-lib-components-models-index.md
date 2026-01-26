---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/components/models/index.ts
type: module
updated: 2026-01-21
status: active
---

# index.ts (models components)

## Purpose

Barrel export for model-related UI components. Provides a single import point for components used on the models page including download progress, model cards, filtering, and model badges.

## Exports

- `DownloadProgressTile` - Tile showing download progress
- `ModelCard` - Card displaying model info and actions
- `ModelToggle` - Toggle between local/remote model views
- `FilterModal` - Modal for advanced filtering
- `FilterState` - Filter state interface
- `ARCHITECTURE_OPTIONS` - Available architecture filter values
- `QUANTIZATION_OPTIONS` - Available quantization filter values
- `createEmptyFilters` - Factory for empty filter state
- `FilterChips` - Display active filter chips
- `ArchitectureBadge` - Badge showing model architecture
- `MultimodalBadge` - Badge indicating multimodal capability
- `QuantizationBadge` - Badge showing quantization level
- `ModelBadges` - Container for model badges
- `ModelSpecs` - Display model specifications

## Dependencies

- [[frontend-src-lib-components-models-filter-types]] - Filter type definitions

## Used By

TBD

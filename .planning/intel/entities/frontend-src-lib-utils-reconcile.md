---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/utils/reconcile.ts
type: util
updated: 2026-01-21
status: active
---

# reconcile.ts

## Purpose

Provides reconciliation utilities for efficient reactive state updates in Svelte 5. These utilities compare data and mutate arrays/collections in-place, leveraging Svelte 5's fine-grained reactivity through proxies. Only items that have actually changed trigger re-renders, dramatically improving performance for frequently-updated lists like running servers.

## Exports

- `ReconcileArrayOptions<T>` - Options interface for array reconciliation
- `shallowEqual<T>(a, b) -> boolean` - Shallow equality check for objects
- `reconcileArray<T>(target, source, options) -> boolean` - Reconcile array in-place
- `reconcileSet<T>(target, values) -> boolean` - Reconcile Set in-place
- `reconcileMap<K, V>(target, entries, isEqual?) -> boolean` - Reconcile Map in-place
- `deepEqual<T>(a, b) -> boolean` - One-level deep equality check

## Dependencies

None

## Used By

TBD

## Notes

Key for performance optimization in stores. Instead of replacing arrays (which triggers full re-renders), reconciliation updates items in-place so only changed items cause component updates.

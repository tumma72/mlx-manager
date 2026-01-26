---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/stores/downloads.svelte.ts
type: hook
updated: 2026-01-21
status: active
---

# downloads.svelte.ts

## Purpose

Svelte 5 runes-based store for model download progress tracking. Manages download state globally so it persists across navigation. Uses Server-Sent Events (SSE) for real-time progress updates from the backend. Supports reconnecting to existing downloads after page reload.

## Exports

- `DownloadState` - Interface for download progress state
- `DownloadsStore` - Class with download management methods
- `downloadsStore` - Singleton store instance

## Dependencies

- [[frontend-src-lib-api-client]] - API client for starting downloads

## Used By

TBD

## Notes

Key methods: startDownload() initiates download and SSE connection, reconnect() resumes tracking existing downloads, loadActiveDownloads() fetches active downloads from backend on init. Uses Map with reassignment pattern for Svelte 5 reactivity.

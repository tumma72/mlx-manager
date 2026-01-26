---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/menubar.py
type: module
updated: 2026-01-21
status: active
---

# menubar.py

## Purpose

macOS menubar (status bar) application for MLX Manager using rumps. Provides a native system tray icon with menu options to open the dashboard, start/stop the server, view status, and quit. Auto-starts the server on launch and periodically checks health. Allows managing MLX Manager without terminal access.

## Exports

- `MLXManagerApp` - rumps.App subclass for the menubar app
- `run_menubar()` - Entry point to run the menubar app
- `main()` - CLI entry point

## Dependencies

- rumps - macOS menubar framework
- httpx - HTTP client for health checks
- subprocess - Server process management
- webbrowser - Opening dashboard in browser

## Used By

TBD

## Notes

Uses rumps.timer decorator for periodic health checks (every 5 seconds). Menu items are dynamically enabled/disabled based on server state. Quit dialog asks whether to stop the server.

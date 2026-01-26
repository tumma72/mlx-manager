---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/manager_launchd.py
type: service
updated: 2026-01-21
status: active
---

# manager_launchd.py

## Purpose

Manages the launchd service for MLX Manager itself (not individual MLX servers). Provides functions to install, uninstall, and check status of the MLX Manager application as a macOS launchd service that starts automatically at user login.

## Exports

- `LABEL` - Launchd service label constant ("com.mlx-manager.app")
- `get_plist_path() -> Path` - Get the plist file path
- `install_manager_service(host, port) -> str` - Install MLX Manager as launchd service
- `uninstall_manager_service() -> bool` - Remove the launchd service
- `is_service_installed() -> bool` - Check if service plist exists
- `is_service_running() -> bool` - Check if service is running via launchctl

## Dependencies

- plistlib - macOS plist generation
- subprocess - launchctl commands

## Used By

TBD

## Notes

Configures KeepAlive to restart on crash but not on successful exit. Uses ThrottleInterval of 30 seconds to prevent rapid restart loops.

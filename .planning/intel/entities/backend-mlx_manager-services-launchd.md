---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/launchd.py
type: service
updated: 2026-01-21
status: active
---

# launchd.py

## Purpose

Manages macOS launchd service configuration for individual MLX server profiles (not the MLX Manager app itself). Generates plist files, installs/uninstalls services, and queries service status. Enables profiles to auto-start at user login.

## Exports

- `LaunchdManager` - Class for launchd operations
- `launchd_manager` - Singleton instance

## Dependencies

- [[backend-mlx_manager-models]] - ServerProfile for configuration
- [[backend-mlx_manager-types]] - LaunchdStatus type
- [[backend-mlx_manager-utils-command_builder]] - Build server command
- plistlib - macOS plist generation
- subprocess - launchctl commands

## Used By

TBD

## Notes

Label format: com.mlx-manager.{sanitized-profile-name}. Configures KeepAlive to restart on crash, ThrottleInterval of 30 seconds, and logs to /tmp/{label}.log.

---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/cli.py
type: module
updated: 2026-01-21
status: active
---

# cli.py

## Purpose

Provides the command-line interface for MLX Manager using Typer. Supports commands for starting the web server, installing/uninstalling as a macOS launchd service, checking running server status, launching the menubar app, and displaying version information. This is the main entry point when running `mlx-manager` from the terminal.

## Exports

- `app` - Typer application instance
- `serve()` - Start the web server with uvicorn
- `install_service()` - Install as launchd service
- `uninstall_service()` - Remove launchd service
- `status()` - Show running MLX servers
- `menubar()` - Launch status bar app
- `version()` - Show version info
- `main()` - CLI entry point

## Dependencies

- [[backend-mlx_manager-services-manager_launchd]] - Launchd service management
- [[backend-mlx_manager-menubar]] - Menubar application
- typer - CLI framework
- rich - Terminal output formatting
- uvicorn - ASGI server

## Used By

TBD

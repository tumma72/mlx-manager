# Changelog

All notable changes to MLX Model Manager will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2025-01-19

### Added

- **Models Panel UX improvements**
  - Search/filter bar now stays anchored at the top when scrolling
  - Download progress consolidated into a single tile at top of list
  - Non-downloading model tiles no longer show progress bars

- **Server Panel Redesign**
  - New searchable profile dropdown with keyboard navigation
  - Running servers displayed as rich metric tiles with real-time stats
  - Server tiles show memory usage, CPU/GPU utilization, and uptime
  - Stop/Restart buttons with loading states and error handling
  - "Restarting..." badge during restart operations (tile stays mounted)

- **Scroll preservation** during polling updates - list position maintained

- **Comprehensive test coverage** - 864 tests (452 backend + 412 frontend)

### Fixed

- Server tile no longer disappears briefly during restart
- Duplicate server tiles eliminated during startup
- UI flickering reduced via reconciliation and centralized polling
- ProfileSelector enables Start button on keyboard Enter selection

### Changed

- Version number now managed from single `VERSION` file at project root
- Frontend coverage branch threshold adjusted to 90% (Svelte 5 compiled template artifacts)

## [1.0.2] - 2025-01-15

### Added

- MiniMax M2.1 model support with tool call and reasoning parsers
- Parser options support for various model architectures (Qwen3, GLM-4, Nemotron, etc.)

### Fixed

- Server failure detection improved for exit code 0 errors
- UI flickering eliminated with reconciliation and centralized polling

## [1.0.1] - 2025-01-14

### Fixed

- Server startup and error handling improvements
- Homebrew formula fixes:
  - Removed deprecated `bottle :unneeded` syntax
  - Fixed pip install in post_install phase
  - Proper dependency installation

### Changed

- Switched to Gist-based coverage badge
- E2E test improvements for CI stability

## [1.0.0] - 2025-01-08

### Added

- Initial release of MLX Model Manager
- Web UI for browsing and downloading MLX models from HuggingFace
- Server profile management with configurable parameters
- Start/stop/restart server instances
- macOS menubar app with quick access
- launchd service integration for auto-start
- CLI commands: `mlx-manager serve`, `mlx-manager menubar`, `mlx-manager status`

[1.0.3]: https://github.com/tumma72/mlx-manager/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/tumma72/mlx-manager/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/tumma72/mlx-manager/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tumma72/mlx-manager/releases/tag/v1.0.0

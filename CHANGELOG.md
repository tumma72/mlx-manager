# Changelog

All notable changes to MLX Model Manager will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] - 2025-01-20

### Added

- **User Authentication System**
  - JWT-based authentication with secure token handling
  - User registration and login pages
  - Admin user management page (approve, disable, delete users)
  - Password reset functionality for admins
  - First user automatically becomes administrator
  - Protected routes requiring authentication
  - Auto-redirect to login on session expiration

- **Model Discovery & Filtering**
  - Model characteristics API detecting architecture, quantization, and capabilities
  - Architecture badges (Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, etc.)
  - Quantization level display (4-bit, 8-bit)
  - Multimodal capability detection (vision models)
  - Filter modal with Architecture, Capabilities, and Quantization sections
  - Toggle between "My Models" and "HuggingFace" search modes
  - Local models now display characteristics badges

- **UI Improvements**
  - User profile dropdown in navbar (replaces email + logout button)
  - Standardized button styles across ProfileCard and ServerTile
  - Custom confirmation dialogs for destructive actions
  - Last admin protection (cannot disable/delete sole administrator)

### Changed

- All API endpoints now require authentication
- Navbar shows pending user count badge for admins
- Model search now finds all MLX models (not just mlx-community)

### Fixed

- Auth store properly clears on 401 responses
- Login redirect loop resolved
- Profile filter searches both name and model_path

### Security

- PyJWT for token handling (replaces abandoned python-jose)
- pwdlib with Argon2 for password hashing (Python 3.13+ compatible)
- Secure password validation (minimum 8 characters)

### Tests

- 1075 total tests (532 backend + 543 frontend)
- Auth store at 100% branch coverage
- Comprehensive auth API endpoint testing

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

[1.0.4]: https://github.com/tumma72/mlx-manager/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/tumma72/mlx-manager/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/tumma72/mlx-manager/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/tumma72/mlx-manager/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tumma72/mlx-manager/releases/tag/v1.0.0

# Changelog

All notable changes to MLX Model Manager will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-03-04

### Fixed

- Frontend `StartingTile.svelte` `$effect` infinite loop causing broken tests
- Stdlib logging format bug in probe sweeps (`sweeps.py`)
- Clean shutdown: dispose database engine and suppress `resource_tracker` warning
- Clear stale polling state on component unmount and HMR reload

### Changed

- Backend test coverage: 95.54% to 96.55% (3216 tests, +70 new)
- Frontend test coverage: 1077 to 1238 tests (+161 new)
- 4 new frontend test files: AuditLogPanel, HuggingFaceSettings, ModelPoolSettings, TimeoutSettings
- P3 OpenAPI annotations: line-length lint fixes

## [1.1.0] - 2026-01-26

The major feature release introducing the **embedded MLX inference server** -- a fully self-contained, high-speed inference engine with multi-protocol API support.

### Added

- **Embedded MLX Inference Server**
  - Fully self-contained inference engine mounted at `/v1`, no external server dependencies
  - Multi-protocol API: both OpenAI and Anthropic compatibility from a single server
  - Model pool with LRU eviction: load multiple models simultaneously, auto-evict when memory is low
  - Unified adapter architecture: single `ModelAdapter` handles text, vision, embeddings, and audio
  - Smart model detection: auto-detects model type, family, and capabilities from `config.json`
  - 8 model families: Qwen, GLM-4, Llama, Gemma, Mistral/Devstral/Magistral, Liquid, Whisper, Kokoro
  - 4 model types: text generation (mlx-lm), vision (mlx-vlm), embeddings (mlx-embeddings), audio TTS/STT (mlx-audio)
  - Model probing: automatically discovers tool-calling, thinking, and streaming capabilities
  - Continuous batching (experimental) with prefix caching and priority scheduling
  - Structured output (JSON mode) with schema validation
  - Anthropic protocol translation with 140+ unit tests for bidirectional conversion
  - Cloud routing: seamlessly route requests to OpenAI/Anthropic APIs when local models can't handle them

- **Observability & Operations**
  - Audit logging: privacy-first request metadata logging (no prompt/response content stored)
  - WebSocket live streaming for real-time audit log monitoring
  - Audit log export in JSONL and CSV formats
  - Prometheus metrics: request latency, throughput, model memory, pool cache hits/misses
  - LogFire integration for distributed tracing
  - RFC 7807 structured error responses with request ID correlation
  - Request ID propagation (client-provided or auto-generated)
  - Graceful shutdown with configurable drain timeout for in-flight requests
  - Model loading progress via Server-Sent Events

- **Security & Rate Limiting**
  - Admin token authentication for `/v1/admin/*` endpoints
  - Per-IP token bucket rate limiter with configurable RPM
  - Input validation with image size limits and path traversal protection
  - Per-endpoint configurable timeouts (chat, completions, embeddings)

- **Data Model Redesign**
  - Capability polymorphism: single-table `model_capabilities` with type discriminator
  - Profile polymorphism: `ExecutionProfile` with STI (inference, audio, base types)
  - All dataclasses converted to Pydantic BaseModel throughout the codebase
  - Shared package (`mlx_manager/shared/`) for cross-module entities

- **Chat Multimodal Support**
  - Image attachments via button click or drag-and-drop
  - Video attachments with 2-minute duration limit
  - Text file attachments (.txt, .md, .json, .yaml, etc.)
  - Extensionless text files supported (README, Makefile, Dockerfile)

- **Thinking Model Support**
  - Collapsible thinking panel for reasoning models (Qwen3, GLM-4, DeepSeek, etc.)
  - "Thought for Xs" duration display
  - Dual detection: server-side reasoning_parser and raw `<think>` tag parsing

- **MCP Tool Integration**
  - Built-in calculator and weather mock tools for testing
  - Tools toggle button in chat interface
  - Tool execution loop with 3-depth limit
  - AST-based safe calculator (no eval/exec)

- **Profile Enhancements**
  - Model description field changed to multi-line textarea
  - New system prompt field for default chat context

### Changed

- Architecture refactored from external mlx_lm server dependency to embedded inference
- Server profiles renamed to Execution Profiles with type-specific parameters
- Chat input replaced with auto-resizing textarea
- Frontend stores migrated to Svelte 5 runes (`$state`, `$derived`)

### Fixed

- **BUGFIX-01**: Silent exception handlers now log errors (no more `except: pass`)
- **BUGFIX-02**: Server log files cleaned up on crash/exit (4 exit paths covered)
- **BUGFIX-03**: API validation uses HTTPException (no assertions in routers)
- **BUGFIX-04**: Server status polling optimized (early-exit prevents unnecessary re-renders)
- **BUGFIX-05**: Console.log debug statements removed from production code
- **BUGFIX-06**: Models marked as started but failing to load handled with retry
- **BUGFIX-07**: CPU gauge shows actual values; memory gauge reflects real model size

## [1.0.4] - 2025-01-20

### Added

- **User Authentication System**
  - JWT-based authentication with secure token handling
  - User registration and login pages
  - Admin user management page (approve, disable, delete users)
  - Password reset functionality for admins
  - First user automatically becomes administrator
  - Protected routes requiring authentication

- **Model Discovery & Filtering**
  - Model characteristics API detecting architecture, quantization, and capabilities
  - Architecture badges (Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, etc.)
  - Quantization level display (4-bit, 8-bit)
  - Multimodal capability detection (vision models)
  - Filter modal with Architecture, Capabilities, and Quantization sections
  - Toggle between "My Models" and "HuggingFace" search modes

- **UI Improvements**
  - User profile dropdown in navbar
  - Standardized button styles
  - Custom confirmation dialogs for destructive actions
  - Last admin protection

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

## [1.0.3] - 2025-01-19

### Added

- **Models Panel UX improvements**
  - Search/filter bar anchored at top when scrolling
  - Download progress consolidated into single tile
  - Non-downloading model tiles no longer show progress bars

- **Server Panel Redesign**
  - Searchable profile dropdown with keyboard navigation
  - Running servers displayed as rich metric tiles with real-time stats
  - Server tiles show memory usage, CPU/GPU utilization, and uptime

- **Scroll preservation** during polling updates

### Fixed

- Server tile no longer disappears briefly during restart
- Duplicate server tiles eliminated during startup
- UI flickering reduced via reconciliation and centralized polling

### Changed

- Version number now managed from single `VERSION` file at project root

## [1.0.2] - 2025-01-15

### Added

- MiniMax M2.1 model support with tool call and reasoning parsers
- Parser options support for various model architectures

### Fixed

- Server failure detection improved for exit code 0 errors
- UI flickering eliminated with reconciliation and centralized polling

## [1.0.1] - 2025-01-14

### Fixed

- Server startup and error handling improvements
- Homebrew formula fixes

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

[1.2.0]: https://github.com/tumma72/mlx-manager/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/tumma72/mlx-manager/compare/v1.0.4...v1.1.0
[1.0.4]: https://github.com/tumma72/mlx-manager/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/tumma72/mlx-manager/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/tumma72/mlx-manager/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/tumma72/mlx-manager/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tumma72/mlx-manager/releases/tag/v1.0.0

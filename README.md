# MLX Manager

[![CI](https://github.com/tumma72/mlx-manager/actions/workflows/ci.yml/badge.svg)](https://github.com/tumma72/mlx-manager/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tumma72/bfb62663af32da529734c79e0e67fa23/raw/mlx-manager-coverage.json)](https://github.com/tumma72/mlx-manager)
[![PyPI version](https://img.shields.io/pypi/v/mlx-manager.svg)](https://pypi.org/project/mlx-manager/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python 3.11-3.12](https://img.shields.io/badge/Python-3.11--3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Svelte 5](https://img.shields.io/badge/Svelte-5-ff3e00.svg?logo=svelte&logoColor=white)](https://svelte.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

**Run local LLMs on your Mac with one command.** MLX Manager makes it easy to download, configure, and serve MLX-optimized language models on Apple Silicon—no terminal expertise required.

## Why MLX Manager?

Running local LLMs typically requires juggling multiple tools, config files, and terminal commands. MLX Manager gives you:

- **One-click model downloads** from HuggingFace MLX models (mlx-community, lmstudio-community, and more)
- **Smart model discovery** - filter by architecture (Llama, Qwen, Mistral), quantization (4-bit, 8-bit), and capabilities (multimodal, tool use)
- **Visual server management** - start, stop, and monitor models with real-time CPU/memory metrics
- **Rich chat interface** - test models with image/video/text attachments, thinking model support, and MCP tool integration
- **User authentication** - secure multi-user access with admin controls
- **OpenAI-compatible API** - use your local models with any OpenAI client
- **Background service** - models auto-start on login via macOS launchd
- **Menubar app** - quick access from your Mac's status bar

## Quick Start

### Install

```bash
# Homebrew (recommended)
brew tap tumma72/mlx-manager https://github.com/tumma72/mlx-manager
brew install mlx-manager

# Or via pip
pip install mlx-manager
```

### Run

```bash
mlx-manager serve
```

Open http://localhost:8080 and you're ready to:

1. **Register** - Create your account (first user becomes admin)
2. **Browse** - Search HuggingFace for MLX-optimized models
3. **Filter** - Find models by architecture, quantization, or capabilities
4. **Download** - One-click download with progress tracking
5. **Configure** - Create a server profile with custom settings
6. **Run** - Start serving and chat with your model

## Features

### Model Discovery

Browse and filter models with rich metadata:
- **Architecture badges** - Llama, Qwen, Mistral, Gemma, Phi, and more
- **Quantization info** - 4-bit, 8-bit quantization levels
- **Capability detection** - Multimodal (vision), tool use support
- **Toggle view** - Switch between your downloaded models and HuggingFace search

### User Management

Secure multi-user support:
- **JWT authentication** - Secure token-based auth
- **Admin controls** - Approve/disable users, manage permissions
- **First-user admin** - Initial user automatically becomes administrator

### Server Monitoring

Real-time server metrics:
- Memory usage and CPU/GPU utilization
- Server uptime tracking
- One-click start/stop/restart controls

### Chat Interface

Rich conversation experience:
- **Multimodal support** - Attach images, videos, and text files via drag-drop or button
- **Thinking models** - Collapsible thinking panel for reasoning models (Qwen3, GLM-4, DeepSeek)
- **MCP tools** - Built-in calculator and weather tools for testing tool-use models
- **System prompts** - Configure default context per server profile

## System Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12
- 8GB+ RAM (16GB+ recommended for larger models)

## Commands

```bash
mlx-manager serve            # Start the web server
mlx-manager menubar          # Launch menubar app
mlx-manager install-service  # Auto-start on login
mlx-manager status           # Show running servers
```

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_MANAGER_DATABASE_PATH` | `~/.mlx-manager/mlx-manager.db` | Database location |
| `MLX_MANAGER_DEFAULT_PORT_START` | `10240` | Starting port for servers |
| `MLX_MANAGER_JWT_SECRET` | Auto-generated | JWT signing secret |

## Development

```bash
git clone https://github.com/tumma72/mlx-manager.git
cd mlx-manager
make install-dev  # Install dependencies
make dev          # Start dev servers
make test         # Run tests (1000+ tests)
```

## License

MIT

## Acknowledgments

Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), and [mlx-embeddings](https://github.com/ml-explore/mlx-embeddings).

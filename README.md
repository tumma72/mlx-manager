# MLX Manager

[![CI](https://github.com/tumma72/mlx-manager/actions/workflows/ci.yml/badge.svg)](https://github.com/tumma72/mlx-manager/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tumma72/bfb62663af32da529734c79e0e67fa23/raw/mlx-manager-coverage.json)](https://github.com/tumma72/mlx-manager)
[![PyPI version](https://img.shields.io/pypi/v/mlx-manager.svg)](https://pypi.org/project/mlx-manager/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Svelte 5](https://img.shields.io/badge/Svelte-5-ff3e00.svg?logo=svelte&logoColor=white)](https://svelte.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

**Run local LLMs on your Mac with one command.** MLX Manager makes it easy to download, configure, and serve MLX-optimized language models on Apple Silicon—no terminal expertise required.

## Why MLX Manager?

Running local LLMs typically requires juggling multiple tools, config files, and terminal commands. MLX Manager gives you:

- **One-click model downloads** from HuggingFace's [mlx-community](https://huggingface.co/mlx-community)
- **Visual server management** - start, stop, and monitor models from a web UI
- **OpenAI-compatible API** - use your local models with any OpenAI client
- **Background service** - models auto-start on login via macOS launchd
- **Menubar app** - quick access from your Mac's status bar

## Quick Start

### Install

```bash
# Homebrew (recommended)
brew install tumma72/tap/mlx-manager

# Or via pip
pip install mlx-manager
```

### Run

```bash
mlx-manager serve
```

Open http://localhost:8080 and you're ready to:

1. **Browse** - Search mlx-community models
2. **Download** - One-click download to your Mac
3. **Configure** - Create a server profile
4. **Run** - Start serving and chat with your model

## System Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
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

## Development

```bash
git clone https://github.com/tumma72/mlx-manager.git
cd mlx-manager
make install-dev  # Install dependencies
make dev          # Start dev servers
make test         # Run tests
```

## License

MIT

## Acknowledgments

Built on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-examples).

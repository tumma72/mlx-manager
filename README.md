# MLX Manager

[![CI](https://github.com/tumma72/mlx-manager/actions/workflows/ci.yml/badge.svg)](https://github.com/tumma72/mlx-manager/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tumma72/bfb62663af32da529734c79e0e67fa23/raw/mlx-manager-coverage.json)](https://github.com/tumma72/mlx-manager)
[![PyPI version](https://img.shields.io/pypi/v/mlx-manager.svg)](https://pypi.org/project/mlx-manager/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python 3.11-3.12](https://img.shields.io/badge/Python-3.11--3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Svelte 5](https://img.shields.io/badge/Svelte-5-ff3e00.svg?logo=svelte&logoColor=white)](https://svelte.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

**Run and serve local LLMs on your Mac with one command.** MLX Manager provides a web UI for managing MLX-optimized models on Apple Silicon, with an embedded high-performance inference server exposing both OpenAI and Anthropic-compatible APIs.

## Why MLX Manager?

Running local LLMs typically requires juggling multiple tools, config files, and terminal commands. MLX Manager gives you:

- **One-click model downloads** from HuggingFace MLX models (mlx-community, lmstudio-community, and more)
- **Smart model discovery** - filter by architecture (Llama, Qwen, Mistral), quantization (4-bit, 8-bit), and capabilities (multimodal, tool use)
- **Embedded inference server** - high-speed MLX inference engine with OpenAI and Anthropic API compatibility
- **Model pool with LRU eviction** - load multiple models simultaneously, auto-evict when memory is low
- **4 model types** - text generation, vision, embeddings, and audio (TTS/STT)
- **Visual server management** - start, stop, and monitor models with real-time CPU/memory metrics
- **Rich chat interface** - test models with image/video/text attachments, thinking model support, and MCP tool integration
- **Cloud routing** - seamlessly route requests to OpenAI/Anthropic APIs when local models can't handle them
- **User authentication** - secure multi-user access with JWT auth and admin controls
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

Open http://localhost:10242 and you're ready to:

1. **Register** - Create your account (first user becomes admin)
2. **Browse** - Search HuggingFace for MLX-optimized models
3. **Filter** - Find models by architecture, quantization, or capabilities
4. **Download** - One-click download with progress tracking
5. **Configure** - Create a server profile with custom settings
6. **Run** - Start serving and chat with your model

### Use as an API

Once a model is loaded, use it with any OpenAI or Anthropic client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10242/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:10242/v1", api_key="not-needed")
message = client.messages.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Embedded Inference Server

MLX Manager includes a fully self-contained inference server mounted at `/v1`. No external inference backends needed.

### Multi-Protocol API

| Protocol | Endpoint | Description |
|----------|----------|-------------|
| OpenAI | `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| OpenAI | `POST /v1/completions` | Legacy text completions |
| OpenAI | `POST /v1/embeddings` | Text embeddings |
| OpenAI | `POST /v1/audio/speech` | Text-to-speech |
| OpenAI | `POST /v1/audio/transcriptions` | Speech-to-text |
| Anthropic | `POST /v1/messages` | Anthropic Messages API |
| Both | `GET /v1/models` | List available models |

### Key Capabilities

- **Unified adapter architecture** - single ModelAdapter handles all model types with family-specific configs
- **8 model families** - Qwen, GLM-4, Llama, Gemma, Mistral/Devstral/Magistral, Liquid, Whisper, Kokoro
- **Smart model detection** - auto-detects model type, family, and capabilities from config.json
- **Model probing** - discovers tool-calling, thinking, and streaming capabilities at load time
- **Continuous batching** (experimental) with prefix caching and priority scheduling
- **Structured output** - JSON mode with schema validation
- **Anthropic protocol translation** - bidirectional conversion between OpenAI and Anthropic formats
- **Cloud routing** - route to OpenAI/Anthropic cloud APIs when local models can't handle the request

### Observability

- **Audit logging** - privacy-first request metadata logging with WebSocket live streaming
- **Prometheus metrics** - request latency, throughput, model memory, pool cache hits/misses
- **LogFire integration** - distributed tracing with Pydantic LogFire
- **RFC 7807 errors** - structured error responses with request ID correlation

See [docs/MLX_SERVER.md](docs/MLX_SERVER.md) for the full configuration reference, security guide, metrics list, and API documentation.

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
- **Rate limiting** - Per-IP request throttling with token bucket algorithm

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

### MLX Server Configuration

The embedded MLX inference server accepts `MLX_SERVER_*` environment variables. All settings are opt-in with safe defaults -- zero configuration needed for local use.

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_SERVER_ADMIN_TOKEN` | *none* | Bearer token for `/v1/admin/*` endpoints |
| `MLX_SERVER_RATE_LIMIT_RPM` | `0` (off) | Requests per minute per IP |
| `MLX_SERVER_METRICS_ENABLED` | `false` | Enable Prometheus metrics at `/v1/admin/metrics` |
| `MLX_SERVER_MAX_MEMORY_GB` | `0` (auto) | Model pool memory limit (0 = 75% of device RAM) |
| `MLX_SERVER_MAX_MODELS` | `4` | Max models loaded simultaneously |
| `MLX_SERVER_TIMEOUT_CHAT_SECONDS` | `900` | Chat completions timeout |
| `MLX_SERVER_DRAIN_TIMEOUT_SECONDS` | `30` | Graceful shutdown drain timeout |

See [docs/MLX_SERVER.md](docs/MLX_SERVER.md) for the full configuration reference, security guide, metrics list, and API documentation.

## Development

```bash
git clone https://github.com/tumma72/mlx-manager.git
cd mlx-manager
make install-dev  # Install dependencies
make dev          # Start dev servers
make test         # Run tests (4400+ tests)
```

## License

MIT

## Acknowledgments

Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [mlx-embeddings](https://github.com/ml-explore/mlx-embeddings), and [mlx-audio](https://github.com/Blaizzy/mlx-audio).

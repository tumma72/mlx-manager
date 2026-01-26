# Stack Research: Unified API Gateway

**Project:** MLX Manager v1.2 - Unified API Gateway
**Researched:** 2026-01-26
**Overall Confidence:** MEDIUM-HIGH

## Executive Summary

The unified API gateway transforms MLX Manager into a routing layer that proxies requests to local (mlx-openai-server, vLLM-MLX) or cloud (OpenAI, Anthropic) backends. The existing FastAPI + SQLModel stack provides a solid foundation. New additions focus on HTTP proxying, secure credential storage, and cloud API clients.

**Key Findings:**
- **vLLM-MLX:** Experimental/development-stage, NOT recommended for production use
- **API Gateway Pattern:** Use httpx.AsyncClient with lifespan management (no new framework needed)
- **Credential Storage:** Application-level encryption with Python cryptography.fernet
- **Anthropic Integration:** Official SDK (anthropic 0.76.0) with native async support

## vLLM-MLX Evaluation

### Overview
[vLLM-MLX](https://github.com/waybarrios/vllm-mlx) attempts to bring vLLM's advanced features (continuous batching, paged KV cache) to Apple Silicon using MLX as the backend. It provides OpenAI-compatible API with 400+ tok/s performance claims.

### Maturity Assessment

**Indicators:**
- 210 GitHub stars, 22 forks, 6 contributors (moderate adoption)
- 74 commits on main branch (actively maintained)
- Latest release: v0.2.4 (January 23, 2026)
- Feature set: Continuous batching, MCP tool calling, multimodal support

**Concerns:**
- **Early-stage project:** No production deployment documentation found
- **Apple Silicon only:** Platform-restricted, not cross-platform
- **External dependencies:** Requires system-level tools (espeak-ng for audio)
- **Community pattern:** [Industry article](https://medium.com/@michael.hannecke/when-your-dev-and-prod-llm-backends-dont-match-and-why-that-s-okay-3bf2cb1c55c2) (Dec 2025) recommends "Ollama/MLX on Mac for fast local development and vLLM on NVIDIA for production serving"

### API Differences from mlx-openai-server

| Feature | mlx-openai-server | vLLM-MLX |
|---------|------------------|----------|
| **Maturity** | 197 stars, 703 commits | 210 stars, 74 commits |
| **OpenAI Compatible** | Yes | Yes |
| **Continuous Batching** | No (single-request) | Yes (multi-tenant) |
| **CLI Pattern** | `mlx-openai-server launch --model-path X --model-type Y` | `vllm-mlx serve [model] --port 8000` |
| **Model Types** | 6 types (lm, multimodal, image-gen, embeddings, whisper) | Focus on LLM + vision-language |
| **Quantization** | 4/8/16-bit options | Not explicitly documented |
| **Request Queueing** | Built-in queue system | Continuous batching |
| **Production Readiness** | Active development, used in production | Experimental, dev/testing use |

### CLI Parameter Comparison

**mlx-openai-server (current):**
```bash
mlx-openai-server launch \
  --model-path <path-or-hf-repo> \
  --model-type <lm|multimodal|image-generation|embeddings|whisper> \
  --config-name <config> \
  --quantize <4|8|16> \
  --max-concurrency <num> \
  --context-length <num>
```

**vLLM-MLX:**
```bash
vllm-mlx serve [model-name] \
  --port 8000 \
  --continuous-batching \
  --api-key <key>
```

### Recommendation: DEFER vLLM-MLX

**Do NOT add vLLM-MLX to v1.2 for these reasons:**

1. **Maturity:** Too experimental for production use (74 commits vs 703 for mlx-openai-server)
2. **Complexity:** Continuous batching adds architectural complexity without proven stability
3. **Integration risk:** Different CLI patterns require separate subprocess management code
4. **Limited advantage:** For single-user/small-scale use (MLX Manager's target), batching provides minimal benefit
5. **Compatibility:** mlx-openai-server already provides OpenAI-compatible API with broader model type support

**When to reconsider:**
- vLLM-MLX reaches 1.0+ release with production documentation
- Community adoption shows stable production deployments
- User requests specifically need continuous batching features
- Project matures beyond experimental stage (6-12 months from now)

**Alternative:** Stick with mlx-openai-server (already in dependencies at v1.4.0+) which offers proven stability, comprehensive model type support, and active maintenance (703 commits).

## API Gateway Patterns in FastAPI

### Approach: Native httpx.AsyncClient (No Additional Framework)

FastAPI's existing httpx dependency (v0.28.0+ already in project) provides everything needed for API gateway functionality. No specialized gateway library required.

### Pattern: Lifespan-Managed AsyncClient

**Recommended pattern from [FastAPI + httpx best practices](https://medium.com/@benshearlaw/how-to-use-httpx-request-client-with-fastapi-16255a9984a4):**

```python
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create clients for each backend
    app.openai_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        timeout=60.0
    )
    app.anthropic_client = httpx.AsyncClient(
        base_url="https://api.anthropic.com/v1",
        timeout=60.0
    )
    app.local_client = httpx.AsyncClient(timeout=60.0)  # For local servers

    yield

    # Clean up on shutdown
    await app.openai_client.aclose()
    await app.anthropic_client.aclose()
    await app.local_client.aclose()

app = FastAPI(lifespan=lifespan)
```

**Benefits:**
- Connection pooling and keep-alive for cloud APIs
- No socket leaks
- Proper resource cleanup
- Zero additional dependencies

### Routing Logic

```python
async def route_request(model_name: str, request: ChatRequest) -> Response:
    """Route request to appropriate backend based on model name."""

    # Cloud providers
    if model_name.startswith("gpt-"):
        return await proxy_to_openai(request)
    elif model_name.startswith("claude-"):
        return await proxy_to_anthropic(request)

    # Local backends (mlx-openai-server instances)
    profile = await get_profile_for_model(model_name)
    if profile:
        return await proxy_to_local(profile, request)

    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
```

### Streaming Support

httpx supports streaming responses natively:

```python
async def proxy_streaming(backend_url: str, headers: dict, body: dict):
    async with app.local_client.stream(
        "POST",
        f"{backend_url}/v1/chat/completions",
        headers=headers,
        json=body
    ) as response:
        async for chunk in response.aiter_bytes():
            yield chunk
```

### Library Evaluation: fastapi-proxy-lib

**NOT RECOMMENDED** despite appearing in search results:

- **Version:** 0.3.0 (March 25, 2025)
- **Features:** HTTP/WebSocket proxy, streaming support
- **Why avoid:** Adds unnecessary abstraction layer when httpx.AsyncClient provides sufficient functionality
- **Complexity:** Our use case (routing to known backends) doesn't need generic proxy capabilities
- **Maintainability:** Direct httpx usage is more transparent and easier to debug

## Secure API Key Storage

### Approach: Application-Level Encryption with SQLite

Store encrypted keys in the existing aiosqlite database using Python's cryptography library for symmetric encryption.

### Recommended Solution: cryptography.fernet

**Why Fernet:**
- Part of [Python cryptography package](https://cryptography.io/en/latest/fernet/) (standard library-adjacent)
- Symmetric encryption (AES-128-CBC + HMAC-SHA256)
- Simple API designed for developers (no cryptography expertise required)
- Authenticated encryption (prevents tampering)
- Battle-tested and widely adopted

**Installation:**
```bash
pip install cryptography>=46.0.0
```

Current version: 46.0.3 (October 2025), supports Python 3.8+.

### Implementation Pattern

```python
from cryptography.fernet import Fernet
from pydantic_settings import BaseSettings
import base64
import os

class Settings(BaseSettings):
    # Master encryption key (stored in environment, NOT in database)
    encryption_key: str = os.getenv("MLX_MANAGER_ENCRYPTION_KEY", "")

    def get_fernet(self) -> Fernet:
        """Get Fernet cipher with master key."""
        if not self.encryption_key:
            # Generate on first run and prompt user to save
            key = Fernet.generate_key()
            print(f"Generated encryption key: {key.decode()}")
            print("Save this to MLX_MANAGER_ENCRYPTION_KEY environment variable!")
            return Fernet(key)
        return Fernet(self.encryption_key.encode())

def encrypt_api_key(plaintext: str, settings: Settings) -> str:
    """Encrypt API key for storage."""
    fernet = settings.get_fernet()
    encrypted = fernet.encrypt(plaintext.encode())
    return base64.b64encode(encrypted).decode()

def decrypt_api_key(encrypted: str, settings: Settings) -> str:
    """Decrypt API key for use."""
    fernet = settings.get_fernet()
    encrypted_bytes = base64.b64decode(encrypted.encode())
    return fernet.decrypt(encrypted_bytes).decode()
```

### Database Schema Addition

```python
# In models.py
class CloudProvider(SQLModel, table=True):
    """Store encrypted cloud provider API keys."""
    __tablename__ = "cloud_providers"

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(index=True)  # "openai", "anthropic"
    api_key_encrypted: str  # Fernet-encrypted key
    base_url: str | None = None  # Optional custom endpoint
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### Security Best Practices

Based on [OpenAI key safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) and [GitGuardian secrets management](https://blog.gitguardian.com/secrets-api-management/):

1. **Master key management:**
   - Store MLX_MANAGER_ENCRYPTION_KEY in environment variable
   - Generate unique key per installation
   - Never commit to git
   - Document in setup instructions

2. **Key rotation:**
   - Provide CLI command to rotate master key
   - Re-encrypt all stored keys with new master key
   - Log rotation events

3. **Access control:**
   - Require authentication to view/modify cloud provider keys
   - Audit log for key usage
   - Rate limiting on key operations

4. **Least privilege:**
   - Store only necessary permissions with keys
   - Separate read-only vs read-write keys

### Why NOT Use Alternatives

| Alternative | Why Not |
|-------------|---------|
| **SQLCipher** | Encrypts entire database file; overkill when only API keys need encryption; requires compiling custom sqlite |
| **AWS KMS/Vault** | External dependency; not suitable for desktop app; requires cloud services |
| **OS Keychain** | macOS-only (Keyring); Linux/future Windows support blocked; adds platform-specific code |
| **Environment variables only** | No persistence; user must set keys every restart; poor UX |

## Anthropic API Integration

### Approach: Official SDK (anthropic)

Use the [official Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) rather than direct API calls.

**Current version:** 0.76.0 (released January 13, 2026)
**Python requirement:** >=3.9 (compatible with project's 3.11+ requirement)

### Why SDK Over Direct API

| Aspect | SDK | Direct API |
|--------|-----|-----------|
| **Type safety** | Full Pydantic models | Manual typing |
| **Error handling** | Built-in retry logic | Manual implementation |
| **Authentication** | Automatic header management | Manual header construction |
| **Streaming** | Native support | Manual SSE parsing |
| **Version management** | Automatic API version header | Manual version tracking |
| **Maintenance** | Anthropic maintains | We maintain |

### Installation

```bash
pip install anthropic>=0.76.0
```

The SDK uses httpx (already in dependencies) as its HTTP client, so no additional HTTP library needed.

### Async Client Usage

```python
import os
from anthropic import AsyncAnthropic

# Initialize at app startup (in lifespan)
anthropic_client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # From encrypted storage
)

# Use in route handlers
async def call_claude(prompt: str):
    message = await anthropic_client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        model="claude-sonnet-4-5-20250929",
    )
    return message.content
```

### Streaming Support

```python
async def stream_claude(prompt: str):
    """Stream Claude response using SDK helpers."""
    async with anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        model="claude-sonnet-4-5-20250929",
    ) as stream:
        async for text in stream.text_stream:
            yield text
```

### Message Format Translation

**OpenAI format → Anthropic format:**

```python
def translate_openai_to_anthropic(openai_request: dict) -> dict:
    """Translate OpenAI chat request to Anthropic format."""
    return {
        "model": openai_request["model"].replace("gpt-", "claude-"),  # Example
        "messages": openai_request["messages"],  # Same format
        "max_tokens": openai_request.get("max_tokens", 1024),
        "temperature": openai_request.get("temperature", 1.0),
        "stream": openai_request.get("stream", False),
    }
```

**Key differences:**
- `max_tokens` is **required** in Anthropic (optional in OpenAI)
- System messages: Anthropic has top-level `system` parameter (not in messages array)
- Tool/function calling: Different format (Anthropic uses `tools` array)
- Response format: Anthropic wraps content in `content` array with `type` field

### FastAPI Integration Pattern

```python
from fastapi import FastAPI, Request, Depends
from anthropic import AsyncAnthropic

async def get_anthropic_client(request: Request) -> AsyncAnthropic:
    """Dependency to get Anthropic client from app state."""
    return request.app.anthropic_client

@app.post("/v1/chat/completions")
async def unified_chat(
    request: ChatCompletionRequest,
    anthropic: AsyncAnthropic = Depends(get_anthropic_client)
):
    if request.model.startswith("claude-"):
        # Route to Anthropic
        response = await anthropic.messages.create(...)
        return translate_anthropic_to_openai(response)
    # ... other routing logic
```

## Recommended Stack Additions

### Core Dependencies

| Library | Version | Purpose | Add to |
|---------|---------|---------|--------|
| anthropic | >=0.76.0 | Anthropic API client | dependencies |
| cryptography | >=46.0.0 | Encrypt API keys (Fernet) | dependencies |

### No Additional Dependencies Needed

These existing dependencies already provide required functionality:

| Library | Current Version | Gateway Use |
|---------|----------------|-------------|
| httpx | >=0.28.0 | HTTP proxying to local/cloud backends |
| fastapi | >=0.115.0 | API gateway routing, lifespan management |
| aiosqlite | >=0.20.0 | Store encrypted credentials |
| pydantic-settings | >=2.0.0 | Master encryption key from environment |

### Installation Commands

```bash
# Update pyproject.toml dependencies array:
dependencies = [
    # ... existing dependencies ...
    "anthropic>=0.76.0",
    "cryptography>=46.0.0",
]
```

```bash
# Install in development environment:
cd backend
pip install -e ".[dev]"
```

## What NOT to Add

### Rejected: fastapi-proxy-lib

**Why skip:**
- Unnecessary abstraction over httpx
- Our routing logic is specific, not generic proxy
- Direct httpx usage is clearer and more maintainable
- Adds dependency without clear benefit

### Rejected: vLLM-MLX (for v1.2)

**Why defer:**
- Experimental/early-stage maturity
- Only 74 commits vs mlx-openai-server's 703
- No production deployment patterns documented
- Minimal benefit for single-user desktop app
- Adds complexity without proven stability

**When to reconsider:** v1.3+ if project matures and users request batching features

### Rejected: Full Database Encryption (SQLCipher)

**Why avoid:**
- Only API keys need encryption, not entire database
- Requires compiling custom sqlite3 binary
- Platform compatibility issues
- Application-level encryption (Fernet) is sufficient

### Rejected: External Secrets Managers (Vault, AWS KMS)

**Why avoid:**
- Desktop app shouldn't depend on cloud services
- User experience: must set up external services
- Network dependency for key retrieval
- Over-engineering for use case

### Rejected: Direct API Integration for Anthropic

**Why use SDK instead:**
- Official SDK maintained by Anthropic
- Type safety and error handling built-in
- Streaming support without SSE parsing
- API version management handled automatically
- Future compatibility guaranteed

## Integration with Existing Stack

### Lifespan Management (main.py)

Current lifespan structure:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.create_engine(get_settings().database_path)
    await db.create_tables()
    server_manager.start_health_checker()
    yield
    await server_manager.stop_all_servers()
```

**Add after existing setup:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Existing setup
    db.create_engine(get_settings().database_path)
    await db.create_tables()
    server_manager.start_health_checker()

    # NEW: Initialize cloud API clients
    app.anthropic_client = AsyncAnthropic(
        api_key=await get_decrypted_key("anthropic")
    )
    app.openai_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        timeout=60.0
    )

    yield

    # Existing cleanup
    await server_manager.stop_all_servers()

    # NEW: Close cloud clients
    await app.anthropic_client.close()
    await app.openai_client.aclose()
```

### New Services

Create `services/gateway.py`:
```python
"""API Gateway service for routing to local/cloud backends."""
from typing import Protocol
from anthropic import AsyncAnthropic
import httpx

class GatewayService:
    """Routes requests to appropriate backend."""

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        openai_client: httpx.AsyncClient,
        local_client: httpx.AsyncClient
    ):
        self.anthropic = anthropic_client
        self.openai = openai_client
        self.local = local_client

    async def route_chat_completion(self, model: str, messages: list):
        """Route chat completion to correct backend."""
        if model.startswith("claude-"):
            return await self._call_anthropic(model, messages)
        elif model.startswith("gpt-"):
            return await self._call_openai(model, messages)
        else:
            return await self._call_local(model, messages)
```

### New Routers

Create `routers/gateway.py`:
```python
"""Unified API gateway endpoints."""
from fastapi import APIRouter, Depends, Request
from services.gateway import GatewayService

router = APIRouter(prefix="/v1", tags=["gateway"])

@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    gateway: GatewayService = Depends(get_gateway_service)
):
    """Unified chat completions endpoint (OpenAI-compatible)."""
    return await gateway.route_chat_completion(
        model=request.model,
        messages=request.messages
    )
```

### Database Migrations

Add Alembic migration for cloud_providers table:
```python
# migrations/versions/xxx_add_cloud_providers.py
def upgrade():
    op.create_table(
        'cloud_providers',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('provider', sa.String(), nullable=False),
        sa.Column('api_key_encrypted', sa.Text(), nullable=False),
        sa.Column('base_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_cloud_providers_provider', 'cloud_providers', ['provider'])
```

## Architecture Recommendations

### Request Flow

```
Client Request
    ↓
FastAPI Gateway Endpoint (/v1/chat/completions)
    ↓
GatewayService.route_chat_completion()
    ↓
Model Name Analysis
    ↓
├─ "claude-*" → AnthropicClient → Anthropic API
├─ "gpt-*" → httpx.AsyncClient → OpenAI API
└─ Local model → httpx.AsyncClient → mlx-openai-server (profile-based)
```

### Service Layer Separation

```
routers/gateway.py           # FastAPI routes
    ↓ depends on
services/gateway.py          # Routing logic
    ↓ uses
services/cloud_providers.py  # Credential management (encrypt/decrypt)
    ↓ uses
models.py                    # CloudProvider SQLModel
```

### Configuration Management

```python
# config.py additions
class Settings(BaseSettings):
    # Existing settings...

    # NEW: Gateway settings
    encryption_key: str = ""  # Master key for API key encryption
    gateway_timeout: int = 60  # Timeout for cloud API calls
    gateway_max_retries: int = 3  # Retry failed cloud requests

    class Config:
        env_prefix = "MLX_MANAGER_"
```

## Confidence Assessment

| Component | Confidence | Source |
|-----------|-----------|--------|
| httpx proxy pattern | HIGH | Official FastAPI docs + Context7 + recent articles |
| Anthropic SDK | HIGH | Official PyPI (v0.76.0, Jan 2026) + GitHub docs |
| Fernet encryption | HIGH | Official cryptography docs + widespread usage |
| vLLM-MLX maturity | MEDIUM | GitHub inspection + community articles (Dec 2025) |
| mlx-openai-server stability | MEDIUM-HIGH | GitHub inspection (703 commits, active) |

## Open Questions

1. **Model name mapping:** How to handle model aliases (e.g., user says "claude-opus" but API needs "claude-opus-4-5-20251101")?
2. **Rate limiting:** Should gateway implement rate limiting for cloud APIs?
3. **Cost tracking:** Should gateway track token usage/costs per provider?
4. **Fallback logic:** If primary provider fails, automatically fallback to alternative?
5. **Request validation:** Validate requests before forwarding to backends?

These questions should be addressed during roadmap/implementation planning.

## Sources

### vLLM-MLX
- [vLLM-MLX GitHub Repository](https://github.com/waybarrios/vllm-mlx)
- [When Your Dev and Prod LLM Backends Don't Match](https://medium.com/@michael.hannecke/when-your-dev-and-prod-llm-backends-dont-match-and-why-that-s-okay-3bf2cb1c55c2)
- [A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp](https://arxiv.org/pdf/2511.05502)

### mlx-openai-server
- [mlx-openai-server GitHub Repository](https://github.com/cubist38/mlx-openai-server)
- [mlx-openai-server PyPI](https://pypi.org/project/mlx-openai-server/)

### API Gateway Patterns
- [API Gateway Pattern and FastAPI](https://medium.com/@saveriomazza/api-gateway-pattern-and-fastapi-fe8ea09e9620)
- [Complete Guide to Deploying FastAPI in Production](https://blog.greeden.me/en/2026/01/20/complete-guide-to-deploying-fastapi-in-production-reliable-operations-with-uvicorn-multi-workers-docker-and-a-reverse-proxy/)
- [Best way to make Async Requests with FastAPI](https://medium.com/@benshearlaw/how-to-use-httpx-request-client-with-fastapi-16255a9984a4)
- [fastapi-proxy-lib](https://github.com/WSH032/fastapi-proxy-lib)
- [Dependency Injection in FastAPI: 2026 Playbook](https://thelinuxcode.com/dependency-injection-in-fastapi-2026-playbook-for-modular-testable-apis/)

### Secure Key Storage
- [Best Practices for API Key Safety - OpenAI](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
- [API Key Management Best Practices - GitGuardian](https://blog.gitguardian.com/secrets-api-management/)
- [Securing Sensitive Data in Python](https://systemweakness.com/securing-sensitive-data-in-python-best-practices-for-storing-api-keys-and-credentials-2bee9ede57ee)
- [Basic Security Practices for SQLite](https://dev.to/stephenc222/basic-security-practices-for-sqlite-safeguarding-your-data-23lh)
- [Encrypted SQLite Databases with Python and SQLCipher](https://charlesleifer.com/blog/encrypted-sqlite-databases-with-python-and-sqlcipher/)

### Anthropic SDK
- [Anthropic Python SDK GitHub](https://github.com/anthropics/anthropic-sdk-python)
- [Anthropic Python SDK PyPI](https://pypi.org/project/anthropic/)
- [Claude API Client SDKs](https://docs.claude.com/en/api/client-sdks)

### Cryptography
- [Fernet (symmetric encryption) - Cryptography Docs](https://cryptography.io/en/latest/fernet/)
- [cryptography PyPI](https://pypi.org/project/cryptography/)

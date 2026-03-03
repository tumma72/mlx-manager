# MLX Server -- Operations & Configuration Reference

MLX Server is an OpenAI and Anthropic-compatible inference server for MLX models on Apple Silicon. It runs locally on your Mac, loading models on demand into a managed memory pool with LRU eviction, and serves them through standard API endpoints.

When used as part of MLX Manager, it runs embedded (mounted at `/v1`). It can also run standalone for dedicated inference workloads.

## Table of Contents

- [Configuration Reference](#configuration-reference)
- [Security](#security)
- [Monitoring and Observability](#monitoring-and-observability)
- [Error Handling](#error-handling)
- [Graceful Shutdown](#graceful-shutdown)
- [Model Loading Progress](#model-loading-progress)
- [API Compatibility](#api-compatibility)
- [Admin Endpoints](#admin-endpoints)

---

## Configuration Reference

All settings use the `MLX_SERVER_` environment variable prefix. Every setting has a safe default and is opt-in -- a zero-configuration deployment works out of the box.

Example `.env` file:

```bash
MLX_SERVER_ADMIN_TOKEN=my-secret-token
MLX_SERVER_RATE_LIMIT_RPM=60
MLX_SERVER_METRICS_ENABLED=true
MLX_SERVER_MAX_MODELS=2
```

### Server Binding

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_HOST` | string | `127.0.0.1` | Host to bind the server to. Localhost-only by default for security. |
| `MLX_SERVER_PORT` | int | `10242` | Port to bind the server to. |

### Model Pool

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_MAX_MEMORY_GB` | float | `0.0` | Maximum memory (GB) for the model pool. `0` = auto-detect 75% of device memory. |
| `MLX_SERVER_MAX_MODELS` | int | `4` | Maximum number of models kept loaded simultaneously. LRU eviction removes least-recently-used models when the limit is reached. |
| `MLX_SERVER_MAX_CACHE_SIZE_GB` | float | `8.0` | Maximum GPU cache size in GB. Range: 1.0--128.0. |
| `MLX_SERVER_AVAILABLE_MODELS` | JSON list | `["mlx-community/Llama-3.2-3B-Instruct-4bit"]` | List of model IDs the server is configured to serve. Models are loaded on demand when first requested. |
| `MLX_SERVER_DEFAULT_MODEL` | string | `null` | Default model ID when a request omits the `model` field. |

### Generation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_DEFAULT_MAX_TOKENS` | int | `4096` | Default maximum tokens for generation when not specified in the request. |

### Timeouts

Per-endpoint timeouts prevent runaway requests from consuming resources indefinitely.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_TIMEOUT_CHAT_SECONDS` | float | `900.0` | Timeout for `/v1/chat/completions` (15 minutes). |
| `MLX_SERVER_TIMEOUT_COMPLETIONS_SECONDS` | float | `600.0` | Timeout for `/v1/completions` (10 minutes). |
| `MLX_SERVER_TIMEOUT_EMBEDDINGS_SECONDS` | float | `120.0` | Timeout for `/v1/embeddings` (2 minutes). |

When a timeout is exceeded, the server returns HTTP 408 with a `TimeoutProblem` response body (see [Error Handling](#error-handling)).

For streaming requests, a timeout SSE error event is sent before closing the stream.

### Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_ADMIN_TOKEN` | string | `null` | Bearer token for `/v1/admin/*` endpoints. When unset, admin endpoints are open. |

### Rate Limiting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_RATE_LIMIT_RPM` | int | `0` | Requests per minute per IP address. `0` = disabled (no overhead). |

### Graceful Shutdown

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_DRAIN_TIMEOUT_SECONDS` | float | `30.0` | Seconds to wait for in-flight requests to complete during shutdown. Range: 1.0--300.0. |

### Observability

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_METRICS_ENABLED` | bool | `false` | Enable Prometheus metrics at `/v1/admin/metrics`. |
| `MLX_SERVER_LOGFIRE_ENABLED` | bool | `true` | Enable Pydantic LogFire instrumentation. Uses `if-token-present` mode so it works without a token during development. |
| `MLX_SERVER_LOGFIRE_TOKEN` | string | `null` | LogFire API token. Also accepts the standard `LOGFIRE_TOKEN` env var. |

### Audit Logging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_DATABASE_PATH` | string | `~/.mlx-manager/mlx-server.db` | Path to the audit log SQLite database. When embedded in MLX Manager, the shared `mlx-manager.db` is used instead. |
| `MLX_SERVER_AUDIT_RETENTION_DAYS` | int | `30` | Days to retain audit log entries before cleanup. Range: 1--365. |

Audit logging is privacy-first: only request metadata (model, endpoint, duration, token counts, status) is stored. No prompt or response content is logged.

### Input Validation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_MAX_IMAGE_SIZE_MB` | int | `20` | Maximum decoded base64 image size in MB. Range: 1--100. |

### Batching (Experimental)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_ENABLE_BATCHING` | bool | `false` | Enable continuous batching for text requests. Experimental -- falls back to direct inference if the scheduler is unavailable. |
| `MLX_SERVER_BATCH_BLOCK_POOL_SIZE` | int | `1000` | Number of KV cache blocks per model for batching. |
| `MLX_SERVER_BATCH_MAX_BATCH_SIZE` | int | `8` | Maximum concurrent requests per batch. |

### Other

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MLX_SERVER_ENVIRONMENT` | string | `development` | Environment name (`development` or `production`). |

---

## Security

### Admin Token Authentication

Admin endpoints under `/v1/admin/*` can be protected with a Bearer token:

```bash
export MLX_SERVER_ADMIN_TOKEN=my-secret-token
```

Clients must include the token in requests:

```bash
curl -H "Authorization: Bearer my-secret-token" \
  http://localhost:10242/v1/admin/models/status
```

When `MLX_SERVER_ADMIN_TOKEN` is not set, admin endpoints are open. This is appropriate for local development but should be changed for any shared or networked deployment.

### Rate Limiting

Enable per-IP rate limiting to prevent abuse:

```bash
export MLX_SERVER_RATE_LIMIT_RPM=60
```

The rate limiter uses a token bucket algorithm. Each IP gets a bucket with capacity equal to the RPM limit, refilling at RPM/60 tokens per second. When a request is rate-limited, the server returns HTTP 429 with:

- `Retry-After` header indicating seconds until the next token is available
- `X-RateLimit-Limit` header with the configured RPM
- `X-RateLimit-Remaining` header with approximate remaining tokens

Successful responses also include `X-RateLimit-Limit` and `X-RateLimit-Remaining` headers.

Stale per-IP buckets (inactive for 2+ minutes) are automatically cleaned up to prevent memory growth.

### Input Validation

- **Image size limits**: Decoded base64 images are validated against `MLX_SERVER_MAX_IMAGE_SIZE_MB` (default 20 MB).
- **Request body validation**: All request bodies are validated using Pydantic models with strict type checking and field constraints.
- **Path traversal protection**: Model IDs and file paths are validated to prevent directory traversal attacks.

### Network Binding

By default, the server binds to `127.0.0.1` (localhost only). To expose the server on a network, set `MLX_SERVER_HOST=0.0.0.0`, but always combine this with an admin token and rate limiting.

---

## Monitoring and Observability

### Prometheus Metrics

Enable the metrics endpoint:

```bash
export MLX_SERVER_METRICS_ENABLED=true
```

Metrics are served at `GET /v1/admin/metrics` in Prometheus text exposition format. If an admin token is configured, the metrics endpoint requires authentication.

**Available metrics:**

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `mlx_request_latency_seconds` | Histogram | Request latency in seconds | -- |
| `mlx_active_requests` | Gauge | Number of currently active requests | -- |
| `mlx_token_throughput_total` | Counter | Total tokens generated | -- |
| `mlx_model_load_duration_seconds` | Histogram | Model loading duration in seconds | -- |
| `mlx_model_memory_bytes` | Gauge | Memory used by loaded models | -- |
| `mlx_pool_cache_hits_total` | Counter | Model pool cache hits (model already loaded) | -- |
| `mlx_pool_cache_misses_total` | Counter | Model pool cache misses (model needs loading) | -- |

Histogram bucket boundaries for request latency: 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s, 60s.

Histogram bucket boundaries for model load duration: 100ms, 500ms, 1s, 2.5s, 5s, 10s, 30s, 60s, 120s.

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: mlx-server
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:10242']
    metrics_path: /v1/admin/metrics
    # If admin_token is set:
    authorization:
      type: Bearer
      credentials: my-secret-token
```

### Request ID Propagation

Every request gets a unique ID in the format `req_{12-hex-chars}` (e.g., `req_a1b2c3d4e5f6`).

- **Client-provided**: Send an `X-Request-ID` header and it will be propagated through the system.
- **Auto-generated**: If no header is provided, a UUID-based ID is generated.
- **Response header**: Every response includes `X-Request-ID` for correlation.
- **Error responses**: All RFC 7807 error bodies include the `request_id` field for log correlation.

### LogFire Integration

LogFire is enabled by default in `if-token-present` mode, meaning it works silently without a token during development and activates when a token is provided.

```bash
export MLX_SERVER_LOGFIRE_TOKEN=your-logfire-token
# or use the standard env var:
export LOGFIRE_TOKEN=your-logfire-token
```

LogFire instruments:
- FastAPI request/response tracing
- httpx HTTP client calls
- OpenAI/Anthropic client calls (when available)

### Health Endpoint

```
GET /health
```

Returns the server's health status.

**Normal operation:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

**During shutdown drain:**
```json
HTTP 503
{
  "status": "draining",
  "version": "0.1.0",
  "active_requests": 3
}
```

The health endpoint is always accessible, even during graceful shutdown, so load balancers can detect the draining state.

---

## Error Handling

All API errors are returned in [RFC 7807 Problem Details](https://datatracker.ietf.org/doc/html/rfc7807) format:

```json
{
  "type": "https://mlx-manager.dev/errors/not-found",
  "title": "Not Found",
  "status": 404,
  "detail": "Model 'nonexistent-model' not found.",
  "instance": "/v1/chat/completions",
  "request_id": "req_a1b2c3d4e5f6",
  "error_code": "resource_not_found"
}
```

### Response Fields

| Field | Description |
|-------|-------------|
| `type` | URI identifying the problem type (stable, suitable for programmatic matching) |
| `title` | Human-readable summary of the problem class |
| `status` | HTTP status code |
| `detail` | Human-readable explanation specific to this occurrence |
| `instance` | The request path where the error occurred |
| `request_id` | Unique ID for log correlation |
| `error_code` | Machine-readable code for client-side error handling |
| `errors` | Array of field-level validation errors (422 responses only) |

### Error Codes

The `error_code` field provides stable, machine-readable codes for client error handling:

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `validation_error` | 422 | Request validation failed |
| `field_max_length_exceeded` | 422 | A field exceeded its maximum length |
| `invalid_request` | 400 | Malformed request |
| `model_not_found` | 404 | Requested model not found in available models |
| `resource_not_found` | 404 | Generic resource not found |
| `unauthorized` | 401 | Missing or invalid authentication |
| `forbidden` | 403 | Valid authentication but insufficient permissions |
| `rate_limited` | 429 | Rate limit exceeded |
| `request_timeout` | 408 | Request exceeded configured timeout |
| `internal_error` | 500 | Unexpected server error |
| `service_unavailable` | 503 | Server shutting down or otherwise unavailable |
| `inference_error` | 500 | Error during model inference |
| `generation_error` | 500 | Error during text generation |

### Timeout Errors

Timeout responses include the configured timeout value:

```json
{
  "type": "https://mlx-manager.dev/errors/timeout",
  "title": "Request Timeout",
  "status": 408,
  "detail": "Request timed out after 900.0 seconds",
  "instance": "/v1/chat/completions",
  "request_id": "req_a1b2c3d4e5f6",
  "error_code": "request_timeout",
  "timeout_seconds": 900.0
}
```

### Validation Errors

Validation failures include per-field details:

```json
{
  "type": "https://mlx-manager.dev/errors/validation-error",
  "title": "Validation Error",
  "status": 422,
  "detail": "Request validation failed with 1 error(s)",
  "request_id": "req_a1b2c3d4e5f6",
  "error_code": "validation_error",
  "errors": [
    {
      "field": "body.model",
      "message": "Field required",
      "type": "missing"
    }
  ]
}
```

---

## Graceful Shutdown

When the server receives SIGTERM:

1. **Drain starts**: The shutdown middleware begins rejecting new requests with HTTP 503 and a `Retry-After: 5` header.
2. **Health endpoint stays up**: `GET /health` continues responding (with status `"draining"` and HTTP 503) so load balancers can detect the state.
3. **In-flight requests complete**: Existing requests are allowed to finish within the drain timeout.
4. **Drain timeout**: If requests do not complete within `MLX_SERVER_DRAIN_TIMEOUT_SECONDS` (default 30s), the server proceeds with shutdown regardless.
5. **Cleanup**: The model pool is cleaned up and resources are released.

```
                      SIGTERM
                         |
                         v
              +---------------------+
              |   Start drain       |
              |   New requests: 503 |
              +---------------------+
                         |
              +---------------------+
              |  Wait for in-flight |
              |  (up to 30s)        |
              +---------------------+
                    |           |
                 drained     timeout
                    |           |
              +---------------------+
              |  Cleanup & exit     |
              +---------------------+
```

---

## Model Loading Progress

Subscribe to real-time model loading progress via Server-Sent Events:

```
GET /v1/admin/models/{model_id}/loading-progress
```

If an admin token is configured, this endpoint requires authentication.

### Event Types

| Event | Description |
|-------|-------------|
| `download_progress` | Model files being downloaded. Includes `progress` (0--100). |
| `weights_loading` | Model weights being loaded into memory. |
| `adapter_init` | Model adapter being initialized (family detection, parser setup). |
| `ready` | Model is fully loaded and ready for inference. Terminal event. |
| `error` | Loading failed. Terminal event. |

### Event Format

Each SSE event has the following JSON payload:

```json
{
  "event": "weights_loading",
  "model_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "progress": 45.0,
  "message": "Loading weights...",
  "timestamp": 1709123456.789
}
```

### Behavior

- **Late joiners** receive the most recent event on connect for catch-up.
- **Terminal events** (`ready`, `error`) automatically close the stream.
- **Connection timeout**: Streams are closed after 5 minutes to prevent orphaned connections.

### Example

```bash
curl -N http://localhost:10242/v1/admin/models/mlx-community/Llama-3.2-3B-Instruct-4bit/loading-progress
```

---

## API Compatibility

MLX Server implements two API protocols, both pointing at the same local inference engine.

### OpenAI-Compatible Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming and non-streaming) |
| POST | `/v1/completions` | Legacy text completions (streaming and non-streaming) |
| POST | `/v1/embeddings` | Text embeddings |
| POST | `/v1/audio/speech` | Text-to-speech |
| POST | `/v1/audio/transcriptions` | Speech-to-text |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{model_id}` | Get model info |

These endpoints follow the [OpenAI API reference](https://platform.openai.com/docs/api-reference) format. You can use them with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:10242/v1",
    api_key="not-needed",  # no auth on inference endpoints
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Supported features:
- Streaming (`stream: true`) with SSE
- Tool/function calling
- Structured output (JSON mode with schema validation)
- Vision/multimodal inputs (images via base64 or URL)
- Thinking/reasoning model support
- Temperature, top_p, max_tokens, stop sequences, and other sampling parameters

### Anthropic-Compatible Endpoint

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/messages` | Anthropic Messages API |

This endpoint accepts [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) format requests and returns Anthropic-format responses:

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:10242/v1",
    api_key="not-needed",
)

message = client.messages.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

Supported features:
- Streaming (`stream: true`) with SSE
- Tool use
- System messages (separate field, per Anthropic spec)
- Vision/multimodal inputs

---

## Admin Endpoints

All admin endpoints are under `/v1/admin/` and can be protected with `MLX_SERVER_ADMIN_TOKEN`.

### Model Pool Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/admin/models/status` | Get loaded models, memory usage, and pool config |
| POST | `/v1/admin/models/load/{model_id}` | Preload a model (protected from LRU eviction) |
| POST | `/v1/admin/models/unload/{model_id}` | Unload a model to free memory |
| GET | `/v1/admin/models/{model_id}/loading-progress` | SSE stream of model loading progress |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/admin/health` | Admin health check |
| GET | `/v1/admin/metrics` | Prometheus metrics (requires `metrics_enabled=true`) |

### Audit Logs

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/admin/audit-logs` | Query audit logs with filtering and pagination |
| GET | `/v1/admin/audit-logs/stats` | Aggregate statistics (counts by status, backend, unique models) |
| GET | `/v1/admin/audit-logs/export` | Export logs in JSONL or CSV format |
| WS | `/v1/admin/ws/audit-logs` | WebSocket for real-time audit log streaming |

**Audit log query parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Filter by model name |
| `backend_type` | string | Filter by backend (`local`, `openai`, `anthropic`) |
| `status` | string | Filter by status (`success`, `error`, `timeout`) |
| `start_time` | datetime | Start of time range |
| `end_time` | datetime | End of time range |
| `limit` | int | Max results (default 100, max 1000) |
| `offset` | int | Pagination offset |

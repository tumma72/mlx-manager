# Pitfalls Research: Unified API Gateway

**Project:** MLX Model Manager v1.2 - Unified API Gateway
**Researched:** 2026-01-26
**Context:** Adding API gateway/proxy features to existing mlx-manager (subprocess-based LLM server management)

## Critical Pitfalls

### Pitfall 1: Orphaned Subprocess Cleanup on Gateway Crash

**What goes wrong:** When the gateway process crashes or is terminated ungracefully, mlx-openai-server subprocesses become orphaned and continue consuming memory/GPU resources.

**Why it happens:** Python's subprocess module doesn't automatically kill child processes when the parent exits. The existing `ServerManager.cleanup()` only runs during graceful shutdown via FastAPI's lifespan handler.

**Consequences:**
- Orphaned processes consume system resources indefinitely
- Users can't reclaim resources without manual `kill` commands
- Multiple orphaned instances can exhaust available memory/ports
- Process tracking becomes inconsistent across restarts

**Prevention:**
1. **Subprocess group management**: Start subprocesses with `start_new_session=True` on Unix to create a new process group, then kill entire group on cleanup
2. **Heartbeat monitoring**: Implement parent process ID (ppid) monitoring in subprocesses using libraries like `python-orphanage` to auto-terminate when parent dies
3. **PID file tracking**: Write subprocess PIDs to files in `~/.mlx-manager/pids/` so recovery scripts can clean up orphans
4. **Startup orphan detection**: On gateway startup, scan for existing mlx-openai-server processes and either adopt or kill them

**Detection:**
- `ps aux | grep mlx-openai-server` shows processes without corresponding ServerManager entries
- Memory usage continues growing after gateway restart
- Port conflicts occur when trying to start "new" servers

**Phase assignment:** Phase 1 (Core Gateway Infrastructure) - must be solved before multi-backend routing

**Sources:**
- [Ungraceful shutdown of python script can leave orphan processes](https://github.com/facebookresearch/CompilerGym/issues/326)
- [python-orphanage library](https://github.com/tonyseek/python-orphanage)
- [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html)

---

### Pitfall 2: Race Conditions in On-Demand Model Loading

**What goes wrong:** Multiple concurrent requests to the same model trigger duplicate startup attempts, causing crashes, port conflicts, or wasted resources loading the same model multiple times.

**Why it happens:**
- Request arrives → Gateway checks if server running → Not running → Start server → Send request
- Second request arrives before first server finishes starting → Same logic triggers duplicate start
- The existing `ServerManager.start_server()` checks `if profile.id in self.processes` but this has a TOCTOU (time-of-check-time-of-use) vulnerability

**Consequences:**
- Port conflicts when second instance tries to bind same port
- Memory exhaustion loading same model twice
- Request timeouts while duplicate startups fail and retry
- Database inconsistencies with multiple `RunningInstance` records

**Prevention:**
1. **Startup state machine**: Add "starting" state between "stopped" and "running" with per-profile locks
   ```python
   self._startup_locks: dict[int, asyncio.Lock] = {}
   async with self._get_or_create_lock(profile_id):
       if self.is_running(profile_id):
           return  # Already started by another request
       await self._do_start(profile_id)
   ```

2. **Request queuing**: Queue requests during startup and flush queue once server becomes healthy
   ```python
   self._pending_requests: dict[int, list[Request]] = {}
   ```

3. **Startup timeout with cleanup**: If server doesn't become healthy within timeout (e.g., 60s), kill the process and fail all queued requests

4. **Atomic database operations**: Use database transactions with `SELECT FOR UPDATE` to claim exclusive startup rights

**Detection:**
- Multiple "Starting server for profile..." log entries for same profile_id within 10 seconds
- Port binding errors in server logs
- Duplicate PID entries in `processes` dict (impossible but indicates logic race)
- Frontend shows "Starting..." indefinitely

**Phase assignment:** Phase 2 (On-Demand Model Loading) - core requirement

**Sources:**
- [vLLM race condition in request ID recycling](https://github.com/vllm-project/vllm/issues/23697)
- [vLLM V1 engine crashes with concurrent requests](https://github.com/vllm-project/vllm/issues/25991)

---

### Pitfall 3: Streaming Timeout Cascades

**What goes wrong:** Long-running LLM streaming requests get terminated by intermediate timeouts in the proxy layer, load balancers, or client libraries, causing incomplete responses and poor user experience.

**Why it happens:** Multiple timeout layers interact:
- FastAPI default timeout: None (but underlying httpx client has timeouts)
- Nginx/reverse proxy default: 60s idle timeout
- AWS API Gateway: 30s (edge-optimized) or 5min (regional) idle timeout
- Browser fetch() default: varies by browser
- LLM generation can take 30s+ for long outputs

**Consequences:**
- Users see partial responses with no error indication
- Frontend thinks request succeeded but received truncated data
- Retry loops amplify the problem (each retry times out)
- No way to distinguish "timeout" from "generation complete"

**Prevention:**
1. **SSE heartbeats**: Send periodic comment frames during idle periods
   ```python
   # In proxy streaming logic
   last_chunk_time = time.time()
   async for chunk in backend_stream:
       yield chunk
       last_chunk_time = time.time()

   # Separate heartbeat task
   if time.time() - last_chunk_time > 15:
       yield ": heartbeat\n\n"
   ```

2. **Explicit timeout configuration hierarchy**:
   - Gateway → Backend: 15min timeout for generation
   - Gateway → Client: No timeout (streaming responses)
   - Client: User-configurable timeout with default 10min

3. **Timeout headers**: Include `X-Timeout-Strategy: stream-keep-alive` in responses to inform proxies

4. **Graceful degradation**: Return partial results with `X-Generation-Incomplete: timeout` header

**Detection:**
- Streaming responses cut off mid-sentence
- Client-side errors like "net::ERR_EMPTY_RESPONSE"
- Backend logs show completion but frontend received partial data
- Higher error rates around 30s, 60s, 300s mark (common timeout values)

**Phase assignment:** Phase 3 (Streaming Optimization) - after basic proxy works

**Sources:**
- [FastAPI SSE proxy pitfalls](https://github.com/fastapi/fastapi/discussions/10701)
- [AWS API Gateway streaming timeouts](https://aws.amazon.com/blogs/compute/building-responsive-apis-with-amazon-api-gateway-response-streaming/)
- [Nginx SSE buffering issues](https://medium.com/@bhagyarana80/fastapi-streaming-responses-real-time-without-websockets-bc6b071f5d9e)

---

### Pitfall 4: API Key Leakage in Logs and Errors

**What goes wrong:** API keys for OpenAI/Anthropic appear in application logs, error messages, database dumps, or request traces, exposing credentials to anyone with log access.

**Why it happens:**
- Default logging includes request/response bodies
- Exception messages include full request context
- Database queries logged by ORMs include credential values
- Frontend error reporting sends full API response

**Consequences:**
- Compromised API keys lead to unauthorized usage and billing
- Compliance violations (PCI DSS, SOC 2, GDPR)
- Keys committed to version control via log files
- Support tickets include sensitive credentials in error traces

**Prevention:**
1. **Structured logging with redaction**:
   ```python
   import logging
   from logging import Filter

   class RedactAPIKeysFilter(Filter):
       def filter(self, record):
           if hasattr(record, 'msg'):
               record.msg = re.sub(r'(api[_-]?key["\s:=]+)[\w-]+', r'\1***REDACTED***', str(record.msg))
           return True

   logger.addFilter(RedactAPIKeysFilter())
   ```

2. **Environment variable validation**: Never log env vars directly; use `***SET***` or `***UNSET***` placeholders

3. **Database encryption at rest**: Store API keys encrypted with user-specific keys, not application-wide keys

4. **Masked error responses**: Strip sensitive fields from error responses:
   ```python
   SENSITIVE_FIELDS = {"api_key", "authorization", "x-api-key"}
   def sanitize_error(error_data: dict) -> dict:
       return {k: "***" if k.lower() in SENSITIVE_FIELDS else v
               for k, v in error_data.items()}
   ```

5. **Pre-commit hooks**: Add git hooks to scan for API key patterns before commit

**Detection:**
- Search logs for `sk-`, `Bearer`, `Authorization:` patterns
- Review error monitoring dashboards (Sentry, DataDog) for credential exposure
- Check database dumps for plaintext API keys
- Audit git history for accidentally committed credentials

**Phase assignment:** Phase 1 (Core Gateway) - security must be foundational

**Sources:**
- [FastAPI security best practices](https://escape.tech/blog/how-to-secure-fastapi-api/)
- [Secure credential storage with keyring](https://medium.com/@forsytheryan/securely-storing-credentials-in-python-with-keyring-d8972c3bd25f)

---

### Pitfall 5: Model Name Routing Ambiguity

**What goes wrong:** Multiple backends support models with similar names (e.g., "gpt-4", "gpt-4-turbo"), causing requests to route to wrong backend or fail with "model not found" despite the model being available.

**Why it happens:**
- No canonical model registry mapping names to backends
- Model aliases and versions create namespace collisions
- User configuration uses informal names ("gpt4" vs "gpt-4")
- Backend capabilities change over time (new models added)

**Consequences:**
- Requests fail with "model not found" despite model being available
- Requests route to wrong backend (local vs cloud)
- Cost optimization fails (expensive backend used instead of cheap local)
- Poor user experience with unclear error messages

**Prevention:**
1. **Explicit routing configuration**: Require users to map model names to backends
   ```json
   {
     "models": {
       "gpt-4-turbo": {"backend": "openai", "model_id": "gpt-4-turbo-2024-04-09"},
       "gpt-4-local": {"backend": "mlx", "profile_id": 42},
       "claude-3-opus": {"backend": "anthropic", "model_id": "claude-3-opus-20240229"}
     }
   }
   ```

2. **Prefix-based namespacing**: Reserve prefixes for backends
   - `mlx/*` → Local MLX models
   - `openai/*` → OpenAI cloud
   - `anthropic/*` → Anthropic cloud
   - `vllm/*` → vLLM-MLX servers

3. **Fallback chains**: Allow ordered backend attempts
   ```json
   "gpt-4": ["mlx/mlx-community/gpt-4", "openai/gpt-4-turbo"]
   ```

4. **Model registry endpoint**: `/v1/gateway/models` returns all available models with backend info

**Detection:**
- 404 "model not found" errors despite model being configured
- Unexpected backend usage in cost tracking
- User reports "I have that model locally but it's using the API"

**Phase assignment:** Phase 2 (Model Routing Logic)

**Sources:**
- [LLM gateway routing challenges](https://dev.to/palapalapala/making-ai-agent-configurations-stable-with-an-llm-gateway-2jf1)
- [LiteLLM proxy configuration](https://docs.litellm.ai/docs/proxy/timeout)

---

## Integration Pitfalls

### Pitfall 6: Existing ServerManager Lifecycle Interference

**What goes wrong:** The new gateway's model auto-start logic conflicts with existing profile-based server management, causing servers to start/stop unexpectedly or status to become inconsistent.

**Why it happens:**
- Existing code assumes one server per profile, managed explicitly via UI
- Gateway assumes servers are ephemeral, started on-demand
- Health checker polls servers but doesn't know about gateway's auto-stop
- Frontend UI shows server as "running" but gateway stopped it due to inactivity

**Consequences:**
- User starts server via UI, gateway stops it, user confused
- Gateway auto-starts server, interferes with manual server management
- Database `RunningInstance` records become stale
- Resource accounting broken (can't track what gateway vs user started)

**Prevention:**
1. **Server ownership tagging**: Add `started_by` field to `RunningInstance`
   ```python
   class RunningInstance(SQLModel, table=True):
       started_by: str  # "user" | "gateway" | "launchd"
       auto_stop_eligible: bool = False  # Gateway can stop these
   ```

2. **Mutually exclusive modes**: Gateway management disables manual start/stop UI
   - Config option: `gateway_mode: bool` (default: False)
   - When True, hide Start/Stop buttons in UI
   - Show "Managed by Gateway" status instead

3. **Hybrid mode with clear ownership**: Allow both but require explicit handoff
   - Manual servers: Never auto-stop, excluded from gateway routing
   - Gateway servers: Auto-start/stop, hidden from server panel
   - "Promote to Manual" action transfers ownership

4. **Separate port ranges**: Gateway uses different port range (10300-10399) vs manual (10240-10299)

**Detection:**
- User reports "server keeps stopping by itself"
- `RunningInstance` records without corresponding processes
- Port conflicts between gateway and manual servers
- Health checker shows inconsistent status

**Phase assignment:** Phase 1 (Architecture Design) - must decide governance model early

---

### Pitfall 7: Background Task Cleanup on Shutdown

**What goes wrong:** FastAPI's `BackgroundTasks` doesn't track long-running operations (model downloads, server startups), causing incomplete cleanup on shutdown and orphaned tasks.

**Why it happens:**
- `BackgroundTasks` has no cancellation mechanism
- Once response is sent, task runs independently
- Shutdown sequence (lifespan handler) doesn't wait for background tasks
- No task registry to track active operations

**Current code already handles this well:**
- `_download_tasks` list tracks running downloads
- `cancel_download_tasks()` in lifespan shutdown cancels them
- But gateway adds more background operations: health checks, auto-stop timers, request queuing

**Consequences:**
- Downloads continue after shutdown, corrupting state
- Model startups complete after gateway stopped, orphaned processes
- Database writes fail due to closed connection
- Resource leaks (open file handles, network connections)

**Prevention:**
1. **Central task registry** (extend existing pattern):
   ```python
   class GatewayTaskManager:
       def __init__(self):
           self.tasks: dict[str, asyncio.Task] = {}

       def create_task(self, name: str, coro):
           task = asyncio.create_task(coro)
           self.tasks[name] = task
           task.add_done_callback(lambda t: self.tasks.pop(name, None))
           return task

       async def cancel_all(self):
           for task in list(self.tasks.values()):
               task.cancel()
           await asyncio.gather(*self.tasks.values(), return_exceptions=True)
   ```

2. **Graceful cancellation handlers**:
   ```python
   async def auto_stop_timer(profile_id: int):
       try:
           await asyncio.sleep(idle_timeout)
           await stop_server(profile_id)
       except asyncio.CancelledError:
           logger.info(f"Auto-stop cancelled for {profile_id}")
           raise  # Re-raise to properly mark as cancelled
   ```

3. **Shutdown timeout**: Give tasks grace period (30s) then force-kill
   ```python
   try:
       await asyncio.wait_for(task_manager.cancel_all(), timeout=30)
   except asyncio.TimeoutError:
       logger.warning("Tasks didn't cancel in time, forcing shutdown")
   ```

**Detection:**
- Logs show "Download failed" after shutdown
- Database integrity errors on restart
- Open file descriptors (`lsof`) show leaked handles
- Tasks don't respect `asyncio.CancelledError`

**Phase assignment:** Phase 1 (Core Gateway) - must extend existing cleanup patterns

**Sources:**
- [FastAPI lifespan events best practices](https://craftyourstartup.com/cys-docs/fastapi-startup-and-shutdown-events-guide/)
- [Understanding FastAPI lifespan shutdown](https://dev.turmansolutions.ai/2025/09/27/understanding-fastapis-lifespan-events-proper-initialization-and-shutdown/)

---

### Pitfall 8: Health Check Storm During Auto-Start

**What goes wrong:** Existing `HealthChecker` continuously polls servers, but during gateway auto-start, this creates request storms that slow down model loading and can cause startup failures.

**Why it happens:**
- `HealthChecker` polls every 5 seconds (configurable)
- Model loading takes 10-60s depending on size
- Each health check hits `/v1/models` endpoint
- Concurrent health checks + queued user requests overwhelm starting server

**Consequences:**
- Model loading times increase 2-3x due to request overhead
- Server OOM errors during startup from concurrent requests
- Health checks time out, marked as "unhealthy" despite actually starting
- Cascading retries amplify the problem

**Prevention:**
1. **Startup grace period**: Disable health checks for 60s after start
   ```python
   self._startup_grace: dict[int, float] = {}  # profile_id → timestamp

   async def check_health(self, profile_id: int):
       grace_until = self._startup_grace.get(profile_id, 0)
       if time.time() < grace_until:
           return HealthCheckResult(status="starting", skipped=True)
   ```

2. **Exponential backoff**: Increase poll interval during startup phase
   - Normal: 5s
   - Starting (first 60s): 15s
   - Failing: 30s

3. **Request pause during startup**: Queue incoming requests until first successful health check
   ```python
   if status == "starting":
       await wait_for_healthy(profile_id, timeout=60)
   ```

4. **Lightweight health endpoint**: Use `/health` instead of `/v1/models` if backend supports it (reduces overhead)

**Detection:**
- Health check logs show rapid "unhealthy" → "healthy" transitions
- Model loading logs show timeouts during startup
- Memory spikes during server start
- Startup time varies wildly (10s vs 90s for same model)

**Phase assignment:** Phase 2 (On-Demand Loading) - must coordinate with existing health checker

---

## API Compatibility Pitfalls

### Pitfall 9: OpenAI vs Anthropic Format Mismatches

**What goes wrong:** Clients send OpenAI-format requests, but gateway forwards them to Anthropic backend unchanged, causing 400 errors or semantic mismatches.

**Key differences that break compatibility:**

| Feature | OpenAI | Anthropic | Issue |
|---------|--------|-----------|-------|
| System messages | Any position in messages array | Only initial message, single string | Multi-system messages break |
| Function calling | `tools` array, `strict: true` for schema enforcement | `tools` array, but `strict` ignored | Schema validation not guaranteed |
| Streaming | `stream: true`, `data: [DONE]` terminator | `stream: true`, no `[DONE]` marker | Client libraries expect `[DONE]` |
| Audio input | Supported in vision models | Not supported, silently ignored | Feature degradation |
| Prompt caching | Not in OpenAI format | `cache_control` in messages | Cost optimization broken |
| Thinking/reasoning | `o1` models return `reasoning_content` | Claude returns thinking in separate field | Response structure differs |

**Consequences:**
- Requests fail with cryptic 400 errors
- Features silently degrade (caching disabled, strict validation skipped)
- Client libraries break on missing `[DONE]` terminator
- Costs increase due to disabled prompt caching
- Extended thinking/reasoning output lost

**Prevention:**
1. **Request transformation layer**:
   ```python
   class AnthropicAdapter:
       def transform_request(self, openai_request: dict) -> dict:
           # Concatenate multiple system messages
           system_messages = [m for m in messages if m["role"] == "system"]
           system = "\n".join(m["content"] for m in system_messages)

           # Remove system messages from array
           messages = [m for m in messages if m["role"] != "system"]

           return {"system": system, "messages": messages, ...}
   ```

2. **Response transformation layer**:
   ```python
   async def transform_anthropic_stream(stream):
       async for chunk in stream:
           # Transform Anthropic SSE format to OpenAI format
           yield transform_chunk(chunk)

       # Add OpenAI's [DONE] terminator
       yield "data: [DONE]\n\n"
   ```

3. **Feature compatibility matrix**: Document which features work with which backends
   ```python
   FEATURE_SUPPORT = {
       "anthropic": {
           "streaming": True,
           "function_calling": True,
           "strict_schema": False,  # Ignored
           "audio_input": False,
           "prompt_caching": True,  # But needs transformation
       }
   }
   ```

4. **Capability-based routing**: Refuse to route unsupported features
   ```python
   if request.get("audio") and backend == "anthropic":
       raise UnsupportedFeatureError("Audio input not supported by Anthropic")
   ```

**Detection:**
- 400 errors with "multiple system messages not supported"
- Client libraries hang waiting for `[DONE]`
- Streaming responses format differently between backends
- Higher costs than expected (caching disabled)

**Phase assignment:** Phase 3 (Multi-Backend Compatibility)

**Sources:**
- [Anthropic OpenAI SDK compatibility layer](https://docs.anthropic.com/en/api/openai-sdk)
- [Anthropic extended thinking with tool calling](https://github.com/RooCodeInc/Roo-Code/discussions/1882)
- [LiteLLM reasoning content handling](https://docs.litellm.ai/docs/reasoning_content)

---

### Pitfall 10: Streaming Response Buffering

**What goes wrong:** Proxy buffers entire streaming response before forwarding, defeating the purpose of streaming and causing timeouts for long responses.

**Why it happens:**
- Default HTTP proxies buffer responses to calculate `Content-Length`
- Web frameworks collect response body before sending headers
- Middleware intercepts streaming responses for logging/metrics
- Load balancers buffer to enable retries

**Consequences:**
- First token delay increases from 200ms to 30s+
- Large responses cause memory exhaustion
- Timeouts occur before any data reaches client
- Poor user experience (no progressive rendering)

**Prevention:**
1. **Explicit streaming response type**:
   ```python
   from fastapi.responses import StreamingResponse

   async def proxy_chat(request: ChatRequest):
       async def stream_generator():
           async with httpx.AsyncClient() as client:
               async with client.stream("POST", backend_url, json=request) as resp:
                   async for chunk in resp.aiter_bytes():
                       yield chunk

       return StreamingResponse(stream_generator(), media_type="text/event-stream")
   ```

2. **Disable buffering middleware**: Exclude streaming endpoints from middleware
   ```python
   @app.middleware("http")
   async def log_requests(request: Request, call_next):
       if request.url.path.startswith("/v1/chat"):
           return await call_next(request)  # Skip logging
       # ... normal logging
   ```

3. **Nginx configuration** (if deployed behind nginx):
   ```nginx
   location /v1/chat {
       proxy_buffering off;
       proxy_cache off;
       proxy_read_timeout 15m;
       chunked_transfer_encoding on;
   }
   ```

4. **Client-side streaming detection**:
   ```python
   if request.stream:
       return StreamingResponse(...)
   else:
       # Collect full response for non-streaming
       return JSONResponse(...)
   ```

**Detection:**
- Network tab shows no data for 30s, then entire response at once
- Memory usage spikes during streaming requests
- Response headers include `Content-Length` instead of `Transfer-Encoding: chunked`
- First token latency matches total generation time

**Phase assignment:** Phase 3 (Streaming Optimization)

**Sources:**
- [How streaming LLM APIs work](https://til.simonwillison.net/llms/streaming-llm-apis)
- [KrakenD streaming limitations](https://www.krakend.io/docs/enterprise/endpoints/streaming/)

---

## Security Pitfalls

### Pitfall 11: Insecure Credential Storage on macOS

**What goes wrong:** API keys stored in plaintext SQLite database or config files, accessible to any process running as the user, violating principle of least privilege.

**Why it happens:**
- Convenience: easiest to store credentials directly in database
- Existing code stores profile configs in SQLite
- Developers forget that `~/.mlx-manager/` is not access-controlled
- Environment variables seem secure but are easily leaked

**Consequences:**
- Malware can read API keys from database
- Backup files include plaintext credentials
- Log files leak credentials via SQL queries
- Stolen laptop = stolen API keys

**Prevention:**
1. **macOS Keychain integration** (recommended for desktop app):
   ```python
   import keyring

   # Store API key
   keyring.set_password("mlx-manager", f"backend.{backend_id}.api_key", api_key)

   # Retrieve API key
   api_key = keyring.get_password("mlx-manager", f"backend.{backend_id}.api_key")
   ```

2. **Encrypted database fields**: Use SQLAlchemy/SQLModel encrypted types
   ```python
   from sqlalchemy_utils import EncryptedType
   from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

   class Backend(SQLModel, table=True):
       api_key: str | None = Field(
           sa_type=EncryptedType(str, get_encryption_key, AesEngine, 'pkcs5')
       )
   ```

3. **User confirmation prompts**: macOS Keychain prompts user when app accesses keys
   - Better security: user knows when credentials accessed
   - Better UX than typing API key every time

4. **Fallback chain**: Try Keychain → Env var → Prompt user
   ```python
   def get_api_key(backend_id: str) -> str:
       # Try keychain first
       key = keyring.get_password("mlx-manager", f"backend.{backend_id}.api_key")
       if key:
           return key

       # Try environment variable
       key = os.getenv(f"MLX_{backend_id.upper()}_API_KEY")
       if key:
           return key

       # Prompt user (for CLI usage)
       return input(f"Enter API key for {backend_id}: ")
   ```

**Detection:**
- Search database with `strings mlx-manager.db | grep sk-`
- Check if API keys visible in process environment (`ps e`)
- Review backup files for plaintext credentials
- Audit logging for credential exposure

**Phase assignment:** Phase 1 (Core Gateway) - security requirement

**Sources:**
- [Securely storing credentials with Python keyring](https://medium.com/@forsytheryan/securely-storing-credentials-in-python-with-keyring-d8972c3bd25f)
- [Python keyring comprehensive guide](https://coderivers.org/blog/keyring-python/)
- [macOS Keychain security benefits](https://medium.com/jungletronics/how-to-securely-save-credentials-in-python-dd5c6983741a)

---

### Pitfall 12: SSRF Vulnerabilities in Backend URL Configuration

**What goes wrong:** Users configure custom backend URLs, but gateway doesn't validate them, allowing Server-Side Request Forgery (SSRF) attacks to internal services.

**Why it happens:**
- Gateway accepts user-provided backend URLs (e.g., custom vLLM-MLX endpoint)
- No validation on URL scheme, host, or port
- Proxy blindly forwards requests to configured URL
- Attackers can target internal services (cloud metadata, internal APIs)

**Attack scenarios:**
1. **Cloud metadata access**: User sets backend URL to `http://169.254.169.254/latest/meta-data/` (AWS metadata service)
2. **Internal service scanning**: Probe internal network by setting URLs like `http://192.168.1.1:80`
3. **Bypass authentication**: Access internal admin panels not exposed to internet

**Consequences:**
- Leaked cloud credentials (IAM roles, API keys)
- Internal network reconnaissance
- Unauthorized access to internal services
- Data exfiltration via side channels

**Prevention:**
1. **URL allowlist approach**:
   ```python
   ALLOWED_BACKEND_PATTERNS = [
       r"^http://127\.0\.0\.1:\d+$",  # Localhost only
       r"^http://localhost:\d+$",
       r"^https://api\.openai\.com/.*$",  # Official APIs only
       r"^https://api\.anthropic\.com/.*$",
   ]

   def validate_backend_url(url: str) -> bool:
       return any(re.match(pattern, url) for pattern in ALLOWED_BACKEND_PATTERNS)
   ```

2. **Blocklist for dangerous ranges**:
   ```python
   BLOCKED_CIDRS = [
       ipaddress.IPv4Network("10.0.0.0/8"),      # Private
       ipaddress.IPv4Network("172.16.0.0/12"),   # Private
       ipaddress.IPv4Network("192.168.0.0/16"),  # Private
       ipaddress.IPv4Network("169.254.0.0/16"),  # Link-local (AWS metadata)
       ipaddress.IPv4Network("127.0.0.0/8"),     # Loopback (unless explicitly allowed)
   ]

   def is_safe_backend_url(url: str) -> bool:
       hostname = urllib.parse.urlparse(url).hostname
       ip = ipaddress.ip_address(socket.gethostbyname(hostname))
       return not any(ip in cidr for cidr in BLOCKED_CIDRS)
   ```

3. **DNS rebinding protection**: Re-resolve hostname before each request (attackers can change DNS after validation)

4. **Timeout and size limits**: Prevent slow-loris and data exfiltration
   ```python
   async with httpx.AsyncClient(timeout=5.0, max_redirects=0) as client:
       response = await client.post(backend_url, ...)
   ```

**Detection:**
- Unusual backend URLs in configuration (IP addresses instead of domains)
- Requests to non-standard ports
- High request volume to internal IP ranges
- Cloud provider security alerts for metadata access

**Phase assignment:** Phase 1 (Core Gateway) - critical security requirement

**Sources:**
- [OWASP SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)

---

## Performance Pitfalls

### Pitfall 13: Connection Pool Exhaustion

**What goes wrong:** Gateway creates new HTTP connection for each request to backends, exhausting connection pools and causing "connection pool full" errors under load.

**Why it happens:**
- Creating `httpx.AsyncClient()` in each request handler
- No connection pooling or reuse
- Backend servers have connection limits (e.g., 100 concurrent)
- Each connection has overhead (TLS handshake, DNS lookup)

**Consequences:**
- Requests fail with "ConnectionPoolFull" errors
- Latency increases due to connection setup overhead (50-200ms)
- Backend servers hit connection limits and reject new connections
- Gateway can't scale beyond ~100 concurrent requests

**Prevention:**
1. **Singleton HTTP client** with connection pooling:
   ```python
   from httpx import AsyncClient, Limits

   class BackendClient:
       def __init__(self):
           self._clients: dict[str, AsyncClient] = {}

       def get_client(self, backend_id: str) -> AsyncClient:
           if backend_id not in self._clients:
               self._clients[backend_id] = AsyncClient(
                   timeout=300.0,  # 5min for LLM generation
                   limits=Limits(
                       max_connections=100,
                       max_keepalive_connections=20,
                   ),
                   http2=True,  # Enable HTTP/2 multiplexing
               )
           return self._clients[backend_id]

       async def cleanup(self):
           for client in self._clients.values():
               await client.aclose()

   backend_client = BackendClient()  # Singleton
   ```

2. **Connection limit tuning**: Match backend capabilities
   - Local mlx-openai-server: 10-20 connections (single-process)
   - OpenAI API: 100+ connections (distributed backend)
   - vLLM-MLX: 50 connections (depends on `--max-parallel-requests`)

3. **HTTP/2 multiplexing**: Use HTTP/2 to share connections across requests (reduces connection overhead)

4. **Lifespan cleanup**: Close clients gracefully on shutdown
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       yield
       await backend_client.cleanup()
   ```

**Detection:**
- "ConnectionPoolFull" errors in logs
- High latency for new requests (connection setup overhead)
- `netstat -an | grep ESTABLISHED | wc -l` shows excessive connections
- Backend logs show "too many connections" errors

**Phase assignment:** Phase 2 (Performance Optimization)

**Sources:**
- [HTTPX connection pooling documentation](https://www.python-httpx.org/advanced/#pool-limit-configuration)

---

### Pitfall 14: Memory Leaks in Streaming Response Handlers

**What goes wrong:** Streaming responses accumulate in memory when clients disconnect mid-stream, causing memory leaks and eventual OOM crashes.

**Why it happens:**
- Backend continues generating response even after client disconnects
- Generator functions don't check for client disconnection
- Buffered chunks accumulate waiting to be sent
- No timeout or cleanup for abandoned streams

**Consequences:**
- Memory usage grows unbounded under load
- Gateway OOM crashes after hours of operation
- Resources wasted generating responses no one receives
- Backend servers continue processing abandoned requests

**Prevention:**
1. **Client disconnection detection**:
   ```python
   async def stream_with_disconnect_handling(request: Request, backend_stream):
       try:
           async for chunk in backend_stream:
               if await request.is_disconnected():
                   logger.info("Client disconnected, stopping stream")
                   break
               yield chunk
       finally:
           # Cleanup backend resources
           await backend_stream.aclose()
   ```

2. **Streaming timeout**: Abort streams that take too long
   ```python
   async def stream_with_timeout(backend_stream, timeout: float):
       started = time.time()
       async for chunk in backend_stream:
           if time.time() - started > timeout:
               raise TimeoutError("Stream exceeded timeout")
           yield chunk
   ```

3. **Backpressure handling**: Pause backend if client can't keep up
   ```python
   async def stream_with_backpressure(backend_stream):
       buffer = asyncio.Queue(maxsize=10)  # Limit buffered chunks

       async def producer():
           async for chunk in backend_stream:
               await buffer.put(chunk)
           await buffer.put(None)  # Sentinel

       producer_task = asyncio.create_task(producer())
       try:
           while True:
               chunk = await buffer.get()
               if chunk is None:
                   break
               yield chunk
       finally:
           producer_task.cancel()
   ```

4. **Memory monitoring**: Alert when memory usage exceeds threshold
   ```python
   import psutil

   def check_memory_usage():
       process = psutil.Process()
       mem_percent = process.memory_percent()
       if mem_percent > 80:
           logger.warning(f"High memory usage: {mem_percent}%")
   ```

**Detection:**
- Memory usage grows over time (use `htop`, `ps aux`)
- More memory used than expected for request volume
- OOM errors in logs after hours of operation
- Active streaming connections don't correlate with memory usage

**Phase assignment:** Phase 3 (Streaming Robustness)

---

## Prevention Checklist by Phase

### Phase 1: Core Gateway Infrastructure
- [ ] Implement orphaned subprocess cleanup (Pitfall 1)
- [ ] Set up API key secure storage with Keychain (Pitfall 4, 11)
- [ ] Add SSRF validation for backend URLs (Pitfall 12)
- [ ] Extend background task cleanup for gateway tasks (Pitfall 7)
- [ ] Design server ownership model (manual vs gateway) (Pitfall 6)

### Phase 2: On-Demand Model Loading
- [ ] Implement startup locks to prevent race conditions (Pitfall 2)
- [ ] Add startup grace period to health checker (Pitfall 8)
- [ ] Design model name routing with namespacing (Pitfall 5)
- [ ] Configure connection pooling for backends (Pitfall 13)

### Phase 3: Multi-Backend Compatibility
- [ ] Build request/response transformation layer (Pitfall 9)
- [ ] Test OpenAI vs Anthropic format differences (Pitfall 9)
- [ ] Disable response buffering for streaming (Pitfall 10)
- [ ] Implement SSE heartbeats for timeout prevention (Pitfall 3)

### Phase 4: Production Hardening
- [ ] Add client disconnection detection to streams (Pitfall 14)
- [ ] Configure timeout hierarchy (client, gateway, backend) (Pitfall 3)
- [ ] Set up memory monitoring and alerts (Pitfall 14)
- [ ] Audit logging for credential exposure (Pitfall 4, 11)

---

## Testing Recommendations

### Critical Test Scenarios

1. **Orphan cleanup testing**:
   - Kill gateway process with `kill -9`, verify subprocesses terminate
   - Restart gateway, verify orphaned processes adopted or cleaned up

2. **Race condition testing**:
   - Send 10 concurrent requests to same unloaded model
   - Verify only one startup occurs, all requests succeed

3. **Streaming timeout testing**:
   - Generate 5-minute response, verify no intermediate timeouts
   - Disconnect client mid-stream, verify backend stops generating

4. **API compatibility testing**:
   - Send OpenAI-format requests to Anthropic backend
   - Verify correct transformation and response format

5. **Security testing**:
   - Try to configure backend URL as `http://169.254.169.254`
   - Verify SSRF protection rejects dangerous URLs
   - Check database/logs for leaked API keys

6. **Memory leak testing**:
   - Run 1000 streaming requests with random disconnections
   - Verify memory usage returns to baseline

---

## Confidence Assessment

| Pitfall Category | Confidence | Evidence |
|-----------------|------------|----------|
| Subprocess Management | HIGH | Existing ServerManager code reviewed, Python subprocess docs, orphanage library docs |
| Race Conditions | HIGH | vLLM GitHub issues, async lock patterns |
| Streaming/Timeouts | HIGH | FastAPI SSE docs, AWS API Gateway docs, multiple blog posts |
| API Compatibility | HIGH | Official Anthropic compatibility docs, LiteLLM docs |
| Security (Credentials) | HIGH | Python keyring docs, security best practices |
| Security (SSRF) | MEDIUM | OWASP guidelines (no specific 2026 FastAPI examples found) |
| Performance | MEDIUM | HTTPX docs, general patterns (not mlx-specific benchmarks) |
| Integration with existing code | HIGH | Read ServerManager and main.py source code |

---

## Open Questions for Phase-Specific Research

1. **vLLM-MLX specific behavior**: Does vLLM-MLX have same race conditions as vLLM? (Phase 2)
2. **mlx-openai-server streaming**: Does it support chunked streaming or buffer responses? (Phase 3)
3. **macOS Keychain Electron compatibility**: Will keyring work in Electron menubar app? (Phase 1)
4. **Health check endpoints**: Does vLLM-MLX support `/health` or only `/v1/models`? (Phase 2)

---

## Sources

### API Gateway Architecture
- [Microservices Pattern: API Gateway](https://microservices.io/patterns/apigateway.html)
- [API Gateway vs API Proxy differences](https://konghq.com/blog/engineering/api-gateway-vs-api-proxy-understanding-the-differences)
- [Azure API Gateway design patterns](https://learn.microsoft.com/en-us/azure/architecture/microservices/design/gateway)

### LLM Streaming & SSE
- [How to Stream LLM Responses Using SSE](https://apidog.com/blog/stream-llm-responses-using-sse/)
- [Streaming LLM responses in Next.js with SSE](https://upstash.com/blog/sse-streaming-llm-responses)
- [How streaming LLM APIs work](https://til.simonwillison.net/llms/streaming-llm-apis)
- [Complete guide to streaming LLM responses](https://dev.to/hobbada/the-complete-guide-to-streaming-llm-responses-in-web-applications-from-sse-to-real-time-ui-3534)
- [KrakenD HTTP Streaming documentation](https://www.krakend.io/docs/enterprise/endpoints/streaming/)

### API Compatibility
- [Anthropic OpenAI SDK compatibility](https://docs.anthropic.com/en/api/openai-sdk)
- [OpenAI API vs Anthropic API developer guide](https://www.eesel.ai/blog/openai-api-vs-anthropic-api)
- [LiteLLM reasoning content handling](https://docs.litellm.ai/docs/reasoning_content)
- [Anthropic extended thinking compatibility](https://github.com/RooCodeInc/Roo-Code/discussions/1882)

### Subprocess & Process Management
- [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html)
- [Orphaned processes in Python](https://superfastpython.com/orphan-process-in-python/)
- [python-orphanage library](https://github.com/tonyseek/python-orphanage)
- [Ungraceful shutdown leaving orphans](https://github.com/facebookresearch/CompilerGym/issues/326)

### FastAPI Background Tasks & Lifecycle
- [FastAPI lifespan events](https://fastapi.tiangolo.com/advanced/events/)
- [Understanding FastAPI lifespan](https://dev.turmansolutions.ai/2025/09/27/understanding-fastapis-lifespan-events-proper-initialization-and-shutdown/)
- [FastAPI application lifecycle management](https://craftyourstartup.com/cys-docs/fastapi-startup-and-shutdown-events-guide/)
- [FastAPI async task pitfalls](https://leapcell.io/blog/understanding-pitfalls-of-async-task-management-in-fastapi-requests)

### Security & Credentials
- [Securely storing credentials with Python keyring](https://medium.com/@forsytheryan/securely-storing-credentials-in-python-with-keyring-d8972c3bd25f)
- [How to securely save credentials in Python](https://medium.com/jungletronics/how-to-securely-save-credentials-in-python-dd5c6983741a)
- [Keyring Python comprehensive guide](https://coderivers.org/blog/keyring-python/)
- [FastAPI security best practices](https://escape.tech/blog/how-to-secure-fastapi-api/)
- [FastAPI API key authentication](https://medium.com/@agusabdulrahman/fastapi-api-key-authentication-complete-guide-to-securing-your-api-44345f5c9bec)

### Performance & Timeouts
- [AWS API Gateway response streaming](https://aws.amazon.com/blogs/compute/building-responsive-apis-with-amazon-api-gateway-response-streaming/)
- [LiteLLM proxy timeouts](https://docs.litellm.ai/docs/proxy/timeout)
- [Portkey request timeouts](https://portkey.ai/docs/product/ai-gateway/request-timeouts)

### Race Conditions in LLM Servers
- [vLLM race condition in request cancellation](https://github.com/vllm-project/vllm/issues/23697)
- [vLLM V1 engine concurrent request issues](https://github.com/vllm-project/vllm/issues/25991)
- [vLLM KV offloading race conditions](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)

### LLM Gateway Patterns
- [Top 5 LLM Gateways in 2026](https://dev.to/varshithvhegde/top-5-llm-gateways-in-2026-a-deep-dive-comparison-for-production-teams-34d2)
- [Making AI agent configurations stable with gateway](https://dev.to/palapalapala/making-ai-agent-configurations-stable-with-an-llm-gateway-2jf1)
- [LLM orchestration frameworks and gateways](https://research.aimultiple.com/llm-orchestration/)

### Error Handling
- [OpenAI API error codes](https://platform.openai.com/docs/guides/error-codes)
- [Fix OpenAI API key issues 2026](https://theaisurf.com/common-issues-with-openai-api-keys-and-how-to-fix-them/)
- [Troubleshooting API errors and latency](https://help.openai.com/en/articles/1000499-troubleshooting-api-errors-and-latency)

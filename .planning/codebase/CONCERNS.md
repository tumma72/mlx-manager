# Codebase Concerns

**Analysis Date:** 2026-01-27

## Tech Debt

**Hardcoded JWT Secret in Default Config:**
- Issue: Default `jwt_secret` value is `"CHANGE_ME_IN_PRODUCTION"` in `mlx_manager/config.py` line 15
- Files: `mlx_manager/config.py`, `mlx_manager/services/auth_service.py`
- Impact: Development instances with default config expose authentication tokens to trivial compromise. Users who forget to set `MLX_MANAGER_JWT_SECRET` env var run insecure systems.
- Fix approach: Generate a secure random secret on first run if not configured, document requirement in setup steps, add startup warning if using default value

**Manual Database Schema Migration:**
- Issue: Schema changes require manual ALTER TABLE commands in `mlx_manager/database.py:30-56` instead of using Alembic
- Files: `mlx_manager/database.py`
- Impact: Migration brittleness - hardcoded column additions don't handle complex schema evolution, no version tracking, difficult to maintain across releases
- Fix approach: Integrate Alembic (already in dependencies) for tracked migrations

**Download Task Tracking in Memory:**
- Issue: `download_tasks` dict in `mlx_manager/routers/models.py:34` is in-process only
- Files: `mlx_manager/routers/models.py`, `mlx_manager/main.py`
- Impact: Download progress lost on server restart, frontend loses SSE connection and can't recover task state, tasks not resumable across restarts (only database records are)
- Fix approach: Move download task tracking to Redis or persistent queue, implement task resume logic in frontend

**Singleton Services Without Async Cleanup:**
- Issue: Services like `server_manager`, `hf_client`, `health_checker` are module-level singletons without lifecycle management
- Files: `mlx_manager/services/server_manager.py:389`, `mlx_manager/services/hf_client.py:326`, `mlx_manager/services/health_checker.py:85`
- Impact: Resource leaks if services have untracked state or connections, difficult to test in isolation, no dependency injection
- Fix approach: Convert to dependency injection via FastAPI Depends(), implement explicit lifecycle management

**Subprocess Process Resource Leaks:**
- Issue: In `mlx_manager/services/server_manager.py`, log file handles stored in `_log_files` dict but cleanup only happens on explicit stop or cleanup
- Files: `mlx_manager/services/server_manager.py:27-76`
- Impact: If process crashes and status check doesn't happen, file handles remain open, memory accumulates with many profile crashes
- Fix approach: Use context managers for log files, implement guaranteed cleanup with weak references or __del__

**Unsafe Error Pattern Matching:**
- Issue: Process error detection in `mlx_manager/services/server_manager.py:280-328` uses simple string pattern matching ("ERROR", "Error", "failed") on log tails
- Files: `mlx_manager/services/server_manager.py`
- Impact: False positives (error strings in normal output), false negatives (different error formats), brittle across mlx-openai-server versions
- Fix approach: Parse structured log output or implement version-specific error detection

---

## Known Bugs

**Download Task State Inconsistency:**
- Symptoms: Download shows "downloading" in frontend but database shows "completed", or frontend loses connection and shows stale state
- Files: `mlx_manager/routers/models.py`, `mlx_manager/main.py`
- Trigger: Server restart during download, frontend connection drop, rapid status queries
- Workaround: Refresh page to sync state from database, manually mark download as complete via API

**Process Status Race Condition:**
- Symptoms: `get_process_status()` may report process as running when it has actually exited between poll() check and stats collection
- Files: `mlx_manager/services/server_manager.py:269-350`
- Trigger: Rapid-fire status checks while process is exiting
- Workaround: Status polling includes race-safe exit handling but uptime/stats may briefly show old values

**Health Check Cache Invalidation:**
- Symptoms: RunningInstance.health_status not updated if health_checker task is cancelled or slow
- Files: `mlx_manager/services/health_checker.py`
- Trigger: Server under heavy load, many profiles, interval too aggressive
- Workaround: Increase health_check_interval, manually restart health checker

---

## Security Considerations

**Command Injection in Launchd Configuration:**
- Risk: `mlx_manager/services/launchd.py:40-50` builds ProgramArguments list with profile settings
- Files: `mlx_manager/services/launchd.py`, `mlx_manager/utils/command_builder.py`
- Current mitigation: Arguments passed as list not string (safe from shell injection), model_path validated with `validate_model_path()` in `mlx_manager/utils/security.py`
- Recommendations: Validate all profile fields (port, host, log_level, context_length) as integers/allowed values before passing to subprocess, add input sanitization tests

**Model Path Traversal via Symlinks:**
- Risk: `validate_model_path()` in `mlx_manager/utils/security.py:8-23` resolves paths but doesn't prevent symlinks to unsafe locations
- Files: `mlx_manager/utils/security.py`
- Current mitigation: `.resolve()` follows symlinks to real path, then checks against allowed dirs
- Recommendations: Use `readlink()` to verify no symlink traversal before validation, document symlink behavior, consider disallowing symlinks entirely

**HuggingFace API Token in Logs:**
- Risk: If HuggingFace credentials are used, they may appear in debug logs
- Files: `mlx_manager/services/hf_client.py`, `mlx_manager/services/hf_api.py`
- Current mitigation: No explicit token handling in code shown, huggingface_hub library handles tokens via ~/.cache
- Recommendations: Audit huggingface_hub credential handling, add log masking for any HF_TOKEN env vars, never log full error responses from API

**CORS Configuration Too Permissive:**
- Risk: `mlx_manager/main.py:168-182` allows all methods and headers from localhost dev servers
- Files: `mlx_manager/main.py`
- Current mitigation: Restricted to 127.0.0.1 and localhost (not production internet), only localhost dev/preview ports
- Recommendations: For production, restrict to exact origin, tighten methods to [GET, POST, PUT, DELETE], validate headers list

**SQL Injection via String Interpolation in Migration:**
- Risk: `mlx_manager/database.py:54` builds SQL with string formatting: `f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"`
- Files: `mlx_manager/database.py:54`
- Current mitigation: Column names and types are hardcoded in migrations list (not user input), but pattern is dangerous
- Recommendations: Use SQLAlchemy DDL constructs instead of raw SQL strings, refactor to use alembic.op.add_column()

---

## Performance Bottlenecks

**Directory Size Calculation O(n) on Every Download Poll:**
- Problem: `mlx_manager/services/hf_client.py:232-240` walks entire directory tree every 1 second during download to get size
- Files: `mlx_manager/services/hf_client.py`
- Cause: `Path.rglob("*")` enumerates all files recursively
- Improvement path: Track file count/size via inotify or filesystem events, cache size updates, increase poll interval to 5-10s

**Health Check Serial Execution:**
- Problem: `mlx_manager/services/health_checker.py:49-81` checks servers sequentially, slow server blocks others
- Files: `mlx_manager/services/health_checker.py`
- Cause: `await server_manager.check_health(profile)` blocks loop per instance
- Improvement path: Use `asyncio.gather()` with timeout to check all servers concurrently, skip slow servers instead of blocking

**Model Detection Reads config.json Every Request:**
- Problem: `mlx_manager/utils/model_detection.py` re-reads model config.json on each API call for characteristics
- Files: `mlx_manager/utils/model_detection.py`, `mlx_manager/routers/models.py`
- Cause: No caching of extracted characteristics
- Improvement path: Add per-model cache with TTL, invalidate on model delete

**Frontend API Client No Connection Pooling:**
- Problem: Each request in `mlx_manager/routers/chat.py:82` and elsewhere creates new `httpx.AsyncClient`
- Files: Multiple routers use context manager pattern `async with httpx.AsyncClient() as client:`
- Cause: High-frequency requests create/destroy connections to mlx-openai-server
- Improvement path: Use singleton `httpx.AsyncClient` with connection pooling, implement retry logic

---

## Fragile Areas

**Server Subprocess Log File Handling:**
- Files: `mlx_manager/services/server_manager.py`
- Why fragile: Log file opened in write mode, flushed on status check, but file handle stored in mutable dict without locking. If profile crashes between open and close, handle may be left open or read from wrong position.
- Safe modification: Wrap all log file operations in locks, use context managers, implement LRU cache for log file handles with guaranteed cleanup
- Test coverage: Tests exist for start/stop but not for concurrent access or edge cases like disk full

**Authentication Token Expiry Edge Cases:**
- Files: `mlx_manager/routers/auth.py`, `mlx_manager/dependencies.py`, `mlx_manager/services/auth_service.py`
- Why fragile: Token expiry checked via `jwt.decode()` which raises exception, caught and converted to 401. No explicit test for edge cases like negative expiry, expired token refresh, simultaneous logout.
- Safe modification: Add explicit token expiry validation before decode, implement refresh token rotation, test token revocation
- Test coverage: ~95% coverage but token expiry scenarios not explicitly tested

**Download Resume on Server Restart:**
- Files: `mlx_manager/main.py:81-103`, `mlx_manager/database.py:59-95`
- Why fragile: Resume depends on database records and in-memory download_tasks dict sync. If database is updated but task not created, or vice versa, state diverges. No idempotency check.
- Safe modification: Implement task creation idempotency key (download_id), validate database state before task creation, implement download state machine with explicit transitions
- Test coverage: Server restart scenario not tested in CI

**Model Family Detection Version Compatibility:**
- Files: `mlx_manager/utils/model_detection.py:76-240`
- Why fragile: Uses `importlib.metadata.version()` to check mlx-lm version, but version check is simple string comparison. MiniMax requires >= 0.28.4 but no patch version handling.
- Safe modification: Use packaging.version.parse() for semantic version comparison, test version detection against multiple mlx-lm releases
- Test coverage: Version detection not tested with actual version strings

---

## Scaling Limits

**In-Memory Download Task Registry:**
- Current capacity: Limited by available RAM, ~1000 concurrent downloads before performance degrades
- Limit: Grows unbounded until process restart, no pruning of completed tasks
- Scaling path: Move to Redis queue, implement task TTL (24h), batch cleanup of old tasks

**SQLite Database for Concurrent Users:**
- Current capacity: Works for single-user (default), ~5-10 concurrent users before lock contention
- Limit: SQLite has page-level locking, multiple writers block each other
- Scaling path: Migrate to PostgreSQL or MySQL, add connection pooling via pgBouncer

**Single Health Checker Task:**
- Current capacity: 50-100 profiles before health checks start timing out
- Limit: Sequential checking with 30s interval means 1500s+ for full cycle at scale
- Scaling path: Shard health checks across multiple tasks, distribute by profile_id hash

---

## Dependencies at Risk

**huggingface_hub Security Updates:**
- Risk: Upstream library, depends on httpx and transformers which have frequent security patches
- Impact: Vulnerability in HF Hub could expose cached models or auth tokens
- Migration plan: Pin to tested version (>= 0.27.0), monitor releases monthly, implement dependency scanning in CI

**mlx-openai-server Subprocess Dependency:**
- Risk: External process, no version pinning, subprocess communication is fragile
- Impact: Breaking changes in mlx-openai-server API breaks chat/model endpoints
- Migration plan: Pin mlx-openai-server version in pyproject.toml, test against multiple versions, implement version detection at startup

**sqlmodel ORM Stability:**
- Risk: Smaller community than SQLAlchemy, less frequent updates
- Impact: SQL generation bugs, missing features
- Migration plan: Switch to pure SQLAlchemy 2.0 + Pydantic v2 for validation

---

## Missing Critical Features

**No Audit Logging:**
- Problem: Profile creation/deletion, user approval, server start/stop not logged
- Blocks: Security compliance, debugging incidents, detecting unauthorized changes
- Solution: Add audit table with timestamps, user IDs, action types, implement audit log API endpoint

**No Model Usage Metrics:**
- Problem: No tracking of which models are used, how often, resource consumption per model
- Blocks: Performance optimization decisions, capacity planning
- Solution: Add metrics table, hook into chat endpoint, implement dashboard aggregation

**No Backup/Restore for Database:**
- Problem: User database loss or corruption has no recovery
- Blocks: Production deployment
- Solution: Implement automated daily backups to configurable location, implement restore CLI command

---

## Test Coverage Gaps

**Process Crash Edge Cases:**
- What's not tested: Server process exits during startup, during health check, rapid restart cycles
- Files: `mlx_manager/services/server_manager.py`, `mlx_manager/services/health_checker.py`
- Risk: Crashes leave orphan processes, log files, database inconsistencies
- Priority: High - affects production stability

**Concurrent Profile Operations:**
- What's not tested: Creating profile while another is starting, deleting profile while health check runs, concurrent API requests to same profile
- Files: `mlx_manager/routers/profiles.py`, `mlx_manager/routers/servers.py`, `mlx_manager/services/server_manager.py`
- Risk: Race conditions, database locks, inconsistent state
- Priority: High - affects multi-user environments

**Download Interruption Recovery:**
- What's not tested: Aborting download mid-flight, disk full during download, network timeout mid-transfer
- Files: `mlx_manager/routers/models.py`, `mlx_manager/services/hf_client.py`, `mlx_manager/main.py`
- Risk: Partial downloads not cleaned up, download state desynchronized with filesystem
- Priority: Medium - intermittent failures in practice

**Authentication Token Expiry:**
- What's not tested: Token expiry during request, expired token in SSE connection, refresh token flow, simultaneous logout
- Files: `mlx_manager/routers/auth.py`, `mlx_manager/dependencies.py`
- Risk: Stale tokens accepted, infinite sessions, logout ineffective
- Priority: Medium - security boundary

**Frontend Component Error Boundaries:**
- What's not tested: API failures in ProfileSelector, ServerCard, models search, error recovery and fallback UI
- Files: `frontend/src/lib/components/` (multiple)
- Risk: Blank pages, unhandled exceptions in console, poor UX on network errors
- Priority: Low - graceful degradation but incomplete

---

*Concerns audit: 2026-01-27*

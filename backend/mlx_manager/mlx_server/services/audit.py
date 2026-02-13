"""Audit logging service with non-blocking background writes.

PRIVACY REQUIREMENT: This service NEVER receives or stores prompt/response content.
Only request metadata is logged: model, backend, duration, status, tokens.
"""

import asyncio
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from mlx_manager.mlx_server.database import get_session
from mlx_manager.mlx_server.models.audit import AuditLog


class RequestContext(BaseModel):
    """Context for tracking request lifecycle.

    Created at request start, completed at request end.
    """

    request_id: str
    model: str
    endpoint: str
    backend_type: str = "local"
    start_time: float = Field(default_factory=time.time)

    # Set when request completes
    status: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def duration_ms(self) -> int:
        """Calculate duration from start_time to now."""
        return int((time.time() - self.start_time) * 1000)


class AuditService:
    """Service for writing audit logs with background task execution.

    Uses asyncio.create_task for non-blocking writes.
    Maintains an in-memory buffer for WebSocket broadcasting.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._buffer_size = buffer_size
        self._recent_logs: list[dict[str, Any]] = []
        self._subscribers: set[Callable[[dict[str, Any]], None]] = set()

    @asynccontextmanager
    async def track_request(
        self,
        request_id: str,
        model: str,
        endpoint: str,
        backend_type: str = "local",
    ):
        """Context manager for tracking request lifecycle.

        Usage:
            async with audit_service.track_request(req_id, model, endpoint) as ctx:
                # ... do work ...
                ctx.prompt_tokens = 100
                ctx.completion_tokens = 50
        """
        ctx = RequestContext(
            request_id=request_id,
            model=model,
            endpoint=endpoint,
            backend_type=backend_type,
        )
        try:
            yield ctx
            ctx.status = "success"
        except TimeoutError:
            ctx.status = "timeout"
            ctx.error_type = "TimeoutError"
            ctx.error_message = "Request timed out"
            raise
        except Exception as e:
            ctx.status = "error"
            ctx.error_type = type(e).__name__
            ctx.error_message = str(e)[:500]  # Truncate long error messages
            raise
        finally:
            # Always log, even on error
            asyncio.create_task(self._write_log(ctx))

    async def log_request(
        self,
        request_id: str,
        model: str,
        endpoint: str,
        backend_type: str,
        duration_ms: int,
        status: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Log a completed request (alternative to track_request context manager).

        For cases where context manager pattern doesn't fit.
        """
        log_entry = AuditLog(
            request_id=request_id,
            timestamp=datetime.now(UTC),
            model=model,
            endpoint=endpoint,
            backend_type=backend_type,
            duration_ms=duration_ms,
            status=status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error_type=error_type,
            error_message=error_message,
        )
        asyncio.create_task(self._write_log_entry(log_entry))

    async def _write_log(self, ctx: RequestContext) -> None:
        """Write audit log from RequestContext."""
        log_entry = AuditLog(
            request_id=ctx.request_id,
            timestamp=datetime.now(UTC),
            model=ctx.model,
            endpoint=ctx.endpoint,
            backend_type=ctx.backend_type,
            duration_ms=ctx.duration_ms,
            status=ctx.status or "unknown",
            prompt_tokens=ctx.prompt_tokens,
            completion_tokens=ctx.completion_tokens,
            total_tokens=ctx.total_tokens,
            error_type=ctx.error_type,
            error_message=ctx.error_message,
        )
        await self._write_log_entry(log_entry)

    async def _write_log_entry(self, log_entry: AuditLog) -> None:
        """Write log entry to database and notify subscribers."""
        try:
            async with get_session() as session:
                session.add(log_entry)
                await session.commit()
                await session.refresh(log_entry)

            # Add to recent buffer
            log_dict = log_entry.model_dump()
            log_dict["timestamp"] = log_dict["timestamp"].isoformat()
            self._recent_logs.append(log_dict)
            if len(self._recent_logs) > self._buffer_size:
                self._recent_logs.pop(0)

            # Notify subscribers (for WebSocket broadcast)
            for callback in self._subscribers:
                try:
                    callback(log_dict)
                except Exception as e:
                    logger.warning(f"Subscriber callback error: {e}")

        except Exception as e:
            logger.exception(f"Failed to write audit log: {e}")

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to new log entries for live updates."""
        self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Unsubscribe from log entries."""
        self._subscribers.discard(callback)

    def get_recent_logs(self) -> list[dict[str, Any]]:
        """Get recent logs from in-memory buffer."""
        return list(self._recent_logs)


# Global singleton
audit_service = AuditService()

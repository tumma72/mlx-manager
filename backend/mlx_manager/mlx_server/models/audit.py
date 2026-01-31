"""Audit log models for request tracking.

PRIVACY REQUIREMENT: Never store prompt/response content.
Only store metadata: model, backend, duration, status, tokens.
"""

from datetime import UTC, datetime

from pydantic import Field as PydanticField
from sqlmodel import Field, SQLModel


class AuditLogBase(SQLModel):
    """Base audit log fields - metadata only, no content."""

    request_id: str = Field(index=True, description="Request ID for correlation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
        description="Request timestamp",
    )
    model: str = Field(index=True, description="Model ID requested")
    backend_type: str = Field(
        index=True,
        description="Backend: local, openai, anthropic",
    )
    endpoint: str = Field(description="API endpoint path")
    duration_ms: int = Field(description="Request duration in milliseconds")
    status: str = Field(
        index=True,
        description="Request status: success, error, timeout",
    )

    # Token counts (optional - may not be available for all backends)
    prompt_tokens: int | None = Field(default=None, description="Prompt token count")
    completion_tokens: int | None = Field(
        default=None, description="Completion token count"
    )
    total_tokens: int | None = Field(default=None, description="Total token count")

    # Error info (only for failed requests)
    error_type: str | None = Field(default=None, description="Exception type if error")
    error_message: str | None = Field(default=None, description="Error message if error")


class AuditLog(AuditLogBase, table=True):
    """Audit log database table."""

    __tablename__ = "audit_logs"  # type: ignore[assignment]

    id: int | None = Field(default=None, primary_key=True)


class AuditLogResponse(AuditLogBase):
    """API response model for audit log entries."""

    id: int


class AuditLogFilter(SQLModel):
    """Query filters for audit log retrieval."""

    model: str | None = None
    backend_type: str | None = None
    status: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int = PydanticField(default=100, le=1000)
    offset: int = 0

"""RFC 7807/9457 Problem Details models.

Provides structured error responses per RFC 7807:
https://datatracker.ietf.org/doc/html/rfc7807
"""

from enum import StrEnum
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Programmatic error codes for client handling.

    Provides machine-readable codes that clients can switch on,
    independent of human-readable titles/details which may change.
    """

    # Validation
    VALIDATION_ERROR = "validation_error"
    FIELD_MAX_LENGTH_EXCEEDED = "field_max_length_exceeded"
    INVALID_REQUEST = "invalid_request"

    # Resource
    MODEL_NOT_FOUND = "model_not_found"
    RESOURCE_NOT_FOUND = "resource_not_found"

    # Auth
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"

    # Rate limiting
    RATE_LIMITED = "rate_limited"

    # Timeout
    REQUEST_TIMEOUT = "request_timeout"

    # Server
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"

    # Inference
    INFERENCE_ERROR = "inference_error"
    GENERATION_ERROR = "generation_error"


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details response.

    All API errors are returned in this format for consistency.
    Always includes request_id for log correlation.
    """

    type: str = Field(
        default="about:blank",
        description="URI reference identifying the problem type",
    )
    title: str = Field(
        description="Human-readable summary (not occurrence-specific)",
    )
    status: int = Field(
        description="HTTP status code",
    )
    detail: str | None = Field(
        default=None,
        description="Human-readable explanation specific to this occurrence",
    )
    instance: str | None = Field(
        default=None,
        description="URI reference for this specific occurrence",
    )
    request_id: str = Field(
        description="Request ID for log correlation",
    )

    # Extension fields (RFC 7807 allows additional members)
    error_code: str | None = Field(
        default=None,
        description="Programmatic error code for client handling",
    )
    errors: list[dict[str, Any]] | None = Field(
        default=None,
        description="Validation errors (for 422 responses)",
    )


class TimeoutProblem(ProblemDetail):
    """Problem Details for timeout errors."""

    type: str = "https://mlx-manager.dev/errors/timeout"
    title: str = "Request Timeout"
    status: int = 408
    timeout_seconds: float = Field(
        description="Configured timeout that was exceeded",
    )


class TimeoutHTTPException(HTTPException):
    """Custom exception for endpoint timeouts.

    Carries timeout information for proper Problem Details response.
    """

    def __init__(
        self,
        timeout_seconds: float,
        detail: str | None = None,
    ) -> None:
        super().__init__(
            status_code=408,
            detail=detail or f"Request timed out after {timeout_seconds} seconds",
        )
        self.timeout_seconds = timeout_seconds

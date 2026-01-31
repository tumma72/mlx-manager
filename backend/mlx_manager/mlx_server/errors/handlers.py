"""FastAPI exception handlers for RFC 7807 responses.

Registers handlers that convert all exceptions to Problem Details format.
Generates request_id for every error for log correlation.
"""

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlx_manager.mlx_server.errors.problem_details import (
    ProblemDetail,
    TimeoutHTTPException,
    TimeoutProblem,
)

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Generate unique request ID for log correlation."""
    return f"req_{uuid.uuid4().hex[:12]}"


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Handle HTTPException with Problem Details response."""
    request_id = generate_request_id()

    # Log the error with request_id for correlation
    logger.warning(
        f"HTTP {exc.status_code} error: {exc.detail}",
        extra={"request_id": request_id, "path": request.url.path},
    )

    # Map common status codes to problem types
    problem_types = {
        400: "https://mlx-manager.dev/errors/bad-request",
        401: "https://mlx-manager.dev/errors/unauthorized",
        403: "https://mlx-manager.dev/errors/forbidden",
        404: "https://mlx-manager.dev/errors/not-found",
        422: "https://mlx-manager.dev/errors/validation-error",
        429: "https://mlx-manager.dev/errors/rate-limited",
        500: "https://mlx-manager.dev/errors/internal-error",
        503: "https://mlx-manager.dev/errors/service-unavailable",
    }

    problem_titles = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        422: "Validation Error",
        429: "Too Many Requests",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }

    problem = ProblemDetail(
        type=problem_types.get(exc.status_code, "about:blank"),
        title=problem_titles.get(exc.status_code, "Error"),
        status=exc.status_code,
        detail=str(exc.detail) if exc.detail else None,
        instance=str(request.url.path),
        request_id=request_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        headers={"X-Request-ID": request_id},
    )


async def timeout_exception_handler(
    request: Request,
    exc: TimeoutHTTPException,
) -> JSONResponse:
    """Handle TimeoutHTTPException with specialized Problem Details."""
    request_id = generate_request_id()

    logger.warning(
        f"Request timeout after {exc.timeout_seconds}s: {request.url.path}",
        extra={"request_id": request_id},
    )

    problem = TimeoutProblem(
        detail=exc.detail,
        instance=str(request.url.path),
        request_id=request_id,
        timeout_seconds=exc.timeout_seconds,
    )

    return JSONResponse(
        status_code=408,
        content=problem.model_dump(exclude_none=True),
        headers={"X-Request-ID": request_id},
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors with Problem Details."""
    request_id = generate_request_id()

    # Extract validation errors
    errors: list[dict[str, Any]] = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown"),
            }
        )

    logger.warning(
        f"Validation error: {len(errors)} errors",
        extra={"request_id": request_id, "path": request.url.path},
    )

    problem = ProblemDetail(
        type="https://mlx-manager.dev/errors/validation-error",
        title="Validation Error",
        status=422,
        detail=f"Request validation failed with {len(errors)} error(s)",
        instance=str(request.url.path),
        request_id=request_id,
        errors=errors,
    )

    return JSONResponse(
        status_code=422,
        content=problem.model_dump(exclude_none=True),
        headers={"X-Request-ID": request_id},
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions without exposing internals."""
    request_id = generate_request_id()

    # Log full exception for debugging, but don't expose to client
    logger.exception(
        f"Unhandled exception: {type(exc).__name__}",
        extra={"request_id": request_id, "path": request.url.path},
    )

    problem = ProblemDetail(
        type="https://mlx-manager.dev/errors/internal-error",
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred. Please try again later.",
        instance=str(request.url.path),
        request_id=request_id,
    )

    return JSONResponse(
        status_code=500,
        content=problem.model_dump(exclude_none=True),
        headers={"X-Request-ID": request_id},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app.

    Call this after creating the FastAPI app but before adding routes.
    """
    # Type ignores needed because FastAPI's type stubs expect generic Exception handlers
    # but specific exception types work correctly at runtime
    app.add_exception_handler(
        TimeoutHTTPException, timeout_exception_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        HTTPException, http_exception_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        RequestValidationError, validation_exception_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("RFC 7807 error handlers registered")

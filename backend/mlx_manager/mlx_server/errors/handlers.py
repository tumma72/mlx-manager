"""FastAPI exception handlers for RFC 7807 responses.

Registers handlers that convert all exceptions to Problem Details format.
Generates request_id for every error for log correlation.
"""

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from mlx_manager.mlx_server.errors.problem_details import (
    ErrorCode,
    ProblemDetail,
    TimeoutHTTPException,
    TimeoutProblem,
)

# Map HTTP status codes to programmatic error codes
ERROR_CODE_MAP: dict[int, ErrorCode] = {
    400: ErrorCode.INVALID_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.RESOURCE_NOT_FOUND,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    500: ErrorCode.INTERNAL_ERROR,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}


def generate_request_id() -> str:
    """Generate unique request ID for log correlation.

    Used as fallback when RequestIDMiddleware has not set request.state.request_id.
    """
    return f"req_{uuid.uuid4().hex[:12]}"


def _get_request_id(request: Request) -> str:
    """Get request ID from middleware, or generate fallback."""
    try:
        return request.state.request_id  # type: ignore[no-any-return]
    except AttributeError:
        return generate_request_id()


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Handle HTTPException with Problem Details response."""
    request_id = _get_request_id(request)

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

    error_code = ERROR_CODE_MAP.get(exc.status_code)

    problem = ProblemDetail(
        type=problem_types.get(exc.status_code, "about:blank"),
        title=problem_titles.get(exc.status_code, "Error"),
        status=exc.status_code,
        detail=str(exc.detail) if exc.detail else None,
        instance=str(request.url.path),
        request_id=request_id,
        error_code=error_code,
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
    request_id = _get_request_id(request)

    logger.warning(
        f"Request timeout after {exc.timeout_seconds}s: {request.url.path}",
        extra={"request_id": request_id},
    )

    problem = TimeoutProblem(
        detail=exc.detail,
        instance=str(request.url.path),
        request_id=request_id,
        timeout_seconds=exc.timeout_seconds,
        error_code=ErrorCode.REQUEST_TIMEOUT,
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
    request_id = _get_request_id(request)

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
        error_code=ErrorCode.VALIDATION_ERROR,
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
    request_id = _get_request_id(request)

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
        error_code=ErrorCode.INTERNAL_ERROR,
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
        TimeoutHTTPException,
        timeout_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        HTTPException,
        http_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("RFC 7807 error handlers registered")

"""LogFire configuration for MLX Manager.

Configures observability with reduced verbosity for normal operation.
Set LOGFIRE_CONSOLE_VERBOSE=true to enable verbose console output.
"""

import os
from typing import TYPE_CHECKING

import logfire
from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine

_configured = False


def configure_logfire(
    service_name: str = "mlx-manager",
    service_version: str | None = None,
) -> None:
    """Configure LogFire with service metadata.

    MUST be called before creating FastAPI app or any instrumented clients.
    Uses send_to_logfire='if-token-present' for offline development.

    Console output is disabled by default to reduce noise. Set
    LOGFIRE_CONSOLE_VERBOSE=true to enable verbose console output.
    """
    global _configured
    if _configured:
        return

    # Check if verbose console output is requested
    verbose = os.environ.get("LOGFIRE_CONSOLE_VERBOSE", "").lower() in ("true", "1", "yes")

    logfire.configure(
        service_name=service_name,
        service_version=service_version,
        send_to_logfire="if-token-present",  # Offline mode without token
        console=logfire.ConsoleOptions(verbose=verbose) if verbose else False,
    )
    _configured = True
    console_mode = "verbose" if verbose else "disabled"
    logger.info(f"LogFire configured for {service_name} (console={console_mode})")


def instrument_fastapi(app: "FastAPI") -> None:
    """Add FastAPI instrumentation for request tracing."""
    logfire.instrument_fastapi(app)
    logger.info("LogFire FastAPI instrumentation enabled")


def instrument_httpx() -> None:
    """Add HTTPX instrumentation for outbound HTTP tracing.

    Instruments ALL httpx clients globally.
    """
    logfire.instrument_httpx()
    logger.info("LogFire HTTPX instrumentation enabled")


def instrument_sqlalchemy(engine: "AsyncEngine") -> None:
    """Add SQLAlchemy instrumentation for database query tracing."""
    logfire.instrument_sqlalchemy(engine=engine)
    logger.info("LogFire SQLAlchemy instrumentation enabled")

"""LogFire configuration for MLX Inference Server."""

from typing import TYPE_CHECKING

import logfire
from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine

_configured = False


def configure_logfire(
    service_name: str = "mlx-server",
    service_version: str | None = None,
) -> None:
    """Configure LogFire with service metadata.

    MUST be called before creating FastAPI app or any instrumented clients.
    Uses send_to_logfire='if-token-present' for offline development.
    """
    global _configured
    if _configured:
        return

    logfire.configure(
        service_name=service_name,
        service_version=service_version,
        send_to_logfire="if-token-present",  # Offline mode without token
    )
    _configured = True
    logger.info(f"LogFire configured for {service_name}")


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


def instrument_llm_clients() -> None:
    """Add OpenAI and Anthropic instrumentation for LLM token tracking.

    Captures request duration, token usage, and exceptions.
    Only instruments clients that are installed (openai/anthropic packages optional).
    """
    instrumented = []
    try:
        logfire.instrument_openai()
        instrumented.append("OpenAI")
    except Exception:
        logger.debug("OpenAI client not available for instrumentation")

    try:
        logfire.instrument_anthropic()
        instrumented.append("Anthropic")
    except Exception:
        logger.debug("Anthropic client not available for instrumentation")

    if instrumented:
        logger.info(f"LogFire LLM client instrumentation enabled: {', '.join(instrumented)}")

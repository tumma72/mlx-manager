"""Loguru-based logging configuration.

Provides:
- Separate log files for mlx-server and mlx-manager
- Configurable log levels via environment variables
- Automatic rotation and retention
- Intercept handler for standard logging compatibility

Environment Variables:
- MLX_MANAGER_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
- MLX_MANAGER_LOG_DIR: Log directory path. Default: logs/
"""

import logging
import os
import sys
from pathlib import Path

from loguru import logger

# Environment variables
LOG_LEVEL = os.environ.get("MLX_MANAGER_LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(os.environ.get("MLX_MANAGER_LOG_DIR", "logs"))

# Track if logging has been configured to avoid duplicate setup
_logging_configured = False


def setup_logging() -> None:
    """Configure Loguru logging with separate files.

    Sets up:
    - Console output (stderr) with colorized format
    - MLX Server log file for inference operations
    - MLX Manager log file for app operations

    Log files are rotated at 10 MB and retained for 7 days.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console output (stderr)
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # MLX Server log file (inference, models, adapters)
    logger.add(
        LOG_DIR / "mlx-server.log",
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        filter=lambda record: record["name"].startswith("mlx_manager.mlx_server"),
    )

    # MLX Manager log file (app, routers, services)
    logger.add(
        LOG_DIR / "mlx-manager.log",
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        filter=lambda record: not record["name"].startswith("mlx_manager.mlx_server"),
    )


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to Loguru.

    This handler allows third-party libraries that use standard logging
    to have their logs properly routed through Loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by forwarding to Loguru."""
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def intercept_standard_logging() -> None:
    """Redirect all standard logging to Loguru.

    This ensures that third-party libraries using standard logging
    have their output routed through our Loguru configuration.
    """
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party loggers
    for name in ["httpx", "httpcore", "uvicorn.access", "huggingface_hub"]:
        logging.getLogger(name).setLevel(logging.WARNING)

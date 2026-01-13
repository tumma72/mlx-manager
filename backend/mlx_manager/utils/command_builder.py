"""Utilities for building mlx-openai-server commands."""

import shutil
import sys
from pathlib import Path

from mlx_manager.models import ServerProfile


def find_mlx_openai_server() -> str:
    """Find the mlx-openai-server executable.

    First checks the same directory as the Python executable (for venv installs),
    then falls back to system PATH.
    """
    # Check alongside the Python executable (handles venv correctly)
    python_dir = Path(sys.executable).parent
    local_cmd = python_dir / "mlx-openai-server"
    if local_cmd.exists():
        return str(local_cmd)

    # Fall back to PATH lookup
    path_cmd = shutil.which("mlx-openai-server")
    if path_cmd:
        return path_cmd

    raise RuntimeError(
        "mlx-openai-server not found. Please install it with: pip install mlx-openai-server"
    )


def build_mlx_server_command(profile: ServerProfile) -> list[str]:
    """Build the mlx-openai-server command from profile.

    Note: mlx-openai-server CLI uses 'launch' subcommand and supports only:
    --model-path, --model-type (lm|multimodal), --port, --host,
    --max-concurrency, --queue-timeout, --queue-size
    """
    # Map our model types to mlx-openai-server supported types
    # mlx-openai-server only supports 'lm' and 'multimodal'
    model_type = profile.model_type
    if model_type not in ("lm", "multimodal"):
        model_type = "lm"  # Default to lm for unsupported types

    return [
        find_mlx_openai_server(),
        "launch",  # Required subcommand
        "--model-path",
        profile.model_path,
        "--model-type",
        model_type,
        "--port",
        str(profile.port),
        "--host",
        profile.host,
        "--max-concurrency",
        str(profile.max_concurrency),
        "--queue-timeout",
        str(profile.queue_timeout),
        "--queue-size",
        str(profile.queue_size),
    ]

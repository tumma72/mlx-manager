"""Security utilities."""

from pathlib import Path

from app.config import settings


def validate_model_path(path: str) -> bool:
    """Ensure model path is within allowed directories."""
    resolved = Path(path).resolve()

    for allowed in settings.allowed_model_dirs:
        try:
            resolved.relative_to(Path(allowed).resolve())
            return True
        except ValueError:
            continue

    # Also allow HuggingFace model IDs (e.g., mlx-community/model-name)
    if "/" in path and not path.startswith("/"):
        return True

    return False

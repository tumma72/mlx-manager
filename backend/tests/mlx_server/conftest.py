"""Shared fixtures for MLX Server tests.

The validate_model_available function was added to all router endpoints.
Unit tests that call endpoint functions directly use test model IDs that
are not in the default available_models list, so we bypass the validation
globally for all mlx_server tests.

The validation logic itself is tested in test_validation.py.
"""

from unittest.mock import patch

import pytest

# All modules that import validate_model_available at the top level.
_VALIDATE_MODEL_TARGETS = [
    "mlx_manager.mlx_server.api.v1.chat.validate_model_available",
    "mlx_manager.mlx_server.api.v1.completions.validate_model_available",
    "mlx_manager.mlx_server.api.v1.messages.validate_model_available",
    "mlx_manager.mlx_server.api.v1.embeddings.validate_model_available",
    "mlx_manager.mlx_server.api.v1.speech.validate_model_available",
    "mlx_manager.mlx_server.api.v1.transcriptions.validate_model_available",
]


def _passthrough_validate(model: str | None) -> str:
    """Passthrough: returns the model as-is (or a default)."""
    return model or "test-model"


@pytest.fixture(autouse=True)
def _bypass_model_validation():
    """Bypass validate_model_available in all router endpoint modules.

    This is safe for tests that do not call router endpoints -- the
    mock simply replaces a function that would not be reached anyway.
    """
    patches = [
        patch(target, side_effect=_passthrough_validate) for target in _VALIDATE_MODEL_TARGETS
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()

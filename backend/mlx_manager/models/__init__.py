"""Models package - re-exports all domain models for backward compatibility."""

from mlx_manager.models.capabilities import *  # noqa: F401,F403
from mlx_manager.models.dto import *  # noqa: F401,F403
from mlx_manager.models.entities import *  # noqa: F401,F403
from mlx_manager.models.enums import *  # noqa: F401,F403
from mlx_manager.models.profiles import *  # noqa: F401,F403
from mlx_manager.models.profiles import (  # noqa: F401
    ExecutionProfile as ServerProfile,
)
from mlx_manager.models.profiles import (  # noqa: F401
    ExecutionProfileCreate as ServerProfileCreate,
)
from mlx_manager.models.profiles import (  # noqa: F401
    ExecutionProfileResponse as ServerProfileResponse,
)
from mlx_manager.models.profiles import (  # noqa: F401
    ExecutionProfileUpdate as ServerProfileUpdate,
)
from mlx_manager.models.value_objects import *  # noqa: F401,F403

# Lazy re-exports from shared package to avoid circular imports.
# shared.cloud_entities imports models.enums which would trigger this __init__.py.
_SHARED_NAMES = {
    "API_TYPE_FOR_BACKEND",
    "DEFAULT_BASE_URLS",
    "BackendMapping",
    "CloudCredential",
}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy import of shared cloud entities for backward compatibility."""
    if name in _SHARED_NAMES:
        import mlx_manager.shared.cloud_entities as _cloud

        return getattr(_cloud, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

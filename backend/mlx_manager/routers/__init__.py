"""API routers."""

from mlx_manager.routers.models import router as models_router
from mlx_manager.routers.profiles import router as profiles_router
from mlx_manager.routers.servers import router as servers_router
from mlx_manager.routers.system import router as system_router

__all__ = ["models_router", "profiles_router", "servers_router", "system_router"]

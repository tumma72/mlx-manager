"""API routers."""

from app.routers.models import router as models_router
from app.routers.profiles import router as profiles_router
from app.routers.servers import router as servers_router
from app.routers.system import router as system_router

__all__ = ["models_router", "profiles_router", "servers_router", "system_router"]

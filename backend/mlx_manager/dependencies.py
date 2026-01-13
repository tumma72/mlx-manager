"""FastAPI dependencies for common operations."""

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.models import ServerProfile


async def get_profile_or_404(
    profile_id: int,
    session: AsyncSession = Depends(get_db),
) -> ServerProfile:
    """Get a profile by ID or raise 404."""
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

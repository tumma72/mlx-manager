"""Server profiles API router.

With the embedded MLX server, profiles no longer need port/host configuration.
Profiles are now primarily for storing model configuration and generation parameters.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user, get_profile_or_404
from mlx_manager.models import (
    ServerProfile,
    ServerProfileCreate,
    ServerProfileResponse,
    ServerProfileUpdate,
    User,
)

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.get("", response_model=list[ServerProfileResponse])
async def list_profiles(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """List all server profiles."""
    result = await session.execute(select(ServerProfile))
    profiles = result.scalars().all()
    return profiles


@router.get("/{profile_id}", response_model=ServerProfileResponse)
async def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile: ServerProfile = Depends(get_profile_or_404),
):
    """Get a specific profile."""
    return profile


@router.post("", response_model=ServerProfileResponse, status_code=201)
async def create_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_data: ServerProfileCreate,
    session: AsyncSession = Depends(get_db),
):
    """Create a new server profile."""
    # Check for unique name
    result = await session.execute(
        select(ServerProfile).where(ServerProfile.name == profile_data.name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Profile name already exists")

    profile = ServerProfile.model_validate(profile_data)
    session.add(profile)
    await session.commit()
    await session.refresh(profile)

    return profile


@router.put("/{profile_id}", response_model=ServerProfileResponse)
async def update_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
    profile_data: ServerProfileUpdate,
    session: AsyncSession = Depends(get_db),
):
    """Update a server profile."""
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Check for name conflict
    if profile_data.name and profile_data.name != profile.name:
        result = await session.execute(
            select(ServerProfile).where(ServerProfile.name == profile_data.name)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Profile name already exists")

    # Update fields
    update_data = profile_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(profile, key, value)

    profile.updated_at = datetime.now(tz=UTC)
    session.add(profile)
    await session.commit()
    await session.refresh(profile)

    return profile


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile: ServerProfile = Depends(get_profile_or_404),
    session: AsyncSession = Depends(get_db),
):
    """Delete a server profile."""
    await session.delete(profile)
    await session.commit()


@router.post("/{profile_id}/duplicate", response_model=ServerProfileResponse, status_code=201)
async def duplicate_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    new_name: str,
    profile: ServerProfile = Depends(get_profile_or_404),
    session: AsyncSession = Depends(get_db),
):
    """Duplicate a profile with a new name."""
    # Check for name conflict
    result = await session.execute(select(ServerProfile).where(ServerProfile.name == new_name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Profile name already exists")

    # Create new profile with simplified fields (embedded server)
    new_profile = ServerProfile(
        name=new_name,
        description=profile.description,
        model_path=profile.model_path,
        model_type=profile.model_type,
        context_length=profile.context_length,
        auto_start=False,  # Don't copy auto_start
        system_prompt=profile.system_prompt,
        # Generation parameters
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        top_p=profile.top_p,
        # Tool calling
        enable_prompt_injection=profile.enable_prompt_injection,
    )

    session.add(new_profile)
    await session.commit()
    await session.refresh(new_profile)

    return new_profile

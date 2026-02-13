"""Execution profiles API router.

Profiles configure execution defaults for models: inference parameters
(text/vision), audio settings, or just model selection (embeddings).
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user, get_profile_or_404
from mlx_manager.models import Model, User
from mlx_manager.models.enums import ProfileType
from mlx_manager.models.profiles import (
    ExecutionProfile,
    ExecutionProfileCreate,
    ExecutionProfileResponse,
    ExecutionProfileUpdate,
    profile_to_response,
    profile_type_for_model_type,
)

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.get("", response_model=list[ExecutionProfileResponse])
async def list_profiles(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """List all execution profiles."""
    result = await session.execute(
        select(ExecutionProfile).options(selectinload(ExecutionProfile.model))  # type: ignore[arg-type]
    )
    profiles = result.scalars().all()
    return [profile_to_response(p) for p in profiles]


@router.get("/{profile_id}", response_model=ExecutionProfileResponse)
async def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile: ExecutionProfile = Depends(get_profile_or_404),
):
    """Get a specific profile."""
    return profile_to_response(profile)


@router.post("", response_model=ExecutionProfileResponse, status_code=201)
async def create_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_data: ExecutionProfileCreate,
    session: AsyncSession = Depends(get_db),
):
    """Create a new execution profile."""
    # Check for unique name
    result = await session.execute(
        select(ExecutionProfile).where(ExecutionProfile.name == profile_data.name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Profile name already exists")

    # Validate model exists
    model = await session.get(Model, profile_data.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Determine profile_type from model
    p_type = profile_type_for_model_type(model.model_type)

    # Validate cross-type consistency
    _validate_profile_fields(p_type, profile_data)

    # Flatten DTOs into entity columns
    profile = ExecutionProfile(
        name=profile_data.name,
        description=profile_data.description,
        model_id=profile_data.model_id,
        profile_type=p_type,
        auto_start=profile_data.auto_start,
    )
    _apply_dto_to_entity(profile, profile_data, p_type)

    session.add(profile)
    await session.commit()
    await session.refresh(profile, ["model"])

    return profile_to_response(profile)


@router.put("/{profile_id}", response_model=ExecutionProfileResponse)
async def update_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
    profile_data: ExecutionProfileUpdate,
    session: AsyncSession = Depends(get_db),
):
    """Update an execution profile."""
    result = await session.execute(
        select(ExecutionProfile)
        .where(ExecutionProfile.id == profile_id)
        .options(selectinload(ExecutionProfile.model))  # type: ignore[arg-type]
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Check for name conflict
    if profile_data.name and profile_data.name != profile.name:
        result = await session.execute(
            select(ExecutionProfile).where(ExecutionProfile.name == profile_data.name)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Profile name already exists")

    # If model_id changes, update profile_type and validate
    p_type = profile.profile_type
    if profile_data.model_id is not None:
        model = await session.get(Model, profile_data.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        p_type = profile_type_for_model_type(model.model_type)
        profile.profile_type = p_type
        profile.model_id = profile_data.model_id

    # Validate cross-type consistency
    _validate_profile_fields(p_type, profile_data)

    # Apply scalar updates
    if profile_data.name is not None:
        profile.name = profile_data.name
    if profile_data.description is not None:
        profile.description = profile_data.description
    if profile_data.auto_start is not None:
        profile.auto_start = profile_data.auto_start

    # Apply type-specific updates from value objects
    _apply_dto_to_entity(profile, profile_data, p_type)

    profile.updated_at = datetime.now(tz=UTC)
    session.add(profile)
    await session.commit()
    await session.refresh(profile, ["model"])

    return profile_to_response(profile)


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    profile: ExecutionProfile = Depends(get_profile_or_404),
    session: AsyncSession = Depends(get_db),
):
    """Delete an execution profile."""
    await session.delete(profile)
    await session.commit()


@router.post("/{profile_id}/duplicate", response_model=ExecutionProfileResponse, status_code=201)
async def duplicate_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    new_name: str,
    profile: ExecutionProfile = Depends(get_profile_or_404),
    session: AsyncSession = Depends(get_db),
):
    """Duplicate a profile with a new name."""
    # Check for name conflict
    result = await session.execute(
        select(ExecutionProfile).where(ExecutionProfile.name == new_name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Profile name already exists")

    new_profile = ExecutionProfile(
        name=new_name,
        description=profile.description,
        model_id=profile.model_id,
        profile_type=profile.profile_type,
        auto_start=False,  # Don't copy auto_start
        # Inference defaults
        default_temperature=profile.default_temperature,
        default_max_tokens=profile.default_max_tokens,
        default_top_p=profile.default_top_p,
        default_context_length=profile.default_context_length,
        default_system_prompt=profile.default_system_prompt,
        default_enable_tool_injection=profile.default_enable_tool_injection,
        # Audio defaults
        default_tts_voice=profile.default_tts_voice,
        default_tts_speed=profile.default_tts_speed,
        default_tts_sample_rate=profile.default_tts_sample_rate,
        default_stt_language=profile.default_stt_language,
    )

    session.add(new_profile)
    await session.commit()
    await session.refresh(new_profile, ["model"])

    return profile_to_response(new_profile)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_profile_fields(
    profile_type: str,
    dto: ExecutionProfileCreate | ExecutionProfileUpdate,
) -> None:
    """Validate that DTO fields match the profile type."""
    if profile_type == ProfileType.AUDIO:
        if dto.inference is not None or dto.context is not None:
            raise HTTPException(
                status_code=422,
                detail="Audio profiles cannot have inference or context parameters",
            )
    elif profile_type in (ProfileType.INFERENCE, ProfileType.BASE):
        if dto.audio is not None:
            raise HTTPException(
                status_code=422,
                detail="Inference profiles cannot have audio parameters",
            )


def _apply_dto_to_entity(
    profile: ExecutionProfile,
    dto: ExecutionProfileCreate | ExecutionProfileUpdate,
    profile_type: str,
) -> None:
    """Flatten value objects from DTO into entity columns."""
    if profile_type in (ProfileType.INFERENCE, ProfileType.BASE):
        if dto.inference is not None:
            profile.default_temperature = dto.inference.temperature
            profile.default_max_tokens = dto.inference.max_tokens
            profile.default_top_p = dto.inference.top_p
        if dto.context is not None:
            profile.default_context_length = dto.context.context_length
            profile.default_system_prompt = dto.context.system_prompt
            profile.default_enable_tool_injection = dto.context.enable_tool_injection
    elif profile_type == ProfileType.AUDIO:
        if dto.audio is not None:
            profile.default_tts_voice = dto.audio.tts_voice
            profile.default_tts_speed = dto.audio.tts_speed
            profile.default_tts_sample_rate = dto.audio.tts_sample_rate
            profile.default_stt_language = dto.audio.stt_language

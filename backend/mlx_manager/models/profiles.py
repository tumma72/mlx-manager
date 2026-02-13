"""Execution profile entity and DTOs.

Replaces the flat ``ServerProfile`` with a polymorphic ``ExecutionProfile``
that uses Single Table Inheritance with a ``profile_type`` discriminator.

Profile types:
- ``inference`` (TEXT_GEN, VISION): temperature, max_tokens, system_prompt, etc.
- ``audio`` (AUDIO): tts_voice, tts_speed, stt_language, etc.
- ``base`` (EMBEDDINGS): model selection only, no extra defaults.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel
from sqlmodel import Field, Relationship, SQLModel

from mlx_manager.models.enums import ProfileType
from mlx_manager.models.value_objects import AudioDefaults, InferenceContext, InferenceParams

if TYPE_CHECKING:
    from mlx_manager.models.entities import Model


# ---------------------------------------------------------------------------
# Domain entity (SQLModel table)
# ---------------------------------------------------------------------------


class ExecutionProfile(SQLModel, table=True):
    """Execution profile with type-specific defaults.

    Single-table design: all inference + audio default fields live in one
    ``execution_profiles`` table.  The ``profile_type`` discriminator
    identifies which subset of nullable columns is populated.
    """

    __tablename__ = "execution_profiles"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    model_id: int | None = Field(default=None, foreign_key="models.id")
    profile_type: str = Field(default=ProfileType.INFERENCE)
    auto_start: bool = Field(default=False)
    launchd_installed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    # --- Inference defaults (TEXT_GEN, VISION) ---
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    default_max_tokens: int | None = Field(default=None, ge=1, le=128000)
    default_top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    default_context_length: int | None = None
    default_system_prompt: str | None = None
    default_enable_tool_injection: bool = Field(default=False)

    # --- Audio defaults (AUDIO) ---
    default_tts_voice: str | None = None
    default_tts_speed: float | None = Field(default=None, ge=0.25, le=4.0)
    default_tts_sample_rate: int | None = None
    default_stt_language: str | None = None

    # Relationship
    model: Optional["Model"] = Relationship(back_populates="profiles")


# ---------------------------------------------------------------------------
# DTOs for API requests/responses
# ---------------------------------------------------------------------------


class ExecutionProfileCreate(BaseModel):
    """Create an execution profile. Server determines profile_type from model."""

    name: str
    description: str | None = None
    model_id: int
    auto_start: bool = False
    # Type-specific defaults (server validates against model_type)
    inference: InferenceParams | None = None
    context: InferenceContext | None = None
    audio: AudioDefaults | None = None


class ExecutionProfileUpdate(BaseModel):
    """Partial update. Only provided fields are changed."""

    name: str | None = None
    description: str | None = None
    model_id: int | None = None
    auto_start: bool | None = None
    inference: InferenceParams | None = None
    context: InferenceContext | None = None
    audio: AudioDefaults | None = None


class ExecutionProfileResponse(BaseModel):
    """Full profile with denormalized model info and type-specific defaults."""

    id: int
    name: str
    description: str | None = None
    model_id: int | None = None
    model_repo_id: str | None = None
    model_type: str | None = None
    profile_type: str
    auto_start: bool
    launchd_installed: bool
    inference: InferenceParams | None = None
    context: InferenceContext | None = None
    audio: AudioDefaults | None = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

MODEL_TYPE_TO_PROFILE_TYPE: dict[str | None, str] = {
    "text-gen": ProfileType.INFERENCE,
    "vision": ProfileType.INFERENCE,
    "audio": ProfileType.AUDIO,
    "embeddings": ProfileType.BASE,
    None: ProfileType.INFERENCE,
}


def profile_type_for_model_type(model_type: str | None) -> str:
    """Determine profile_type from a model's model_type."""
    return MODEL_TYPE_TO_PROFILE_TYPE.get(model_type, ProfileType.INFERENCE)


def profile_to_response(profile: ExecutionProfile) -> ExecutionProfileResponse:
    """Build a response DTO from an ExecutionProfile entity."""
    # Hydrate value objects from flat columns
    inference: InferenceParams | None = None
    context: InferenceContext | None = None
    audio: AudioDefaults | None = None

    if profile.profile_type in (ProfileType.INFERENCE, ProfileType.BASE):
        inference = InferenceParams(
            temperature=profile.default_temperature,
            max_tokens=profile.default_max_tokens,
            top_p=profile.default_top_p,
        )
        context = InferenceContext(
            context_length=profile.default_context_length,
            system_prompt=profile.default_system_prompt,
            enable_tool_injection=profile.default_enable_tool_injection,
        )
    elif profile.profile_type == ProfileType.AUDIO:
        audio = AudioDefaults(
            tts_voice=profile.default_tts_voice,
            tts_speed=profile.default_tts_speed,
            tts_sample_rate=profile.default_tts_sample_rate,
            stt_language=profile.default_stt_language,
        )

    return ExecutionProfileResponse(
        id=profile.id,  # type: ignore[arg-type]
        name=profile.name,
        description=profile.description,
        model_id=profile.model_id,
        model_repo_id=profile.model.repo_id if profile.model else None,
        model_type=profile.model.model_type if profile.model else None,
        profile_type=profile.profile_type,
        auto_start=profile.auto_start,
        launchd_installed=profile.launchd_installed,
        inference=inference,
        context=context,
        audio=audio,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )

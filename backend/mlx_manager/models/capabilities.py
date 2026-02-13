"""Capability model for type-specific probe results.

Separates type-specific capability fields from the ``Model`` table into a
dedicated ``model_capabilities`` table with a ``capability_type`` discriminator.

Single-table design: all capability fields live in one table with nullable
columns.  The ``capability_type`` column identifies which subset of fields
is meaningful for a given model (text-gen, vision, embeddings, audio).
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from mlx_manager.models.entities import Model


class ModelCapabilities(SQLModel, table=True):
    """Capability record for a probed model.

    One row per model.  The ``capability_type`` discriminator tells the
    consumer which subset of nullable columns is populated.
    """

    __tablename__ = "model_capabilities"

    id: int | None = Field(default=None, primary_key=True)
    model_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("models.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
    )
    capability_type: str = Field(
        sa_column=Column(String, nullable=False, default="base"),
    )

    # Probe metadata
    probed_at: datetime | None = None
    probe_version: int | None = None

    # Shared across text/vision
    model_family: str | None = None

    # --- Text-gen fields ---
    supports_native_tools: bool | None = None
    supports_thinking: bool | None = None
    tool_format: str | None = None
    practical_max_tokens: int | None = None
    tool_parser_id: str | None = None
    thinking_parser_id: str | None = None

    # --- Vision fields ---
    supports_multi_image: bool | None = None
    supports_video: bool | None = None

    # --- Embedding fields ---
    embedding_dimensions: int | None = None
    max_sequence_length: int | None = None
    is_normalized: bool | None = None

    # --- Audio fields ---
    supports_tts: bool | None = None
    supports_stt: bool | None = None

    # Relationship back to Model
    model: Optional["Model"] = Relationship(back_populates="capabilities")


# ---------------------------------------------------------------------------
# DTO for API responses
# ---------------------------------------------------------------------------


class CapabilitiesResponse(BaseModel):
    """Flat DTO for capability data in API responses.

    Uses a type discriminator + all-optional fields so the frontend
    can branch on ``capability_type`` while keeping a single type.
    """

    capability_type: str
    probed_at: datetime | None = None
    probe_version: int | None = None
    model_family: str | None = None

    # Text-gen / Vision
    supports_native_tools: bool | None = None
    supports_thinking: bool | None = None
    tool_format: str | None = None
    practical_max_tokens: int | None = None
    tool_parser_id: str | None = None
    thinking_parser_id: str | None = None

    # Vision
    supports_multi_image: bool | None = None
    supports_video: bool | None = None

    # Embeddings
    embedding_dimensions: int | None = None
    max_sequence_length: int | None = None
    is_normalized: bool | None = None

    # Audio
    supports_tts: bool | None = None
    supports_stt: bool | None = None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def capabilities_to_response(caps: ModelCapabilities) -> CapabilitiesResponse:
    """Serialize a ModelCapabilities instance to a flat DTO."""
    return CapabilitiesResponse(
        capability_type=caps.capability_type,
        probed_at=caps.probed_at,
        probe_version=caps.probe_version,
        model_family=caps.model_family,
        supports_native_tools=caps.supports_native_tools,
        supports_thinking=caps.supports_thinking,
        tool_format=caps.tool_format,
        practical_max_tokens=caps.practical_max_tokens,
        tool_parser_id=caps.tool_parser_id,
        thinking_parser_id=caps.thinking_parser_id,
        supports_multi_image=caps.supports_multi_image,
        supports_video=caps.supports_video,
        embedding_dimensions=caps.embedding_dimensions,
        max_sequence_length=caps.max_sequence_length,
        is_normalized=caps.is_normalized,
        supports_tts=caps.supports_tts,
        supports_stt=caps.supports_stt,
    )

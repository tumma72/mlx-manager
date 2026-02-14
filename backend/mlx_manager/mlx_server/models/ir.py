"""Intermediate Representation types for adapter pipeline.

These protocol-neutral types flow through the 3-layer adapter pipeline:
  Layer 1 (ModelAdapter)  -> PreparedInput
  Layer 2 (StreamProcessor) -> StreamEvent, TextResult
  Layer 3 (ProtocolFormatter) -> protocol-specific responses

All types use Pydantic BaseModel for consistency with the project's data model standard.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# -- Input IR ------------------------------------------------------------------


class PreparedInput(BaseModel):
    """Model-ready input after adapter processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    token_ids: list[int] | None = None
    stop_token_ids: list[int] | None = None
    pixel_values: Any | None = None  # Vision: preprocessed images
    generation_params: dict[str, Any] | None = None


# -- Stream IR -----------------------------------------------------------------


class StreamEvent(BaseModel):
    """Single event emitted during streaming.

    Fields mirror the OpenAI o1/o3 reasoning model API spec:
    - content: Regular response text
    - reasoning_content: Text inside thinking tags (thinking phase)
    - tool_call_delta: Incremental tool call data (future use)
    - is_complete: True when a thinking pattern ends (transition point)
    """

    type: Literal["content", "reasoning_content", "tool_call_delta"] | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_call_delta: dict[str, Any] | None = None
    is_complete: bool = False


# -- Output IR -----------------------------------------------------------------


class AdapterResult(BaseModel):
    """Base result type for all adapters."""

    finish_reason: str = "stop"


class TextResult(AdapterResult):
    """Text generation result (TEXT_GEN and VISION)."""

    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"


class EmbeddingResult(AdapterResult):
    """Embedding generation result."""

    embeddings: list[list[float]] = Field(default_factory=list)
    dimensions: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"


class AudioResult(AdapterResult):
    """TTS audio generation result."""

    audio_bytes: bytes = b""
    sample_rate: int = 0
    format: str = ""
    finish_reason: str = "stop"


class TranscriptionResult(AdapterResult):
    """STT transcription result."""

    text: str = ""
    segments: list[dict[str, Any]] | None = None
    language: str | None = None
    finish_reason: str = "stop"

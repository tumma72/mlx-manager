"""Intermediate Representation types for adapter pipeline.

These protocol-neutral types flow through the 3-layer adapter pipeline:
  Layer 1 (ModelAdapter)  -> PreparedInput
  Layer 2 (StreamProcessor) -> StreamEvent, TextResult
  Layer 3 (ProtocolFormatter) -> protocol-specific responses

All types use Pydantic BaseModel for consistency with the project's data model standard.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlx_manager.models.enums import ApiType
from mlx_manager.models.value_objects import InferenceParams

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


# -- Request IR ----------------------------------------------------------------


class InternalRequest(BaseModel):
    """Protocol-neutral request IR used between protocol translators and router/inference.

    Created by ProtocolFormatter.parse_request() from protocol-specific requests.
    Carries the original request and protocol for passthrough optimization in cloud routing.
    """

    model: str
    messages: list[dict[str, Any]]
    params: InferenceParams
    stream: bool = False
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    images: list[str] | None = None  # Base64 data URLs for vision
    original_request: Any | None = None  # The raw protocol-specific request object
    original_protocol: ApiType | None = None  # Which protocol created this IR

    model_config = ConfigDict(arbitrary_types_allowed=True)


class InferenceResult(BaseModel):
    """Non-streaming inference result with token counts."""

    result: TextResult
    prompt_tokens: int
    completion_tokens: int


class RoutingOutcome(BaseModel):
    """Tagged union result from the backend router.

    Exactly one of the four fields should be set:
    - ir_result: non-streaming local or cross-protocol result (needs formatting)
    - ir_stream: streaming local or cross-protocol result (needs formatting)
    - raw_response: same-protocol passthrough non-streaming (forward as-is)
    - raw_stream: same-protocol passthrough streaming (forward as-is)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ir_result: InferenceResult | None = None
    ir_stream: AsyncGenerator | None = None
    raw_response: dict[str, Any] | BaseModel | None = None
    raw_stream: AsyncGenerator | None = None

    @property
    def is_passthrough(self) -> bool:
        """True if this outcome carries a raw same-protocol response."""
        return self.raw_response is not None or self.raw_stream is not None

    @property
    def is_streaming(self) -> bool:
        """True if this outcome carries a streaming response."""
        return self.ir_stream is not None or self.raw_stream is not None

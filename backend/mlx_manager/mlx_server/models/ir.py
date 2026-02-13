"""Intermediate Representation types for adapter pipeline.

These protocol-neutral types flow through the 3-layer adapter pipeline:
  Layer 1 (ModelAdapter)  -> PreparedInput
  Layer 2 (StreamProcessor) -> StreamEvent, TextResult
  Layer 3 (ProtocolFormatter) -> protocol-specific responses

All types are simple dataclasses for minimal overhead and serialization ease.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# -- Input IR ------------------------------------------------------------------


@dataclass
class PreparedInput:
    """Model-ready input after adapter processing."""

    prompt: str
    token_ids: list[int] | None = None
    stop_token_ids: list[int] | None = None
    pixel_values: Any | None = None  # Vision: preprocessed images
    generation_params: dict[str, Any] | None = None


# -- Stream IR -----------------------------------------------------------------


@dataclass
class StreamEvent:
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


@dataclass
class AdapterResult(ABC):
    """Base result type for all adapters."""

    finish_reason: str = "stop"

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class TextResult(AdapterResult):
    """Text generation result (TEXT_GEN and VISION)."""

    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
        }


@dataclass
class EmbeddingResult(AdapterResult):
    """Embedding generation result."""

    embeddings: list[list[float]] = field(default_factory=list)
    dimensions: int = 0
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddings": self.embeddings,
            "dimensions": self.dimensions,
            "finish_reason": self.finish_reason,
        }


@dataclass
class AudioResult(AdapterResult):
    """TTS audio generation result."""

    audio_bytes: bytes = b""
    sample_rate: int = 0
    format: str = ""
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_bytes": self.audio_bytes,
            "sample_rate": self.sample_rate,
            "format": self.format,
            "finish_reason": self.finish_reason,
        }


@dataclass
class TranscriptionResult(AdapterResult):
    """STT transcription result."""

    text: str = ""
    segments: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "segments": self.segments,
            "finish_reason": self.finish_reason,
        }

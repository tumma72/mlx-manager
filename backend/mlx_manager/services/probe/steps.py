"""Probe step and result dataclasses.

These are the building blocks shared across all probe strategies.
ProbeStep is yielded as SSE events for streaming progress to the UI.
ProbeResult accumulates capabilities discovered during probing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProbeStep:
    """A single step in the probe process, yielded as an SSE event."""

    step: str
    status: str  # "running", "completed", "failed", "skipped"
    capability: str | None = None
    value: Any = None
    error: str | None = None

    def to_sse(self) -> str:
        """Serialize to SSE event data."""
        data: dict[str, Any] = {"step": self.step, "status": self.status}
        if self.capability is not None:
            data["capability"] = self.capability
        if self.value is not None:
            data["value"] = self.value
        if self.error is not None:
            data["error"] = self.error
        return f"data: {json.dumps(data)}\n\n"


@dataclass
class ProbeResult:
    """Accumulated probe results across all capability checks.

    Fields are set by type-specific strategies. Irrelevant fields
    remain None and are not persisted to the database.
    """

    # Text-gen capabilities
    supports_native_tools: bool | None = None
    supports_thinking: bool | None = None
    tool_format: str | None = None
    practical_max_tokens: int | None = None

    # Composable adapter fields
    model_family: str | None = None
    tool_parser_id: str | None = None
    thinking_parser_id: str | None = None

    # Vision capabilities
    supports_multi_image: bool | None = None
    supports_video: bool | None = None

    # Embeddings capabilities
    embedding_dimensions: int | None = None
    max_sequence_length: int | None = None
    is_normalized: bool | None = None

    # Audio capabilities
    supports_tts: bool | None = None
    supports_stt: bool | None = None

    # Metadata
    model_type: str | None = None
    errors: list[str] = field(default_factory=list)

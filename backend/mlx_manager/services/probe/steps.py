"""Probe step and result models.

These are the building blocks shared across all probe strategies.
ProbeStep is yielded as SSE events for streaming progress to the UI.
ProbeResult accumulates capabilities discovered during probing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DiagnosticLevel(StrEnum):
    """Severity level for probe diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ACTION_NEEDED = "action_needed"


class DiagnosticCategory(StrEnum):
    """Category of probe diagnostic."""

    FAMILY = "family"
    TOOL_DIALECT = "tool_dialect"
    THINKING_DIALECT = "thinking_dialect"
    TYPE = "type"
    UNSUPPORTED = "unsupported"


class ProbeDiagnostic(BaseModel):
    """A diagnostic message produced during probing."""

    level: DiagnosticLevel
    category: DiagnosticCategory
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ProbeStep(BaseModel):
    """A single step in the probe process, yielded as an SSE event."""

    step: str
    status: str  # "running", "completed", "failed", "skipped"
    capability: str | None = None
    value: Any = None
    error: str | None = None
    details: dict[str, Any] | None = None
    diagnostics: list[ProbeDiagnostic] | None = None

    def to_sse(self) -> str:
        """Serialize to SSE event data."""
        data: dict[str, Any] = {"step": self.step, "status": self.status}
        if self.capability is not None:
            data["capability"] = self.capability
        if self.value is not None:
            data["value"] = self.value
        if self.error is not None:
            data["error"] = self.error
        if self.details is not None:
            data["details"] = self.details
        if self.diagnostics:
            data["diagnostics"] = [d.model_dump() for d in self.diagnostics]
        return f"data: {json.dumps(data)}\n\n"


class TagDiscovery(BaseModel):
    """A tag pattern discovered in model output with parser mappings."""

    name: str  # e.g. "TOOL_CALLS", "tool_call", "think"
    style: str  # "xml", "bracket", or "special"
    paired: bool  # True if closing tag found
    matched_parsers: list[str] = Field(default_factory=list)


class ProbeResult(BaseModel):
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

    # Template options
    template_params: dict[str, Any] | None = None

    # Tag discovery
    discovered_tool_tags: list[dict[str, Any]] | None = None
    discovered_thinking_tags: list[dict[str, Any]] | None = None

    # Metadata
    model_type: str | None = None
    errors: list[str] = Field(default_factory=list)
    diagnostics: list[ProbeDiagnostic] = Field(default_factory=list)


class StepContext:
    """Mutable context for probe step lifecycle.

    Used with the probe_step() async context manager to eliminate
    the repetitive try/yield-running/yield-completed/except/yield-failed
    boilerplate in probe strategies.

    Usage in an async generator::

        async with probe_step("check_X", "cap_name") as ctx:
            yield ctx.running  # Emit "running" step to SSE
            result = do_work()
            ctx.value = result
        yield ctx.result  # Emit "completed" or "failed" step
    """

    __slots__ = ("step", "capability", "value", "details", "diagnostics", "_failed", "_error")

    def __init__(self, step: str, capability: str | None = None) -> None:
        self.step = step
        self.capability = capability
        self.value: Any = None
        self.details: dict[str, Any] | None = None
        self.diagnostics: list[ProbeDiagnostic] | None = None
        self._failed: bool = False
        self._error: str | None = None

    @property
    def running(self) -> ProbeStep:
        """The 'running' step to yield before starting work."""
        return ProbeStep(step=self.step, status="running")

    @property
    def result(self) -> ProbeStep:
        """The 'completed' or 'failed' step to yield after the context exits."""
        if self._failed:
            return ProbeStep(step=self.step, status="failed", error=self._error)
        return ProbeStep(
            step=self.step,
            status="completed",
            capability=self.capability,
            value=self.value,
            details=self.details,
            diagnostics=self.diagnostics,
        )

    def fail(self, error: str) -> None:
        """Explicitly mark this step as failed (e.g. for boolean checks)."""
        self._failed = True
        self._error = error


@asynccontextmanager
async def probe_step(step: str, capability: str | None = None) -> AsyncGenerator[StepContext, None]:
    """Async context manager for probe step lifecycle.

    Catches exceptions inside the block and records them as failures.
    The caller is responsible for yielding ``ctx.running`` (before work)
    and ``ctx.result`` (after the block exits).
    """
    ctx = StepContext(step=step, capability=capability)
    try:
        yield ctx
    except Exception as e:
        ctx._failed = True
        ctx._error = str(e)

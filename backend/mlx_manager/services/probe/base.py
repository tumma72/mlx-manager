"""Base probe classes providing shared hierarchy and generative probing.

BaseProbe is the minimal ABC matching the ProbeStrategy protocol.
GenerativeProbe extends it with _generate() for adapter-based generation
used by the ProbingCoordinator's sweep methods.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .steps import (
    ProbeResult,
    ProbeStep,
    TagDiscovery,
)

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.mlx_server.models.pool import LoadedModel


class BaseProbe(ABC):
    """Abstract base for all probe strategies.

    Provides the minimal interface matching the ProbeStrategy protocol.
    """

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """The model type this strategy handles."""
        ...

    @abstractmethod
    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        """Run type-specific probe steps."""
        ...  # pragma: no cover
        if False:  # pragma: no cover
            yield ProbeStep(step="", status="")


# ---------------------------------------------------------------------------
# Constants shared by generative probes (text-gen + vision)
# ---------------------------------------------------------------------------

_TOOL_PROBE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Tokyo",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

_TOOL_PROBE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
]


def _build_generic_tool_prompt(tools: list[dict[str, Any]]) -> str:
    """Build a format-agnostic tool prompt — no format instructions.

    The model responds in whatever tool-calling format it was trained on,
    which we then discover by sweeping all registered parsers.
    """
    return (
        "You have access to the following tools. Use them when relevant.\n\n"
        f"Tools:\n{json.dumps(tools, indent=2)}"
    )


_KNOWN_BENIGN_TAGS: frozenset[str] = frozenset(
    {
        "think",
        "thinking",
        "reasoning",
        "reflection",
        "code",
        "output",
        "result",
        "step",
        "answer",
        "solution",
    }
)

# Backward-compat alias used by old diagnostic code paths
_KNOWN_XML_TAGS: frozenset[str] = _KNOWN_BENIGN_TAGS | frozenset(
    {"tool_call", "tool_response", "function_call"}
)


@dataclass
class DetectedTag:
    """A tag pattern discovered in model output."""

    name: str  # e.g. "TOOL_CALLS", "tool_call"
    style: str  # "xml" (<tag>) or "bracket" ([TAG])
    paired: bool  # True if closing tag found


def _detect_all_tags(output: str) -> list[DetectedTag]:
    """Scan output for ANY tag-like patterns — XML, bracket, and special token styles.

    XML:     <tag>...</tag>          matched by  <([a-zA-Z_][\\w-]*)>
    Bracket: [TAG]...[/TAG]         matched by  \\[([A-Z_][\\w_]*)\\]
    Special: <|token|>...<|token|>   matched by  <\\|([a-zA-Z_][\\w_]*)\\|>
    """
    tags: list[DetectedTag] = []
    seen: set[str] = set()

    # XML-style: <tag>...</tag>
    for match in re.finditer(r"<([a-zA-Z_][\w-]*)(?:\s[^>]*)?>", output):
        tag_name = match.group(1)
        key = f"xml:{tag_name.lower()}"
        if key in seen:
            continue
        seen.add(key)
        has_close = bool(re.search(rf"</{re.escape(tag_name)}>", output, re.IGNORECASE))
        tags.append(DetectedTag(name=tag_name.lower(), style="xml", paired=has_close))

    # Bracket-style: [TAG] and optionally [/TAG]
    for match in re.finditer(r"\[([A-Z][A-Z_\d]*)\]", output):
        tag_name = match.group(1)
        key = f"bracket:{tag_name}"
        if key in seen:
            continue
        seen.add(key)
        has_close = bool(re.search(rf"\[/{re.escape(tag_name)}\]", output, re.IGNORECASE))
        tags.append(DetectedTag(name=tag_name, style="bracket", paired=has_close))

    # Special token style: <|token_name|>
    for match in re.finditer(r"<\|([a-zA-Z_][\w_]*)\|>", output):
        token_name = match.group(1)
        key = f"special:{token_name.lower()}"
        if key in seen:
            continue
        seen.add(key)
        # Check for a second occurrence (special tokens often appear as open/close pairs)
        all_occurrences = list(re.finditer(rf"<\|{re.escape(token_name)}\|>", output))
        has_pair = len(all_occurrences) >= 2
        tags.append(DetectedTag(name=token_name.lower(), style="special", paired=has_pair))

    return tags


def _build_marker_to_parsers(
    parsers: dict[str, type],
) -> dict[str, list[str]]:
    """Build {start_marker: [parser_ids]} lookup from registered parsers' stream_markers.

    Multiple parsers can share the same marker (e.g. hermes, glm4_native, glm4_xml
    all use ``<tool_call>``).
    """
    marker_map: dict[str, list[str]] = {}
    for pid, pcls in parsers.items():
        if pid == "null":
            continue
        for start, _end in pcls().stream_markers:
            if not start:
                continue
            marker_map.setdefault(start, []).append(pid)
    return marker_map


def _scan_for_known_markers(
    output: str,
    marker_map: dict[str, list[str]],
) -> set[str]:
    """Direct string search for ALL registered start markers in output.

    Returns the set of parser_ids whose start markers appear in the output.
    This catches markers that regex-based ``_detect_all_tags()`` might miss,
    e.g. ``<function=`` (not a standard XML tag), ``<|python_tag|>``
    (special token delimiters), ``[TOOL_CALLS]`` (safety net).
    """
    found_parsers: set[str] = set()
    for marker, pids in marker_map.items():
        if marker in output:
            found_parsers.update(pids)
    return found_parsers


def _discover_and_map_tags(
    output: str,
    parsers: dict[str, type],
) -> list[TagDiscovery]:
    """Primary tag discovery pipeline: regex detection + direct marker scan + merge.

    1. ``_detect_all_tags(output)`` — generic regex-based detection
    2. ``_build_marker_to_parsers(parsers)`` — marker → parser lookup
    3. ``_scan_for_known_markers(output, marker_map)`` — direct marker scan
    4. Merge: enrich each detected tag with ``matched_parsers`` list
    5. Add any parsers found by direct scan but not by regex detection
    """
    detected = _detect_all_tags(output)
    marker_map = _build_marker_to_parsers(parsers)
    direct_parsers = _scan_for_known_markers(output, marker_map)

    # Build reverse lookup: parser_id → which markers matched
    parsers_covered_by_tags: set[str] = set()
    discoveries: list[TagDiscovery] = []

    for tag in detected:
        # Build possible marker strings for this tag
        candidates: list[str] = []
        if tag.style == "xml":
            candidates = [f"<{tag.name}>", f"<|{tag.name}|>"]
        elif tag.style == "bracket":
            candidates = [f"[{tag.name}]"]
        elif tag.style == "special":
            candidates = [f"<|{tag.name}|>"]

        matched: list[str] = []
        for candidate in candidates:
            if candidate in marker_map:
                matched.extend(marker_map[candidate])
                parsers_covered_by_tags.update(marker_map[candidate])

        discoveries.append(
            TagDiscovery(
                name=tag.name,
                style=tag.style,
                paired=tag.paired,
                matched_parsers=sorted(set(matched)),
            )
        )

    # Add parsers found by direct scan but not covered by regex-detected tags
    uncovered = direct_parsers - parsers_covered_by_tags
    if uncovered:
        # Map these back to their markers for reporting
        for pid in sorted(uncovered):
            for marker, pids in marker_map.items():
                if pid in pids:
                    # Determine style from marker format
                    if marker.startswith("<|") and marker.endswith("|>"):
                        style = "special"
                        name = marker[2:-2].lower()
                    elif marker.startswith("["):
                        style = "bracket"
                        name = marker.strip("[]")
                    elif marker.startswith("<"):
                        style = "xml"
                        name = marker.strip("<>").lower()
                    else:
                        style = "unknown"
                        name = marker
                    discoveries.append(
                        TagDiscovery(
                            name=name,
                            style=style,
                            paired=False,  # Not detected by regex, just the marker
                            matched_parsers=[pid],
                        )
                    )
                    break  # One entry per parser

    return discoveries


class GenerativeProbe(BaseProbe):
    """ABC for probes that can generate text and test thinking/tool capabilities.

    Subclasses implement ``_generate()`` using their backend-specific
    generation path (mlx-lm for text, mlx-vlm for vision).
    """

    async def _generate(
        self,
        loaded: LoadedModel,
        messages: list[dict],
        tools: list[dict] | None = None,
        template_options: dict[str, Any] | None = None,
        max_tokens: int = 800,
    ) -> TextResult:
        """Generate a response using the adapter's full pipeline.

        Uses adapter.generate() which handles prepare_input → generation
        → process_complete for all model types (text and vision).
        Returns the full TextResult so callers can inspect reasoning_content
        directly instead of re-parsing already-cleaned content.
        """
        adapter = loaded.adapter
        if adapter is None:
            msg = "No adapter available for generation"
            raise RuntimeError(msg)

        # Temporarily configure adapter with probe-specific template options
        if template_options is not None:
            adapter.configure(template_options=template_options)

        try:
            result = await adapter.generate(
                model=loaded.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                tools=tools,
            )
            return result
        finally:
            # Reset template options after probe
            if template_options is not None:
                adapter.configure(template_options=None)


# ---------------------------------------------------------------------------
# Module-level helpers (shared by generative probes)
# ---------------------------------------------------------------------------


_THINKING_TAGS = ("think", "thinking", "reasoning", "reflection")

# Bracket-style thinking tags (e.g., Mistral [THINK]...[/THINK])
_BRACKET_THINKING_TAGS = ("THINK",)


def _find_unclosed_thinking_tag(output: str) -> str | None:
    """Check for an opening thinking tag without a matching close tag.

    Returns the tag name if found (e.g. "think" or "THINK"), or None.
    This indicates the model started thinking but its output was
    truncated before the closing tag.
    Checks both XML-style <tag> and bracket-style [TAG] patterns.
    """
    # XML-style
    for tag in _THINKING_TAGS:
        open_re = re.compile(rf"<{tag}>", re.IGNORECASE)
        close_re = re.compile(rf"</{tag}>", re.IGNORECASE)
        if open_re.search(output) and not close_re.search(output):
            return tag
    # Bracket-style
    for tag in _BRACKET_THINKING_TAGS:
        if f"[{tag}]" in output and f"[/{tag}]" not in output:
            return tag
    return None


_GENERIC_THINKING_RE = re.compile(r"^\s*<([a-z_]+)>([\s\S]+?)</\1>", re.IGNORECASE)
_NON_THINKING_TAGS = {"tool_call", "function_call", "code", "output", "result"}


def _detect_unknown_thinking_tags(output: str) -> str | None:
    """Detect tag patterns at start of output that look like thinking wrappers.

    Checks both XML-style (<tag>...</tag>) and bracket-style ([TAG]...[/TAG]).
    """
    # XML-style
    match = _GENERIC_THINKING_RE.match(output)
    if match:
        tag = match.group(1).lower()
        if tag not in _NON_THINKING_TAGS:
            return tag

    # Bracket-style at start of output
    bracket_match = re.match(r"^\s*\[([A-Z][A-Z_\d]*)\]", output)
    if bracket_match:
        tag_name = bracket_match.group(1)
        # Check if it has a closing tag (thinking-like pattern)
        if re.search(rf"\[/{re.escape(tag_name)}\]", output, re.IGNORECASE):
            # Only flag if not already a known thinking tag handled by parsers
            if tag_name.lower() not in _KNOWN_BENIGN_TAGS:
                return tag_name

    return None


def _detect_unknown_xml_tags(output: str) -> set[str]:
    """Scan output for XML-style tags not in the known set.

    Kept for backward compatibility. For generic detection including
    bracket-style tags, use _detect_all_tags() instead.
    """
    tag_pattern = re.compile(r"<([a-zA-Z_][\w-]*)(?:\s[^>]*)?>")
    found_tags: set[str] = set()
    for match in tag_pattern.finditer(output):
        tag_name = match.group(1).lower()
        close_pattern = re.compile(rf"</{re.escape(match.group(1))}>", re.IGNORECASE)
        if close_pattern.search(output):
            found_tags.add(tag_name)

    unknown = found_tags - _KNOWN_XML_TAGS
    return unknown


# Sentencepiece boundary marker left in detokenized output
_SP_MARKER = "\u2581"  # ▁

# Regex: space inside a JSON quoted key, e.g. '" name "' instead of '"name"'
_SPACED_JSON_KEY = re.compile(r'"\s+\w+\s+"')


def _has_tokenization_artifacts(output: str) -> bool:
    """Detect garbled output from broken detokenization.

    Returns True if the output contains sentencepiece boundary markers (▁)
    or has spaces inside JSON string delimiters — strong signals that the
    model architecture is not properly supported by the installed mlx-lm.
    """
    if _SP_MARKER in output:
        return True
    if _SPACED_JSON_KEY.search(output):
        return True
    return False

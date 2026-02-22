"""Base probe classes providing shared hierarchy and generative probing.

BaseProbe is the minimal ABC matching the ProbeStrategy protocol.
GenerativeProbe extends it with shared thinking/tool verification
logic that both TextGenProbe and VisionProbe can reuse.
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
    DiagnosticCategory,
    DiagnosticLevel,
    ProbeDiagnostic,
    ProbeResult,
    ProbeStep,
    TagDiscovery,
)

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
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

    # ------------------------------------------------------------------
    # Shared thinking/tool verification
    # ------------------------------------------------------------------

    async def _verify_thinking_support(
        self,
        loaded: LoadedModel,
        adapter: ModelAdapter,
        template_params: dict[str, Any] | None = None,
    ) -> tuple[bool, str, list[ProbeDiagnostic]]:
        """Detect thinking support: one generation, scan for known tag patterns."""
        from mlx_manager.mlx_server.parsers import THINKING_PARSERS

        diagnostics: list[ProbeDiagnostic] = []
        has_enable_thinking = template_params is not None and "enable_thinking" in template_params

        messages = [
            {
                "role": "user",
                "content": (
                    "A farmer has 3 foxes and 5 chickens. Each fox eats 2 chickens per day. "
                    "After 1 day, how many chickens are left? Explain your reasoning step by step."
                ),
            }
        ]

        template_options = {"enable_thinking": True} if has_enable_thinking else None

        try:
            gen_result = await self._generate(
                loaded,
                messages,
                template_options=template_options,
                max_tokens=2000,
            )
        except Exception as e:
            logger.debug("Thinking probe generation failed: {}", e)
            diagnostics.append(
                ProbeDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    category=DiagnosticCategory.THINKING_DIALECT,
                    message="Thinking probe failed due to generation error",
                    details={"error": str(e)},
                )
            )
            return (False, "null", diagnostics)

        raw_output = gen_result.content

        # Scan raw output against each registered parser
        for parser_id, parser_cls in THINKING_PARSERS.items():
            if parser_id == "null":
                continue
            parser = parser_cls()
            if parser.extract(raw_output) is not None:
                logger.info("Thinking probe: detected (parser={})", parser_id)
                return (True, parser_id, diagnostics)

        # Check for unclosed tags
        unclosed = _find_unclosed_thinking_tag(raw_output)
        if unclosed:
            logger.info("Thinking probe: unclosed <{}> tag, retrying", unclosed)
            try:
                retry_result = await self._generate(
                    loaded,
                    [
                        messages[0],
                        {"role": "assistant", "content": raw_output},
                        {
                            "role": "user",
                            "content": "Please answer the same question more concisely.",
                        },
                    ],
                    template_options=template_options,
                    max_tokens=4000,
                )
                for parser_id, parser_cls in THINKING_PARSERS.items():
                    if parser_id == "null":
                        continue
                    parser = parser_cls()
                    if parser.extract(retry_result.content) is not None:
                        logger.info("Thinking probe: detected on retry (parser={})", parser_id)
                        return (True, parser_id, diagnostics)
            except Exception as e:
                logger.debug("Thinking probe retry failed: {}", e)

            # Retry didn't produce a closed tag either — match unclosed tag
            # against registered parsers' stream markers to identify the parser
            for parser_id, parser_cls in THINKING_PARSERS.items():
                if parser_id == "null":
                    continue
                parser = parser_cls()
                for open_tag, _close_tag in parser.stream_markers:
                    # Normalize tag name: strip XML (<>) or bracket ([]) delimiters
                    marker_name = open_tag.strip("<>").strip("[]").lower()
                    if marker_name == unclosed.lower():
                        logger.info(
                            "Thinking probe: unclosed <{}> matches parser={}",
                            unclosed,
                            parser_id,
                        )
                        return (True, parser_id, diagnostics)

        # Unknown tag detection
        unknown_tag = _detect_unknown_thinking_tags(raw_output)
        if unknown_tag:
            diagnostics.append(
                ProbeDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    category=DiagnosticCategory.THINKING_DIALECT,
                    message=(
                        f"Detected unknown thinking-like tag <{unknown_tag}> in output "
                        f"that doesn't match any registered parser"
                    ),
                    details={
                        "detected_tag": unknown_tag,
                        "registered_parsers": [pid for pid in THINKING_PARSERS if pid != "null"],
                    },
                )
            )
        elif has_enable_thinking:
            diagnostics.append(
                ProbeDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    category=DiagnosticCategory.THINKING_DIALECT,
                    message=(
                        "Template has enable_thinking parameter but probe could not "
                        "verify thinking support — no thinking tags found in output"
                    ),
                    details={
                        "registered_parsers": [pid for pid in THINKING_PARSERS if pid != "null"],
                    },
                )
            )

        return (False, "null", diagnostics)

    async def _verify_tool_support(
        self, loaded: LoadedModel, adapter: ModelAdapter
    ) -> tuple[str | None, str | None, list[ProbeDiagnostic]]:
        """2-step model-agnostic tool verification.

        Step 1 — Template delivery: if tokenizer natively supports tools=,
                 generate once then sweep ALL parsers on the output.
        Step 2 — Generic injection: inject a format-agnostic tool prompt
                 as system message, generate once, sweep ALL parsers.

        Returns (tool_format, tool_parser_id, diagnostics).
        """
        from mlx_manager.mlx_server.parsers import TOOL_PARSERS
        from mlx_manager.mlx_server.utils.template_tools import has_native_tool_support

        tokenizer = loaded.tokenizer
        last_output: str | None = None
        diagnostics: list[ProbeDiagnostic] = []

        # ── Step 1: Template delivery (native tools) ─────────────────
        if adapter.supports_native_tools() or has_native_tool_support(tokenizer):
            try:
                gen_result = await self._generate(
                    loaded, _TOOL_PROBE_MESSAGES, tools=_TOOL_PROBE_TOOL
                )
                last_output = gen_result.content
                for parser_id, parser_cls in TOOL_PARSERS.items():
                    if parser_id == "null":
                        continue
                    if parser_cls().validates(last_output, "get_weather"):
                        logger.info(
                            "Tool probe: template delivery verified (parser={})",
                            parser_id,
                        )
                        return ("template", parser_id, diagnostics)
                logger.debug(
                    "Template delivery produced output but no parser matched: {}",
                    last_output[:200],
                )
            except Exception as e:
                logger.debug("Template delivery generation failed: {}", e)

        # ── Step 2: Generic injection (format-agnostic) ──────────────
        generic_prompt = _build_generic_tool_prompt(_TOOL_PROBE_TOOL)
        try:
            messages_with_tools = [
                {"role": "system", "content": generic_prompt},
                *_TOOL_PROBE_MESSAGES,
            ]
            gen_result = await self._generate(loaded, messages_with_tools)
            last_output = gen_result.content
            for parser_id, parser_cls in TOOL_PARSERS.items():
                if parser_id == "null":
                    continue
                if parser_cls().validates(last_output, "get_weather"):
                    logger.info(
                        "Tool probe: generic injection verified (parser={})",
                        parser_id,
                    )
                    return ("adapter", parser_id, diagnostics)
            logger.debug(
                "Generic injection produced output but no parser matched: {}",
                last_output[:200],
            )
        except Exception as e:
            logger.debug("Generic injection generation failed: {}", e)

        # ── No match: generic tag scan ────────────────────────────────
        registered_parsers = [pid for pid in TOOL_PARSERS if pid != "null"]
        try:
            if last_output is None:
                last_gen = await self._generate(loaded, _TOOL_PROBE_MESSAGES)
                last_output = last_gen.content

            # Check for tokenization artifacts first
            if _has_tokenization_artifacts(last_output):
                diagnostics.append(
                    ProbeDiagnostic(
                        level=DiagnosticLevel.ACTION_NEEDED,
                        category=DiagnosticCategory.UNSUPPORTED,
                        message=(
                            "Model architecture not fully supported by installed "
                            "mlx-lm — output is garbled due to tokenizer "
                            "incompatibility. Tool calling will not work until "
                            "mlx-lm adds proper support for this architecture."
                        ),
                        details={"raw_output_sample": last_output[:300]},
                    )
                )
            else:
                # Generic tag detection
                detected_tags = _detect_all_tags(last_output)
                unmatched = [t for t in detected_tags if t.name.lower() not in _KNOWN_BENIGN_TAGS]
                tool_hints = ["get_weather", '"name"']
                found_hints = [m for m in tool_hints if m in last_output]

                if found_hints:
                    diagnostics.append(
                        ProbeDiagnostic(
                            level=DiagnosticLevel.ACTION_NEEDED,
                            category=DiagnosticCategory.TOOL_DIALECT,
                            message=(
                                "Unknown tool dialect — model produced tool-like output "
                                "but no registered parser could parse it"
                            ),
                            details={
                                "found_hints": found_hints,
                                "raw_output_sample": last_output[:300],
                                "registered_parsers": registered_parsers,
                            },
                        )
                    )
                elif unmatched:
                    diagnostics.append(
                        ProbeDiagnostic(
                            level=DiagnosticLevel.WARNING,
                            category=DiagnosticCategory.TOOL_DIALECT,
                            message=(
                                "Unknown tag patterns in output — "
                                "possible unrecognized tool dialect"
                            ),
                            details={
                                "detected_tags": [(t.name, t.style, t.paired) for t in unmatched],
                                "raw_output_sample": last_output[:300],
                            },
                        )
                    )
                else:
                    logger.info("Tool probe: no tool support detected")
        except Exception:
            logger.info("Tool probe: no tool support detected")

        return (None, None, diagnostics)

    async def _probe_generative_capabilities(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        """Shared orchestrator for family detection + thinking + tools.

        Yields ProbeSteps and populates result with discovered capabilities.
        """
        from mlx_manager.mlx_server.models.adapters import (
            FAMILY_REGISTRY,
            detect_model_family,
        )

        # Step: Detect model family
        yield ProbeStep(step="detect_family", status="running")
        if result.model_family is None:
            result.model_family = detect_model_family(model_id)

        family_diagnostics: list[ProbeDiagnostic] = []
        if result.model_family == "default":
            # Determine architecture for diagnostic context
            architecture = ""
            try:
                from mlx_manager.utils.model_detection import read_model_config

                config = read_model_config(model_id)
                if config:
                    arch_list = config.get("architectures", [])
                    architecture = arch_list[0] if arch_list else ""
            except Exception:
                pass

            diag = ProbeDiagnostic(
                level=DiagnosticLevel.WARNING,
                category=DiagnosticCategory.FAMILY,
                message=(
                    f"No dedicated adapter — using DefaultAdapter. "
                    f"Architecture '{architecture or 'unknown'}' doesn't match "
                    f"any registered family."
                ),
                details={
                    "architecture": architecture,
                    "registered_families": sorted(FAMILY_REGISTRY.keys()),
                },
            )
            family_diagnostics.append(diag)
            result.diagnostics.append(diag)

        yield ProbeStep(
            step="detect_family",
            status="completed",
            capability="model_family",
            value=result.model_family,
            details={"family": result.model_family},
            diagnostics=family_diagnostics or None,
        )

        tokenizer = loaded.tokenizer
        adapter = loaded.adapter

        # Guard: adapter must exist for generation-based probing
        if adapter is None:
            logger.warning(
                "No adapter available for {}; skipping thinking and tool probes",
                model_id,
            )
            yield ProbeStep(step="test_thinking", status="skipped")
            yield ProbeStep(step="test_tools", status="skipped")
            return

        # Discover template parameters
        discovered_params: dict[str, Any] | None = None
        if tokenizer is not None:
            from mlx_manager.mlx_server.utils.template_params import discover_template_params

            discovered_params = discover_template_params(tokenizer)
            if discovered_params:
                result.template_params = discovered_params
                logger.info(
                    "Discovered template params for {}: {}",
                    model_id,
                    list(discovered_params.keys()),
                )

        # Test thinking support
        if tokenizer is not None:
            yield ProbeStep(step="test_thinking", status="running")
            try:
                (
                    supports_thinking,
                    thinking_parser_id,
                    thinking_diags,
                ) = await self._verify_thinking_support(
                    loaded, adapter, template_params=discovered_params
                )
                result.supports_thinking = supports_thinking
                result.thinking_parser_id = thinking_parser_id
                result.diagnostics.extend(thinking_diags)
                yield ProbeStep(
                    step="test_thinking",
                    status="completed",
                    capability="supports_thinking",
                    value=supports_thinking,
                    diagnostics=thinking_diags or None,
                )
            except Exception as e:
                logger.warning("Thinking test failed for {}: {}", model_id, e)
                yield ProbeStep(step="test_thinking", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_thinking", status="skipped")

        # Generation-based tool verification (2-attempt)
        if tokenizer is not None:
            yield ProbeStep(step="test_tools", status="running")
            try:
                tool_format, tool_parser_id, tool_diags = await self._verify_tool_support(
                    loaded, adapter
                )
                result.supports_native_tools = tool_format is not None
                result.tool_format = tool_format
                result.tool_parser_id = tool_parser_id
                result.diagnostics.extend(tool_diags)
                yield ProbeStep(
                    step="test_tools",
                    status="completed",
                    capability="tool_format",
                    value=tool_format,
                    diagnostics=tool_diags or None,
                )
            except Exception as e:
                logger.warning("Tool verification failed for {}: {}", model_id, e)
                yield ProbeStep(step="test_tools", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_tools", status="skipped")


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

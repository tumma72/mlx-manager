"""Thinking and tool sweep functions for generative capability probing.

Extracted from ProbingCoordinator as standalone async functions since they
don't require access to the model pool — they only operate on an already-loaded
model via the strategy's _generate() call.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from mlx_manager.services.probe.steps import (
    DiagnosticCategory,
    DiagnosticLevel,
    ProbeDiagnostic,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _has_structural_tool_evidence(output: str) -> bool:
    """Check for structural evidence of an actual tool call attempt.

    Returns True if the output contains patterns indicating the model
    attempted to emit a tool call in some structured format, not just
    mentioned the tool name in natural language prose.
    """
    # JSON object with "name" key containing the tool name
    if re.search(r'\{\s*"name"\s*:\s*"get_weather"', output):
        return True
    # Function call syntax: get_weather(...)
    if re.search(r"get_weather\s*\(", output):
        return True
    # XML tool tags
    if re.search(r"<(tool_call|function_call|tool_use)[^>]*>", output):
        return True
    # Bracket markers (e.g. [TOOL_CALLS], [TOOL_CALL])
    if re.search(r"\[TOOL_CALL", output, re.IGNORECASE):
        return True
    # JSON array with tool-like structure
    if re.search(r'\[\s*\{\s*"(name|function)"', output):
        return True
    return False


# ------------------------------------------------------------------
# Thinking sweep
# ------------------------------------------------------------------


async def sweep_thinking(
    model_id: str,
    loaded: Any,
    strategy: Any,
    template_params: dict[str, Any] | None,
    family: str | None = None,
    verbose: bool = False,
) -> tuple[bool, str, list[ProbeDiagnostic], list[Any]]:
    """Tag-first thinking detection.

    Phase 1 — CHALLENGE: Generate with reasoning prompt
    Phase 2 — DISCOVER: _discover_and_map_tags on raw output
    Phase 3 — VALIDATE: For matched parsers, run extract(); also sweep ALL parsers
    Phase 4 — UNCLOSED TAG FALLBACK: Retry with more tokens if unclosed tag found
    Phase 5 — REPORT: Include discovered_tags

    Returns (supports_thinking, parser_id, diagnostics, discovered_tags).
    """
    from mlx_manager.mlx_server.parsers import THINKING_PARSERS

    from .base import (
        _detect_unknown_thinking_tags,
        _discover_and_map_tags,
        _find_unclosed_thinking_tag,
    )

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

    # Generate with enable_thinking if the template supports it
    template_options = {"enable_thinking": True} if has_enable_thinking else None

    # ── Phase 1: CHALLENGE ────────────────────────────────────────
    try:
        gen_result = await strategy._generate(
            loaded,
            messages,
            template_options=template_options,
            max_tokens=2000,
        )
    except Exception as e:
        logger.debug("Thinking probe generation failed: %s", e)
        diagnostics.append(
            ProbeDiagnostic(
                level=DiagnosticLevel.WARNING,
                category=DiagnosticCategory.THINKING_DIALECT,
                message="Thinking probe failed due to generation error",
                details={"error": str(e)},
            )
        )
        return (False, "null", diagnostics, [])

    raw_output = gen_result.content

    # When enable_thinking=True, templates inject <think> into the prompt.
    # The model generates "...reasoning...</think>answer" — the opening tag
    # is in the prompt, not the generated text. Prepend <think> so parsers
    # can find paired tags, or _find_unclosed_thinking_tag can detect
    # truncated thinking (model reasoned but output hit max_tokens).
    if has_enable_thinking and raw_output:
        raw_output = "<think>\n" + raw_output

    # ── Phase 2: DISCOVER ─────────────────────────────────────────
    discovered_tags = _discover_and_map_tags(raw_output, THINKING_PARSERS)

    # ── Phase 3: VALIDATE ─────────────────────────────────────────
    # First try parsers identified by tag discovery
    validated_from_tags: set[str] = set()
    for tag in discovered_tags:
        validated_from_tags.update(tag.matched_parsers)

    from .base import _prioritize_parsers, get_family_thinking_parser_id

    family_thinking_id = get_family_thinking_parser_id(family)
    for pid in _prioritize_parsers(validated_from_tags, family_thinking_id):
        if pid in THINKING_PARSERS:
            parser = THINKING_PARSERS[pid]()
            if parser.extract(raw_output) is not None:
                logger.info("Thinking probe: tag-first detected (parser=%s)", pid)
                return (True, pid, diagnostics, discovered_tags)

    # Also sweep ALL parsers directly (some may match without tag detection)
    for parser_id, parser_cls in THINKING_PARSERS.items():
        if parser_id == "null" or parser_id in validated_from_tags:
            continue
        parser = parser_cls()
        if parser.extract(raw_output) is not None:
            logger.info("Thinking probe: full sweep detected (parser=%s)", parser_id)
            return (True, parser_id, diagnostics, discovered_tags)

    # ── Phase 4: UNCLOSED TAG FALLBACK ────────────────────────────
    unclosed = _find_unclosed_thinking_tag(raw_output)
    if unclosed:
        logger.info("Thinking probe: unclosed <%s> tag, retrying with more tokens", unclosed)
        try:
            retry_result = await strategy._generate(
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
            retry_output = retry_result.content
            for parser_id, parser_cls in THINKING_PARSERS.items():
                if parser_id == "null":
                    continue
                parser = parser_cls()
                if parser.extract(retry_output) is not None:
                    logger.info("Thinking probe: detected on retry (parser=%s)", parser_id)
                    return (True, parser_id, diagnostics, discovered_tags)
        except Exception as e:
            logger.debug("Thinking probe retry failed: %s", e)

        # Match unclosed tag against registered parsers' stream markers
        for parser_id, parser_cls in THINKING_PARSERS.items():
            if parser_id == "null":
                continue
            parser = parser_cls()
            for open_tag, _close_tag in parser.stream_markers:
                marker_name = open_tag.strip("<>").strip("[]").lower()
                if marker_name == unclosed.lower():
                    logger.info(
                        "Thinking probe: unclosed <%s> matches parser=%s",
                        unclosed,
                        parser_id,
                    )
                    return (True, parser_id, diagnostics, discovered_tags)

    # ── Phase 5: REPORT ───────────────────────────────────────────
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

    # ── Verbose: raw output sample ────────────────────────────────
    if verbose and raw_output is not None:
        tried_parsers = [pid for pid in THINKING_PARSERS if pid != "null"]
        diagnostics.append(
            ProbeDiagnostic(
                level=DiagnosticLevel.INFO,
                category=DiagnosticCategory.THINKING_DIALECT,
                message="Thinking sweep completed (verbose)",
                details={
                    "raw_output_sample": raw_output[:300],
                    "parser_trials": tried_parsers,
                    "discovered_tags": [t.model_dump() for t in discovered_tags],
                },
            )
        )

    return (False, "null", diagnostics, discovered_tags)


# ------------------------------------------------------------------
# Tool sweep
# ------------------------------------------------------------------


async def sweep_tools(
    model_id: str,
    loaded: Any,
    strategy: Any,
    family: str | None = None,
    verbose: bool = False,
) -> tuple[str | None, str | None, list[ProbeDiagnostic], list[Any]]:
    """Tag-first tool support detection.

    Phase 1 — CHALLENGE: Generic injection (format-agnostic prompt, NO tools= param)
    Phase 2 — DISCOVER: _discover_and_map_tags on raw output
    Phase 3 — VALIDATE: For matched parsers, run validates()
    Phase 4 — FALLBACK: Template delivery only if Phase 2 found NO parser markers
    Phase 5 — REPORT: Include discovered_tags regardless of outcome

    Returns (tool_format, tool_parser_id, diagnostics, discovered_tags).
    """
    from mlx_manager.mlx_server.parsers import TOOL_PARSERS
    from mlx_manager.mlx_server.utils.template_tools import (
        has_native_tool_support,
    )

    from .base import (
        _KNOWN_BENIGN_TAGS,
        _TOOL_PROBE_MESSAGES,
        _TOOL_PROBE_TOOL,
        _build_generic_tool_prompt,
        _discover_and_map_tags,
        _has_tokenization_artifacts,
        _prioritize_parsers,
        get_family_tool_parser_id,
    )

    diagnostics: list[ProbeDiagnostic] = []
    adapter = loaded.adapter
    tokenizer = loaded.tokenizer
    last_output: str | None = None
    all_discovered_tags: list[Any] = []
    family_tool_id = get_family_tool_parser_id(family)

    # ── Phase 1: CHALLENGE (generic injection) ────────────────────
    generic_prompt = _build_generic_tool_prompt(_TOOL_PROBE_TOOL)
    try:
        messages_with_tools = [
            {"role": "system", "content": generic_prompt},
            *_TOOL_PROBE_MESSAGES,
        ]
        gen_result = await strategy._generate(loaded, messages_with_tools)
        last_output = gen_result.content
    except Exception as e:
        logger.debug("Generic injection generation failed: %s", e)

    # ── Phase 2: DISCOVER ─────────────────────────────────────────
    if last_output is not None:
        # Check for tokenization artifacts first
        if _has_tokenization_artifacts(last_output):
            logger.warning(
                "Tool probe: output has tokenization artifacts — "
                "model architecture likely not fully supported by "
                "installed mlx-lm. Raw output: %s",
                last_output[:300],
            )
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
                    details={
                        "raw_output_sample": last_output[:300],
                    },
                )
            )
            return (None, None, diagnostics, [])

        all_discovered_tags = _discover_and_map_tags(last_output, TOOL_PARSERS)

        if all_discovered_tags:
            logger.info(
                "Tool probe: discovered tags: %s",
                [(t.name, t.style, t.matched_parsers) for t in all_discovered_tags],
            )

    # ── Phase 3: VALIDATE ─────────────────────────────────────────
    if last_output is not None:
        # Collect parser_ids matched by tag discovery
        matched_parser_ids: set[str] = set()
        for tag in all_discovered_tags:
            matched_parser_ids.update(tag.matched_parsers)

        # Try tag-matched parsers first (highest confidence)
        if matched_parser_ids:
            for pid in _prioritize_parsers(matched_parser_ids, family_tool_id):
                if pid in TOOL_PARSERS:
                    parser = TOOL_PARSERS[pid]()
                    if parser.validates(last_output, "get_weather"):
                        logger.info("Tool probe: tag-first validated (parser=%s)", pid)
                        return ("detected", pid, diagnostics, all_discovered_tags)

        # Sweep ALL remaining parsers (covers tagless parsers like openai_json)
        for parser_id, parser_cls in TOOL_PARSERS.items():
            if parser_id == "null" or parser_id in matched_parser_ids:
                continue
            if parser_cls().validates(last_output, "get_weather"):
                logger.info(
                    "Tool probe: generic injection full sweep verified (parser=%s)",
                    parser_id,
                )
                return ("adapter", parser_id, diagnostics, all_discovered_tags)

    # ── Phase 4: FALLBACK (template delivery) ─────────────────────
    # Try template delivery if Phases 1-3 didn't find a valid parser.
    # This covers both "no tags found" AND "tags found but validation failed".
    can_try_template = adapter.supports_native_tools() or has_native_tool_support(tokenizer)
    if can_try_template:
        try:
            gen_result = await strategy._generate(
                loaded, _TOOL_PROBE_MESSAGES, tools=_TOOL_PROBE_TOOL
            )
            template_output = gen_result.content

            # Run discovery + validation on template output too
            template_tags = _discover_and_map_tags(template_output, TOOL_PARSERS)
            if template_tags:
                # Merge into all_discovered_tags (dedup by name+style)
                existing_keys = {(t.name, t.style) for t in all_discovered_tags}
                for t in template_tags:
                    if (t.name, t.style) not in existing_keys:
                        all_discovered_tags.append(t)

            # Collect tag-matched parsers from template output for prioritization
            template_matched: set[str] = set()
            for t in template_tags:
                template_matched.update(t.matched_parsers)

            # Try tag-matched parsers first (family-aware order)
            if template_matched:
                for pid in _prioritize_parsers(template_matched, family_tool_id):
                    if pid in TOOL_PARSERS:
                        if TOOL_PARSERS[pid]().validates(template_output, "get_weather"):
                            logger.info("Tool probe: template delivery verified (parser=%s)", pid)
                            return ("template", pid, diagnostics, all_discovered_tags)

            # Sweep remaining parsers not covered by tag discovery
            for parser_id, parser_cls in TOOL_PARSERS.items():
                if parser_id == "null" or parser_id in template_matched:
                    continue
                if parser_cls().validates(template_output, "get_weather"):
                    logger.info(
                        "Tool probe: template delivery verified (parser=%s)",
                        parser_id,
                    )
                    return ("template", parser_id, diagnostics, all_discovered_tags)
            logger.debug(
                "Template delivery produced output but no parser matched: %s",
                template_output[:200] if template_output else "",
            )
        except Exception as e:
            logger.debug("Template delivery with native tools failed: %s", e)

    # ── Phase 5: REPORT ───────────────────────────────────────────
    registered_parsers = [pid for pid in TOOL_PARSERS if pid != "null"]
    scan_output = last_output
    if scan_output is not None:
        # Check for unmatched non-benign tags
        unmatched_tags = [
            {"name": t.name, "style": t.style, "paired": t.paired}
            for t in all_discovered_tags
            if not t.matched_parsers and t.name.lower() not in _KNOWN_BENIGN_TAGS
        ]

        has_structural = _has_structural_tool_evidence(scan_output)

        if has_structural:
            # Structural evidence: model actually tried to emit a tool call
            diagnostics.append(
                ProbeDiagnostic(
                    level=DiagnosticLevel.ACTION_NEEDED,
                    category=DiagnosticCategory.TOOL_DIALECT,
                    message=(
                        "Unknown tool dialect — model produced tool-like markers "
                        "but no registered parser could parse the output"
                    ),
                    details={
                        "detected_tags": unmatched_tags,
                        "raw_output_sample": scan_output[:300],
                        "registered_parsers": registered_parsers,
                    },
                )
            )
        elif "get_weather" in scan_output:
            # Prose mention only — model talked about the tool but didn't
            # emit a structured call. Downgrade to INFO.
            diagnostics.append(
                ProbeDiagnostic(
                    level=DiagnosticLevel.INFO,
                    category=DiagnosticCategory.TOOL_DIALECT,
                    message=(
                        "Model mentioned tool name in prose but did not emit "
                        "a structured tool call — no tool dialect detected"
                    ),
                    details={
                        "raw_output_sample": scan_output[:300],
                        "registered_parsers": registered_parsers,
                    },
                )
            )
        elif unmatched_tags:
            diagnostics.append(
                ProbeDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    category=DiagnosticCategory.TOOL_DIALECT,
                    message=("Unknown tag patterns in output — possible unrecognized tool dialect"),
                    details={
                        "detected_tags": unmatched_tags,
                        "raw_output_sample": scan_output[:300],
                        "registered_parsers": registered_parsers,
                    },
                )
            )
        else:
            logger.info("Tool probe: no tool support detected")
    else:
        logger.info("Tool probe: no tool support detected (no output)")

    # ── Verbose: parser trial details ─────────────────────────────
    if verbose:
        tried_parsers = [pid for pid in TOOL_PARSERS if pid != "null"]
        diagnostics.append(
            ProbeDiagnostic(
                level=DiagnosticLevel.INFO,
                category=DiagnosticCategory.TOOL_DIALECT,
                message="Tool sweep completed (verbose)",
                details={
                    "parser_trials": tried_parsers,
                    "raw_output_sample": scan_output[:300] if scan_output else None,
                    "discovered_tags": [
                        {"name": t.name, "style": t.style, "paired": t.paired}
                        for t in all_discovered_tags
                    ],
                },
            )
        )

    return (None, None, diagnostics, all_discovered_tags)

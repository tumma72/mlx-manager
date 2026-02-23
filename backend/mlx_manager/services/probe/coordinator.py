"""ProbingCoordinator — orchestrates probing through the Profile → Pool → Adapter path.

Replaces the ad-hoc probe_model() orchestration with a design that:
1. Loads models through the normal pool path (register_profile_settings + get_model)
2. Tests capabilities via generation with the null parser (raw text flows through)
3. Keeps the model loaded once; thinking/tool detection scans raw output
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from mlx_manager.services.probe.steps import (
    DiagnosticCategory,
    DiagnosticLevel,
    ProbeDiagnostic,
    ProbeResult,
    ProbeStep,
)

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import ModelPoolManager

logger = logging.getLogger(__name__)


class ProbingCoordinator:
    """Orchestrates capability probing through the normal Profile → Pool → Adapter path."""

    def __init__(self, pool: ModelPoolManager) -> None:
        self._pool = pool

    async def probe(
        self, model_id: str, *, verbose: bool = False
    ) -> AsyncGenerator[ProbeStep, None]:
        """Main entry point. Drop-in replacement for probe_model().

        Yields ProbeStep objects for progressive SSE streaming.
        """
        from mlx_manager.mlx_server.models.detection import detect_model_type_detailed
        from mlx_manager.mlx_server.models.pool import ProfileSettings
        from mlx_manager.mlx_server.models.types import ModelType

        result = ProbeResult()
        was_preloaded = model_id in self._pool._models

        # Save original settings to restore after probing
        original_settings = self._pool._profile_settings.get(model_id)

        # ── Step 1: Detect model type ─────────────────────────────────
        yield ProbeStep(step="detect_type", status="running")
        try:
            detection = detect_model_type_detailed(model_id)
            model_type = detection.model_type
            result.model_type = model_type.value

            step_details: dict[str, object] = {
                "detection_method": detection.detection_method,
                "architecture": detection.architecture,
            }
            step_diagnostics: list[ProbeDiagnostic] = []

            if detection.detection_method == "default":
                diag = ProbeDiagnostic(
                    level=DiagnosticLevel.WARNING,
                    category=DiagnosticCategory.TYPE,
                    message=(
                        f"Model type defaulted to TEXT_GEN — no config, architecture, "
                        f"or name pattern matched for '{model_id}'"
                    ),
                    details={
                        "architecture": detection.architecture,
                        "detection_method": detection.detection_method,
                    },
                )
                step_diagnostics.append(diag)
                result.diagnostics.append(diag)

            yield ProbeStep(
                step="detect_type",
                status="completed",
                capability="model_type",
                value=model_type.value,
                details=step_details,
                diagnostics=step_diagnostics or None,
            )
        except Exception as e:
            logger.error(f"Type detection failed for {model_id}: {e}")
            yield ProbeStep(step="detect_type", status="failed", error=str(e))
            return

        # ── Step 2: Audio pre-validation ──────────────────────────────
        if model_type == ModelType.AUDIO:
            from .audio import _detect_audio_capabilities

            is_tts, is_stt, _ = _detect_audio_capabilities(model_id)
            if not is_tts and not is_stt:
                diag = ProbeDiagnostic(
                    level=DiagnosticLevel.ACTION_NEEDED,
                    category=DiagnosticCategory.UNSUPPORTED,
                    message=(
                        f"Unsupported audio model subtype: {model_id} is detected as audio "
                        "but is not a recognized TTS or STT model (likely an audio codec)."
                    ),
                    details={"model_id": model_id, "is_tts": False, "is_stt": False},
                )
                result.diagnostics.append(diag)
                yield ProbeStep(
                    step="load_model",
                    status="failed",
                    error=(
                        f"Unsupported audio model subtype: {model_id} is detected as audio "
                        "but is not a recognized TTS or STT model (likely an audio codec). "
                        "Audio codec models are not supported for inference."
                    ),
                    diagnostics=[diag],
                )
                return

        # ── Step 3: Load model through normal Profile path ────────────
        yield ProbeStep(step="load_model", status="running")
        try:
            # Register empty probe settings (goes through normal Profile path)
            self._pool.register_profile_settings(model_id, ProfileSettings())
            loaded = await self._pool.get_model(model_id)
            yield ProbeStep(step="load_model", status="completed")
        except Exception as e:
            logger.error(f"Probe failed to load model {model_id}: {e}")
            yield ProbeStep(step="load_model", status="failed", error=str(e))
            return

        # ── Step 3b: Force null parsers so raw output reaches sweep code ──
        from mlx_manager.mlx_server.parsers.thinking import NullThinkingParser
        from mlx_manager.mlx_server.parsers.tool_call import NullToolParser

        original_tool_parser = None
        original_thinking_parser = None
        if loaded.adapter is not None:
            original_tool_parser = loaded.adapter.tool_parser
            original_thinking_parser = loaded.adapter.thinking_parser
            loaded.adapter.configure(
                tool_parser=NullToolParser(),
                thinking_parser=NullThinkingParser(),
            )

        # ── Step 4: Look up strategy for type-specific static checks ──
        from .strategy import get_probe_strategy

        strategy = get_probe_strategy(model_type)
        if strategy is None:
            yield ProbeStep(
                step="find_strategy",
                status="failed",
                error=f"No probe strategy registered for model type: {model_type.value}",
            )
            return

        # ── Step 5: Run type-specific static checks ───────────────────
        try:
            async for step in strategy.probe(model_id, loaded, result):
                yield step
        except Exception as e:
            logger.error(f"Probe strategy failed for {model_id}: {e}")
            yield ProbeStep(step="strategy_error", status="failed", error=str(e))

        # ── Step 6: Parser sweep (TEXT_GEN / VISION only) ─────────────
        if model_type in (ModelType.TEXT_GEN, ModelType.VISION):
            async for step in self._sweep_generative_capabilities(
                model_id, loaded, result, strategy
            ):
                yield step

        # ── Step 7: Save results to DB ────────────────────────────────
        yield ProbeStep(step="save_results", status="running")
        try:
            await _save_capabilities(model_id, result)
            yield ProbeStep(step="save_results", status="completed")
        except Exception as e:
            logger.error(f"Failed to save probe results for {model_id}: {e}")
            yield ProbeStep(step="save_results", status="failed", error=str(e))

        # ── Step 8: Cleanup ───────────────────────────────────────────
        # Restore original settings or unregister probe settings
        if original_settings is not None:
            self._pool.register_profile_settings(model_id, original_settings)
        else:
            self._pool.unregister_profile_settings(model_id)

        if not was_preloaded:
            yield ProbeStep(step="cleanup", status="running")
            try:
                await self._pool.unload_model(model_id)
                yield ProbeStep(step="cleanup", status="completed")
            except Exception as e:
                logger.warning(f"Cleanup failed for {model_id}: {e}")
                yield ProbeStep(step="cleanup", status="failed", error=str(e))
        else:
            # Restore original parsers for preloaded models
            if loaded.adapter is not None and original_tool_parser is not None:
                loaded.adapter.configure(
                    tool_parser=original_tool_parser,
                    thinking_parser=original_thinking_parser,
                )

            # Update capabilities on the already-loaded model
            try:
                from mlx_manager.database import get_session
                from mlx_manager.models import Model

                async with get_session() as session:
                    from sqlalchemy.orm import selectinload
                    from sqlmodel import select

                    stmt = (
                        select(Model)
                        .where(Model.repo_id == model_id)
                        .options(
                            selectinload(Model.capabilities)  # type: ignore[arg-type]
                        )
                    )
                    caps_result = await session.execute(stmt)
                    model_record = caps_result.scalar_one_or_none()
                    if model_record and model_record.capabilities:
                        loaded.capabilities = model_record.capabilities
            except Exception:
                pass
            yield ProbeStep(step="cleanup", status="skipped")

        # ── Final: probe_complete ─────────────────────────────────────
        yield ProbeStep(
            step="probe_complete",
            status="completed",
            details={"result": result.model_dump()},
        )
        logger.info(f"Probe complete for {model_id}: type={result.model_type}")

    # ------------------------------------------------------------------
    # Parser sweep: thinking + tools
    # ------------------------------------------------------------------

    async def _sweep_generative_capabilities(
        self,
        model_id: str,
        loaded: Any,
        result: ProbeResult,
        strategy: Any,
    ) -> AsyncGenerator[ProbeStep, None]:
        """Sweep parser/config combinations for thinking and tool support.

        Uses register_profile_settings() to reconfigure the adapter for each
        test, keeping the model loaded throughout.
        """
        from mlx_manager.mlx_server.models.adapters import (
            FAMILY_REGISTRY,
            detect_model_family,
        )

        adapter = loaded.adapter
        tokenizer = loaded.tokenizer

        # ── Family detection ──────────────────────────────────────────
        yield ProbeStep(step="detect_family", status="running")
        if result.model_family is None:
            result.model_family = detect_model_family(model_id)

        family_diagnostics: list[ProbeDiagnostic] = []
        if result.model_family == "default":
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

        # Guard: adapter + tokenizer required for generation-based probing
        if adapter is None:
            logger.warning(
                "No adapter available for %s; skipping thinking and tool probes",
                model_id,
            )
            yield ProbeStep(step="test_thinking", status="skipped")
            yield ProbeStep(step="test_tools", status="skipped")
            return

        # ── Discover template parameters ──────────────────────────────
        discovered_params: dict[str, Any] | None = None
        if tokenizer is not None:
            from mlx_manager.mlx_server.utils.template_params import (
                discover_template_params,
            )

            discovered_params = discover_template_params(tokenizer)
            if discovered_params:
                result.template_params = discovered_params
                logger.info(
                    "Discovered template params for %s: %s",
                    model_id,
                    list(discovered_params.keys()),
                )

        # ── Thinking sweep ────────────────────────────────────────────
        if tokenizer is not None:
            yield ProbeStep(step="test_thinking", status="running")
            try:
                supports, parser_id, diags, thinking_tags = await self._sweep_thinking(
                    model_id, loaded, strategy, discovered_params
                )
                result.supports_thinking = supports
                result.thinking_parser_id = parser_id
                result.diagnostics.extend(diags)
                result.discovered_thinking_tags = (
                    [t.model_dump() for t in thinking_tags] if thinking_tags else None
                )
                step_details: dict[str, Any] = {}
                if thinking_tags:
                    step_details["discovered_tags"] = [t.model_dump() for t in thinking_tags]
                yield ProbeStep(
                    step="test_thinking",
                    status="completed",
                    capability="supports_thinking",
                    value=supports,
                    details=step_details or None,
                    diagnostics=diags or None,
                )
            except Exception as e:
                logger.warning("Thinking test failed for %s: %s", model_id, e)
                yield ProbeStep(step="test_thinking", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_thinking", status="skipped")

        # ── Tool sweep ────────────────────────────────────────────────
        if tokenizer is not None:
            yield ProbeStep(step="test_tools", status="running")
            try:
                tool_format, tool_parser_id, diags, tool_tags = await self._sweep_tools(
                    model_id, loaded, strategy
                )
                result.supports_native_tools = tool_format is not None
                result.tool_format = tool_format
                result.tool_parser_id = tool_parser_id
                result.diagnostics.extend(diags)
                result.discovered_tool_tags = (
                    [t.model_dump() for t in tool_tags] if tool_tags else None
                )
                tool_details: dict[str, Any] = {}
                if tool_tags:
                    tool_details["discovered_tags"] = [t.model_dump() for t in tool_tags]
                yield ProbeStep(
                    step="test_tools",
                    status="completed",
                    capability="tool_format",
                    value=tool_format,
                    details=tool_details or None,
                    diagnostics=diags or None,
                )
            except Exception as e:
                logger.warning("Tool verification failed for %s: %s", model_id, e)
                yield ProbeStep(step="test_tools", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_tools", status="skipped")

    # ------------------------------------------------------------------
    # Thinking sweep
    # ------------------------------------------------------------------

    async def _sweep_thinking(
        self,
        model_id: str,
        loaded: Any,
        strategy: Any,
        template_params: dict[str, Any] | None,
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

        # ── Phase 2: DISCOVER ─────────────────────────────────────────
        discovered_tags = _discover_and_map_tags(raw_output, THINKING_PARSERS)

        # ── Phase 3: VALIDATE ─────────────────────────────────────────
        # First try parsers identified by tag discovery
        validated_from_tags: set[str] = set()
        for tag in discovered_tags:
            validated_from_tags.update(tag.matched_parsers)

        for pid in sorted(validated_from_tags):
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

        return (False, "null", diagnostics, discovered_tags)

    # ------------------------------------------------------------------
    # Tool sweep
    # ------------------------------------------------------------------

    async def _sweep_tools(
        self,
        model_id: str,
        loaded: Any,
        strategy: Any,
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
        )

        diagnostics: list[ProbeDiagnostic] = []
        adapter = loaded.adapter
        tokenizer = loaded.tokenizer
        last_output: str | None = None
        all_discovered_tags: list[Any] = []

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
                for pid in sorted(matched_parser_ids):
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
        # Only try template delivery if Phase 2 found NO parser markers at all
        has_parser_markers = any(t.matched_parsers for t in all_discovered_tags)
        if not has_parser_markers:
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

                    for parser_id, parser_cls in TOOL_PARSERS.items():
                        if parser_id == "null":
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

            tool_hints = ["get_weather", '"name"']
            found_hints = [m for m in tool_hints if m in scan_output]

            if found_hints:
                diagnostics.append(
                    ProbeDiagnostic(
                        level=DiagnosticLevel.ACTION_NEEDED,
                        category=DiagnosticCategory.TOOL_DIALECT,
                        message=(
                            "Unknown tool dialect — model produced tool-like markers "
                            "but no registered parser could parse the output"
                        ),
                        details={
                            "found_hints": found_hints,
                            "detected_tags": unmatched_tags,
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
                        message=(
                            "Unknown tag patterns in output — possible unrecognized tool dialect"
                        ),
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

        return (None, None, diagnostics, all_discovered_tags)


# ---------------------------------------------------------------------------
# Helpers (reused from service.py — moved here as the coordinator owns saving)
# ---------------------------------------------------------------------------


async def _save_capabilities(model_id: str, result: ProbeResult) -> None:
    """Persist probe results to the database."""
    from mlx_manager.services.model_registry import update_model_capabilities

    caps_dict: dict[str, Any] = {"probe_version": 2}

    field_map = {
        "model_type": result.model_type,
        "supports_native_tools": result.supports_native_tools,
        "supports_thinking": result.supports_thinking,
        "tool_format": result.tool_format,
        "practical_max_tokens": result.practical_max_tokens,
        "model_family": result.model_family,
        "tool_parser_id": result.tool_parser_id,
        "thinking_parser_id": result.thinking_parser_id,
        "supports_multi_image": result.supports_multi_image,
        "supports_video": result.supports_video,
        "embedding_dimensions": result.embedding_dimensions,
        "max_sequence_length": result.max_sequence_length,
        "is_normalized": result.is_normalized,
        "supports_tts": result.supports_tts,
        "supports_stt": result.supports_stt,
    }
    for key, value in field_map.items():
        if value is not None:
            caps_dict[key] = value

    # Template params need JSON string serialization (DB column is str)
    if result.template_params:
        serialized = {}
        for k, v in result.template_params.items():
            serialized[k] = v.model_dump() if hasattr(v, "model_dump") else v
        caps_dict["template_params"] = json.dumps(serialized)

    await update_model_capabilities(model_id, **caps_dict)

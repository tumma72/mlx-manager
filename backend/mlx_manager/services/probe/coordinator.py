"""ProbingCoordinator — orchestrates probing through the Profile → Pool → Adapter path.

Replaces the ad-hoc probe_model() orchestration with a design that:
1. Loads models through the normal pool path (register_profile_settings + get_model)
2. Iterates parser/config combinations via register_profile_settings() reconfiguration
3. Keeps the model loaded once; only the adapter reconfigures per test
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
                supports, parser_id, diags = await self._sweep_thinking(
                    model_id, loaded, strategy, discovered_params
                )
                result.supports_thinking = supports
                result.thinking_parser_id = parser_id
                result.diagnostics.extend(diags)
                yield ProbeStep(
                    step="test_thinking",
                    status="completed",
                    capability="supports_thinking",
                    value=supports,
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
                tool_format, tool_parser_id, diags = await self._sweep_tools(
                    model_id, loaded, strategy
                )
                result.supports_native_tools = tool_format is not None
                result.tool_format = tool_format
                result.tool_parser_id = tool_parser_id
                result.diagnostics.extend(diags)
                yield ProbeStep(
                    step="test_tools",
                    status="completed",
                    capability="tool_format",
                    value=tool_format,
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
    ) -> tuple[bool, str, list[ProbeDiagnostic]]:
        """Test thinking support by iterating parser configurations.

        Uses register_profile_settings() to swap thinking parsers.

        Returns (supports_thinking, thinking_parser_id, diagnostics).
        """
        from mlx_manager.mlx_server.models.pool import ProfileSettings
        from mlx_manager.mlx_server.parsers import THINKING_PARSERS

        from .base import _find_unclosed_thinking_tag

        diagnostics: list[ProbeDiagnostic] = []
        messages = [{"role": "user", "content": "What is 2 + 2? Answer briefly."}]
        has_enable_thinking = template_params is not None and "enable_thinking" in template_params

        # --- Path A: explicit enable_thinking parameter ---
        if has_enable_thinking:
            for parser_id in THINKING_PARSERS:
                if parser_id == "null":
                    continue
                try:
                    self._pool.register_profile_settings(
                        model_id,
                        ProfileSettings(
                            template_options={"enable_thinking": True},
                            thinking_parser_id=parser_id,
                        ),
                    )
                    output = await strategy._generate(
                        loaded,
                        messages,
                        template_options={"enable_thinking": True},
                    )
                    adapter = loaded.adapter
                    thinking_content = adapter.thinking_parser.extract(output)
                    if thinking_content is not None:
                        logger.info("Thinking probe: Path A verified (parser=%s)", parser_id)
                        return (True, parser_id, diagnostics)

                    # Check for unclosed tags (truncation)
                    unclosed_tag = _find_unclosed_thinking_tag(output)
                    if unclosed_tag:
                        logger.info(
                            "Thinking probe: found unclosed <%s> tag, retrying",
                            unclosed_tag,
                        )
                        retry_messages = [
                            {"role": "user", "content": "What is 2 + 2? Answer briefly."},
                            {"role": "assistant", "content": output},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was truncated because you "
                                    "weren't concise enough as requested. Please answer "
                                    "the same question again, keeping your response "
                                    "very brief."
                                ),
                            },
                        ]
                        retry_output = await strategy._generate(
                            loaded,
                            retry_messages,
                            template_options={"enable_thinking": True},
                            max_tokens=2000,
                        )
                        retry_content = adapter.thinking_parser.extract(retry_output)
                        if retry_content is not None:
                            logger.info(
                                "Thinking probe: verified on retry (parser=%s)",
                                parser_id,
                            )
                            return (True, parser_id, diagnostics)
                except Exception as e:
                    logger.debug("Path A thinking with parser %s failed: %s", parser_id, e)

        # --- Path B: always-thinks detection (no enable_thinking) ---
        for parser_id in THINKING_PARSERS:
            if parser_id == "null":
                continue
            try:
                self._pool.register_profile_settings(
                    model_id,
                    ProfileSettings(thinking_parser_id=parser_id),
                )
                output = await strategy._generate(loaded, messages)
                adapter = loaded.adapter
                thinking_content = adapter.thinking_parser.extract(output)
                if thinking_content is not None:
                    logger.info(
                        "Thinking probe: always-thinks detected (parser=%s)",
                        parser_id,
                    )
                    return (True, parser_id, diagnostics)
            except Exception as e:
                logger.debug("Path B thinking with parser %s failed: %s", parser_id, e)
                diagnostics.append(
                    ProbeDiagnostic(
                        level=DiagnosticLevel.WARNING,
                        category=DiagnosticCategory.THINKING_DIALECT,
                        message="Thinking verification failed due to generation error",
                        details={"error": str(e), "parser_id": parser_id},
                    )
                )

        # Neither path found thinking tags
        if has_enable_thinking:
            logger.info("Thinking probe: template has enable_thinking param but no tags found")
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

    # ------------------------------------------------------------------
    # Tool sweep
    # ------------------------------------------------------------------

    async def _sweep_tools(
        self,
        model_id: str,
        loaded: Any,
        strategy: Any,
    ) -> tuple[str | None, str | None, list[ProbeDiagnostic]]:
        """Test tool support by iterating parser configurations.

        Uses register_profile_settings() to swap tool parsers.
        Tests both template delivery (native) and adapter injection delivery.

        Returns (tool_format, tool_parser_id, diagnostics).
        """
        from mlx_manager.mlx_server.models.pool import ProfileSettings
        from mlx_manager.mlx_server.parsers import TOOL_PARSERS
        from mlx_manager.mlx_server.utils.template_tools import (
            has_native_tool_support,
        )

        from .base import _TOOL_PROBE_MESSAGES, _TOOL_PROBE_TOOL, _detect_unknown_xml_tags

        diagnostics: list[ProbeDiagnostic] = []
        adapter = loaded.adapter
        tokenizer = loaded.tokenizer
        last_output: str | None = None

        # ── Attempt 1: Template delivery (native tools) ──────────────
        can_try_template = adapter.supports_native_tools() or has_native_tool_support(tokenizer)
        if can_try_template:
            for parser_id in TOOL_PARSERS:
                if parser_id == "null":
                    continue
                try:
                    self._pool.register_profile_settings(
                        model_id,
                        ProfileSettings(tool_parser_id=parser_id),
                    )
                    last_output = await strategy._generate(
                        loaded, _TOOL_PROBE_MESSAGES, tools=_TOOL_PROBE_TOOL
                    )
                    current_adapter = loaded.adapter
                    if current_adapter.tool_parser.validates(last_output, "get_weather"):
                        logger.info(
                            "Tool probe: template delivery verified (parser=%s)",
                            parser_id,
                        )
                        return ("template", parser_id, diagnostics)
                    logger.debug(
                        "Template delivery with parser %s: no match on: %s",
                        parser_id,
                        last_output[:200] if last_output else "",
                    )
                except Exception as e:
                    logger.debug("Template delivery with parser %s failed: %s", parser_id, e)

        # ── Attempt 2: Adapter injection delivery ─────────────────────
        # Use default adapter config (no specific parser override needed yet)
        self._pool.register_profile_settings(model_id, ProfileSettings())
        tool_prompt = adapter.format_tools_for_prompt(_TOOL_PROBE_TOOL)
        if tool_prompt:
            for parser_id in TOOL_PARSERS:
                if parser_id == "null":
                    continue
                try:
                    self._pool.register_profile_settings(
                        model_id,
                        ProfileSettings(tool_parser_id=parser_id),
                    )
                    messages_with_tools = [
                        {"role": "system", "content": tool_prompt},
                        *_TOOL_PROBE_MESSAGES,
                    ]
                    last_output = await strategy._generate(loaded, messages_with_tools)
                    current_adapter = loaded.adapter
                    if current_adapter.tool_parser.validates(last_output, "get_weather"):
                        logger.info(
                            "Tool probe: adapter delivery verified (parser=%s)",
                            parser_id,
                        )
                        return ("adapter", parser_id, diagnostics)
                    logger.debug(
                        "Adapter delivery with parser %s: no match on: %s",
                        parser_id,
                        last_output[:200] if last_output else "",
                    )
                except Exception as e:
                    logger.debug("Adapter delivery with parser %s failed: %s", parser_id, e)

        # ── No match: diagnostic scan ─────────────────────────────────
        registered_parsers = [pid for pid in TOOL_PARSERS if pid != "null"]
        try:
            if last_output is None:
                # Reset to defaults and generate without tools
                self._pool.register_profile_settings(model_id, ProfileSettings())
                last_output = await strategy._generate(loaded, _TOOL_PROBE_MESSAGES)

            tool_markers = [
                "<tool_call>",
                "</tool_call>",
                "<function=",
                "</function>",
                "get_weather",
                '"name"',
            ]
            found_markers = [m for m in tool_markers if m in last_output]
            if found_markers:
                logger.warning(
                    "Tool probe: model attempted tool use (markers: %s) but "
                    "no parser matched. Raw output: %s",
                    found_markers,
                    last_output[:300],
                )
                diagnostics.append(
                    ProbeDiagnostic(
                        level=DiagnosticLevel.ACTION_NEEDED,
                        category=DiagnosticCategory.TOOL_DIALECT,
                        message=(
                            "Unknown tool dialect — model produced tool-like markers "
                            "but no registered parser could parse the output"
                        ),
                        details={
                            "found_markers": found_markers,
                            "raw_output_sample": last_output[:300],
                            "registered_parsers": registered_parsers,
                        },
                    )
                )
            else:
                unknown_tags = _detect_unknown_xml_tags(last_output)
                if unknown_tags:
                    logger.warning(
                        "Tool probe: unknown XML tags in output: %s",
                        unknown_tags,
                    )
                    diagnostics.append(
                        ProbeDiagnostic(
                            level=DiagnosticLevel.WARNING,
                            category=DiagnosticCategory.TOOL_DIALECT,
                            message=(
                                f"Unknown XML tags in output: {', '.join(sorted(unknown_tags))}"
                            ),
                            details={
                                "unknown_tags": sorted(unknown_tags),
                                "raw_output_sample": last_output[:300],
                            },
                        )
                    )
                else:
                    logger.info("Tool probe: no tool support detected")
        except Exception:
            logger.info("Tool probe: no tool support detected")

        return (None, None, diagnostics)


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

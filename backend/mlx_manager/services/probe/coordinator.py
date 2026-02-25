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
    probe_step,
)
from mlx_manager.services.probe.strategy import ProbeStrategy

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
        async with probe_step("detect_type", "model_type") as ctx:
            yield ctx.running
            detection = detect_model_type_detailed(model_id)
            model_type = detection.model_type
            result.model_type = model_type.value

            ctx.value = model_type.value
            ctx.details = {
                "detection_method": detection.detection_method,
                "architecture": detection.architecture,
            }

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
                ctx.diagnostics = [diag]
                result.diagnostics.append(diag)
        yield ctx.result
        if ctx._failed:
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
        async with probe_step("load_model") as ctx:
            yield ctx.running
            self._pool.register_profile_settings(model_id, ProfileSettings())
            loaded = await self._pool.get_model(model_id)
        yield ctx.result
        if ctx._failed:
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
                # Enable tool injection so prepare_input() still passes tools=
                # to the template even though the null parser is active.
                # Without this, supports_tool_calling() returns False and
                # Phase 4 template delivery silently drops tools.
                enable_tool_injection=True,
            )

        # ── Step 4: Look up strategy for type-specific static checks ──
        from .strategy import get_probe_strategy

        strategy: ProbeStrategy | None = get_probe_strategy(model_type)
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

        # ── Step 6: Parser sweep (GenerativeProbe strategies only) ───
        from .base import GenerativeProbe

        if isinstance(strategy, GenerativeProbe):
            async for step in strategy.sweep_capabilities(
                model_id, loaded, result, verbose=verbose
            ):
                yield step

        # ── Step 7: Save results to DB ────────────────────────────────
        async with probe_step("save_results") as ctx:
            yield ctx.running
            await _save_capabilities(model_id, result)
        yield ctx.result

        # ── Step 8: Cleanup ───────────────────────────────────────────
        # Restore original settings or unregister probe settings
        if original_settings is not None:
            self._pool.register_profile_settings(model_id, original_settings)
        else:
            self._pool.unregister_profile_settings(model_id)

        if not was_preloaded:
            async with probe_step("cleanup") as ctx:
                yield ctx.running
                await self._pool.unload_model(model_id)
            yield ctx.result
        else:
            # Restore original parsers and disable tool injection for preloaded models
            if loaded.adapter is not None and original_tool_parser is not None:
                loaded.adapter.configure(
                    tool_parser=original_tool_parser,
                    thinking_parser=original_thinking_parser,
                    enable_tool_injection=False,
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

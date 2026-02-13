"""Probe orchestrator.

Coordinates the probe lifecycle: detect type, load model, dispatch
to the appropriate strategy, save results, and cleanup.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from loguru import logger

from .steps import ProbeResult, ProbeStep
from .strategy import get_probe_strategy


async def probe_model(model_id: str, *, verbose: bool = False) -> AsyncGenerator[ProbeStep, None]:
    """Probe a model's capabilities using the appropriate type strategy.

    Loads the model, detects its type, dispatches to the matching
    probe strategy, stores results in the DB, and cleans up.

    Yields ProbeStep objects for progressive SSE streaming.

    Args:
        model_id: HuggingFace model path (e.g. "mlx-community/Qwen3-0.6B-4bit-DWQ")
        verbose: If True, include diagnostic details in ProbeStep.details

    Yields:
        ProbeStep objects describing each step's progress
    """
    from mlx_manager.mlx_server.models.detection import detect_model_type
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    result = ProbeResult()
    was_preloaded = model_id in pool._models

    # Step 1: Detect model type
    yield ProbeStep(step="detect_type", status="running")
    try:
        model_type = detect_model_type(model_id)
        result.model_type = model_type.value
        yield ProbeStep(
            step="detect_type",
            status="completed",
            capability="model_type",
            value=model_type.value,
        )
    except Exception as e:
        logger.error(f"Type detection failed for {model_id}: {e}")
        yield ProbeStep(step="detect_type", status="failed", error=str(e))
        return

    # Step 2: Look up strategy
    strategy = get_probe_strategy(model_type)
    if strategy is None:
        yield ProbeStep(
            step="find_strategy",
            status="failed",
            error=f"No probe strategy registered for model type: {model_type.value}",
        )
        return

    # Step 2.5: Pre-validate audio models (codecs are detected as AUDIO but can't be loaded)
    from mlx_manager.mlx_server.models.types import ModelType

    if model_type == ModelType.AUDIO:
        from .audio import _detect_audio_capabilities

        is_tts, is_stt, _ = _detect_audio_capabilities(model_id)
        if not is_tts and not is_stt:
            yield ProbeStep(
                step="load_model",
                status="failed",
                error=(
                    f"Unsupported audio model subtype: {model_id} is detected as audio "
                    "but is not a recognized TTS or STT model (likely an audio codec). "
                    "Audio codec models are not supported for inference."
                ),
            )
            return

    # Step 3: Load model
    yield ProbeStep(step="load_model", status="running")
    try:
        loaded = await pool.get_model(model_id)
        yield ProbeStep(step="load_model", status="completed")
    except Exception as e:
        logger.error(f"Probe failed to load model {model_id}: {e}")
        yield ProbeStep(step="load_model", status="failed", error=str(e))
        return

    # Step 4: Run type-specific probe
    try:
        async for step in strategy.probe(model_id, loaded, result):
            yield step
    except Exception as e:
        logger.error(f"Probe strategy failed for {model_id}: {e}")
        yield ProbeStep(step="strategy_error", status="failed", error=str(e))

    # Step 5: Save results to DB
    yield ProbeStep(step="save_results", status="running")
    try:
        await _save_capabilities(model_id, result)
        yield ProbeStep(step="save_results", status="completed")
    except Exception as e:
        logger.error(f"Failed to save probe results for {model_id}: {e}")
        yield ProbeStep(step="save_results", status="failed", error=str(e))

    # Step 6: Cleanup — unload if model wasn't loaded before probe
    if not was_preloaded:
        yield ProbeStep(step="cleanup", status="running")
        try:
            await pool.unload_model(model_id)
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

    logger.info(f"Probe complete for {model_id}: type={result.model_type}")


async def _save_capabilities(model_id: str, result: ProbeResult) -> None:
    """Save model capabilities to the unified Model table."""
    from mlx_manager.services.model_registry import update_model_capabilities

    caps_dict: dict[str, object] = {}
    if result.model_type is not None:
        caps_dict["model_type"] = result.model_type
    if result.supports_native_tools is not None:
        caps_dict["supports_native_tools"] = result.supports_native_tools
    if result.supports_thinking is not None:
        caps_dict["supports_thinking"] = result.supports_thinking
    if result.tool_format is not None:
        caps_dict["tool_format"] = result.tool_format
    if result.practical_max_tokens is not None:
        caps_dict["practical_max_tokens"] = result.practical_max_tokens
    if result.model_family is not None:
        caps_dict["model_family"] = result.model_family
    if result.tool_parser_id is not None:
        caps_dict["tool_parser_id"] = result.tool_parser_id
    if result.thinking_parser_id is not None:
        caps_dict["thinking_parser_id"] = result.thinking_parser_id
    if result.supports_multi_image is not None:
        caps_dict["supports_multi_image"] = result.supports_multi_image
    if result.supports_video is not None:
        caps_dict["supports_video"] = result.supports_video
    if result.embedding_dimensions is not None:
        caps_dict["embedding_dimensions"] = result.embedding_dimensions
    if result.max_sequence_length is not None:
        caps_dict["max_sequence_length"] = result.max_sequence_length
    if result.is_normalized is not None:
        caps_dict["is_normalized"] = result.is_normalized
    if result.supports_tts is not None:
        caps_dict["supports_tts"] = result.supports_tts
    if result.supports_stt is not None:
        caps_dict["supports_stt"] = result.supports_stt
    caps_dict["probe_version"] = 2
    await update_model_capabilities(model_id, **caps_dict)
    logger.info(f"Saved capabilities for {model_id}")

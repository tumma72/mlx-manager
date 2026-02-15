"""Model probe service.

Probes a model's capabilities by loading it temporarily and testing
for native tool support, thinking support, and practical max tokens.
Results are stored in the DB for use at inference time.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from mlx_manager.services.probe.steps import ProbeResult, ProbeStep


async def probe_model(model_id: str):
    """Probe a model's capabilities.

    Loads the model, tests capabilities, stores results in DB,
    and unloads if the model wasn't already loaded.

    Yields ProbeStep objects for progressive SSE streaming.

    Args:
        model_id: HuggingFace model path (e.g. "mlx-community/Qwen3-0.6B-4bit-DWQ")

    Yields:
        ProbeStep objects describing each step's progress
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    result = ProbeResult()
    was_preloaded = model_id in pool._models

    # Step 1: Load model
    yield ProbeStep(step="load_model", status="running")
    try:
        loaded = await pool.get_model(model_id)
        yield ProbeStep(step="load_model", status="completed")
    except Exception as e:
        logger.error(f"Probe failed to load model {model_id}: {e}")
        yield ProbeStep(step="load_model", status="failed", error=str(e))
        return

    tokenizer = loaded.tokenizer

    # Step 2: Check context length
    yield ProbeStep(step="check_context", status="running")
    try:
        practical_max = _estimate_practical_max_tokens(model_id, loaded)
        result.practical_max_tokens = practical_max
        yield ProbeStep(
            step="check_context",
            status="completed",
            capability="practical_max_tokens",
            value=practical_max,
        )
    except Exception as e:
        logger.warning(f"Context check failed for {model_id}: {e}")
        yield ProbeStep(step="check_context", status="failed", error=str(e))

    # Step 3: Test thinking support
    if tokenizer is not None:
        yield ProbeStep(step="test_thinking", status="running")
        try:
            from mlx_manager.mlx_server.utils.template_tools import has_thinking_support

            supports_thinking = has_thinking_support(tokenizer)
            result.supports_thinking = supports_thinking
            yield ProbeStep(
                step="test_thinking",
                status="completed",
                capability="supports_thinking",
                value=supports_thinking,
            )
        except Exception as e:
            logger.warning(f"Thinking test failed for {model_id}: {e}")
            yield ProbeStep(step="test_thinking", status="failed", error=str(e))
    else:
        yield ProbeStep(step="test_thinking", status="skipped")

    # Step 4: Test native tool support
    if tokenizer is not None:
        yield ProbeStep(step="test_tools", status="running")
        try:
            from mlx_manager.mlx_server.utils.template_tools import has_native_tool_support

            supports_tools = has_native_tool_support(tokenizer)
            result.supports_native_tools = supports_tools
            yield ProbeStep(
                step="test_tools",
                status="completed",
                capability="supports_native_tools",
                value=supports_tools,
            )
        except Exception as e:
            logger.warning(f"Tool test failed for {model_id}: {e}")
            yield ProbeStep(step="test_tools", status="failed", error=str(e))
    else:
        yield ProbeStep(step="test_tools", status="skipped")

    # Step 5: Save results to DB
    yield ProbeStep(step="save_results", status="running")
    try:
        await _save_capabilities(model_id, result)
        yield ProbeStep(step="save_results", status="completed")
    except Exception as e:
        logger.error(f"Failed to save probe results for {model_id}: {e}")
        yield ProbeStep(step="save_results", status="failed", error=str(e))

    # Step 6: Cleanup - unload if model wasn't loaded before probe
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

                caps_result = await session.execute(
                    select(Model)
                    .where(Model.repo_id == model_id)
                    .options(selectinload(Model.capabilities))  # type: ignore[arg-type]
                )
                model_record = caps_result.scalar_one_or_none()
                if model_record and model_record.capabilities:
                    loaded.capabilities = model_record.capabilities
        except Exception:
            pass
        yield ProbeStep(step="cleanup", status="skipped")

    logger.info(
        f"Probe complete for {model_id}: "
        f"native_tools={result.supports_native_tools}, "
        f"thinking={result.supports_thinking}, "
        f"max_tokens={result.practical_max_tokens}"
    )


def _estimate_practical_max_tokens(model_id: str, loaded: Any) -> int | None:
    """Estimate practical max tokens based on model config and available memory.

    Delegates to the shared KV cache estimation utility.
    """
    from mlx_manager.mlx_server.utils.kv_cache import estimate_practical_max_tokens

    return estimate_practical_max_tokens(model_id, loaded.size_gb)


async def _save_capabilities(model_id: str, result: ProbeResult) -> None:
    """Save model capabilities to the unified Model table."""
    from mlx_manager.services.model_registry import update_model_capabilities

    caps_dict: dict[str, object] = {}
    if result.supports_native_tools is not None:
        caps_dict["supports_native_tools"] = result.supports_native_tools
    if result.supports_thinking is not None:
        caps_dict["supports_thinking"] = result.supports_thinking
    if result.practical_max_tokens is not None:
        caps_dict["practical_max_tokens"] = result.practical_max_tokens
    caps_dict["probe_version"] = 2
    await update_model_capabilities(model_id, **caps_dict)
    logger.info(f"Saved capabilities for {model_id}")

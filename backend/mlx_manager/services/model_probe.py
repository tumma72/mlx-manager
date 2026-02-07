"""Model probe service.

Probes a model's capabilities by loading it temporarily and testing
for native tool support, thinking support, and practical max tokens.
Results are stored in the DB for use at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger


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
    """Accumulated probe results."""

    supports_native_tools: bool | None = None
    supports_thinking: bool | None = None
    practical_max_tokens: int | None = None


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
            from mlx_manager.models import ModelCapabilities

            async with get_session() as session:
                from sqlmodel import select

                caps_result = await session.execute(
                    select(ModelCapabilities).where(ModelCapabilities.model_id == model_id)
                )
                caps = caps_result.scalar_one_or_none()
                if caps:
                    loaded.capabilities = caps
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

    Uses the formula:
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  (bytes, fp16)
    available_bytes = (device_memory * 0.75 - model_size_gb - 1.0) * 1e9
    practical_max = min(max_position_embeddings, available_bytes / kv_per_token)
    """
    try:
        # Try to read config.json for model parameters
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        max_pos = config.get(
            "max_position_embeddings",
            config.get("max_sequence_length", config.get("seq_length")),
        )
        if max_pos is None:
            return None

        num_layers = config.get(
            "num_hidden_layers", config.get("n_layer", config.get("num_layers"))
        )
        num_kv_heads = config.get(
            "num_key_value_heads",
            config.get("num_attention_heads", config.get("n_head")),
        )
        head_dim = config.get("head_dim")
        if head_dim is None:
            hidden_size = config.get("hidden_size", config.get("d_model"))
            num_heads = config.get("num_attention_heads", config.get("n_head"))
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        if not all([num_layers, num_kv_heads, head_dim]):
            return int(max_pos)

        # KV cache cost per token in bytes (fp16)
        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

        # Available memory
        from mlx_manager.mlx_server.utils.memory import get_device_memory_gb

        device_memory_gb = get_device_memory_gb()
        model_size_gb = loaded.size_gb

        available_gb = (device_memory_gb * 0.75) - model_size_gb - 1.0
        if available_gb <= 0:
            return int(min(max_pos, 2048))  # Minimum practical context

        available_bytes = available_gb * 1e9
        practical_max = int(min(max_pos, available_bytes / kv_per_token))

        return max(practical_max, 512)  # At least 512 tokens

    except Exception as e:
        logger.debug(f"Could not estimate practical max tokens for {model_id}: {e}")
        return None


async def _save_capabilities(model_id: str, result: ProbeResult) -> None:
    """Upsert model capabilities in the database."""
    from mlx_manager.database import get_session
    from mlx_manager.models import ModelCapabilities

    async with get_session() as session:
        from sqlmodel import select

        existing = await session.execute(
            select(ModelCapabilities).where(ModelCapabilities.model_id == model_id)
        )
        caps = existing.scalar_one_or_none()

        if caps:
            # Update existing
            caps.supports_native_tools = result.supports_native_tools
            caps.supports_thinking = result.supports_thinking
            caps.practical_max_tokens = result.practical_max_tokens
            caps.probed_at = datetime.now(tz=UTC)
            session.add(caps)
        else:
            # Create new
            caps = ModelCapabilities(
                model_id=model_id,
                supports_native_tools=result.supports_native_tools,
                supports_thinking=result.supports_thinking,
                practical_max_tokens=result.practical_max_tokens,
            )
            session.add(caps)

        await session.commit()
        logger.info(f"Saved capabilities for {model_id}")

"""Tests for generation timeout protection in GenerativeProbe._generate().

TDD tests written BEFORE implementation (RED phase).

Verifies:
- GenerativeProbe._generate() accepts a timeout parameter (default 60.0s)
- asyncio.TimeoutError from adapter.generate() raises descriptive TimeoutError
- Custom timeout values are respected
- Template options are reset even when timeout occurs
- VisionProbe._generate() has the same timeout pattern
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.services.probe.base import GenerativeProbe

# ---------------------------------------------------------------------------
# Concrete GenerativeProbe subclass for testing
# ---------------------------------------------------------------------------


class _SimpleProbe(GenerativeProbe):
    """Minimal concrete GenerativeProbe for testing _generate() directly."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TEXT_GEN

    async def probe(self, model_id, loaded, result):
        if False:
            yield  # make it an async generator


# ---------------------------------------------------------------------------
# Test 1: default timeout parameter exists and is 60.0
# ---------------------------------------------------------------------------


def test_generate_default_timeout_is_60s():
    """_generate() signature must include timeout parameter with default 60.0."""
    sig = inspect.signature(GenerativeProbe._generate)
    params = sig.parameters

    assert "timeout" in params, "_generate() must have a 'timeout' parameter"
    assert params["timeout"].default == 60.0, "Default timeout must be 60.0 seconds"


# ---------------------------------------------------------------------------
# Test 2: timeout raises descriptive TimeoutError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_timeout_raises_error():
    """_generate() raises TimeoutError when adapter.generate() hangs past timeout."""
    probe = _SimpleProbe()

    async def slow_generate(**kwargs):
        await asyncio.sleep(100)  # Will be cancelled by timeout

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(side_effect=slow_generate)
    mock_adapter.configure = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(TimeoutError) as exc_info:
        await probe._generate(
            loaded=mock_loaded,
            messages=messages,
            timeout=0.05,  # Very short to trigger quickly in tests
        )

    error_message = str(exc_info.value)
    assert "timed out" in error_message.lower() or "timeout" in error_message.lower()
    # Should include the timeout value in the message
    assert "0.05" in error_message


# ---------------------------------------------------------------------------
# Test 3: custom timeout value is used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_custom_timeout():
    """_generate() uses the provided timeout value when waiting for adapter."""
    probe = _SimpleProbe()

    call_log: list[float] = []

    async def timed_slow_generate(**kwargs):
        start = asyncio.get_event_loop().time()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            call_log.append(asyncio.get_event_loop().time() - start)
            raise

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(side_effect=timed_slow_generate)
    mock_adapter.configure = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    messages = [{"role": "user", "content": "Hello"}]

    start = asyncio.get_event_loop().time()
    with pytest.raises(TimeoutError):
        await probe._generate(
            loaded=mock_loaded,
            messages=messages,
            timeout=0.1,  # 100ms timeout
        )
    elapsed = asyncio.get_event_loop().time() - start

    # Should have timed out around 0.1s (allow 0.5s margin for slow CI)
    assert elapsed < 0.6, f"Timeout took too long: {elapsed:.3f}s (expected ~0.1s)"


# ---------------------------------------------------------------------------
# Test 4: template options reset even on timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_timeout_resets_template_options():
    """Template options are reset via adapter.configure(None) even when timeout occurs."""
    probe = _SimpleProbe()

    async def slow_generate(**kwargs):
        await asyncio.sleep(100)

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(side_effect=slow_generate)
    mock_adapter.configure = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    messages = [{"role": "user", "content": "Hello"}]
    template_options = {"enable_thinking": True}

    with pytest.raises(TimeoutError):
        await probe._generate(
            loaded=mock_loaded,
            messages=messages,
            template_options=template_options,
            timeout=0.05,
        )

    # configure should be called twice: once to set options, once to reset
    assert mock_adapter.configure.call_count == 2
    # Second call must reset to None
    reset_call = mock_adapter.configure.call_args_list[1]
    assert (
        reset_call.kwargs.get("template_options") is None
        or (len(reset_call.args) == 0 and reset_call.kwargs == {"template_options": None})
        or reset_call == ((), {"template_options": None})
    )


# ---------------------------------------------------------------------------
# Test 5: successful generation still works (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_success_without_timeout():
    """_generate() completes normally when adapter finishes within timeout."""
    from mlx_manager.mlx_server.models.ir import TextResult

    probe = _SimpleProbe()

    expected_result = TextResult(content="Hello world", reasoning_content=None)

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(return_value=expected_result)
    mock_adapter.configure = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    messages = [{"role": "user", "content": "Hello"}]

    result = await probe._generate(
        loaded=mock_loaded,
        messages=messages,
        timeout=60.0,
    )

    assert result is expected_result


# ---------------------------------------------------------------------------
# Test 6: VisionProbe._generate() has timeout parameter
# ---------------------------------------------------------------------------


def test_vision_generate_has_timeout_parameter():
    """VisionProbe._generate() must also have a timeout parameter with default 60.0."""
    from mlx_manager.services.probe.vision import VisionProbe

    sig = inspect.signature(VisionProbe._generate)
    params = sig.parameters

    assert "timeout" in params, "VisionProbe._generate() must have a 'timeout' parameter"
    assert params["timeout"].default == 60.0, "VisionProbe default timeout must be 60.0 seconds"


# ---------------------------------------------------------------------------
# Test 7: VisionProbe._generate() raises TimeoutError on hang
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vision_generate_timeout():
    """VisionProbe._generate() raises TimeoutError when adapter hangs past timeout."""
    from mlx_manager.services.probe.vision import VisionProbe

    probe = VisionProbe()

    async def slow_generate(**kwargs):
        await asyncio.sleep(100)

    mock_adapter = MagicMock()
    # Use the async function directly to avoid unawaited coroutine from AsyncMock
    mock_adapter.generate = slow_generate
    mock_adapter.configure = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    messages = [{"role": "user", "content": "Describe this image"}]

    # Patch PIL.Image at the PIL module level since it's imported locally inside _generate
    with patch("PIL.Image") as mock_pil:
        mock_pil.new.return_value = MagicMock()

        with pytest.raises(TimeoutError) as exc_info:
            await probe._generate(
                loaded=mock_loaded,
                messages=messages,
                timeout=0.05,
            )

    error_message = str(exc_info.value)
    assert "timed out" in error_message.lower() or "timeout" in error_message.lower()


# ---------------------------------------------------------------------------
# Test 8: error message includes max_tokens for diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_timeout_error_includes_max_tokens():
    """TimeoutError message includes max_tokens to help diagnose slow generation."""
    probe = _SimpleProbe()

    async def slow_generate(**kwargs):
        await asyncio.sleep(100)

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(side_effect=slow_generate)
    mock_adapter.configure = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(TimeoutError) as exc_info:
        await probe._generate(
            loaded=mock_loaded,
            messages=messages,
            max_tokens=512,
            timeout=0.05,
        )

    error_message = str(exc_info.value)
    assert "512" in error_message, "Error message should include max_tokens value"

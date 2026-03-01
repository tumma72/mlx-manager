"""Tests for the new IR types introduced in the IR-Centric Cloud Routing refactoring.

Covers InternalRequest, InferenceResult, and RoutingOutcome — the three types added
on top of the existing PreparedInput / StreamEvent / AdapterResult hierarchy.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    RoutingOutcome,
    StreamEvent,
    TextResult,
)
from mlx_manager.models.enums import ApiType
from mlx_manager.models.value_objects import InferenceParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_params() -> InferenceParams:
    return InferenceParams()


def _minimal_text_result() -> TextResult:
    return TextResult(content="Hello")


def _minimal_inference_result() -> InferenceResult:
    return InferenceResult(
        result=_minimal_text_result(),
        prompt_tokens=10,
        completion_tokens=5,
    )


async def _async_gen() -> AsyncGenerator[StreamEvent, None]:
    """Trivial async generator for RoutingOutcome stream fields."""
    yield StreamEvent(type="content", content="chunk")


# ---------------------------------------------------------------------------
# InternalRequest
# ---------------------------------------------------------------------------


class TestInternalRequest:
    """Tests for InternalRequest protocol-neutral request IR."""

    def test_minimal_construction(self) -> None:
        """Required fields only — optional fields default to None/False."""
        req = InternalRequest(
            model="mlx-community/Qwen3-0.6B-4bit-DWQ",
            messages=[{"role": "user", "content": "Hi"}],
            params=_minimal_params(),
        )
        assert req.model == "mlx-community/Qwen3-0.6B-4bit-DWQ"
        assert req.messages == [{"role": "user", "content": "Hi"}]
        assert req.stream is False
        assert req.stop is None
        assert req.tools is None
        assert req.images is None
        assert req.original_request is None
        assert req.original_protocol is None

    def test_stream_flag(self) -> None:
        """Streaming flag is stored correctly when set to True."""
        req = InternalRequest(
            model="m",
            messages=[],
            params=_minimal_params(),
            stream=True,
        )
        assert req.stream is True

    def test_all_optional_fields(self) -> None:
        """All optional fields are stored without mutation."""
        tools = [{"type": "function", "function": {"name": "fn"}}]
        images = ["data:image/png;base64,abc"]
        stop = ["<|endoftext|>", "<|im_end|>"]
        raw_request: dict[str, Any] = {"raw": True}

        req = InternalRequest(
            model="m",
            messages=[{"role": "user", "content": "hello"}],
            params=InferenceParams(temperature=0.5, max_tokens=256),
            stream=False,
            stop=stop,
            tools=tools,
            images=images,
            original_request=raw_request,
            original_protocol=ApiType.OPENAI,
        )
        assert req.stop == stop
        assert req.tools == tools
        assert req.images == images
        assert req.original_request is raw_request
        assert req.original_protocol == ApiType.OPENAI

    def test_anthropic_protocol(self) -> None:
        """ApiType.ANTHROPIC is stored correctly in original_protocol."""
        req = InternalRequest(
            model="m",
            messages=[],
            params=_minimal_params(),
            original_protocol=ApiType.ANTHROPIC,
        )
        assert req.original_protocol == ApiType.ANTHROPIC

    def test_params_stored_as_inference_params(self) -> None:
        """InferenceParams value object is preserved on the request."""
        params = InferenceParams(temperature=0.8, max_tokens=512, top_p=0.95)
        req = InternalRequest(model="m", messages=[], params=params)
        assert req.params.temperature == pytest.approx(0.8)
        assert req.params.max_tokens == 512
        assert req.params.top_p == pytest.approx(0.95)

    def test_model_copy_with_update(self) -> None:
        """model_copy(update=...) produces a new instance without mutating the original."""
        req = InternalRequest(
            model="original",
            messages=[{"role": "user", "content": "hi"}],
            params=_minimal_params(),
            stream=False,
        )
        updated = req.model_copy(update={"model": "updated", "stream": True})

        # Updated instance has new values
        assert updated.model == "updated"
        assert updated.stream is True
        # Original is unchanged
        assert req.model == "original"
        assert req.stream is False

    def test_arbitrary_types_allowed(self) -> None:
        """arbitrary_types_allowed lets non-Pydantic objects sit in original_request."""

        class ArbitraryRequest:
            body = "raw bytes"

        obj = ArbitraryRequest()
        req = InternalRequest(
            model="m",
            messages=[],
            params=_minimal_params(),
            original_request=obj,
        )
        assert req.original_request is obj

    def test_messages_multiple_turns(self) -> None:
        """Multi-turn conversation messages are stored in order."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Tell me a joke."},
        ]
        req = InternalRequest(model="m", messages=msgs, params=_minimal_params())
        assert req.messages == msgs
        assert len(req.messages) == 4


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------


class TestInferenceResult:
    """Tests for InferenceResult non-streaming result wrapper."""

    def test_construction(self) -> None:
        """InferenceResult stores TextResult and token counts correctly."""
        text_result = TextResult(content="Answer", finish_reason="stop")
        result = InferenceResult(
            result=text_result,
            prompt_tokens=42,
            completion_tokens=17,
        )
        assert result.result is text_result
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 17

    def test_field_access(self) -> None:
        """Nested TextResult fields are accessible through .result."""
        text_result = TextResult(
            content="The answer is 42",
            reasoning_content="Let me think...",
            tool_calls=None,
            finish_reason="stop",
        )
        ir = InferenceResult(result=text_result, prompt_tokens=8, completion_tokens=12)
        assert ir.result.content == "The answer is 42"
        assert ir.result.reasoning_content == "Let me think..."
        assert ir.result.finish_reason == "stop"

    def test_tool_calls_finish_reason(self) -> None:
        """finish_reason='tool_calls' propagates through the wrapper."""
        text_result = TextResult(
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f"}}],
            finish_reason="tool_calls",
        )
        ir = InferenceResult(result=text_result, prompt_tokens=5, completion_tokens=30)
        assert ir.result.finish_reason == "tool_calls"
        assert ir.result.tool_calls is not None
        assert len(ir.result.tool_calls) == 1

    def test_zero_token_counts(self) -> None:
        """Token counts of zero are valid (e.g. cached prompt)."""
        ir = InferenceResult(
            result=TextResult(content=""),
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert ir.prompt_tokens == 0
        assert ir.completion_tokens == 0


# ---------------------------------------------------------------------------
# RoutingOutcome
# ---------------------------------------------------------------------------


class TestRoutingOutcome:
    """Tests for RoutingOutcome tagged-union result from the backend router."""

    # -- ir_result variant ---------------------------------------------------

    def test_ir_result_variant(self) -> None:
        """ir_result variant stores an InferenceResult; both properties False."""
        outcome = RoutingOutcome(ir_result=_minimal_inference_result())
        assert outcome.ir_result is not None
        assert outcome.ir_stream is None
        assert outcome.raw_response is None
        assert outcome.raw_stream is None

    def test_ir_result_is_not_passthrough(self) -> None:
        """ir_result is local/formatted, not a passthrough raw response."""
        outcome = RoutingOutcome(ir_result=_minimal_inference_result())
        assert outcome.is_passthrough is False

    def test_ir_result_is_not_streaming(self) -> None:
        """ir_result carries a complete non-streaming result."""
        outcome = RoutingOutcome(ir_result=_minimal_inference_result())
        assert outcome.is_streaming is False

    # -- ir_stream variant ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_ir_stream_variant(self) -> None:
        """ir_stream variant stores an AsyncGenerator."""
        gen = _async_gen()
        outcome = RoutingOutcome(ir_stream=gen)
        assert outcome.ir_result is None
        assert outcome.ir_stream is gen
        assert outcome.raw_response is None
        assert outcome.raw_stream is None
        # Drain to avoid resource warning
        async for _ in outcome.ir_stream:
            pass

    @pytest.mark.asyncio
    async def test_ir_stream_is_not_passthrough(self) -> None:
        """ir_stream needs formatting — it is not a raw passthrough."""
        gen = _async_gen()
        outcome = RoutingOutcome(ir_stream=gen)
        assert outcome.is_passthrough is False
        async for _ in outcome.ir_stream:
            pass

    @pytest.mark.asyncio
    async def test_ir_stream_is_streaming(self) -> None:
        """ir_stream is the streaming IR variant."""
        gen = _async_gen()
        outcome = RoutingOutcome(ir_stream=gen)
        assert outcome.is_streaming is True
        async for _ in outcome.ir_stream:
            pass

    # -- raw_response variant ------------------------------------------------

    def test_raw_response_variant_dict(self) -> None:
        """raw_response can hold a plain dict passthrough payload.

        Note: Pydantic copies dict values rather than storing the same reference,
        so equality (==) is used instead of identity (is).
        """
        payload: dict[str, Any] = {"id": "chatcmpl-123", "choices": []}
        outcome = RoutingOutcome(raw_response=payload)
        assert outcome.ir_result is None
        assert outcome.ir_stream is None
        assert outcome.raw_response == payload
        assert outcome.raw_stream is None

    def test_raw_response_is_passthrough(self) -> None:
        """raw_response signals same-protocol passthrough — is_passthrough=True."""
        outcome = RoutingOutcome(raw_response={"ok": True})
        assert outcome.is_passthrough is True

    def test_raw_response_is_not_streaming(self) -> None:
        """raw_response is a complete non-streaming passthrough."""
        outcome = RoutingOutcome(raw_response={"ok": True})
        assert outcome.is_streaming is False

    # -- raw_stream variant --------------------------------------------------

    @pytest.mark.asyncio
    async def test_raw_stream_variant(self) -> None:
        """raw_stream variant stores a passthrough streaming AsyncGenerator."""
        gen = _async_gen()
        outcome = RoutingOutcome(raw_stream=gen)
        assert outcome.ir_result is None
        assert outcome.ir_stream is None
        assert outcome.raw_response is None
        assert outcome.raw_stream is gen
        async for _ in outcome.raw_stream:
            pass

    @pytest.mark.asyncio
    async def test_raw_stream_is_passthrough(self) -> None:
        """raw_stream is a same-protocol passthrough — is_passthrough=True."""
        gen = _async_gen()
        outcome = RoutingOutcome(raw_stream=gen)
        assert outcome.is_passthrough is True
        async for _ in outcome.raw_stream:
            pass

    @pytest.mark.asyncio
    async def test_raw_stream_is_streaming(self) -> None:
        """raw_stream is the passthrough streaming variant — is_streaming=True."""
        gen = _async_gen()
        outcome = RoutingOutcome(raw_stream=gen)
        assert outcome.is_streaming is True
        async for _ in outcome.raw_stream:
            pass

    # -- Empty / default variant ---------------------------------------------

    def test_empty_outcome(self) -> None:
        """All-None outcome is valid Pydantic but both properties read False."""
        outcome = RoutingOutcome()
        assert outcome.ir_result is None
        assert outcome.ir_stream is None
        assert outcome.raw_response is None
        assert outcome.raw_stream is None
        assert outcome.is_passthrough is False
        assert outcome.is_streaming is False

    # -- Property semantics with multiple fields set -------------------------

    @pytest.mark.asyncio
    async def test_is_passthrough_true_when_raw_stream_set(self) -> None:
        """is_passthrough is True whenever raw_stream is present (stream passthrough)."""
        gen = _async_gen()
        outcome = RoutingOutcome(raw_stream=gen)
        assert outcome.is_passthrough is True
        async for _ in outcome.raw_stream:
            pass

    @pytest.mark.asyncio
    async def test_is_streaming_true_when_ir_stream_set(self) -> None:
        """is_streaming is True whenever ir_stream is present (local streaming)."""
        gen = _async_gen()
        outcome = RoutingOutcome(ir_stream=gen)
        assert outcome.is_streaming is True
        async for _ in outcome.ir_stream:
            pass

    def test_is_passthrough_false_for_ir_only(self) -> None:
        """is_passthrough is False when only ir_result is set (local non-passthrough)."""
        outcome = RoutingOutcome(ir_result=_minimal_inference_result())
        assert outcome.is_passthrough is False

    # -- arbitrary_types_allowed (AsyncGenerator is not a Pydantic type) ----

    @pytest.mark.asyncio
    async def test_async_generator_accepted_by_pydantic(self) -> None:
        """RoutingOutcome's model_config allows AsyncGenerator without validation error."""
        gen = _async_gen()
        # Should not raise ValidationError
        outcome = RoutingOutcome(ir_stream=gen)
        assert outcome.ir_stream is gen
        async for _ in outcome.ir_stream:
            pass

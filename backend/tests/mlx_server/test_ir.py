"""Tests for Intermediate Representation types (models/ir.py)."""

from mlx_manager.mlx_server.models.ir import (
    AdapterResult,
    AudioResult,
    EmbeddingResult,
    PreparedInput,
    StreamEvent,
    TextResult,
    TranscriptionResult,
)


class TestPreparedInput:
    """Tests for PreparedInput IR type."""

    def test_minimal(self) -> None:
        inp = PreparedInput(prompt="Hello")
        assert inp.prompt == "Hello"
        assert inp.token_ids is None
        assert inp.stop_token_ids is None
        assert inp.pixel_values is None
        assert inp.generation_params is None

    def test_full(self) -> None:
        inp = PreparedInput(
            prompt="Hi",
            token_ids=[1, 2, 3],
            stop_token_ids=[0],
            pixel_values=[[0.5]],
            generation_params={"temperature": 0.7},
        )
        assert inp.token_ids == [1, 2, 3]
        assert inp.pixel_values == [[0.5]]
        assert inp.generation_params["temperature"] == 0.7


class TestStreamEvent:
    """Tests for StreamEvent IR type."""

    def test_empty(self) -> None:
        event = StreamEvent()
        assert event.type is None
        assert event.content is None
        assert event.reasoning_content is None
        assert event.tool_call_delta is None
        assert event.is_complete is False

    def test_content(self) -> None:
        event = StreamEvent(type="content", content="Hello")
        assert event.type == "content"
        assert event.content == "Hello"

    def test_reasoning(self) -> None:
        event = StreamEvent(type="reasoning_content", reasoning_content="Thinking...")
        assert event.type == "reasoning_content"
        assert event.reasoning_content == "Thinking..."

    def test_tool_call_delta(self) -> None:
        delta = {"name": "get_weather", "arguments": '{"city":'}
        event = StreamEvent(type="tool_call_delta", tool_call_delta=delta)
        assert event.type == "tool_call_delta"
        assert event.tool_call_delta == delta

    def test_is_complete(self) -> None:
        event = StreamEvent(
            type="reasoning_content",
            reasoning_content="Done",
            is_complete=True,
        )
        assert event.is_complete is True


class TestAdapterResultHierarchy:
    """Tests for AdapterResult ABC and concrete subclasses."""

    def test_base_result_defaults(self) -> None:
        result = AdapterResult()
        assert result.finish_reason == "stop"

    def test_text_result_minimal(self) -> None:
        result = TextResult(content="Hello")
        assert result.content == "Hello"
        assert result.finish_reason == "stop"
        assert result.reasoning_content is None
        assert result.tool_calls is None

    def test_text_result_full(self) -> None:
        result = TextResult(
            content="Here's the weather",
            reasoning_content="Let me think...",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather"}}],
            finish_reason="tool_calls",
        )
        d = result.model_dump()
        assert d["content"] == "Here's the weather"
        assert d["reasoning_content"] == "Let me think..."
        assert len(d["tool_calls"]) == 1
        assert d["finish_reason"] == "tool_calls"

    def test_text_result_is_adapter_result(self) -> None:
        result = TextResult(content="test")
        assert isinstance(result, AdapterResult)

    def test_embedding_result(self) -> None:
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            dimensions=2,
        )
        assert isinstance(result, AdapterResult)
        d = result.model_dump()
        assert d["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert d["dimensions"] == 2
        assert d["finish_reason"] == "stop"

    def test_audio_result(self) -> None:
        result = AudioResult(
            audio_bytes=b"\x00\x01",
            sample_rate=24000,
            format="wav",
        )
        assert isinstance(result, AdapterResult)
        d = result.model_dump()
        assert d["audio_bytes"] == b"\x00\x01"
        assert d["sample_rate"] == 24000
        assert d["format"] == "wav"

    def test_transcription_result(self) -> None:
        result = TranscriptionResult(
            text="Hello world",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
        )
        assert isinstance(result, AdapterResult)
        d = result.model_dump()
        assert d["text"] == "Hello world"
        assert len(d["segments"]) == 1

    def test_transcription_result_minimal(self) -> None:
        result = TranscriptionResult(text="Hi")
        assert result.segments is None
        assert result.finish_reason == "stop"

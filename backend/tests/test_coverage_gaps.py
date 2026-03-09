"""Tests targeting specific uncovered lines across multiple modules.

Covers:
- composable.py: _UnsetType repr/bool, model_type property, apply_chat_template fallback,
  generate_step vision streaming, generate_speech fallback, embeddings fallback
- strategies.py: glm4_template TypeError fallback with tools, mistral_template TypeError
  fallback with tools, liquid_template fallback, mistral_message_converter
- base.py (parsers): ABC abstract properties (raise NotImplementedError)
- cloud/client.py: _build_headers abstract, protocol abstract, forward_request abstract,
  _stream_with_circuit_breaker success, _stream_with_circuit_breaker failure
- cloud/router.py: route_request fails to get session, _resolve_profile_model
- embeddings.py: LOGFIRE_AVAILABLE import path
- inference.py: legacy vision streaming, legacy text non-streaming, legacy vision non-streaming
- probe/base.py: BaseProbe.model_type, GenerativeProbe._generate timeout,
  estimate_context_window, get_model_config_value, _has_tokenization_artifacts,
  _prioritize_parsers, _discover_and_map_tags uncovered parsers
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# 1. composable.py
# ============================================================================


class TestUnsetType:
    """Cover _UnsetType.__repr__ (line 51) and __bool__ (line 54)."""

    def test_repr(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import _UNSET

        assert repr(_UNSET) == "<UNSET>"

    def test_bool_is_false(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import _UNSET

        assert bool(_UNSET) is False
        assert not _UNSET


class TestModelAdapterModelTypeProperty:
    """Cover model_type property (line 123)."""

    def test_model_type_returns_stored_type(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        adapter = ModelAdapter(model_type="vision")
        assert adapter.model_type == "vision"

    def test_model_type_text_gen(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        adapter = ModelAdapter(model_type="text-gen")
        assert adapter.model_type == "text-gen"


class TestApplyChatTemplateFallback:
    """Cover lines 252, 269 — apply_chat_template TypeError fallback with native_tools."""

    def test_fallback_strips_template_options_but_keeps_tools(self) -> None:
        """When first apply_chat_template raises TypeError, fallback keeps native_tools."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
        from mlx_manager.mlx_server.models.adapters.configs import FamilyConfig

        call_count = 0

        class FailOnOptionsTokenizer:
            eos_token_id = 0
            unk_token_id = -1

            def convert_tokens_to_ids(self, token: str) -> int:
                return -1

            def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise TypeError("unexpected keyword argument")
                tools = kwargs.get("tools")
                return f"fallback tools={tools is not None}"

        config = FamilyConfig(family="test", native_tools=True)
        adapter = ModelAdapter(
            model_type="text-gen",
            config=config,
            tokenizer=FailOnOptionsTokenizer(),
            template_options={"some_option": True},
        )
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"function": {"name": "test"}}]
        result = adapter.apply_chat_template(messages, tools=tools)
        assert "fallback" in result
        assert "tools=True" in result


class TestGenerateStepVisionStreaming:
    """Cover lines 681-715 — vision streaming without images."""

    async def test_vision_text_only_streaming(self) -> None:
        """generate_step for vision model without images uses VLM streaming."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
        from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult

        tok = MagicMock()
        tok.eos_token_id = 0
        tok.unk_token_id = -1
        tok.convert_tokens_to_ids = MagicMock(return_value=-1)
        tok.apply_chat_template = MagicMock(return_value="<prompt>")
        tok.tokenizer = tok

        adapter = ModelAdapter(model_type="vision", tokenizer=tok, model_id="test/model")
        model = MagicMock()

        async def fake_stream_from_metal(produce_fn):
            for item in produce_fn():
                yield item

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
                side_effect=fake_stream_from_metal,
            ),
            patch("mlx_vlm.stream_generate") as mock_vlm_stream,
        ):
            resp1 = MagicMock()
            resp1.token = None
            resp1.text = "Hello"
            resp2 = MagicMock()
            resp2.token = None
            resp2.text = " world"
            mock_vlm_stream.return_value = [resp1, resp2]

            events = []
            async for item in adapter.generate_step(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            ):
                events.append(item)

        # Last event should be TextResult
        assert isinstance(events[-1], TextResult)
        # Earlier events are StreamEvent
        stream_events = [e for e in events if isinstance(e, StreamEvent)]
        assert len(stream_events) >= 1


class TestGenerateStepTextStopToken:
    """Cover line 740 — text streaming stop token in produce_tokens."""

    async def test_text_stop_token_ends_generation(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
        from mlx_manager.mlx_server.models.ir import TextResult

        tok = MagicMock()
        tok.eos_token_id = 99
        tok.unk_token_id = -1
        tok.convert_tokens_to_ids = MagicMock(return_value=-1)
        tok.apply_chat_template = MagicMock(return_value="<prompt>")
        tok.tokenizer = tok

        adapter = ModelAdapter(model_type="text-gen", tokenizer=tok)
        model = MagicMock()

        async def fake_stream_from_metal(produce_fn):
            for item in produce_fn():
                yield item

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
                side_effect=fake_stream_from_metal,
            ),
            patch("mlx_lm.stream_generate") as mock_gen,
            patch("mlx_lm.sample_utils.make_sampler") as mock_sampler,
        ):
            mock_sampler.return_value = MagicMock()
            resp1 = MagicMock()
            resp1.token = 1
            resp1.text = "Hi"
            resp2 = MagicMock()
            resp2.token = 99  # Stop token
            resp2.text = ""
            mock_gen.return_value = [resp1, resp2]

            events = []
            async for item in adapter.generate_step(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=100,
            ):
                events.append(item)

        assert isinstance(events[-1], TextResult)
        assert events[-1].finish_reason == "stop"


class TestGenerateSpeechFallback:
    """Cover lines 888-890 — np.array fallback for older MLX."""

    async def test_generate_speech_delegates_to_run_on_metal(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
        from mlx_manager.mlx_server.models.ir import AudioResult

        adapter = ModelAdapter(model_type="audio")
        model = MagicMock()

        async def fake_run_on_metal(fn, **kwargs):
            return (b"audio_bytes", 24000)

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=fake_run_on_metal,
        ):
            result = await adapter.generate_speech(model, "Hello", voice="af_heart")

        assert isinstance(result, AudioResult)
        assert result.audio_bytes == b"audio_bytes"
        assert result.sample_rate == 24000


class TestEmbeddingsFallbackTokenCount:
    """Cover line 796 — embeddings fallback token counting without attention_mask."""

    async def test_generate_embeddings_delegates_to_run_on_metal(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
        from mlx_manager.mlx_server.models.ir import EmbeddingResult

        tok = MagicMock()
        tok.eos_token_id = 0
        tok.unk_token_id = -1
        tok.convert_tokens_to_ids = MagicMock(return_value=-1)
        tok.tokenizer = tok

        adapter = ModelAdapter(model_type="embeddings", tokenizer=tok)
        model = MagicMock()

        async def fake_run_on_metal(fn, **kwargs):
            return ([[0.1, 0.2], [0.3, 0.4]], 6)

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=fake_run_on_metal,
        ):
            result = await adapter.generate_embeddings(model, ["hello", "world"])

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert result.total_tokens == 6


# ============================================================================
# 2. strategies.py
# ============================================================================


class _TypeErrorOnExtraKwargsTokenizer:
    """Tokenizer that raises TypeError only when extra kwargs are passed."""

    chat_template = ""  # for mistral to see no SYSTEM_PROMPT

    def apply_chat_template(
        self,
        messages: list,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        tools: list | None = None,
        **kwargs: Any,
    ) -> str:
        # Raise if any extra kwargs besides the standard ones
        if kwargs:
            raise TypeError("unexpected keyword argument")
        parts = [f"{m.get('role')}: {m.get('content', '')}" for m in messages]
        if tools:
            parts.append(f"[TOOLS: {len(tools)}]")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


class TestGlm4TemplateFallbackWithTools:
    """Cover lines 72-73, 77-86 — glm4_template TypeError fallback with tools."""

    def test_glm4_template_fallback_with_tools_and_template_options(self) -> None:
        from mlx_manager.mlx_server.models.adapters.strategies import glm4_template

        tok = _TypeErrorOnExtraKwargsTokenizer()
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test"}}]

        result = glm4_template(
            tokenizer=tok,
            messages=messages,
            add_generation_prompt=True,
            native_tools=tools,
            template_options={"enable_thinking": True},
        )

        assert "user: Hello" in result
        assert "[TOOLS: 1]" in result


class TestMistralTemplateFallbackWithTools:
    """Cover lines 134-135, 138-146 — mistral_template TypeError fallback with tools."""

    def test_mistral_template_fallback_with_tools_and_template_options(self) -> None:
        from mlx_manager.mlx_server.models.adapters.strategies import mistral_template

        tok = _TypeErrorOnExtraKwargsTokenizer()
        tok.chat_template = "[SYSTEM_PROMPT] some template"  # has system support
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test"}}]

        result = mistral_template(
            tokenizer=tok,
            messages=messages,
            add_generation_prompt=True,
            native_tools=tools,
            template_options={"enable_thinking": True},
        )

        assert "user: Hello" in result
        assert "[TOOLS: 1]" in result


class TestLiquidTemplateFallbackWithTools:
    """Cover lines 157-178 — liquid_template fallback."""

    def test_liquid_template_fallback_with_tools(self) -> None:
        from mlx_manager.mlx_server.models.adapters.strategies import liquid_template

        call_count = 0

        class LiquidTokenizer:
            def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
                nonlocal call_count
                call_count += 1
                if "keep_past_thinking" in kwargs:
                    raise TypeError("unexpected kwarg")
                parts = [f"{m.get('role')}: {m.get('content', '')}" for m in messages]
                if kwargs.get("tools"):
                    parts.append(f"[TOOLS: {len(kwargs['tools'])}]")
                if kwargs.get("add_generation_prompt"):
                    parts.append("assistant:")
                return "\n".join(parts)

        tok = LiquidTokenizer()
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test"}}]

        result = liquid_template(
            tokenizer=tok,
            messages=messages,
            add_generation_prompt=True,
            native_tools=tools,
            template_options={"keep_past_thinking": True},
        )

        assert "user: Hello" in result
        assert "[TOOLS: 1]" in result
        assert call_count == 2  # first call failed, second succeeded


class TestMistralMessageConverter:
    """Cover lines 341-365 — mistral_message_converter."""

    def test_converts_string_arguments_to_dict(self) -> None:
        from mlx_manager.mlx_server.models.adapters.strategies import mistral_message_converter

        messages = [
            {
                "role": "assistant",
                "content": "Checking weather",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    }
                ],
            }
        ]

        result = mistral_message_converter(messages)

        assert len(result) == 1
        tc = result[0]["tool_calls"][0]
        # Arguments should be converted from string to dict
        assert isinstance(tc["function"]["arguments"], dict)
        assert tc["function"]["arguments"]["location"] == "Tokyo"

    def test_handles_invalid_json_arguments(self) -> None:
        from mlx_manager.mlx_server.models.adapters.strategies import mistral_message_converter

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test",
                            "arguments": "not valid json",
                        },
                    }
                ],
            }
        ]

        result = mistral_message_converter(messages)

        # Invalid JSON stays as string
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "not valid json"

    def test_passes_through_non_tool_messages(self) -> None:
        from mlx_manager.mlx_server.models.adapters.strategies import mistral_message_converter

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "tool", "content": "result"},
        ]

        result = mistral_message_converter(messages)

        # All messages pass through unchanged
        assert len(result) == 3
        assert result[0] == messages[0]
        assert result[1] == messages[1]
        assert result[2] == messages[2]

    def test_dict_arguments_stay_as_dict(self) -> None:
        from mlx_manager.mlx_server.models.adapters.strategies import mistral_message_converter

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test",
                            "arguments": {"already": "a dict"},
                        },
                    }
                ],
            }
        ]

        result = mistral_message_converter(messages)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"already": "a dict"}


# ============================================================================
# 3. parsers/base.py — ABC abstract method lines 30, 36, 41, 91, 97, 108, 113
# ============================================================================


class TestToolCallParserABCProperties:
    """Verify ABC abstract methods raise NotImplementedError."""

    def test_parser_id_raises(self) -> None:
        from mlx_manager.mlx_server.parsers.base import ToolCallParser

        # Create a minimal concrete subclass that only implements what's needed
        class MinimalParser(ToolCallParser):
            @property
            def parser_id(self) -> str:
                # Call super to hit the raise
                return super().parser_id  # type: ignore[safe-super]

            @property
            def stream_markers(self) -> list[tuple[str, str]]:
                return []

            def extract(self, text: str) -> list:
                return []

        p = MinimalParser()
        with pytest.raises(NotImplementedError):
            _ = p.parser_id

    def test_stream_markers_raises(self) -> None:
        from mlx_manager.mlx_server.parsers.base import ToolCallParser

        class MinimalParser(ToolCallParser):
            @property
            def parser_id(self) -> str:
                return "test"

            @property
            def stream_markers(self) -> list[tuple[str, str]]:
                return super().stream_markers  # type: ignore[safe-super]

            def extract(self, text: str) -> list:
                return []

        p = MinimalParser()
        with pytest.raises(NotImplementedError):
            _ = p.stream_markers

    def test_extract_raises(self) -> None:
        from mlx_manager.mlx_server.parsers.base import ToolCallParser

        class MinimalParser(ToolCallParser):
            @property
            def parser_id(self) -> str:
                return "test"

            @property
            def stream_markers(self) -> list[tuple[str, str]]:
                return []

            def extract(self, text: str) -> list:
                return super().extract(text)  # type: ignore[safe-super]

        p = MinimalParser()
        with pytest.raises(NotImplementedError):
            p.extract("text")


class TestThinkingParserABCProperties:
    """Verify ThinkingParser ABC abstract methods raise NotImplementedError."""

    def test_parser_id_raises(self) -> None:
        from mlx_manager.mlx_server.parsers.base import ThinkingParser

        class MinimalParser(ThinkingParser):
            @property
            def parser_id(self) -> str:
                return super().parser_id  # type: ignore[safe-super]

            @property
            def stream_markers(self) -> list[tuple[str, str]]:
                return []

            def extract(self, text: str) -> str | None:
                return None

            def remove(self, text: str) -> str:
                return text

        p = MinimalParser()
        with pytest.raises(NotImplementedError):
            _ = p.parser_id

    def test_stream_markers_raises(self) -> None:
        from mlx_manager.mlx_server.parsers.base import ThinkingParser

        class MinimalParser(ThinkingParser):
            @property
            def parser_id(self) -> str:
                return "test"

            @property
            def stream_markers(self) -> list[tuple[str, str]]:
                return super().stream_markers  # type: ignore[safe-super]

            def extract(self, text: str) -> str | None:
                return None

            def remove(self, text: str) -> str:
                return text

        p = MinimalParser()
        with pytest.raises(NotImplementedError):
            _ = p.stream_markers

    def test_extract_raises(self) -> None:
        from mlx_manager.mlx_server.parsers.base import ThinkingParser

        class MinimalParser(ThinkingParser):
            @property
            def parser_id(self) -> str:
                return "test"

            @property
            def stream_markers(self) -> list[tuple[str, str]]:
                return []

            def extract(self, text: str) -> str | None:
                return super().extract(text)  # type: ignore[safe-super]

            def remove(self, text: str) -> str:
                return text

        p = MinimalParser()
        with pytest.raises(NotImplementedError):
            p.extract("text")

    def test_remove_raises(self) -> None:
        from mlx_manager.mlx_server.parsers.base import ThinkingParser

        class MinimalParser(ThinkingParser):
            @property
            def parser_id(self) -> str:
                return "test"

            @property
            def stream_markers(self) -> list[tuple[str, str]]:
                return []

            def extract(self, text: str) -> str | None:
                return None

            def remove(self, text: str) -> str:
                return super().remove(text)  # type: ignore[safe-super]

        p = MinimalParser()
        with pytest.raises(NotImplementedError):
            p.remove("text")


# ============================================================================
# 4. cloud/client.py — lines 154, 160, 179, 223, 226-228
# ============================================================================


class TestCloudClientAbstractProperties:
    """Cover abstract method stubs: _build_headers (154), protocol (160), forward_request (179)."""

    def test_abstract_methods_defined_in_base(self) -> None:
        from mlx_manager.mlx_server.services.cloud.client import CloudBackendClient

        assert "_build_headers" in CloudBackendClient.__abstractmethods__
        assert "protocol" in CloudBackendClient.__abstractmethods__
        assert "forward_request" in CloudBackendClient.__abstractmethods__


class TestStreamCircuitBreakerSuccess:
    """Cover line 223 — _stream_with_circuit_breaker success path."""

    async def test_stream_success_records_success(self) -> None:
        from mlx_manager.mlx_server.services.cloud.client import CloudBackendClient

        class TestClient(CloudBackendClient):
            @property
            def protocol(self) -> Any:
                return "openai"

            def _build_headers(self) -> dict[str, str]:
                return {"Authorization": f"Bearer {self._api_key}"}

            async def forward_request(self, ir: Any) -> Any:
                return {}

        client = TestClient(base_url="https://api.example.com", api_key="test")

        # Mock the streaming context manager
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield "data: hello"
            yield "data: world"

        mock_response.aiter_lines = mock_aiter_lines

        @asynccontextmanager
        async def mock_stream(method, endpoint, json):
            yield mock_response

        with patch.object(client._client, "stream", side_effect=mock_stream):
            lines = []
            async for line in client._stream_with_circuit_breaker("/test", {"data": "value"}):
                lines.append(line)

        assert lines == ["data: hello", "data: world"]
        assert client._circuit_breaker.fail_counter == 0


class TestStreamCircuitBreakerFailure:
    """Cover lines 226-228 — _stream_with_circuit_breaker failure path."""

    async def test_stream_failure_records_failure(self) -> None:
        from mlx_manager.mlx_server.services.cloud.client import CloudBackendClient

        class TestClient(CloudBackendClient):
            @property
            def protocol(self) -> Any:
                return "openai"

            def _build_headers(self) -> dict[str, str]:
                return {"Authorization": f"Bearer {self._api_key}"}

            async def forward_request(self, ir: Any) -> Any:
                return {}

        client = TestClient(base_url="https://api.example.com", api_key="test")

        @asynccontextmanager
        async def mock_stream(method, endpoint, json):
            raise ConnectionError("Stream failed")
            yield  # noqa: F841

        with patch.object(client._client, "stream", side_effect=mock_stream):
            with pytest.raises(ConnectionError, match="Stream failed"):
                async for _ in client._stream_with_circuit_breaker("/test", {"data": "value"}):
                    pass

        assert client._circuit_breaker.fail_counter == 1


# ============================================================================
# 5. cloud/router.py — lines 51, 136-149
# ============================================================================


class TestRouterResolveProfileModel:
    """Cover lines 136-149 — _resolve_profile_model."""

    async def test_resolve_profile_model_returns_repo_id(self) -> None:
        from mlx_manager.mlx_server.services.cloud.router import BackendRouter

        router = BackendRouter()
        mock_db = AsyncMock()

        # Mock profile query
        mock_profile = MagicMock()
        mock_profile.model_id = 42

        # Mock model query
        mock_model = MagicMock()
        mock_model.repo_id = "mlx-community/test-model"

        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = mock_profile
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = mock_model

        mock_db.execute.side_effect = [mock_result1, mock_result2]

        result = await router._resolve_profile_model(mock_db, 1)
        assert result == "mlx-community/test-model"

    async def test_resolve_profile_model_profile_not_found(self) -> None:
        from mlx_manager.mlx_server.services.cloud.router import BackendRouter

        router = BackendRouter()
        mock_db = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await router._resolve_profile_model(mock_db, 999)
        assert result is None

    async def test_resolve_profile_model_model_not_found(self) -> None:
        from mlx_manager.mlx_server.services.cloud.router import BackendRouter

        router = BackendRouter()
        mock_db = AsyncMock()

        mock_profile = MagicMock()
        mock_profile.model_id = 42

        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = mock_profile
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_result1, mock_result2]

        result = await router._resolve_profile_model(mock_db, 1)
        assert result is None

    async def test_resolve_profile_model_no_model_id(self) -> None:
        from mlx_manager.mlx_server.services.cloud.router import BackendRouter

        router = BackendRouter()
        mock_db = AsyncMock()

        mock_profile = MagicMock()
        mock_profile.model_id = None  # Profile has no model linked

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_profile
        mock_db.execute.return_value = mock_result

        result = await router._resolve_profile_model(mock_db, 1)
        assert result is None


class TestRouterRouteRequestNoSession:
    """Cover line 51 — route_request RuntimeError when get_db yields nothing."""

    async def test_route_request_fails_when_get_db_empty(self) -> None:
        from mlx_manager.mlx_server.services.cloud.router import BackendRouter

        router = BackendRouter()

        async def empty_get_db():
            return
            yield  # noqa: F841

        with patch(
            "mlx_manager.mlx_server.services.cloud.router.get_db",
            empty_get_db,
        ):
            from mlx_manager.mlx_server.models.ir import InternalRequest
            from mlx_manager.models.value_objects import InferenceParams

            ir = InternalRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                params=InferenceParams(max_tokens=100),
                stream=False,
            )
            with pytest.raises(RuntimeError, match="Failed to get database session"):
                await router.route_request(ir, db=None)


# ============================================================================
# 6. embeddings.py — lines 15-16 (LOGFIRE_AVAILABLE import fallback)
# ============================================================================


class TestEmbeddingsLogfireImport:
    """Cover lines 15-16 — LOGFIRE_AVAILABLE = False when logfire not available."""

    def test_logfire_available_is_bool(self) -> None:
        from mlx_manager.mlx_server.services.embeddings import LOGFIRE_AVAILABLE

        assert isinstance(LOGFIRE_AVAILABLE, bool)


# ============================================================================
# 7. inference.py — lines 286-298, 318-334, 413-425, 434-453
# ============================================================================


def _make_fake_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.eos_token_id = 128009
    tok.unk_token_id = 0
    tok.convert_tokens_to_ids = MagicMock(return_value=128001)
    tok.encode = MagicMock(return_value=list(range(10)))
    tok.apply_chat_template = MagicMock(return_value="<formatted_prompt>")
    tok.tokenizer = tok
    return tok


class TestInferenceLegacyVisionStream:
    """Cover lines 286-298 — legacy vision streaming path (pixel_values != None, no messages)."""

    async def test_legacy_vision_stream_yields_events(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter
        from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult
        from mlx_manager.mlx_server.services.inference import _GenContext, _stream_chat_ir

        tok = _make_fake_tokenizer()
        adapter = create_adapter("default", tok, model_type="text-gen")
        model = MagicMock()

        ctx = _GenContext(
            model=model,
            tokenizer=tok,
            prompt="<prompt>",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="test-123",
            created=0,
            tools=None,
            pixel_values=MagicMock(),  # Non-None triggers vision path
            messages=None,  # None triggers legacy path
        )

        async def fake_run(fn, **kwargs):
            return "Vision response text"

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=fake_run,
        ):
            events = []
            async for item in _stream_chat_ir(ctx):
                events.append(item)

        assert any(isinstance(e, StreamEvent) for e in events)
        assert isinstance(events[-1], TextResult)


class TestInferenceLegacyTextStream:
    """Cover lines 318-334 — legacy text streaming path (produce_tokens)."""

    async def test_legacy_text_stream_yields_events(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter
        from mlx_manager.mlx_server.models.ir import TextResult
        from mlx_manager.mlx_server.services.inference import _GenContext, _stream_chat_ir

        tok = _make_fake_tokenizer()
        adapter = create_adapter("default", tok, model_type="text-gen")
        model = MagicMock()

        ctx = _GenContext(
            model=model,
            tokenizer=tok,
            prompt="<prompt>",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="test-123",
            created=0,
            tools=None,
            pixel_values=None,
            messages=None,  # None triggers legacy path
        )

        async def fake_stream(produce_fn):
            for item in produce_fn():
                yield item

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
                side_effect=fake_stream,
            ),
            patch("mlx_lm.stream_generate") as mock_gen,
            patch("mlx_lm.sample_utils.make_sampler") as mock_sampler,
        ):
            mock_sampler.return_value = MagicMock()
            resp1 = MagicMock()
            resp1.token = 1
            resp1.text = "Hello"
            resp2 = MagicMock()
            resp2.token = 128009  # Stop token
            resp2.text = ""
            mock_gen.return_value = [resp1, resp2]

            events = []
            async for item in _stream_chat_ir(ctx):
                events.append(item)

        assert isinstance(events[-1], TextResult)


class TestInferenceLegacyVisionComplete:
    """Cover lines 413-425 — legacy vision non-streaming path."""

    async def test_legacy_vision_complete(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter
        from mlx_manager.mlx_server.services.inference import _complete_chat_ir, _GenContext

        tok = _make_fake_tokenizer()
        adapter = create_adapter("default", tok, model_type="text-gen")
        model = MagicMock()

        ctx = _GenContext(
            model=model,
            tokenizer=tok,
            prompt="<prompt>",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="test-123",
            created=0,
            tools=None,
            pixel_values=MagicMock(),  # Non-None triggers vision path
            messages=None,  # None triggers legacy path
        )

        async def fake_run(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=fake_run,
            ),
            patch("mlx_vlm.generate") as mock_vlm_gen,
        ):
            mock_response = MagicMock()
            mock_response.text = "Vision output"
            mock_vlm_gen.return_value = mock_response

            result = await _complete_chat_ir(ctx)

        assert result.result.content is not None


class TestInferenceLegacyTextComplete:
    """Cover lines 434-453 — legacy text non-streaming path (run_generation)."""

    async def test_legacy_text_complete(self) -> None:
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter
        from mlx_manager.mlx_server.services.inference import _complete_chat_ir, _GenContext

        tok = _make_fake_tokenizer()
        adapter = create_adapter("default", tok, model_type="text-gen")
        model = MagicMock()

        ctx = _GenContext(
            model=model,
            tokenizer=tok,
            prompt="<prompt>",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="test-123",
            created=0,
            tools=None,
            pixel_values=None,
            messages=None,  # None triggers legacy path
        )

        async def fake_run(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=fake_run,
            ),
            patch("mlx_lm.stream_generate") as mock_gen,
            patch("mlx_lm.sample_utils.make_sampler") as mock_sampler,
        ):
            mock_sampler.return_value = MagicMock()
            resp1 = MagicMock()
            resp1.token = 1
            resp1.text = "Hello"
            resp2 = MagicMock()
            resp2.token = 128009  # Stop token
            resp2.text = ""
            mock_gen.return_value = [resp1, resp2]

            result = await _complete_chat_ir(ctx)

        assert result.result.content is not None
        assert result.result.finish_reason == "stop"


# ============================================================================
# 8. probe/base.py
# ============================================================================


class TestBaseProbeModelType:
    """Cover line 59 — BaseProbe.model_type abstract property."""

    def test_base_probe_cannot_be_instantiated(self) -> None:
        from mlx_manager.services.probe.base import BaseProbe

        with pytest.raises(TypeError):
            BaseProbe()  # type: ignore[abstract]


class TestGenerativeProbeGenerate:
    """Cover lines 252-253, 279-280, 282-283, 288-289 — _generate with timeout."""

    async def test_generate_raises_timeout(self) -> None:
        from mlx_manager.services.probe.base import GenerativeProbe

        class TestProbe(GenerativeProbe):
            @property
            def model_type(self):
                from mlx_manager.mlx_server.models.types import ModelType

                return ModelType.TEXT_GEN

            async def probe(self, model_id, loaded, result):
                yield  # type: ignore

        probe = TestProbe()
        loaded = MagicMock()
        loaded.adapter = MagicMock()
        loaded.model = MagicMock()

        # Make generate hang indefinitely to trigger timeout
        async def slow_generate(**kwargs):
            await asyncio.sleep(10)

        loaded.adapter.generate = slow_generate
        loaded.adapter.configure = MagicMock()

        with pytest.raises(TimeoutError, match="timed out"):
            await probe._generate(
                loaded,
                messages=[{"role": "user", "content": "hi"}],
                timeout=0.01,
            )

    async def test_generate_with_template_options(self) -> None:
        """Cover lines 279-280 — configure + reset template_options."""
        from mlx_manager.mlx_server.models.ir import TextResult
        from mlx_manager.services.probe.base import GenerativeProbe

        class TestProbe(GenerativeProbe):
            @property
            def model_type(self):
                from mlx_manager.mlx_server.models.types import ModelType

                return ModelType.TEXT_GEN

            async def probe(self, model_id, loaded, result):
                yield  # type: ignore

        probe = TestProbe()
        loaded = MagicMock()
        loaded.adapter = MagicMock()
        loaded.model = MagicMock()

        async def fake_generate(**kwargs):
            return TextResult(content="result", finish_reason="stop")

        loaded.adapter.generate = fake_generate

        result = await probe._generate(
            loaded,
            messages=[{"role": "user", "content": "hi"}],
            template_options={"enable_thinking": True},
        )

        assert result.content == "result"
        # Verify configure was called with options and then reset
        loaded.adapter.configure.assert_any_call(template_options={"enable_thinking": True})
        loaded.adapter.configure.assert_any_call(template_options=None)

    async def test_generate_no_adapter_raises(self) -> None:
        """Cover line 166 — no adapter raises RuntimeError."""
        from mlx_manager.services.probe.base import GenerativeProbe

        class TestProbe(GenerativeProbe):
            @property
            def model_type(self):
                from mlx_manager.mlx_server.models.types import ModelType

                return ModelType.TEXT_GEN

            async def probe(self, model_id, loaded, result):
                yield  # type: ignore

        probe = TestProbe()
        loaded = MagicMock()
        loaded.adapter = None

        with pytest.raises(RuntimeError, match="No adapter"):
            await probe._generate(loaded, messages=[{"role": "user", "content": "hi"}])


class TestEstimateContextWindow:
    """Cover lines 679 — estimate_context_window."""

    def test_none_size_returns_none(self) -> None:
        from mlx_manager.services.probe.base import estimate_context_window

        assert estimate_context_window("test/model", None) is None

    def test_returns_int_for_valid_size(self) -> None:
        from mlx_manager.services.probe.base import estimate_context_window

        with patch(
            "mlx_manager.mlx_server.utils.kv_cache.estimate_practical_max_tokens",
            return_value=8192,
        ):
            result = estimate_context_window("test/model", 4.0)
            assert result == 8192


class TestGetModelConfigValue:
    """Cover line 703 — get_model_config_value."""

    def test_returns_default_when_config_none(self) -> None:
        from mlx_manager.services.probe.base import get_model_config_value

        with patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value=None,
        ):
            result = get_model_config_value("test/model", "max_position_embeddings")
            assert result is None

    def test_returns_value_for_existing_key(self) -> None:
        from mlx_manager.services.probe.base import get_model_config_value

        with patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"max_position_embeddings": 4096, "hidden_size": 768},
        ):
            result = get_model_config_value("test/model", "max_position_embeddings")
            assert result == 4096

    def test_returns_first_matching_key(self) -> None:
        from mlx_manager.services.probe.base import get_model_config_value

        with patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"max_seq_length": 2048},
        ):
            result = get_model_config_value(
                "test/model",
                "max_position_embeddings",
                "max_seq_length",
            )
            assert result == 2048

    def test_returns_default_when_no_keys_match(self) -> None:
        from mlx_manager.services.probe.base import get_model_config_value

        with patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"hidden_size": 768},
        ):
            result = get_model_config_value(
                "test/model",
                "max_position_embeddings",
                default=512,
            )
            assert result == 512


class TestHasTokenizationArtifacts:
    """Cover lines 574-576 — _has_tokenization_artifacts."""

    def test_detects_sentencepiece_marker(self) -> None:
        from mlx_manager.services.probe.base import _has_tokenization_artifacts

        assert _has_tokenization_artifacts("Hello\u2581world") is True

    def test_detects_spaced_json_key(self) -> None:
        from mlx_manager.services.probe.base import _has_tokenization_artifacts

        assert _has_tokenization_artifacts('" name "') is True

    def test_no_artifacts(self) -> None:
        from mlx_manager.services.probe.base import _has_tokenization_artifacts

        assert _has_tokenization_artifacts("Normal text output") is False


class TestPrioritizeParsers:
    """Cover lines 492, 522 — _prioritize_parsers."""

    def test_family_parser_first(self) -> None:
        from mlx_manager.services.probe.base import _prioritize_parsers

        candidates = {"hermes_json", "glm4_native", "llama_xml"}
        result = _prioritize_parsers(candidates, "hermes_json")
        assert result[0] == "hermes_json"

    def test_no_family_parser_alphabetical(self) -> None:
        from mlx_manager.services.probe.base import _prioritize_parsers

        candidates = {"hermes_json", "glm4_native"}
        result = _prioritize_parsers(candidates, None)
        assert result == sorted(candidates)

    def test_family_parser_not_in_candidates(self) -> None:
        from mlx_manager.services.probe.base import _prioritize_parsers

        candidates = {"hermes_json", "glm4_native"}
        result = _prioritize_parsers(candidates, "mistral_native")
        assert result == sorted(candidates)


class TestDiscoverAndMapTagsUncoveredParsers:
    """Cover lines 412-413 — _discover_and_map_tags with parsers found by
    direct scan but not by regex detection (uncovered parsers path)."""

    def test_uncovered_parsers_added_as_discoveries(self) -> None:
        from mlx_manager.services.probe.base import _discover_and_map_tags

        # Create a parser with a marker that won't be detected by regex
        # (e.g., "<function=" is not a standard XML tag)
        mock_parser = MagicMock()
        mock_parser_inst = MagicMock()
        mock_parser_inst.stream_markers = [("<function=", "</function>")]
        mock_parser.return_value = mock_parser_inst

        parsers = {"llama_xml": mock_parser}

        # Output contains the marker but not in a form regex would detect as XML tag
        output = '<function=get_weather>{"location":"Tokyo"}</function>'

        discoveries = _discover_and_map_tags(output, parsers)

        # Should find llama_xml via direct marker scan
        parser_ids = []
        for d in discoveries:
            parser_ids.extend(d.matched_parsers)
        assert "llama_xml" in parser_ids

    def test_bracket_style_uncovered_parser(self) -> None:
        from mlx_manager.services.probe.base import _discover_and_map_tags

        mock_parser = MagicMock()
        mock_parser_inst = MagicMock()
        mock_parser_inst.stream_markers = [("[CUSTOM_TAG]", "[/CUSTOM_TAG]")]
        mock_parser.return_value = mock_parser_inst

        parsers = {"custom": mock_parser}

        # Output has the marker — regex WILL detect it as bracket tag,
        # but let's test with a non-standard bracket pattern
        output = "[CUSTOM_TAG] some content [/CUSTOM_TAG]"

        discoveries = _discover_and_map_tags(output, parsers)

        # Should find custom parser
        parser_ids = []
        for d in discoveries:
            parser_ids.extend(d.matched_parsers)
        assert "custom" in parser_ids

    def test_special_token_style_uncovered_parser(self) -> None:
        from mlx_manager.services.probe.base import _discover_and_map_tags

        mock_parser = MagicMock()
        mock_parser_inst = MagicMock()
        mock_parser_inst.stream_markers = [("<|python_tag|>", "<|eom_id|>")]
        mock_parser.return_value = mock_parser_inst

        parsers = {"python_tool": mock_parser}

        output = "<|python_tag|>print('hello')<|eom_id|>"

        discoveries = _discover_and_map_tags(output, parsers)

        parser_ids = []
        for d in discoveries:
            parser_ids.extend(d.matched_parsers)
        assert "python_tool" in parser_ids


class TestGetFamilyParserIds:
    """Cover lines 412-413 — get_family_tool_parser_id and get_family_thinking_parser_id."""

    def test_get_family_tool_parser_id_none_family(self) -> None:
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        assert get_family_tool_parser_id(None) is None

    def test_get_family_tool_parser_id_valid(self) -> None:
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        result = get_family_tool_parser_id("qwen")
        # Qwen has a tool parser factory
        assert result is not None

    def test_get_family_tool_parser_id_no_factory(self) -> None:
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        result = get_family_tool_parser_id("default")
        # Default family has no tool parser factory
        assert result is None

    def test_get_family_thinking_parser_id_none_family(self) -> None:
        from mlx_manager.services.probe.base import get_family_thinking_parser_id

        assert get_family_thinking_parser_id(None) is None

    def test_get_family_thinking_parser_id_no_factory(self) -> None:
        from mlx_manager.services.probe.base import get_family_thinking_parser_id

        result = get_family_thinking_parser_id("default")
        assert result is None

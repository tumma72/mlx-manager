"""Tests for Pydantic field-level body size limits at the API layer.

Covers Gap 4: verifying that Pydantic Field(max_length=N) constraints on
ChatCompletionRequest fields are enforced at the API boundary and produce
422 responses with meaningful error details.

Fields tested:
- messages: Field(..., max_length=1024)        → more than 1024 messages → 422
- tools: Field(default=None, max_length=256)   → more than 256 tools → 422
- stop (via model_validator, max 16 entries)   → more than 16 stop sequences → 422
"""

import pytest

from mlx_manager.mlx_server.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDefinition,
    Tool,
)

# ---------------------------------------------------------------------------
# Direct Pydantic validation tests (no HTTP layer required)
# ---------------------------------------------------------------------------


class TestMessageCountLimit:
    """messages field has max_length=1024."""

    def test_exactly_1024_messages_is_valid(self):
        """1024 messages is at the limit and must be accepted."""
        messages = [ChatMessage(role="user", content=f"Message {i}") for i in range(1024)]
        req = ChatCompletionRequest(model="test-model", messages=messages)
        assert len(req.messages) == 1024

    def test_1025_messages_raises_validation_error(self):
        """1025 messages exceeds max_length=1024 and must raise a validation error."""
        from pydantic import ValidationError

        messages = [ChatMessage(role="user", content=f"Message {i}") for i in range(1025)]
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(model="test-model", messages=messages)

        errors = exc_info.value.errors()
        # At least one error should reference messages or list_too_long
        error_types = {e["type"] for e in errors}
        assert "too_long" in error_types or any(
            "message" in str(e).lower()
            or "max_length" in str(e).lower()
            or "too_long" in str(e).lower()
            for e in errors
        )

    def test_0_messages_is_accepted_by_pydantic(self):
        """Empty messages list is accepted by Pydantic (no min_length constraint).

        The business logic layer validates message count, but the schema itself
        only enforces max_length=1024, not a minimum.
        """
        req = ChatCompletionRequest(model="test-model", messages=[])
        assert len(req.messages) == 0

    def test_1_message_is_valid(self):
        """Single message is valid."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert len(req.messages) == 1


class TestToolCountLimit:
    """tools field has max_length=256."""

    def _make_tool(self, i: int) -> Tool:
        return Tool(
            function=FunctionDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
            )
        )

    def test_exactly_256_tools_is_valid(self):
        """256 tools is at the limit and must be accepted."""
        tools = [self._make_tool(i) for i in range(256)]
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            tools=tools,
        )
        assert len(req.tools) == 256

    def test_257_tools_raises_validation_error(self):
        """257 tools exceeds max_length=256 and must raise a validation error."""
        from pydantic import ValidationError

        tools = [self._make_tool(i) for i in range(257)]
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hello")],
                tools=tools,
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "too_long" in error_types or any(
            "too_long" in str(e).lower() or "max_length" in str(e).lower() for e in errors
        )

    def test_no_tools_is_valid(self):
        """tools=None (default) is valid."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            tools=None,
        )
        assert req.tools is None

    def test_1_tool_is_valid(self):
        """Single tool is valid."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            tools=[self._make_tool(0)],
        )
        assert len(req.tools) == 1


class TestStopSequenceLimit:
    """stop field (list variant) allows at most 16 entries via model_validator."""

    def test_exactly_16_stop_sequences_is_valid(self):
        """16 stop sequences is at the limit and must be accepted."""
        stop = [f"stop_{i}" for i in range(16)]
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stop=stop,
        )
        assert len(req.stop) == 16

    def test_17_stop_sequences_raises_validation_error(self):
        """17 stop sequences exceeds the limit and must raise a validation error."""
        from pydantic import ValidationError

        stop = [f"stop_{i}" for i in range(17)]
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hello")],
                stop=stop,
            )

        # The model_validator raises ValueError("stop may contain at most 16 entries")
        errors = exc_info.value.errors()
        assert any("16" in str(e) or "stop" in str(e).lower() for e in errors)

    def test_string_stop_is_valid(self):
        """Single string stop sequence is always valid."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stop="<|endoftext|>",
        )
        assert req.stop == "<|endoftext|>"

    def test_none_stop_is_valid(self):
        """stop=None (default) is valid."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.stop is None

    def test_1_stop_sequence_is_valid(self):
        """Single-element stop list is valid."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stop=["<|end|>"],
        )
        assert req.stop == ["<|end|>"]


# ---------------------------------------------------------------------------
# API-level tests via HTTP (verify 422 comes back from the endpoint)
# ---------------------------------------------------------------------------


class TestBodyLimitsViaAPI:
    """Pydantic limits translate to 422 at the HTTP layer."""

    @pytest.fixture()
    def client(self):
        """FastAPI TestClient with embedded app (no real model pool needed)."""
        from fastapi.testclient import TestClient

        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)
        return TestClient(app, raise_server_exceptions=False)

    def test_too_many_messages_returns_422(self, client):
        """POST /chat/completions with 1025 messages returns 422."""
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(1025)]
        response = client.post(
            "/chat/completions",
            json={"model": "test-model", "messages": messages},
        )
        assert response.status_code == 422

    def test_too_many_tools_returns_422(self, client):
        """POST /chat/completions with 257 tools returns 422."""
        tools = [
            {"type": "function", "function": {"name": f"fn_{i}", "description": f"fn {i}"}}
            for i in range(257)
        ]
        response = client.post(
            "/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": tools,
            },
        )
        assert response.status_code == 422

    def test_too_many_stop_sequences_returns_422(self, client):
        """POST /chat/completions with 17 stop sequences returns 422."""
        stop = [f"stop_{i}" for i in range(17)]
        response = client.post(
            "/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stop": stop,
            },
        )
        assert response.status_code == 422

    def test_valid_request_passes_validation(self, client):
        """A request within limits passes Pydantic validation (may fail for other reasons)."""
        response = client.post(
            "/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stop": ["<|end|>"],
            },
        )
        # 422 = Pydantic rejection (bad); anything else means validation passed
        assert response.status_code != 422

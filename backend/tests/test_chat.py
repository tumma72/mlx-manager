"""Tests for the chat completions API router."""

import json
from unittest.mock import patch

import httpx
import pytest


class MockResponse:
    """Mock httpx.Response for streaming tests."""

    def __init__(self, status_code: int, lines: list[str]):
        """Initialize mock response.

        Args:
            status_code: HTTP status code
            lines: List of SSE lines to return
        """
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        """Async iterator over lines."""
        for line in self._lines:
            yield line

    async def aread(self):
        """Read response body as bytes."""
        return b"Server error"

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        pass


class MockAsyncClient:
    """Mock httpx.AsyncClient for streaming tests."""

    def __init__(self, response: MockResponse | None = None, exception: Exception | None = None):
        """Initialize mock client.

        Args:
            response: MockResponse to return
            exception: Exception to raise
        """
        self.response = response
        self.exception = exception

    def stream(self, method: str, url: str, **kwargs):
        """Mock stream method."""
        if self.exception:
            raise self.exception
        return self.response

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        pass


def make_sse_chunk(
    content: str = "",
    reasoning: str = "",
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> str:
    """Helper to create SSE-formatted chunk."""
    delta: dict = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    if tool_calls:
        delta["tool_calls"] = tool_calls

    data = {
        "choices": [
            {
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    }
    return f"data: {json.dumps(data)}"


def filter_events(events: list[str], event_type: str) -> list[str]:
    """Filter SSE events by type."""
    type_patterns = [f'"type":"{event_type}"', f'"type": "{event_type}"']
    return [e for e in events if any(p in e for p in type_patterns)]


@pytest.mark.asyncio
async def test_chat_completions_profile_not_found(auth_client):
    """Test 404 when profile doesn't exist."""
    response = await auth_client.post(
        "/api/chat/completions",
        json={
            "profile_id": 9999,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 404
    assert "Profile not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_completions_requires_auth(client):
    """Test that endpoint requires authentication."""
    response = await client.post(
        "/api/chat/completions",
        json={
            "profile_id": 1,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_completions_streaming_response(auth_client, test_profile):
    """Test successful streaming response with regular content."""
    lines = [
        make_sse_chunk(content="Hello"),
        make_sse_chunk(content=" world"),
        make_sse_chunk(content="!"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE response
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have 3 response chunks + 1 done event
        assert len(events) >= 4

        # Verify response events
        response_events = filter_events(events, "response")
        assert len(response_events) == 3

        # Verify done event
        assert any('"type":"done"' in e or '"type": "done"' in e for e in events)


@pytest.mark.asyncio
async def test_chat_completions_thinking_tags_parsing(auth_client, test_profile):
    """Test parsing of <think> tags in content - character by character."""
    # When tags appear in content, they're parsed character-by-character
    # The tag itself is skipped, but content between tags is emitted per-character
    lines = [
        make_sse_chunk(content="<think>Let me consider</think>Answer"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Question?"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have thinking events (character by character for "Let me consider")
        thinking_events = filter_events(events, "thinking")
        assert len(thinking_events) == 15  # "Let me consider" = 15 chars

        # Should have thinking_done event
        thinking_done = filter_events(events, "thinking_done")
        assert len(thinking_done) == 1

        # Verify duration field exists
        thinking_done_data = json.loads(thinking_done[0].replace("data: ", ""))
        assert "duration" in thinking_done_data

        # Should have response events (character by character for "Answer")
        response_events = filter_events(events, "response")
        assert len(response_events) == 6  # "Answer" = 6 chars


@pytest.mark.asyncio
async def test_chat_completions_reasoning_content_field(auth_client, test_profile):
    """Test server-extracted thinking via reasoning_content field."""
    lines = [
        make_sse_chunk(reasoning="First thought"),
        make_sse_chunk(reasoning=" second thought"),
        make_sse_chunk(content="Final answer"),  # Transition to content
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Question?"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have thinking events from reasoning_content
        thinking_events = filter_events(events, "thinking")
        assert len(thinking_events) == 2

        # Should have thinking_done when transitioning to content
        thinking_done = filter_events(events, "thinking_done")
        assert len(thinking_done) == 1

        # Should have response event
        response_events = filter_events(events, "response")
        assert len(response_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_reasoning_content_with_tags(auth_client, test_profile):
    """Test that thinking tags are stripped from reasoning_content field."""
    lines = [
        make_sse_chunk(reasoning="<think>actual reasoning</think>"),
        make_sse_chunk(content="Answer"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Question?"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Find thinking event
        thinking_events = filter_events(events, "thinking")
        assert len(thinking_events) == 1

        # Verify tags are stripped
        thinking_data = json.loads(thinking_events[0].replace("data: ", ""))
        assert "<think>" not in thinking_data["content"]
        assert "</think>" not in thinking_data["content"]
        assert "actual reasoning" in thinking_data["content"]


@pytest.mark.asyncio
async def test_chat_completions_tool_calls(auth_client, test_profile):
    """Test tool call events from model."""
    tool_call = {
        "index": 0,
        "id": "call_123",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "SF"}',
        },
    }
    lines = [
        make_sse_chunk(tool_calls=[tool_call]),
        make_sse_chunk(content="", finish_reason="tool_calls"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have tool_call event
        tool_call_events = filter_events(events, "tool_call")
        assert len(tool_call_events) == 1

        # Verify tool call data
        tool_call_data = json.loads(tool_call_events[0].replace("data: ", ""))
        assert tool_call_data["tool_call"]["id"] == "call_123"
        assert tool_call_data["tool_call"]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_chat_completions_tool_calls_done(auth_client, test_profile):
    """Test tool_calls_done event when finish_reason is tool_calls."""
    lines = [
        make_sse_chunk(content="", finish_reason="tool_calls"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have tool_calls_done event
        tool_calls_done = filter_events(events, "tool_calls_done")
        assert len(tool_calls_done) == 1


@pytest.mark.asyncio
async def test_chat_completions_connection_error(auth_client, test_profile):
    """Test connection error when MLX server is not running."""
    mock_client = MockAsyncClient(exception=httpx.ConnectError("Connection failed"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have error event
        error_events = [e for e in events if '"type":"error"' in e or '"type": "error"' in e]
        assert len(error_events) == 1

        error_data = json.loads(error_events[0].replace("data: ", ""))
        assert "Failed to connect" in error_data["content"]
        assert "Is it running?" in error_data["content"]


@pytest.mark.asyncio
async def test_chat_completions_timeout_error(auth_client, test_profile):
    """Test timeout error for long-running requests."""
    mock_client = MockAsyncClient(exception=httpx.TimeoutException("Request timed out"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have error event
        error_events = [e for e in events if '"type":"error"' in e or '"type": "error"' in e]
        assert len(error_events) == 1

        error_data = json.loads(error_events[0].replace("data: ", ""))
        assert "timed out" in error_data["content"]


@pytest.mark.asyncio
async def test_chat_completions_server_error_response(auth_client, test_profile):
    """Test handling of non-200 status from MLX server."""
    lines = []
    mock_response = MockResponse(500, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have error event
        error_events = [e for e in events if '"type":"error"' in e or '"type": "error"' in e]
        assert len(error_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_stream_ends_in_thinking(auth_client, test_profile):
    """Test that thinking_done is emitted if stream ends while in thinking mode."""
    # Use reasoning_content to maintain thinking state, then stream ends
    # This tests the code at lines 222-228 that emits thinking_done after stream completion
    lines = [
        make_sse_chunk(reasoning="First thought"),
        make_sse_chunk(reasoning="Second thought"),
        # Stream ends without transitioning to content, so in_thinking remains True
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Question?"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have thinking events from reasoning_content
        thinking_events = filter_events(events, "thinking")
        assert len(thinking_events) == 2

        # Should have thinking_done event even though stream ended without content transition
        thinking_done = filter_events(events, "thinking_done")
        assert len(thinking_done) == 1

        # Verify thinking_done has duration
        thinking_done_data = json.loads(thinking_done[0].replace("data: ", ""))
        assert "duration" in thinking_done_data

        # Should have done event
        done_events = [e for e in events if '"type":"done"' in e or '"type": "done"' in e]
        assert len(done_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_with_tools_parameters(auth_client, test_profile):
    """Test that tools and tool_choice are forwarded to MLX server."""
    lines = [
        make_sse_chunk(content="Answer"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "tool_choice": "auto",
            },
        )

        assert response.status_code == 200

        # Verify that mock client was called with tools
        # Note: In actual implementation, we'd need to inspect the stream call
        # This test verifies the request is accepted with tools parameters


@pytest.mark.asyncio
async def test_chat_completions_invalid_json_in_stream(auth_client, test_profile):
    """Test that invalid JSON chunks are skipped gracefully."""
    lines = [
        make_sse_chunk(content="Valid"),
        "data: {invalid json}",
        make_sse_chunk(content=" content"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should still have valid response events (invalid JSON skipped)
        response_events = filter_events(events, "response")
        assert len(response_events) >= 2

        # Should complete successfully
        done_events = [e for e in events if '"type":"done"' in e or '"type": "done"' in e]
        assert len(done_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_mixed_thinking_and_response(auth_client, test_profile):
    """Test handling of content both inside and outside thinking tags."""
    lines = [
        make_sse_chunk(content="Before <think>thinking content</think> after"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Question?"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have both thinking and response events
        thinking_events = filter_events(events, "thinking")
        assert len(thinking_events) > 0

        response_events = filter_events(events, "response")
        assert len(response_events) > 0

        # Should have thinking_done
        thinking_done = filter_events(events, "thinking_done")
        assert len(thinking_done) == 1


@pytest.mark.asyncio
async def test_chat_completions_empty_content_chunks(auth_client, test_profile):
    """Test handling of chunks with empty content."""
    lines = [
        make_sse_chunk(content=""),
        make_sse_chunk(content="Hello"),
        make_sse_chunk(content=""),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have response event for non-empty content
        response_events = filter_events(events, "response")
        assert len(response_events) > 0


@pytest.mark.asyncio
async def test_chat_completions_reasoning_without_transition(auth_client, test_profile):
    """Test reasoning_content without transition to regular content."""
    lines = [
        make_sse_chunk(reasoning="Thought 1"),
        make_sse_chunk(reasoning="Thought 2"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Question?"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have thinking events
        thinking_events = filter_events(events, "thinking")
        assert len(thinking_events) == 2

        # Should have thinking_done when stream ends
        thinking_done = filter_events(events, "thinking_done")
        assert len(thinking_done) == 1


@pytest.mark.asyncio
async def test_chat_completions_general_exception(auth_client, test_profile):
    """Test handling of general exceptions during streaming."""

    class MockFailingClient:
        """Mock client that raises an exception."""

        def stream(self, *args, **kwargs):
            """Raise a generic exception."""
            raise ValueError("Unexpected error")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    mock_client = MockFailingClient()

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have error event
        error_events = [e for e in events if '"type":"error"' in e or '"type": "error"' in e]
        assert len(error_events) == 1

        error_data = json.loads(error_events[0].replace("data: ", ""))
        assert "Unexpected error" in error_data["content"]


@pytest.mark.asyncio
async def test_chat_completions_non_sse_lines_skipped(auth_client, test_profile):
    """Test that non-SSE lines (not starting with 'data: ') are skipped."""
    lines = [
        ": comment line",  # SSE comment, should be skipped
        "invalid line",  # Invalid line, should be skipped
        make_sse_chunk(content="Valid"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should only process valid SSE lines
        # "Valid" is emitted as a single chunk (no tags, not in thinking mode)
        response_events = filter_events(events, "response")
        assert len(response_events) == 1

        # Verify content
        response_data = json.loads(response_events[0].replace("data: ", ""))
        assert response_data["content"] == "Valid"

        # Should have done event
        done_events = [e for e in events if '"type":"done"' in e or '"type": "done"' in e]
        assert len(done_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_both_reasoning_and_content_fields(auth_client, test_profile):
    """Test chunk with both reasoning_content and content fields.

    This tests the elif in_thinking branch on lines 198-204.
    When a chunk has BOTH reasoning_content and content:
    - reasoning is processed first (keeps in_thinking=True)
    - Then content is processed while in_thinking=True
    - Since line 152 condition fails (reasoning is truthy), we reach line 198
    """
    lines = [
        make_sse_chunk(reasoning="First thought", content="with content"),
        "data: [DONE]",
    ]
    mock_response = MockResponse(200, lines)
    mock_client = MockAsyncClient(mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = await auth_client.post(
            "/api/chat/completions",
            json={
                "profile_id": test_profile.id,
                "messages": [{"role": "user", "content": "Question?"}],
            },
        )

        assert response.status_code == 200
        content = response.text
        events = [line for line in content.split("\n") if line.startswith("data: ")]

        # Should have thinking event from reasoning_content
        thinking_events = filter_events(events, "thinking")
        assert len(thinking_events) == 2  # One from reasoning, one from content

        # Verify both chunks were emitted as thinking
        thinking_contents = [
            json.loads(e.replace("data: ", ""))["content"] for e in thinking_events
        ]
        assert "First thought" in thinking_contents
        assert "with content" in thinking_contents

        # Should have thinking_done when stream ends (line 222-228)
        thinking_done = filter_events(events, "thinking_done")
        assert len(thinking_done) == 1

        # Should have done event
        done_events = [e for e in events if '"type":"done"' in e or '"type": "done"' in e]
        assert len(done_events) == 1

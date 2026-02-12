"""Tests for the chat completions API router.

With the embedded MLX Server, the chat router calls generate_chat_completion()
and generate_vision_completion() directly instead of proxying to an external
server process.

Note: The inference service now handles thinking tag extraction and returns
OpenAI-compatible deltas with reasoning_content. The chat router forwards
these structured events without character-by-character parsing.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest


async def make_async_generator(chunks):
    """Helper to create an async generator from a list of chunks."""
    for chunk in chunks:
        yield chunk


def make_chunk(
    content: str = "",
    reasoning: str = "",
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> dict:
    """Helper to create an inference chunk dict."""
    delta: dict = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    if tool_calls:
        delta["tool_calls"] = tool_calls

    return {
        "choices": [
            {
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    }


def filter_events(events: list[str], event_type: str) -> list[str]:
    """Filter SSE events by type."""
    type_patterns = [f'"type":"{event_type}"', f'"type": "{event_type}"']
    return [e for e in events if any(p in e for p in type_patterns)]


def create_mock_inference(chunks):
    """Create an async mock that returns an async generator when awaited.

    This mimics how generate_chat_completion works: it's an async function
    that when called with stream=True and awaited, returns an async generator.
    The key is that async def functions return coroutines that must be awaited.
    """

    async def mock_gen():
        for chunk in chunks:
            yield chunk

    async def mock_coro(*args, **kwargs):
        # Return the async generator (not await it)
        return mock_gen()

    # Use AsyncMock so that mock() returns a coroutine that can be awaited
    return AsyncMock(side_effect=mock_coro)


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
async def test_chat_completions_streaming_response(auth_client, api_test_profile):
    """Test successful streaming response with regular content.

    The chat router forwards content token-by-token from the inference service.
    Each chunk from inference becomes one response event.
    """
    chunks = [
        make_chunk(content="Hello"),
        make_chunk(content=" world"),
        make_chunk(content="!"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            # Parse SSE response
            content = response.text
            events = [line for line in content.split("\n") if line.startswith("data: ")]

            # Verify we have response events (one per chunk from inference)
            response_events = filter_events(events, "response")
            assert len(response_events) == 3  # One per chunk

            # Verify combined content matches expected
            combined_content = ""
            for event in response_events:
                data = json.loads(event.replace("data: ", ""))
                combined_content += data.get("content", "")
            assert combined_content == "Hello world!"

            # Verify done event
            assert any('"type":"done"' in e or '"type": "done"' in e for e in events)


@pytest.mark.asyncio
async def test_chat_completions_thinking_via_reasoning_content(auth_client, api_test_profile):
    """Test thinking content via reasoning_content field from inference.

    The inference service extracts thinking content and sends it in the
    reasoning_content field. Chat router forwards this as thinking events.
    """
    chunks = [
        make_chunk(reasoning="Let me consider"),
        make_chunk(content="Answer"),  # Transition to content
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
                    "messages": [{"role": "user", "content": "Question?"}],
                },
            )

            assert response.status_code == 200
            content = response.text
            events = [line for line in content.split("\n") if line.startswith("data: ")]

            # Should have thinking event from reasoning_content
            thinking_events = filter_events(events, "thinking")
            assert len(thinking_events) == 1

            # Verify thinking content
            thinking_data = json.loads(thinking_events[0].replace("data: ", ""))
            assert thinking_data["content"] == "Let me consider"

            # Should have thinking_done event (transition to content)
            thinking_done = filter_events(events, "thinking_done")
            assert len(thinking_done) == 1

            # Verify duration field exists
            thinking_done_data = json.loads(thinking_done[0].replace("data: ", ""))
            assert "duration" in thinking_done_data

            # Should have response event for "Answer"
            response_events = filter_events(events, "response")
            assert len(response_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_reasoning_content_field(auth_client, api_test_profile):
    """Test server-extracted thinking via reasoning_content field."""
    chunks = [
        make_chunk(reasoning="First thought"),
        make_chunk(reasoning=" second thought"),
        make_chunk(content="Final answer"),  # Transition to content
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
                    "messages": [{"role": "user", "content": "Question?"}],
                },
            )

            assert response.status_code == 200
            content = response.text
            events = [line for line in content.split("\n") if line.startswith("data: ")]

            # Should have thinking events from reasoning_content (one per chunk)
            thinking_events = filter_events(events, "thinking")
            assert len(thinking_events) == 2

            # Should have thinking_done when transitioning to content
            thinking_done = filter_events(events, "thinking_done")
            assert len(thinking_done) == 1

            # Should have response event (one per chunk)
            response_events = filter_events(events, "response")
            assert len(response_events) == 1

            # Verify content
            response_data = json.loads(response_events[0].replace("data: ", ""))
            assert response_data["content"] == "Final answer"


@pytest.mark.asyncio
async def test_chat_completions_reasoning_content_forwarded(auth_client, api_test_profile):
    """Test that reasoning_content is forwarded as-is to thinking events.

    The inference service now handles tag extraction, so chat router
    just forwards reasoning_content directly.
    """
    chunks = [
        make_chunk(reasoning="actual reasoning"),
        make_chunk(content="Answer"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
                    "messages": [{"role": "user", "content": "Question?"}],
                },
            )

            assert response.status_code == 200
            content = response.text
            events = [line for line in content.split("\n") if line.startswith("data: ")]

            # Find thinking event
            thinking_events = filter_events(events, "thinking")
            assert len(thinking_events) == 1

            # Verify reasoning content is forwarded
            thinking_data = json.loads(thinking_events[0].replace("data: ", ""))
            assert thinking_data["content"] == "actual reasoning"


@pytest.mark.asyncio
async def test_chat_completions_tool_calls(auth_client, api_test_profile):
    """Test tool call events from model."""
    tool_call = {
        "index": 0,
        "id": "call_123",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "SF"}',
        },
    }
    chunks = [
        make_chunk(tool_calls=[tool_call]),
        make_chunk(content="", finish_reason="tool_calls"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
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
async def test_chat_completions_tool_calls_done(auth_client, api_test_profile):
    """Test tool_calls_done event when finish_reason is tool_calls."""
    chunks = [
        make_chunk(content="", finish_reason="tool_calls"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
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
async def test_chat_completions_model_not_found(auth_client, api_test_profile):
    """Test FileNotFoundError when model is not downloaded."""

    async def mock_raise(*args, **kwargs):
        raise FileNotFoundError("Model not found: mlx-community/test-model")

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        AsyncMock(side_effect=mock_raise),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
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
            assert "Model not available" in error_data["content"]


@pytest.mark.asyncio
async def test_chat_completions_stream_ends_in_thinking(auth_client, api_test_profile):
    """Test that thinking_done is emitted if stream ends while in thinking mode."""
    # Use reasoning_content to maintain thinking state, then stream ends
    chunks = [
        make_chunk(reasoning="First thought"),
        make_chunk(reasoning="Second thought"),
        # Stream ends without transitioning to content, so in_thinking remains True
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
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
async def test_chat_completions_with_tools_parameters(auth_client, api_test_profile):
    """Test that tools and tool_choice are forwarded to inference service."""
    chunks = [
        make_chunk(content="Answer"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                    "tool_choice": "auto",
                },
            )

            assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_mixed_thinking_and_response(auth_client, api_test_profile):
    """Test handling of interleaved reasoning and content.

    The inference service handles tag extraction and sends separate events
    for reasoning_content and content.
    """
    chunks = [
        make_chunk(content="Before"),
        make_chunk(reasoning="thinking content"),
        make_chunk(content="after"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
                    "messages": [{"role": "user", "content": "Question?"}],
                },
            )

            assert response.status_code == 200
            content = response.text
            events = [line for line in content.split("\n") if line.startswith("data: ")]

            # Should have both thinking and response events
            thinking_events = filter_events(events, "thinking")
            assert len(thinking_events) == 1

            response_events = filter_events(events, "response")
            assert len(response_events) == 2  # "Before" and "after"

            # Should have thinking_done
            thinking_done = filter_events(events, "thinking_done")
            assert len(thinking_done) == 1


@pytest.mark.asyncio
async def test_chat_completions_empty_content_chunks(auth_client, api_test_profile):
    """Test handling of chunks with empty content."""
    chunks = [
        make_chunk(content=""),
        make_chunk(content="Hello"),
        make_chunk(content=""),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
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
async def test_chat_completions_reasoning_without_transition(auth_client, api_test_profile):
    """Test reasoning_content without transition to regular content."""
    chunks = [
        make_chunk(reasoning="Thought 1"),
        make_chunk(reasoning="Thought 2"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
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
async def test_chat_completions_general_exception(auth_client, api_test_profile):
    """Test handling of general exceptions during inference."""

    async def mock_raise(*args, **kwargs):
        raise ValueError("Unexpected error")

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        AsyncMock(side_effect=mock_raise),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
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
async def test_chat_completions_both_reasoning_and_content_fields(auth_client, api_test_profile):
    """Test chunk with both reasoning_content and content fields.

    When a chunk has BOTH reasoning_content and content:
    - reasoning is emitted as thinking event
    - content is emitted as response event
    - thinking_done is NOT emitted (no transition, simultaneous)
    """
    chunks = [
        make_chunk(reasoning="First thought", content="with content"),
    ]

    with patch(
        "mlx_manager.mlx_server.services.inference.generate_chat_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value="lm",
        ):
            response = await auth_client.post(
                "/api/chat/completions",
                json={
                    "profile_id": api_test_profile.id,
                    "messages": [{"role": "user", "content": "Question?"}],
                },
            )

            assert response.status_code == 200
            content = response.text
            events = [line for line in content.split("\n") if line.startswith("data: ")]

            # Should have thinking event from reasoning_content
            thinking_events = filter_events(events, "thinking")
            assert len(thinking_events) == 1

            # Verify reasoning content was emitted
            thinking_data = json.loads(thinking_events[0].replace("data: ", ""))
            assert thinking_data["content"] == "First thought"

            # Should have response event from content
            response_events = filter_events(events, "response")
            assert len(response_events) == 1

            response_data = json.loads(response_events[0].replace("data: ", ""))
            assert response_data["content"] == "with content"

            # Should have thinking_done when stream ends (still in thinking mode)
            thinking_done = filter_events(events, "thinking_done")
            assert len(thinking_done) == 1

            # Should have done event
            done_events = [e for e in events if '"type":"done"' in e or '"type": "done"' in e]
            assert len(done_events) == 1


@pytest.mark.asyncio
async def test_chat_completions_vision_model(auth_client, api_test_profile):
    """Test that vision models use generate_vision_completion."""
    from mlx_manager.mlx_server.models.types import ModelType

    chunks = [
        make_chunk(content="I see a cat in the image."),
    ]

    with patch(
        "mlx_manager.mlx_server.services.vision.generate_vision_completion",
        create_mock_inference(chunks),
    ):
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=ModelType.VISION,
        ):
            with patch(
                "mlx_manager.mlx_server.services.image_processor.preprocess_images",
                new_callable=AsyncMock,
                return_value=[],
            ):
                response = await auth_client.post(
                    "/api/chat/completions",
                    json={
                        "profile_id": api_test_profile.id,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What's in this image?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "data:image/png;base64,abc123"},
                                    },
                                ],
                            }
                        ],
                    },
                )

                assert response.status_code == 200
                content = response.text
                events = [line for line in content.split("\n") if line.startswith("data: ")]

                # Should have response event (one per chunk)
                response_events = filter_events(events, "response")
                assert len(response_events) == 1

                # Verify content
                response_data = json.loads(response_events[0].replace("data: ", ""))
                assert response_data["content"] == "I see a cat in the image."

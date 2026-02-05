"""End-to-end cross-protocol tests: OpenAI vs Anthropic API.

Tests send identical prompts through both:
- POST /v1/chat/completions (OpenAI format)
- POST /v1/messages (Anthropic format)

And verify that both:
1. Return valid, non-empty responses
2. Conform to their respective API specifications
3. Produce semantically equivalent content from the same model

Run:
  pytest -m e2e_anthropic -v
"""

import json
from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).parent.parent / "fixtures" / "golden" / "prompts"

WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'",
                }
            },
            "required": ["location"],
        },
    },
}


def load_prompt(name: str) -> str:
    """Load a golden prompt fixture."""
    return (PROMPTS_DIR / f"{name}.txt").read_text().strip()


def build_openai_request(
    model: str,
    prompt: str,
    system: str | None = None,
    tools: list | None = None,
    max_tokens: int = 128,
    stream: bool = False,
) -> dict:
    """Build OpenAI chat completion request."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    req: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.1,
    }
    if tools:
        req["tools"] = tools
    return req


def build_anthropic_request(
    model: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 128,
    stream: bool = False,
) -> dict:
    """Build Anthropic messages request."""
    req: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.1,
    }
    if system:
        req["system"] = system
    return req


# --------------------------------------------------
# Non-streaming comparison tests
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_anthropic
class TestCrossProtocolSimple:
    """Compare simple prompt responses across protocols."""

    @pytest.mark.anyio
    async def test_greeting_openai(self, app_client, text_model_quick):
        """OpenAI endpoint returns valid greeting response."""
        prompt = load_prompt("simple_greeting")
        request = build_openai_request(text_model_quick, prompt)
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) >= 1
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0, "OpenAI response should not be empty"

    @pytest.mark.anyio
    async def test_greeting_anthropic(self, app_client, text_model_quick):
        """Anthropic endpoint returns valid greeting response."""
        prompt = load_prompt("simple_greeting")
        request = build_anthropic_request(text_model_quick, prompt)
        response = await app_client.post("/v1/messages", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) >= 1
        assert data["content"][0]["type"] == "text"
        text = data["content"][0]["text"]
        assert len(text) > 0, "Anthropic response should not be empty"

    @pytest.mark.anyio
    async def test_both_produce_similar_output(self, app_client, text_model_quick):
        """Both protocols should produce non-empty responses from same model."""
        prompt = load_prompt("factual_question")

        # Send via OpenAI
        oai_req = build_openai_request(text_model_quick, prompt)
        oai_resp = await app_client.post("/v1/chat/completions", json=oai_req)
        assert oai_resp.status_code == 200
        oai_content = oai_resp.json()["choices"][0]["message"]["content"]

        # Send via Anthropic
        ant_req = build_anthropic_request(text_model_quick, prompt)
        ant_resp = await app_client.post("/v1/messages", json=ant_req)
        assert ant_resp.status_code == 200
        ant_content = ant_resp.json()["content"][0]["text"]

        # Both should produce non-empty answers
        assert len(oai_content) > 0
        assert len(ant_content) > 0

        # Both should mention "Paris" (factual question about France's capital)
        assert "paris" in oai_content.lower(), f"OpenAI missed 'Paris': {oai_content}"
        assert "paris" in ant_content.lower(), f"Anthropic missed 'Paris': {ant_content}"


# --------------------------------------------------
# System message handling
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_anthropic
class TestCrossProtocolSystemMessages:
    """Verify system messages work correctly in both formats."""

    @pytest.mark.anyio
    async def test_system_message_openai(self, app_client, text_model_quick):
        """OpenAI: system message in messages array."""
        prompt = load_prompt("system_instruction")
        # Higher max_tokens: thinking models need room for reasoning + visible output
        request = build_openai_request(
            text_model_quick,
            prompt,
            system="You are a translator. Always respond in the target language only.",
            max_tokens=512,
        )
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.anyio
    async def test_system_message_anthropic(self, app_client, text_model_quick):
        """Anthropic: system message as separate field."""
        prompt = load_prompt("system_instruction")
        # Higher max_tokens: thinking models need room for reasoning + visible output
        request = build_anthropic_request(
            text_model_quick,
            prompt,
            system="You are a translator. Always respond in the target language only.",
            max_tokens=512,
        )
        response = await app_client.post("/v1/messages", json=request)

        assert response.status_code == 200
        text = response.json()["content"][0]["text"]
        assert len(text) > 0


# --------------------------------------------------
# Streaming comparison
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_anthropic
class TestCrossProtocolStreaming:
    """Verify streaming works for both protocols."""

    @pytest.mark.anyio
    async def test_streaming_openai(self, app_client, text_model_quick):
        """OpenAI streaming returns SSE chunks with delta content."""
        prompt = load_prompt("simple_greeting")
        request = build_openai_request(text_model_quick, prompt, stream=True)

        async with app_client.stream("POST", "/v1/chat/completions", json=request) as response:
            assert response.status_code == 200

            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    assert chunk["object"] == "chat.completion.chunk"
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        chunks.append(delta["content"])

            assert len(chunks) > 0, "Expected streaming chunks"

    @pytest.mark.anyio
    async def test_streaming_anthropic(self, app_client, text_model_quick):
        """Anthropic streaming returns SSE with proper event types."""
        prompt = load_prompt("simple_greeting")
        request = build_anthropic_request(text_model_quick, prompt, stream=True)

        async with app_client.stream("POST", "/v1/messages", json=request) as response:
            assert response.status_code == 200

            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        events.append(event)
                    except json.JSONDecodeError:
                        pass

            # Should have message_start, content_block events, message_stop
            event_types = [e.get("type") for e in events]
            assert "message_start" in event_types, f"Missing message_start, got: {event_types}"


# --------------------------------------------------
# Tool calling (OpenAI only -- Anthropic tool calling
# through local models is not yet implemented)
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_anthropic
class TestCrossProtocolToolCalling:
    """Test tool calling via OpenAI endpoint with same prompts."""

    @pytest.mark.anyio
    async def test_tool_call_openai(self, app_client, text_model_quick):
        """OpenAI endpoint should trigger tool call for weather request."""
        prompt = load_prompt("tool_call_request")
        request = build_openai_request(
            text_model_quick,
            prompt,
            tools=[WEATHER_TOOL_OPENAI],
            max_tokens=256,
        )
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()
        choice = data["choices"][0]

        # Model should either produce a tool call or mention weather in content
        has_tool_call = choice["message"].get("tool_calls") is not None
        has_content = choice["message"].get("content") and len(choice["message"]["content"]) > 0
        assert has_tool_call or has_content, "Expected either a tool call or content in response"

        # If tool call present, validate structure
        if has_tool_call:
            tc = choice["message"]["tool_calls"][0]
            assert tc["type"] == "function"
            assert tc["function"]["name"] == "get_weather"
            args = json.loads(tc["function"]["arguments"])
            assert "location" in args


# --------------------------------------------------
# Response structure validation
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_anthropic
class TestProtocolResponseStructure:
    """Validate response structures match their respective API specs."""

    @pytest.mark.anyio
    async def test_openai_response_structure(self, app_client, text_model_quick):
        """OpenAI response has all required fields."""
        prompt = load_prompt("factual_question")
        request = build_openai_request(text_model_quick, prompt)
        response = await app_client.post("/v1/chat/completions", json=request)

        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "finish_reason" in choice

        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    @pytest.mark.anyio
    async def test_anthropic_response_structure(self, app_client, text_model_quick):
        """Anthropic response has all required fields."""
        prompt = load_prompt("factual_question")
        request = build_anthropic_request(text_model_quick, prompt)
        response = await app_client.post("/v1/messages", json=request)

        data = response.json()
        assert "id" in data
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert "content" in data
        assert "model" in data
        assert "stop_reason" in data
        assert "usage" in data

        # Content is array of blocks
        assert isinstance(data["content"], list)
        assert len(data["content"]) >= 1
        assert data["content"][0]["type"] == "text"

        # Usage uses Anthropic field names
        usage = data["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage

    @pytest.mark.anyio
    async def test_stop_reason_translation(self, app_client, text_model_quick):
        """Stop reasons are correctly translated between protocols."""
        prompt = load_prompt("simple_greeting")

        # OpenAI
        oai_req = build_openai_request(text_model_quick, prompt)
        oai_resp = await app_client.post("/v1/chat/completions", json=oai_req)
        oai_finish = oai_resp.json()["choices"][0]["finish_reason"]

        # Anthropic
        ant_req = build_anthropic_request(text_model_quick, prompt)
        ant_resp = await app_client.post("/v1/messages", json=ant_req)
        ant_stop = ant_resp.json()["stop_reason"]

        # Validate both have valid stop reasons for their protocol
        assert oai_finish in ("stop", "length", "tool_calls", "content_filter")
        assert ant_stop in ("end_turn", "max_tokens", "stop_sequence", "tool_use", None)

        # If OpenAI says "stop", Anthropic should say "end_turn"
        if oai_finish == "stop":
            assert ant_stop == "end_turn"
        elif oai_finish == "length":
            assert ant_stop == "max_tokens"

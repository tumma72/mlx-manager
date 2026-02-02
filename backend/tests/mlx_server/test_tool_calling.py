"""Tests for tool calling functionality.

Tests the tool call parsers for Llama, Qwen, and GLM4 formats.
"""

import json

from mlx_manager.mlx_server.models.adapters.parsers.glm4 import GLM4ToolParser
from mlx_manager.mlx_server.models.adapters.parsers.llama import LlamaToolParser
from mlx_manager.mlx_server.models.adapters.parsers.qwen import QwenToolParser


class TestLlamaToolParser:
    """Tests for LlamaToolParser."""

    def test_parse_xml_style_tool_call(self):
        """Parse Llama XML-style function call."""
        parser = LlamaToolParser()
        text = '<function=get_weather>{"city": "San Francisco"}</function>'

        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[0]["function"]["arguments"] == '{"city": "San Francisco"}'
        assert calls[0]["id"].startswith("call_")

    def test_parse_multiple_xml_tool_calls(self):
        """Parse multiple XML-style function calls."""
        parser = LlamaToolParser()
        text = (
            '<function=get_weather>{"city": "SF"}</function>'
            "Some text in between"
            '<function=get_time>{"timezone": "PST"}</function>'
        )

        calls = parser.parse(text)

        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[1]["function"]["name"] == "get_time"

    def test_parse_python_style_tool_call(self):
        """Parse Llama Python-style tool call."""
        parser = LlamaToolParser()
        text = '<|python_tag|>weather.get(city="San Francisco")<|eom_id|>'

        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "weather.get"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["city"] == "San Francisco"

    def test_parse_no_tool_calls(self):
        """Return empty list when no tool calls present."""
        parser = LlamaToolParser()
        text = "Just a regular response without any tool calls."

        calls = parser.parse(text)

        assert calls == []

    def test_parse_invalid_json_arguments(self):
        """Handle invalid JSON in arguments gracefully."""
        parser = LlamaToolParser()
        text = "<function=get_weather>{invalid json}</function>"

        calls = parser.parse(text)

        # Should still return the call with raw arguments
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_format_tools(self):
        """Format tools for Llama prompt."""
        parser = LlamaToolParser()
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            }
        ]

        result = parser.format_tools(tools)

        assert "get_weather" in result
        assert "Get current weather" in result
        assert "<function=function_name>" in result

    def test_format_tools_empty(self):
        """Return empty string for empty tools list."""
        parser = LlamaToolParser()

        result = parser.format_tools([])

        assert result == ""


class TestQwenToolParser:
    """Tests for QwenToolParser (Hermes format)."""

    def test_parse_hermes_style_tool_call(self):
        """Parse Qwen Hermes-style tool call."""
        parser = QwenToolParser()
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'

        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "get_weather"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["city"] == "Tokyo"

    def test_parse_multiple_hermes_tool_calls(self):
        """Parse multiple Hermes-style tool calls."""
        parser = QwenToolParser()
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
            "Let me also check the time."
            '<tool_call>{"name": "get_time", "arguments": {"timezone": "JST"}}</tool_call>'
        )

        calls = parser.parse(text)

        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[1]["function"]["name"] == "get_time"

    def test_parse_no_tool_calls(self):
        """Return empty list when no tool calls present."""
        parser = QwenToolParser()
        text = "Here's the weather forecast for today."

        calls = parser.parse(text)

        assert calls == []

    def test_parse_invalid_json(self):
        """Skip invalid JSON tool calls."""
        parser = QwenToolParser()
        text = "<tool_call>{not valid json}</tool_call>"

        calls = parser.parse(text)

        assert calls == []

    def test_format_tools(self):
        """Format tools for Qwen prompt (Hermes style)."""
        parser = QwenToolParser()
        tools = [
            {
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            }
        ]

        result = parser.format_tools(tools)

        assert "<tools>" in result
        assert "search" in result
        assert "Search the web" in result
        assert "<tool_call>" in result

    def test_format_tools_empty(self):
        """Return empty string for empty tools list."""
        parser = QwenToolParser()

        result = parser.format_tools([])

        assert result == ""


class TestGLM4ToolParser:
    """Tests for GLM4ToolParser (XML nested format)."""

    def test_parse_xml_nested_tool_call(self):
        """Parse GLM4 XML nested tool call."""
        parser = GLM4ToolParser()
        text = (
            "<tool_call>"
            "<name>calculate</name>"
            '<arguments>{"expression": "2+2"}</arguments>'
            "</tool_call>"
        )

        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "calculate"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["expression"] == "2+2"

    def test_parse_json_format_tool_call(self):
        """Parse GLM4 JSON format tool call."""
        parser = GLM4ToolParser()
        text = '<tool_call>{"name": "calculate", "arguments": {"x": 5}}</tool_call>'

        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "calculate"

    def test_deduplication_removes_duplicate_calls(self):
        """GLM4 parser deduplicates duplicate tool calls."""
        parser = GLM4ToolParser()
        # GLM4 sometimes outputs duplicate calls
        text = (
            "<tool_call><name>get_data</name>"
            '<arguments>{"id": 1}</arguments></tool_call>'
            "<tool_call><name>get_data</name>"
            '<arguments>{"id": 1}</arguments></tool_call>'
        )

        calls = parser.parse(text)

        # Should only have one call after deduplication
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_data"

    def test_deduplication_keeps_different_calls(self):
        """GLM4 parser keeps different tool calls."""
        parser = GLM4ToolParser()
        text = (
            "<tool_call><name>get_data</name>"
            '<arguments>{"id": 1}</arguments></tool_call>'
            "<tool_call><name>get_data</name>"
            '<arguments>{"id": 2}</arguments></tool_call>'
        )

        calls = parser.parse(text)

        # Different arguments = different calls
        assert len(calls) == 2

    def test_parse_no_tool_calls(self):
        """Return empty list when no tool calls present."""
        parser = GLM4ToolParser()
        text = "I'll help you with that calculation."

        calls = parser.parse(text)

        assert calls == []

    def test_format_tools(self):
        """Format tools for GLM4 prompt."""
        parser = GLM4ToolParser()
        tools = [
            {
                "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                    },
                }
            }
        ]

        result = parser.format_tools(tools)

        assert "<tool>" in result
        assert "<name>send_email</name>" in result
        assert "Send an email" in result
        assert "<tool_call>" in result

    def test_format_tools_empty(self):
        """Return empty string for empty tools list."""
        parser = GLM4ToolParser()

        result = parser.format_tools([])

        assert result == ""


class TestAdapterToolCallingSupport:
    """Tests for adapter tool calling support methods."""

    def test_llama_adapter_supports_tool_calling(self):
        """Llama adapter supports tool calling."""
        from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter

        adapter = LlamaAdapter()

        assert adapter.supports_tool_calling() is True

    def test_qwen_adapter_supports_tool_calling(self):
        """Qwen adapter supports tool calling."""
        from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter

        adapter = QwenAdapter()

        assert adapter.supports_tool_calling() is True

    def test_default_adapter_does_not_support_tool_calling(self):
        """Default adapter does not support tool calling."""
        from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

        adapter = DefaultAdapter()

        assert adapter.supports_tool_calling() is False

    def test_llama_adapter_parse_tool_calls(self):
        """Llama adapter delegates to LlamaToolParser."""
        from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter

        adapter = LlamaAdapter()
        text = '<function=test>{"arg": "value"}</function>'

        calls = adapter.parse_tool_calls(text)

        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "test"

    def test_llama_adapter_parse_no_tool_calls(self):
        """Llama adapter returns None when no tool calls found."""
        from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter

        adapter = LlamaAdapter()
        text = "No tool calls here."

        calls = adapter.parse_tool_calls(text)

        assert calls is None

    def test_qwen_adapter_format_tools(self):
        """Qwen adapter formats tools using QwenToolParser."""
        from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter

        adapter = QwenAdapter()
        tools = [{"function": {"name": "test", "description": "Test function"}}]

        result = adapter.format_tools_for_prompt(tools)

        assert "test" in result
        assert "<tools>" in result


class TestToolCallIdGeneration:
    """Tests for unique tool call ID generation."""

    def test_llama_parser_generates_unique_ids(self):
        """Each parsed tool call gets a unique ID."""
        parser = LlamaToolParser()
        text = '<function=func1>{"a": 1}</function><function=func2>{"b": 2}</function>'

        calls = parser.parse(text)

        assert len(calls) == 2
        assert calls[0]["id"] != calls[1]["id"]
        assert calls[0]["id"].startswith("call_")
        assert calls[1]["id"].startswith("call_")

    def test_qwen_parser_generates_unique_ids(self):
        """Qwen parser generates unique IDs."""
        parser = QwenToolParser()
        text = (
            '<tool_call>{"name": "f1", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "f2", "arguments": {}}</tool_call>'
        )

        calls = parser.parse(text)

        assert len(calls) == 2
        assert calls[0]["id"] != calls[1]["id"]

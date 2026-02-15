"""Tests for adapter tool calling support.

Tool call parsing is now handled by adapters via injected parsers.
This file tests adapter-level tool calling configuration: support flags and
prompt formatting.
"""

from mlx_manager.mlx_server.models.adapters.composable import create_adapter


class MockTokenizer:
    """Mock tokenizer for adapter initialization."""

    def __init__(self):
        self.eos_token_id = 100
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 200  # Mock token ID


class TestAdapterToolCallingSupport:
    """Tests for adapter tool calling support methods."""

    def test_llama_adapter_supports_tool_calling(self):
        """Llama adapter supports tool calling."""
        adapter = create_adapter("llama", MockTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_qwen_adapter_supports_tool_calling(self):
        """Qwen adapter supports tool calling."""
        adapter = create_adapter("qwen", MockTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_glm4_adapter_supports_tool_calling(self):
        """GLM4 adapter supports tool calling."""
        adapter = create_adapter("glm4", MockTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_default_adapter_does_not_support_tool_calling(self):
        """Default adapter does not support tool calling."""
        adapter = create_adapter("default", MockTokenizer())
        assert adapter.supports_tool_calling() is False


class TestAdapterToolFormatting:
    """Tests for adapter tool prompt formatting.

    NOTE: The actual tool formatting moved to individual parsers.
    These tests verify the adapter delegates correctly.
    """

    def test_llama_adapter_format_tools(self):
        """Llama adapter formats tools for prompt injection."""
        adapter = create_adapter("llama", MockTokenizer())
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

        result = adapter.format_tools_for_prompt(tools)

        assert "get_weather" in result
        assert "Get current weather" in result
        assert "<function=function_name>" in result

    def test_llama_adapter_format_tools_empty(self):
        """Llama adapter returns empty string for no tools."""
        adapter = create_adapter("llama", MockTokenizer())
        result = adapter.format_tools_for_prompt([])
        assert result == ""

    def test_qwen_adapter_format_tools(self):
        """Qwen adapter formats tools using Hermes style."""
        adapter = create_adapter("qwen", MockTokenizer())
        tools = [
            {
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            }
        ]

        result = adapter.format_tools_for_prompt(tools)

        assert "<tools>" in result
        assert "search" in result
        assert "Search the web" in result
        assert "<tool_call>" in result

    def test_qwen_adapter_format_tools_empty(self):
        """Qwen adapter returns empty string for no tools."""
        adapter = create_adapter("qwen", MockTokenizer())
        result = adapter.format_tools_for_prompt([])
        assert result == ""

    def test_glm4_adapter_format_tools(self):
        """GLM4 adapter formats tools using XML style."""
        adapter = create_adapter("glm4", MockTokenizer())
        tools = [
            {
                "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "body": {"type": "string"},
                        },
                    },
                }
            }
        ]

        result = adapter.format_tools_for_prompt(tools)

        # GLM4 uses XML format
        assert "<tool>" in result
        assert "<name>send_email</name>" in result
        assert "Send an email" in result
        assert "<tool_call>" in result

    def test_glm4_adapter_format_tools_empty(self):
        """GLM4 adapter returns empty string for no tools."""
        adapter = create_adapter("glm4", MockTokenizer())
        result = adapter.format_tools_for_prompt([])
        assert result == ""

    def test_default_adapter_format_tools(self):
        """Default adapter returns empty string for tools."""
        adapter = create_adapter("default", MockTokenizer())
        tools = [{"function": {"name": "test", "description": "Test"}}]

        result = adapter.format_tools_for_prompt(tools)
        assert result == ""

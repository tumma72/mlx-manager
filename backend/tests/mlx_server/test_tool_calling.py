"""Tests for adapter tool calling support.

Tool call parsing is now handled by ResponseProcessor (tested in test_response_processor.py).
This file tests adapter-level tool calling configuration: support flags and prompt formatting.
"""


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

    def test_glm4_adapter_supports_tool_calling(self):
        """GLM4 adapter supports tool calling."""
        from mlx_manager.mlx_server.models.adapters.glm4 import GLM4Adapter

        adapter = GLM4Adapter()

        assert adapter.supports_tool_calling() is True

    def test_default_adapter_does_not_support_tool_calling(self):
        """Default adapter does not support tool calling."""
        from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

        adapter = DefaultAdapter()

        assert adapter.supports_tool_calling() is False


class TestAdapterToolFormatting:
    """Tests for adapter tool prompt formatting."""

    def test_llama_adapter_format_tools(self):
        """Llama adapter formats tools for prompt injection."""
        from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter

        adapter = LlamaAdapter()
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
        from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter

        adapter = LlamaAdapter()

        result = adapter.format_tools_for_prompt([])

        assert result == ""

    def test_qwen_adapter_format_tools(self):
        """Qwen adapter formats tools using Hermes style."""
        from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter

        adapter = QwenAdapter()
        tools = [
            {
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
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
        from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter

        adapter = QwenAdapter()

        result = adapter.format_tools_for_prompt([])

        assert result == ""

    def test_glm4_adapter_format_tools(self):
        """GLM4 adapter formats tools using JSON style (similar to Qwen/Hermes)."""
        from mlx_manager.mlx_server.models.adapters.glm4 import GLM4Adapter

        adapter = GLM4Adapter()
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

        result = adapter.format_tools_for_prompt(tools)

        # GLM-4 now uses JSON format like Qwen/Hermes
        assert "<tools>" in result
        assert '"name": "send_email"' in result
        assert "Send an email" in result
        assert "<tool_call>" in result

    def test_glm4_adapter_format_tools_empty(self):
        """GLM4 adapter returns empty string for no tools."""
        from mlx_manager.mlx_server.models.adapters.glm4 import GLM4Adapter

        adapter = GLM4Adapter()

        result = adapter.format_tools_for_prompt([])

        assert result == ""

    def test_default_adapter_format_tools(self):
        """Default adapter returns empty string for tools."""
        from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

        adapter = DefaultAdapter()
        tools = [{"function": {"name": "test", "description": "Test"}}]

        result = adapter.format_tools_for_prompt(tools)

        assert result == ""

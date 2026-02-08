"""Tests for model family adapters."""

from unittest.mock import MagicMock

from mlx_manager.mlx_server.models.adapters import get_adapter, get_supported_families
from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter
from mlx_manager.mlx_server.models.adapters.gemma import GemmaAdapter
from mlx_manager.mlx_server.models.adapters.glm4 import GLM4Adapter
from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter
from mlx_manager.mlx_server.models.adapters.mistral import MistralAdapter
from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter


class TestAdapterRegistry:
    """Tests for adapter registry."""

    def test_supported_families_includes_all(self):
        """Verify all expected families are registered."""
        families = get_supported_families()
        assert "llama" in families
        assert "qwen" in families
        assert "mistral" in families
        assert "gemma" in families
        assert "default" in families

    def test_get_adapter_qwen(self):
        """Verify Qwen models get QwenAdapter."""
        adapter = get_adapter("mlx-community/Qwen2.5-7B-Instruct-4bit")
        assert adapter.family == "qwen"

    def test_get_adapter_mistral(self):
        """Verify Mistral models get MistralAdapter."""
        adapter = get_adapter("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        assert adapter.family == "mistral"

    def test_get_adapter_mixtral(self):
        """Verify Mixtral models also get MistralAdapter."""
        adapter = get_adapter("mlx-community/Mixtral-8x7B-Instruct-4bit")
        assert adapter.family == "mistral"

    def test_get_adapter_gemma(self):
        """Verify Gemma models get GemmaAdapter."""
        adapter = get_adapter("mlx-community/gemma-2-9b-it-4bit")
        assert adapter.family == "gemma"


class TestQwenAdapter:
    """Tests for QwenAdapter."""

    def test_family(self):
        """Verify family property returns 'qwen'."""
        adapter = QwenAdapter()
        assert adapter.family == "qwen"

    def test_get_stop_tokens_includes_im_end(self):
        """Verify <|im_end|> is included in stop tokens."""
        adapter = QwenAdapter()
        # Use spec to prevent auto-creation of .tokenizer attribute
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = 200  # <|im_end|>

        stop_tokens = adapter.get_stop_tokens(tokenizer)

        assert 100 in stop_tokens  # eos
        assert 200 in stop_tokens  # <|im_end|>
        tokenizer.convert_tokens_to_ids.assert_called_with("<|im_end|>")

    def test_get_stop_tokens_handles_missing_im_end(self):
        """Verify graceful handling when <|im_end|> is not available."""
        adapter = QwenAdapter()
        # Use spec to prevent auto-creation of .tokenizer attribute
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.side_effect = Exception("Token not found")

        stop_tokens = adapter.get_stop_tokens(tokenizer)

        assert stop_tokens == [100]  # Only eos

    def test_apply_chat_template(self):
        """Verify chat template is applied using tokenizer with thinking mode."""
        adapter = QwenAdapter()
        # Use spec=[] to prevent auto-creation of .tokenizer attribute
        # This ensures getattr(tokenizer, "tokenizer", tokenizer) returns tokenizer itself
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "formatted"

        messages = [{"role": "user", "content": "Hello"}]
        result = adapter.apply_chat_template(tokenizer, messages)

        assert result == "formatted"
        # Qwen adapter tries enable_thinking=True for Qwen3 thinking mode
        tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

    def test_get_stop_tokens_with_processor(self):
        """Verify Processor objects (vision models) are handled correctly."""
        adapter = QwenAdapter()
        # Simulate a Processor that wraps a tokenizer
        inner_tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        inner_tokenizer.eos_token_id = 100
        inner_tokenizer.unk_token_id = 0
        inner_tokenizer.convert_tokens_to_ids.return_value = 200  # <|im_end|>

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        stop_tokens = adapter.get_stop_tokens(processor)

        assert 100 in stop_tokens  # eos
        assert 200 in stop_tokens  # <|im_end|>
        inner_tokenizer.convert_tokens_to_ids.assert_called_with("<|im_end|>")


class TestMistralAdapter:
    """Tests for MistralAdapter."""

    def test_family(self):
        """Verify family property returns 'mistral'."""
        adapter = MistralAdapter()
        assert adapter.family == "mistral"

    def test_apply_chat_template_prepends_system_message(self):
        """Verify system message is prepended to first user message."""
        adapter = MistralAdapter()
        # Use spec=[] to prevent auto-creation of .tokenizer attribute
        # This ensures getattr(tokenizer, "tokenizer", tokenizer) returns tokenizer itself
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "formatted"

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        adapter.apply_chat_template(tokenizer, messages)

        # Check that system was merged into user message
        call_args = tokenizer.apply_chat_template.call_args
        processed_messages = call_args[0][0]
        assert len(processed_messages) == 1
        assert processed_messages[0]["role"] == "user"
        assert "You are helpful." in processed_messages[0]["content"]
        assert "Hello" in processed_messages[0]["content"]

    def test_apply_chat_template_no_system_message(self):
        """Verify messages without system are passed through."""
        adapter = MistralAdapter()
        # Use spec=[] to prevent auto-creation of .tokenizer attribute
        # This ensures getattr(tokenizer, "tokenizer", tokenizer) returns tokenizer itself
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "formatted"

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        adapter.apply_chat_template(tokenizer, messages)

        call_args = tokenizer.apply_chat_template.call_args
        processed_messages = call_args[0][0]
        assert len(processed_messages) == 1
        assert processed_messages[0] == {"role": "user", "content": "Hello"}

    def test_get_stop_tokens(self):
        """Verify only eos_token_id is returned."""
        adapter = MistralAdapter()
        # Use spec to prevent auto-creation of .tokenizer attribute
        tokenizer = MagicMock(spec=["eos_token_id"])
        tokenizer.eos_token_id = 100

        stop_tokens = adapter.get_stop_tokens(tokenizer)

        assert stop_tokens == [100]

    def test_get_stop_tokens_with_processor(self):
        """Verify Processor objects (vision models) are handled correctly."""
        adapter = MistralAdapter()
        # Simulate a Processor that wraps a tokenizer
        inner_tokenizer = MagicMock(spec=["eos_token_id"])
        inner_tokenizer.eos_token_id = 100

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        stop_tokens = adapter.get_stop_tokens(processor)

        assert stop_tokens == [100]


class TestGemmaAdapter:
    """Tests for GemmaAdapter."""

    def test_family(self):
        """Verify family property returns 'gemma'."""
        adapter = GemmaAdapter()
        assert adapter.family == "gemma"

    def test_get_stop_tokens_includes_end_of_turn(self):
        """Verify <end_of_turn> is included in stop tokens."""
        adapter = GemmaAdapter()
        # Use spec to prevent auto-creation of .tokenizer attribute
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = 300  # <end_of_turn>

        stop_tokens = adapter.get_stop_tokens(tokenizer)

        assert 100 in stop_tokens  # eos
        assert 300 in stop_tokens  # <end_of_turn>
        tokenizer.convert_tokens_to_ids.assert_called_with("<end_of_turn>")

    def test_get_stop_tokens_handles_missing_end_of_turn(self):
        """Verify graceful handling when <end_of_turn> is not available."""
        adapter = GemmaAdapter()
        # Use spec to prevent auto-creation of .tokenizer attribute
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.side_effect = Exception("Token not found")

        stop_tokens = adapter.get_stop_tokens(tokenizer)

        assert stop_tokens == [100]  # Only eos

    def test_apply_chat_template(self):
        """Verify chat template is applied using tokenizer."""
        adapter = GemmaAdapter()
        # Use spec=[] to prevent auto-creation of .tokenizer attribute
        # This ensures getattr(tokenizer, "tokenizer", tokenizer) returns tokenizer itself
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "formatted"

        messages = [{"role": "user", "content": "Hello"}]
        result = adapter.apply_chat_template(tokenizer, messages)

        assert result == "formatted"
        tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def test_get_stop_tokens_with_processor(self):
        """Verify Processor objects (vision models) are handled correctly."""
        adapter = GemmaAdapter()
        # Simulate a Processor that wraps a tokenizer
        inner_tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        inner_tokenizer.eos_token_id = 100
        inner_tokenizer.unk_token_id = 0
        inner_tokenizer.convert_tokens_to_ids.return_value = 300  # <end_of_turn>

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        stop_tokens = adapter.get_stop_tokens(processor)

        assert 100 in stop_tokens  # eos
        assert 300 in stop_tokens  # <end_of_turn>
        inner_tokenizer.convert_tokens_to_ids.assert_called_with("<end_of_turn>")


class TestAdapterReasoningSupport:
    """Tests for adapter reasoning mode support flags.

    Migrated from test_reasoning.py. Reasoning extraction is handled by
    ResponseProcessor (tested in test_response_processor.py). These tests
    verify the adapter-level supports_reasoning_mode() contract.
    """

    def test_llama_adapter_supports_reasoning_mode(self):
        """Llama adapter supports reasoning mode."""
        adapter = LlamaAdapter()
        assert adapter.supports_reasoning_mode() is True

    def test_qwen_adapter_supports_reasoning_mode(self):
        """Qwen adapter supports reasoning mode."""
        adapter = QwenAdapter()
        assert adapter.supports_reasoning_mode() is True

    def test_glm4_adapter_supports_reasoning_mode(self):
        """GLM4 adapter supports reasoning mode."""
        adapter = GLM4Adapter()
        assert adapter.supports_reasoning_mode() is True

    def test_default_adapter_does_not_support_reasoning_mode(self):
        """Default adapter does not support reasoning mode."""
        adapter = DefaultAdapter()
        assert adapter.supports_reasoning_mode() is False

    def test_gemma_adapter_does_not_support_reasoning_mode(self):
        """Gemma adapter does not support reasoning mode (inherits default)."""
        adapter = GemmaAdapter()
        assert adapter.supports_reasoning_mode() is False

    def test_mistral_adapter_does_not_support_reasoning_mode(self):
        """Mistral adapter does not support reasoning mode (inherits default)."""
        adapter = MistralAdapter()
        assert adapter.supports_reasoning_mode() is False


class TestConvertMessages:
    """Tests for adapter convert_messages() implementations.

    Verifies that tool-capable adapters correctly convert tool messages
    to formats their tokenizers can handle (P2: adapter-driven, P3: no data loss).
    """

    # --- Shared test messages ---

    TOOL_RESULT_MSG = {
        "role": "tool",
        "content": '{"temp": 72}',
        "tool_call_id": "call_abc123",
    }

    ASSISTANT_WITH_TOOL_CALLS_MSG = {
        "role": "assistant",
        "content": "Let me check the weather.",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                },
            }
        ],
    }

    REGULAR_USER_MSG = {"role": "user", "content": "Hello"}
    REGULAR_ASSISTANT_MSG = {"role": "assistant", "content": "Hi there!"}

    # --- DefaultAdapter ---

    def test_default_passes_regular_messages_through(self):
        """DefaultAdapter preserves regular messages unchanged."""
        adapter = DefaultAdapter()
        messages = [self.REGULAR_USER_MSG, self.REGULAR_ASSISTANT_MSG]

        result = adapter.convert_messages(messages)

        assert result == messages

    def test_default_converts_tool_result_to_user(self):
        """DefaultAdapter converts tool result to user message."""
        adapter = DefaultAdapter()
        messages = [self.TOOL_RESULT_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_abc123" in result[0]["content"]
        assert '{"temp": 72}' in result[0]["content"]

    def test_default_converts_assistant_tool_calls_to_text(self):
        """DefaultAdapter converts assistant tool_calls to text."""
        adapter = DefaultAdapter()
        messages = [self.ASSISTANT_WITH_TOOL_CALLS_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "get_weather" in result[0]["content"]
        assert "tool_calls" not in result[0]

    # --- QwenAdapter ---

    def test_qwen_converts_tool_result_to_user(self):
        """QwenAdapter converts tool result to user message."""
        adapter = QwenAdapter()
        messages = [self.TOOL_RESULT_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_abc123" in result[0]["content"]
        assert '{"temp": 72}' in result[0]["content"]

    def test_qwen_converts_assistant_tool_calls_to_hermes(self):
        """QwenAdapter converts tool_calls to Hermes <tool_call> format."""
        adapter = QwenAdapter()
        messages = [self.ASSISTANT_WITH_TOOL_CALLS_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "<tool_call>" in result[0]["content"]
        assert "get_weather" in result[0]["content"]
        assert "tool_calls" not in result[0]

    def test_qwen_preserves_regular_messages(self):
        """QwenAdapter passes regular messages unchanged."""
        adapter = QwenAdapter()
        messages = [self.REGULAR_USER_MSG, self.REGULAR_ASSISTANT_MSG]

        result = adapter.convert_messages(messages)

        assert result == messages

    def test_qwen_multi_turn_tool_use(self):
        """QwenAdapter handles full multi-turn tool use conversation."""
        adapter = QwenAdapter()
        messages = [
            {"role": "system", "content": "You have tools."},
            {"role": "user", "content": "What's the weather?"},
            self.ASSISTANT_WITH_TOOL_CALLS_MSG,
            self.TOOL_RESULT_MSG,
        ]

        result = adapter.convert_messages(messages)

        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert "<tool_call>" in result[2]["content"]
        assert result[3]["role"] == "user"  # tool -> user
        assert "Tool Result" in result[3]["content"]

    # --- LlamaAdapter ---

    def test_llama_converts_tool_result_to_user(self):
        """LlamaAdapter converts tool result to user message."""
        adapter = LlamaAdapter()
        messages = [self.TOOL_RESULT_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_abc123" in result[0]["content"]
        assert '{"temp": 72}' in result[0]["content"]

    def test_llama_converts_assistant_tool_calls_to_function_tags(self):
        """LlamaAdapter converts tool_calls to <function=name> format."""
        adapter = LlamaAdapter()
        messages = [self.ASSISTANT_WITH_TOOL_CALLS_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "<function=get_weather>" in result[0]["content"]
        assert "</function>" in result[0]["content"]
        assert "tool_calls" not in result[0]

    def test_llama_preserves_regular_messages(self):
        """LlamaAdapter passes regular messages unchanged."""
        adapter = LlamaAdapter()
        messages = [self.REGULAR_USER_MSG, self.REGULAR_ASSISTANT_MSG]

        result = adapter.convert_messages(messages)

        assert result == messages

    def test_llama_multi_turn_tool_use(self):
        """LlamaAdapter handles full multi-turn tool use conversation."""
        adapter = LlamaAdapter()
        messages = [
            {"role": "user", "content": "What's the weather?"},
            self.ASSISTANT_WITH_TOOL_CALLS_MSG,
            self.TOOL_RESULT_MSG,
        ]

        result = adapter.convert_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert "<function=get_weather>" in result[1]["content"]
        assert result[2]["role"] == "user"  # tool -> user
        assert "Tool Result" in result[2]["content"]

    # --- GLM4Adapter (existing, verify contract) ---

    def test_glm4_converts_tool_result_to_user(self):
        """GLM4Adapter converts tool result to user message."""
        adapter = GLM4Adapter()
        messages = [self.TOOL_RESULT_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_abc123" in result[0]["content"]

    def test_glm4_converts_assistant_tool_calls_to_text(self):
        """GLM4Adapter converts tool_calls to inline text."""
        adapter = GLM4Adapter()
        messages = [self.ASSISTANT_WITH_TOOL_CALLS_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "get_weather" in result[0]["content"]

    # --- Non-tool adapters inherit safe default ---

    def test_gemma_safely_handles_tool_messages(self):
        """GemmaAdapter (inherits DefaultAdapter) safely converts tool messages."""
        adapter = GemmaAdapter()
        messages = [self.TOOL_RESULT_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_mistral_safely_handles_tool_messages(self):
        """MistralAdapter (inherits DefaultAdapter) safely converts tool messages."""
        adapter = MistralAdapter()
        messages = [self.TOOL_RESULT_MSG]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"


# --- GLM4 Adapter Comprehensive Tests ---


class TestGLM4Adapter:
    """Comprehensive tests for GLM4Adapter to bring coverage to 90%+."""

    def test_family_property(self):
        """GLM4Adapter.family returns 'glm4'."""
        adapter = GLM4Adapter()
        assert adapter.family == "glm4"

    def test_supports_tool_calling(self):
        """GLM4Adapter supports tool calling."""
        adapter = GLM4Adapter()
        assert adapter.supports_tool_calling() is True

    def test_supports_reasoning_mode(self):
        """GLM4Adapter supports reasoning mode."""
        adapter = GLM4Adapter()
        assert adapter.supports_reasoning_mode() is True

    def test_get_tool_call_stop_tokens_returns_empty(self):
        """GLM4 tool call markers are multi-token; returns empty list."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["eos_token_id"])
        tokenizer.eos_token_id = 100
        assert adapter.get_tool_call_stop_tokens(tokenizer) == []

    def test_get_stop_tokens_includes_special_tokens(self):
        """GLM4 stop tokens include eos and special tokens."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        # Simulate: <|user|>=200, <|observation|>=201, <|endoftext|>=202
        tokenizer.convert_tokens_to_ids.side_effect = [200, 201, 202]

        stop_tokens = adapter.get_stop_tokens(tokenizer)

        assert 100 in stop_tokens  # eos
        assert 200 in stop_tokens  # <|user|>
        assert 201 in stop_tokens  # <|observation|>
        assert 202 in stop_tokens  # <|endoftext|>

    def test_get_stop_tokens_with_processor(self):
        """GLM4 stop tokens work with Processor wrapping tokenizer."""
        adapter = GLM4Adapter()
        inner_tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        inner_tokenizer.eos_token_id = 100
        inner_tokenizer.unk_token_id = 0
        inner_tokenizer.convert_tokens_to_ids.side_effect = Exception("Not found")

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        stop_tokens = adapter.get_stop_tokens(processor)
        assert stop_tokens == [100]  # Only eos

    def test_get_stop_tokens_skips_unk_tokens(self):
        """GLM4 stop tokens skip tokens that resolve to unk."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = 0  # All return unk

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens == [100]  # Only eos, unk skipped

    def test_get_stop_tokens_skips_none(self):
        """GLM4 stop tokens skip tokens that resolve to None."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = None

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens == [100]

    def test_get_stop_tokens_no_duplicates(self):
        """GLM4 stop tokens don't include duplicate eos."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        # All special tokens resolve to same as eos
        tokenizer.convert_tokens_to_ids.return_value = 100

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens.count(100) == 1

    def test_format_tools_for_prompt_single_tool(self):
        """Format a single tool for GLM4 system prompt."""
        adapter = GLM4Adapter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]

        result = adapter.format_tools_for_prompt(tools)

        assert "<tool>" in result
        assert "<name>get_weather</name>" in result
        assert "<description>Get weather info</description>" in result
        assert "<parameters>" in result
        assert "tool_call" in result.lower()

    def test_format_tools_for_prompt_empty_list(self):
        """Empty tool list returns empty string."""
        adapter = GLM4Adapter()
        assert adapter.format_tools_for_prompt([]) == ""

    def test_format_tools_for_prompt_multiple_tools(self):
        """Format multiple tools for GLM4 system prompt."""
        adapter = GLM4Adapter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func_a",
                    "description": "A",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "func_b",
                    "description": "B",
                    "parameters": {},
                },
            },
        ]

        result = adapter.format_tools_for_prompt(tools)

        assert "<name>func_a</name>" in result
        assert "<name>func_b</name>" in result

    def test_apply_chat_template_with_tools_ignores_tools(self):
        """GLM4 apply_chat_template ignores tools parameter (uses standard template)."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "standard_result"

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]

        result = adapter.apply_chat_template(tokenizer, messages, tools=tools)

        assert result == "standard_result"
        # tools not passed to tokenizer
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, add_generation_prompt=True, tokenize=False
        )

    def test_apply_chat_template_without_tools(self):
        """GLM4 apply_chat_template without tools uses standard template."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "standard_result"

        messages = [{"role": "user", "content": "Hello"}]
        result = adapter.apply_chat_template(tokenizer, messages)

        assert result == "standard_result"

    def test_apply_chat_template_fallback_manual(self):
        """GLM4 falls back to manual ChatML format when tokenizer fails."""
        from mlx_manager.mlx_server.utils.template_tools import clear_native_tools_cache

        clear_native_tools_cache()

        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.side_effect = Exception("No template")

        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
        ]

        result = adapter.apply_chat_template(tokenizer, messages)

        assert "<|system|>" in result
        assert "You are a helper." in result
        assert "<|user|>" in result
        assert "Hello" in result
        assert "<|assistant|>" in result

    def test_apply_chat_template_manual_no_generation_prompt(self):
        """GLM4 manual fallback without add_generation_prompt."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.side_effect = Exception("No template")

        messages = [{"role": "user", "content": "Hello"}]
        result = adapter.apply_chat_template(tokenizer, messages, add_generation_prompt=False)

        assert "<|assistant|>" not in result
        assert "<|user|>" in result
        assert "Hello" in result

    def test_has_native_tool_support_always_false(self):
        """has_native_tool_support returns False (capabilities-based now)."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=["apply_chat_template", "chat_template"])
        tokenizer.chat_template = "{% if tools %}{{ tools }}{% endif %}"

        assert adapter.has_native_tool_support(tokenizer) is False

    def test_has_native_tool_support_false_when_no_template(self):
        """has_native_tool_support returns False when no template."""
        adapter = GLM4Adapter()
        tokenizer = MagicMock(spec=[])

        assert adapter.has_native_tool_support(tokenizer) is False

    def test_convert_messages_tool_result(self):
        """GLM4 converts tool result to user message."""
        adapter = GLM4Adapter()
        messages = [
            {
                "role": "tool",
                "content": '{"temp": 72}',
                "tool_call_id": "call_abc",
            }
        ]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "call_abc" in result[0]["content"]
        assert '{"temp": 72}' in result[0]["content"]
        assert "Tool Result" in result[0]["content"]

    def test_convert_messages_assistant_tool_calls(self):
        """GLM4 converts assistant tool_calls to text format."""
        adapter = GLM4Adapter()
        messages = [
            {
                "role": "assistant",
                "content": "Checking weather.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Rome"}',
                        },
                    }
                ],
            }
        ]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "<tool_call>" in result[0]["content"]
        assert "get_weather" in result[0]["content"]
        assert "Rome" in result[0]["content"]
        assert "Checking weather." in result[0]["content"]

    def test_convert_messages_passthrough(self):
        """GLM4 passes regular messages through unchanged."""
        adapter = GLM4Adapter()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        result = adapter.convert_messages(messages)

        assert result == messages


# --- Llama Adapter Comprehensive Tests ---


class TestLlamaAdapterComprehensive:
    """Comprehensive tests for LlamaAdapter to bring coverage to 90%+."""

    def test_family_property(self):
        """LlamaAdapter.family returns 'llama'."""
        adapter = LlamaAdapter()
        assert adapter.family == "llama"

    def test_supports_tool_calling(self):
        """LlamaAdapter supports tool calling."""
        adapter = LlamaAdapter()
        assert adapter.supports_tool_calling() is True

    def test_supports_reasoning_mode(self):
        """LlamaAdapter supports reasoning mode."""
        adapter = LlamaAdapter()
        assert adapter.supports_reasoning_mode() is True

    def test_apply_chat_template(self):
        """Llama apply_chat_template uses tokenizer template."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "llama_formatted"

        messages = [{"role": "user", "content": "Hello"}]
        result = adapter.apply_chat_template(tokenizer, messages)

        assert result == "llama_formatted"
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, add_generation_prompt=True, tokenize=False
        )

    def test_apply_chat_template_with_processor(self):
        """Llama apply_chat_template unwraps Processor objects."""
        adapter = LlamaAdapter()
        inner_tokenizer = MagicMock(spec=["apply_chat_template"])
        inner_tokenizer.apply_chat_template.return_value = "formatted"

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        messages = [{"role": "user", "content": "Hi"}]
        result = adapter.apply_chat_template(processor, messages)

        assert result == "formatted"
        inner_tokenizer.apply_chat_template.assert_called_once()

    def test_apply_chat_template_tools_ignored(self):
        """Llama apply_chat_template ignores tools parameter."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "formatted"

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = adapter.apply_chat_template(tokenizer, messages, tools=tools)

        # tools not passed to tokenizer
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, add_generation_prompt=True, tokenize=False
        )
        assert result == "formatted"

    def test_get_stop_tokens_includes_eot_id(self):
        """Llama stop tokens include <|eot_id|>."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 128009
        tokenizer.unk_token_id = 0

        def convert(token):
            if token == "<|eot_id|>":
                return 128001
            if token == "<|end_of_turn|>":
                return 128002
            return 0

        tokenizer.convert_tokens_to_ids.side_effect = convert

        stop_tokens = adapter.get_stop_tokens(tokenizer)

        assert 128009 in stop_tokens  # eos
        assert 128001 in stop_tokens  # eot_id
        assert 128002 in stop_tokens  # end_of_turn

    def test_get_stop_tokens_handles_missing_eot_id(self):
        """Llama stop tokens handle missing <|eot_id|>."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.side_effect = Exception("Token not found")

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens == [100]

    def test_get_stop_tokens_skips_unk(self):
        """Llama stop tokens skip tokens that resolve to unk_token_id."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = 0  # unk

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens == [100]

    def test_get_stop_tokens_skips_none(self):
        """Llama stop tokens skip tokens that resolve to None."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = None

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens == [100]

    def test_get_stop_tokens_with_processor(self):
        """Llama stop tokens unwrap Processor objects."""
        adapter = LlamaAdapter()
        inner_tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        inner_tokenizer.eos_token_id = 100
        inner_tokenizer.unk_token_id = 0
        inner_tokenizer.convert_tokens_to_ids.return_value = 200

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        stop_tokens = adapter.get_stop_tokens(processor)
        assert 200 in stop_tokens

    def test_get_stop_tokens_no_duplicate_end_turn(self):
        """Llama stop tokens don't duplicate if end_turn already present."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0

        def convert(token):
            if token == "<|eot_id|>":
                return 200
            if token == "<|end_of_turn|>":
                return 200  # Same as eot_id
            return 0

        tokenizer.convert_tokens_to_ids.side_effect = convert

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens.count(200) == 1  # No duplicate

    def test_is_stop_token(self):
        """is_stop_token correctly checks token ID."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.eos_token_id = 100
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = 0  # unk

        assert adapter.is_stop_token(100, tokenizer) is True
        assert adapter.is_stop_token(999, tokenizer) is False

    def test_format_tools_for_prompt_single_tool(self):
        """Format a single tool for Llama system prompt."""
        adapter = LlamaAdapter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]

        result = adapter.format_tools_for_prompt(tools)

        assert "get_weather" in result
        assert "Get weather" in result
        assert "parameters" in result
        assert '<function=function_name>{"param": "value"}</function>' in result

    def test_format_tools_for_prompt_empty_list(self):
        """Empty tool list returns empty string."""
        adapter = LlamaAdapter()
        assert adapter.format_tools_for_prompt([]) == ""

    def test_get_tool_call_stop_tokens(self):
        """Llama tool call stop tokens include <|eom_id|>."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = 500  # <|eom_id|>

        stop_tokens = adapter.get_tool_call_stop_tokens(tokenizer)

        assert 500 in stop_tokens

    def test_get_tool_call_stop_tokens_missing_eom(self):
        """Llama tool call stop tokens handle missing <|eom_id|>."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.side_effect = Exception("Not found")

        stop_tokens = adapter.get_tool_call_stop_tokens(tokenizer)
        assert stop_tokens == []

    def test_get_tool_call_stop_tokens_unk(self):
        """Llama tool call stop tokens skip if eom resolves to unk."""
        adapter = LlamaAdapter()
        tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        tokenizer.unk_token_id = 0
        tokenizer.convert_tokens_to_ids.return_value = 0  # unk

        stop_tokens = adapter.get_tool_call_stop_tokens(tokenizer)
        assert stop_tokens == []

    def test_get_tool_call_stop_tokens_with_processor(self):
        """Llama tool call stop tokens unwrap Processor objects."""
        adapter = LlamaAdapter()
        inner_tokenizer = MagicMock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        inner_tokenizer.unk_token_id = 0
        inner_tokenizer.convert_tokens_to_ids.return_value = 500

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        stop_tokens = adapter.get_tool_call_stop_tokens(processor)
        assert 500 in stop_tokens


# --- DefaultAdapter Comprehensive Tests ---


class TestDefaultAdapterComprehensive:
    """Comprehensive tests for DefaultAdapter to bring coverage to 90%+."""

    def test_family_property(self):
        """DefaultAdapter.family returns 'default'."""
        adapter = DefaultAdapter()
        assert adapter.family == "default"

    def test_apply_chat_template(self):
        """DefaultAdapter uses tokenizer's built-in template."""
        adapter = DefaultAdapter()
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template.return_value = "default_formatted"

        messages = [{"role": "user", "content": "Hello"}]
        result = adapter.apply_chat_template(tokenizer, messages)

        assert result == "default_formatted"

    def test_apply_chat_template_with_processor(self):
        """DefaultAdapter unwraps Processor objects."""
        adapter = DefaultAdapter()
        inner_tokenizer = MagicMock(spec=["apply_chat_template"])
        inner_tokenizer.apply_chat_template.return_value = "formatted"

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        messages = [{"role": "user", "content": "Hi"}]
        result = adapter.apply_chat_template(processor, messages)
        assert result == "formatted"
        inner_tokenizer.apply_chat_template.assert_called_once()

    def test_get_stop_tokens(self):
        """DefaultAdapter returns only eos_token_id."""
        adapter = DefaultAdapter()
        tokenizer = MagicMock(spec=["eos_token_id"])
        tokenizer.eos_token_id = 100

        stop_tokens = adapter.get_stop_tokens(tokenizer)
        assert stop_tokens == [100]

    def test_get_stop_tokens_with_processor(self):
        """DefaultAdapter stop tokens unwrap Processor objects."""
        adapter = DefaultAdapter()
        inner_tokenizer = MagicMock(spec=["eos_token_id"])
        inner_tokenizer.eos_token_id = 100

        processor = MagicMock()
        processor.tokenizer = inner_tokenizer

        stop_tokens = adapter.get_stop_tokens(processor)
        assert stop_tokens == [100]

    def test_supports_tool_calling(self):
        """DefaultAdapter does not support tool calling."""
        adapter = DefaultAdapter()
        assert adapter.supports_tool_calling() is False

    def test_format_tools_for_prompt(self):
        """DefaultAdapter returns empty string for tools."""
        adapter = DefaultAdapter()
        assert adapter.format_tools_for_prompt([{"function": {"name": "test"}}]) == ""

    def test_get_tool_call_stop_tokens(self):
        """DefaultAdapter returns empty list for tool stop tokens."""
        adapter = DefaultAdapter()
        tokenizer = MagicMock()
        assert adapter.get_tool_call_stop_tokens(tokenizer) == []

    def test_has_native_tool_support_always_false(self):
        """DefaultAdapter always returns False for has_native_tool_support."""
        adapter = DefaultAdapter()
        tokenizer = MagicMock(spec=[])
        assert adapter.has_native_tool_support(tokenizer) is False

    def test_supports_reasoning_mode(self):
        """DefaultAdapter does not support reasoning mode."""
        adapter = DefaultAdapter()
        assert adapter.supports_reasoning_mode() is False

    def test_clean_response_removes_special_tokens(self):
        """clean_response removes COMMON_SPECIAL_TOKENS."""
        adapter = DefaultAdapter()
        text = "Hello<|endoftext|> world<|im_end|>"

        result = adapter.clean_response(text)

        assert "<|endoftext|>" not in result
        assert "<|im_end|>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_clean_response_collapses_newlines(self):
        """clean_response collapses excessive newlines."""
        adapter = DefaultAdapter()
        text = "Line 1\n\n\n\n\nLine 2"

        result = adapter.clean_response(text)

        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_clean_response_strips_whitespace(self):
        """clean_response strips leading/trailing whitespace."""
        adapter = DefaultAdapter()
        text = "   \n\nContent here\n\n   "

        result = adapter.clean_response(text)

        assert result == "Content here"

    def test_clean_response_removes_all_common_tokens(self):
        """clean_response removes all tokens in COMMON_SPECIAL_TOKENS list."""
        from mlx_manager.mlx_server.models.adapters.base import COMMON_SPECIAL_TOKENS

        adapter = DefaultAdapter()
        text = "start" + "".join(COMMON_SPECIAL_TOKENS) + "end"

        result = adapter.clean_response(text)

        for token in COMMON_SPECIAL_TOKENS:
            assert token not in result
        assert "start" in result
        assert "end" in result

    def test_convert_messages_assistant_tool_calls_no_content(self):
        """DefaultAdapter handles assistant with tool_calls but no content."""
        adapter = DefaultAdapter()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ]

        result = adapter.convert_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "test_func" in result[0]["content"]
        assert "tool_calls" not in result[0]


# --- Registry Comprehensive Tests ---


class TestRegistryComprehensive:
    """Comprehensive tests for adapter registry to bring coverage to 90%+."""

    def test_detect_model_family_glm4(self):
        """detect_model_family returns 'glm4' for GLM models."""
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("mlx-community/GLM-4.7-Flash-4bit") == "glm4"
        assert detect_model_family("THUDM/chatglm-6b") == "glm4"
        assert detect_model_family("some/glm-model") == "glm4"

    def test_detect_model_family_phi(self):
        """detect_model_family returns 'phi' for Phi models."""
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("microsoft/phi-3-mini-4k-instruct") == "phi"
        assert detect_model_family("mlx-community/Phi-3.5-mini-instruct") == "phi"

    def test_detect_model_family_default(self):
        """detect_model_family returns 'default' for unknown models."""
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("unknown/model-xyz") == "default"

    def test_detect_model_family_llama(self):
        """detect_model_family returns 'llama' for Llama models."""
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("mlx-community/Llama-3.2-3B") == "llama"
        assert detect_model_family("codellama/CodeLlama-7b") == "llama"

    def test_get_adapter_glm4(self):
        """get_adapter returns GLM4Adapter for GLM models."""
        from mlx_manager.mlx_server.models.adapters.registry import get_adapter

        adapter = get_adapter("mlx-community/GLM-4.7-Flash-4bit")
        assert adapter.family == "glm4"

    def test_get_adapter_llama(self):
        """get_adapter returns LlamaAdapter for Llama models."""
        from mlx_manager.mlx_server.models.adapters.registry import get_adapter

        adapter = get_adapter("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert adapter.family == "llama"

    def test_get_adapter_unknown_falls_back_to_default(self):
        """get_adapter returns DefaultAdapter for unknown model families."""
        from mlx_manager.mlx_server.models.adapters.registry import get_adapter

        adapter = get_adapter("totally-unknown/model-abc")
        assert adapter.family == "default"

    def test_get_adapter_phi_falls_back_to_default(self):
        """get_adapter returns DefaultAdapter for phi (no phi adapter registered)."""
        from mlx_manager.mlx_server.models.adapters.registry import get_adapter

        adapter = get_adapter("microsoft/phi-3-mini-4k-instruct")
        # phi detected but no registered adapter -> falls back to default
        assert adapter.family == "default"

    def test_register_adapter(self):
        """register_adapter adds a custom adapter."""
        from mlx_manager.mlx_server.models.adapters.registry import (
            _ADAPTERS,
            register_adapter,
        )

        # Create a mock adapter
        mock_adapter = MagicMock()
        mock_adapter.family = "custom"

        register_adapter("custom", mock_adapter)

        assert _ADAPTERS["custom"] is mock_adapter

        # Cleanup
        del _ADAPTERS["custom"]

    def test_get_supported_families(self):
        """get_supported_families includes all registered families."""
        families = get_supported_families()
        assert "llama" in families
        assert "qwen" in families
        assert "mistral" in families
        assert "gemma" in families
        assert "glm4" in families
        assert "default" in families

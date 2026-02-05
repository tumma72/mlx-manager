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

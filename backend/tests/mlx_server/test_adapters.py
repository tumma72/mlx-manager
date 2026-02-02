"""Tests for model family adapters."""

from unittest.mock import MagicMock

from mlx_manager.mlx_server.models.adapters import get_adapter, get_supported_families
from mlx_manager.mlx_server.models.adapters.gemma import GemmaAdapter
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
        tokenizer = MagicMock()
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
        tokenizer = MagicMock()
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
        tokenizer = MagicMock()
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

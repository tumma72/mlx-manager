"""Tests for model family detection and audio adapter creation.

NOTE: Text/vision adapter-specific tests are in test_composable_adapters.py.
This file tests detect_model_family() and audio adapter basics.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.adapters import detect_model_family


class TestModelFamilyDetection:
    """Tests for detect_model_family function."""

    def test_detect_llama_family(self):
        """Verify Llama models are detected correctly."""
        assert detect_model_family("mlx-community/Llama-3.2-3B-Instruct-4bit") == "llama"
        assert detect_model_family("mlx-community/CodeLlama-7b-hf-4bit") == "llama"

    def test_detect_qwen_family(self):
        """Verify Qwen models are detected correctly."""
        assert detect_model_family("mlx-community/Qwen2.5-7B-Instruct-4bit") == "qwen"
        assert detect_model_family("mlx-community/Qwen3-VL-8B-Instruct-MLX-4bit") == "qwen"

    def test_detect_mistral_family(self):
        """Verify Mistral/Mixtral models are detected correctly."""
        assert detect_model_family("mlx-community/Mistral-7B-Instruct-v0.3-4bit") == "mistral"
        assert detect_model_family("mlx-community/Mixtral-8x7B-Instruct-4bit") == "mistral"

    def test_detect_gemma_family(self):
        """Verify Gemma models are detected correctly."""
        assert detect_model_family("mlx-community/gemma-2-9b-it-4bit") == "gemma"
        assert detect_model_family("mlx-community/gemma-3-27b-it-4bit-DWQ") == "gemma"

    def test_detect_glm4_family(self):
        """Verify GLM4 models are detected correctly."""
        assert detect_model_family("mlx-community/GLM-4.7-Flash-4bit") == "glm4"
        assert detect_model_family("mlx-community/chatglm3-6b-4bit") == "glm4"

    def test_detect_phi_family(self):
        """Verify Phi models are detected correctly."""
        assert detect_model_family("mlx-community/Phi-3-mini-4k-instruct-4bit") == "phi"

    def test_detect_iquest_as_qwen_family(self):
        """Verify IQuest-Coder models are detected as Qwen family."""
        assert detect_model_family("mlx-community/IQuest-Coder-V1-40B-Instruct-4bit") == "qwen"
        assert detect_model_family("mlx-community/IQuest-Coder-V1-40B-Loop-Instruct-4bit") == "qwen"
        assert detect_model_family("IQuestLab/IQuest-Coder-V1-40B-Instruct") == "qwen"

    def test_detect_unknown_family(self):
        """Verify unknown models fall back to default."""
        assert detect_model_family("mlx-community/unknown-model-4bit") == "default"

    def test_detect_whisper_family(self):
        """Verify Whisper models are detected correctly."""
        assert detect_model_family("mlx-community/whisper-large-v3-turbo") == "whisper"
        assert detect_model_family("openai/whisper-small") == "whisper"

    def test_detect_kokoro_family(self):
        """Verify Kokoro models are detected correctly."""
        assert detect_model_family("mlx-community/Kokoro-82M-bf16") == "kokoro"
        assert detect_model_family("mlx-community/Kokoro-82M-4bit") == "kokoro"


class TestAudioAdapterCreation:
    """Tests for audio adapter creation with tokenizer=None."""

    def test_create_whisper_adapter_no_tokenizer(self):
        """Audio adapters should work with tokenizer=None."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter(family="whisper", tokenizer=None)
        assert adapter.family == "whisper"
        assert adapter.stop_tokens == []
        assert adapter.tool_parser.parser_id == "null"
        assert adapter.thinking_parser.parser_id == "null"

    def test_create_kokoro_adapter_no_tokenizer(self):
        """Kokoro adapter works with tokenizer=None."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter(family="kokoro", tokenizer=None)
        assert adapter.family == "kokoro"
        assert adapter.stop_tokens == []

    def test_create_audio_default_adapter(self):
        """Default audio adapter works with tokenizer=None."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter(family="audio_default", tokenizer=None)
        assert adapter.family == "audio_default"
        assert adapter.stop_tokens == []

    @pytest.mark.asyncio
    async def test_whisper_post_load_configure_fixes_processor(self):
        """WhisperAdapter.post_load_configure loads processor when missing."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter(family="whisper", tokenizer=None)

        mock_model = MagicMock()
        mock_model._processor = None  # Missing processor

        mock_processor = MagicMock()
        with patch(
            "mlx_manager.mlx_server.models.adapters.strategies.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=mock_processor,
        ):
            await adapter.post_load_configure(mock_model, "mlx-community/whisper-large-v3-turbo")

        assert mock_model._processor is mock_processor

    @pytest.mark.asyncio
    async def test_whisper_post_load_configure_skips_when_processor_ok(self):
        """WhisperAdapter.post_load_configure is a no-op when processor works."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter(family="whisper", tokenizer=None)

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 51865  # Normal Whisper vocab size
        mock_processor = MagicMock()
        mock_processor.tokenizer = mock_tokenizer

        mock_model = MagicMock()
        mock_model._processor = mock_processor

        with patch(
            "mlx_manager.mlx_server.models.adapters.strategies.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_thread:
            await adapter.post_load_configure(mock_model, "mlx-community/whisper-large-v3-turbo")

        # Should not attempt to load from canonical repo
        mock_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_whisper_post_load_configure_handles_failure(self):
        """WhisperAdapter.post_load_configure handles load failure gracefully."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter(family="whisper", tokenizer=None)

        mock_model = MagicMock()
        mock_model._processor = None

        with patch(
            "mlx_manager.mlx_server.models.adapters.strategies.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            # Should not raise
            await adapter.post_load_configure(mock_model, "mlx-community/whisper-large-v3-turbo")

        # Processor should remain None
        assert mock_model._processor is None

    @pytest.mark.asyncio
    async def test_text_adapter_post_load_configure_noop(self):
        """Text adapters' post_load_configure is a no-op."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        adapter = create_adapter(family="default", tokenizer=mock_tokenizer)

        mock_model = MagicMock()
        # Should complete without error
        await adapter.post_load_configure(mock_model, "test/model")

"""Tests for model family detection.

NOTE: Adapter-specific tests are in test_composable_adapters.py.
This file only tests detect_model_family() which remains in registry.py.
"""

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

    def test_detect_unknown_family(self):
        """Verify unknown models fall back to default."""
        assert detect_model_family("mlx-community/unknown-model-4bit") == "default"

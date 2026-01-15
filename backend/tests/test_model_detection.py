"""Tests for the model detection utility."""

import json
from unittest.mock import patch

from mlx_manager.utils.model_detection import (
    AVAILABLE_PARSERS,
    MODEL_PARSER_CONFIGS,
    detect_model_family,
    get_local_model_path,
    get_model_detection_info,
    get_parser_options,
    read_model_config,
)


class TestModelParserConfigs:
    """Tests for model parser configuration constants."""

    def test_minimax_config_exists(self):
        """Test MiniMax parser config exists."""
        assert "minimax" in MODEL_PARSER_CONFIGS
        config = MODEL_PARSER_CONFIGS["minimax"]
        assert config["tool_call_parser"] == "minimax_m2"
        assert config["reasoning_parser"] == "minimax_m2"
        assert config["message_converter"] == "minimax_m2"

    def test_qwen_config_exists(self):
        """Test Qwen parser config exists."""
        assert "qwen" in MODEL_PARSER_CONFIGS
        config = MODEL_PARSER_CONFIGS["qwen"]
        assert config["tool_call_parser"] == "qwen3"
        assert config["reasoning_parser"] == "qwen3"
        assert config["message_converter"] == "qwen3"

    def test_glm_config_exists(self):
        """Test GLM parser config exists."""
        assert "glm" in MODEL_PARSER_CONFIGS
        config = MODEL_PARSER_CONFIGS["glm"]
        assert config["tool_call_parser"] == "glm4"
        assert config["reasoning_parser"] == "glm4"
        assert config["message_converter"] == "glm4"

    def test_available_parsers(self):
        """Test available parsers list contains expected values."""
        assert "minimax_m2" in AVAILABLE_PARSERS
        assert "qwen3" in AVAILABLE_PARSERS
        assert "glm4" in AVAILABLE_PARSERS
        assert "hermes" in AVAILABLE_PARSERS
        assert "llama" in AVAILABLE_PARSERS
        assert "mistral" in AVAILABLE_PARSERS


class TestGetLocalModelPath:
    """Tests for get_local_model_path function."""

    def test_returns_none_when_model_not_downloaded(self, tmp_path):
        """Test returns None when model is not in cache."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_local_model_path("mlx-community/nonexistent-model")
            assert result is None

    def test_returns_snapshot_path_when_downloaded(self, tmp_path):
        """Test returns snapshot path when model is downloaded."""
        # Create mock cache structure
        model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots"
        snapshot_dir = model_dir / "abc123"
        snapshot_dir.mkdir(parents=True)

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_local_model_path("mlx-community/test-model")
            assert result == snapshot_dir

    def test_returns_most_recent_snapshot(self, tmp_path):
        """Test returns most recent snapshot when multiple exist."""
        import time

        # Create mock cache structure with multiple snapshots
        model_dir = tmp_path / "models--mlx-community--multi-model" / "snapshots"
        old_snapshot = model_dir / "old123"
        new_snapshot = model_dir / "new456"
        old_snapshot.mkdir(parents=True)
        time.sleep(0.01)  # Ensure different mtime
        new_snapshot.mkdir(parents=True)

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_local_model_path("mlx-community/multi-model")
            assert result == new_snapshot


class TestReadModelConfig:
    """Tests for read_model_config function."""

    def test_returns_none_when_model_not_downloaded(self, tmp_path):
        """Test returns None when model is not downloaded."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = read_model_config("mlx-community/nonexistent")
            assert result is None

    def test_returns_none_when_config_not_exists(self, tmp_path):
        """Test returns None when config.json doesn't exist."""
        # Create model directory without config.json
        model_dir = tmp_path / "models--mlx-community--no-config" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = read_model_config("mlx-community/no-config")
            assert result is None

    def test_returns_config_when_exists(self, tmp_path):
        """Test returns parsed config.json contents."""
        # Create model directory with config.json
        model_dir = tmp_path / "models--mlx-community--has-config" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        config = {"model_type": "minimax", "architectures": ["MinimaxLMForCausalLM"]}
        (model_dir / "config.json").write_text(json.dumps(config))

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = read_model_config("mlx-community/has-config")
            assert result == config

    def test_returns_none_on_invalid_json(self, tmp_path):
        """Test returns None when config.json is invalid."""
        model_dir = tmp_path / "models--mlx-community--bad-json" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("not valid json")

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = read_model_config("mlx-community/bad-json")
            assert result is None


class TestDetectModelFamily:
    """Tests for detect_model_family function."""

    def test_detects_minimax_from_model_type(self, tmp_path):
        """Test detects MiniMax from model_type field."""
        model_dir = tmp_path / "models--mlx-community--MiniMax-M2" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        config = {"model_type": "minimax"}
        (model_dir / "config.json").write_text(json.dumps(config))

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = detect_model_family("mlx-community/MiniMax-M2")
            assert result == "minimax"

    def test_detects_qwen_from_model_type(self, tmp_path):
        """Test detects Qwen from model_type field."""
        model_dir = tmp_path / "models--mlx-community--Qwen3-8B" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        config = {"model_type": "qwen2"}
        (model_dir / "config.json").write_text(json.dumps(config))

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            # Note: qwen2 contains "qwen" so it should match
            result = detect_model_family("mlx-community/Qwen3-8B")
            assert result == "qwen"

    def test_detects_glm_from_architectures(self, tmp_path):
        """Test detects GLM from architectures field."""
        model_dir = tmp_path / "models--mlx-community--GLM4-9B" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        config = {"architectures": ["Glm4ForCausalLM"]}
        (model_dir / "config.json").write_text(json.dumps(config))

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = detect_model_family("mlx-community/GLM4-9B")
            assert result == "glm"

    def test_detects_from_model_path_fallback(self, tmp_path):
        """Test detects from model path when not downloaded."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            # Model not downloaded, should fall back to path matching
            result = detect_model_family("mlx-community/MiniMax-M2.1-3bit")
            assert result == "minimax"

    def test_detects_qwen_from_path(self, tmp_path):
        """Test detects Qwen from model path."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = detect_model_family("mlx-community/Qwen2.5-72B-4bit")
            assert result == "qwen"

    def test_returns_none_for_unknown_model(self, tmp_path):
        """Test returns None for unknown model family."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = detect_model_family("mlx-community/Llama-3-8B-4bit")
            assert result is None


class TestGetParserOptions:
    """Tests for get_parser_options function."""

    def test_returns_minimax_options(self, tmp_path):
        """Test returns MiniMax parser options."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_parser_options("mlx-community/MiniMax-M2.1-3bit")
            assert result == MODEL_PARSER_CONFIGS["minimax"]

    def test_returns_empty_dict_for_unknown(self, tmp_path):
        """Test returns empty dict for unknown model."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_parser_options("mlx-community/Llama-3-8B")
            assert result == {}


class TestGetModelDetectionInfo:
    """Tests for get_model_detection_info function."""

    def test_returns_full_detection_info(self, tmp_path):
        """Test returns complete detection info."""
        # Create downloaded model
        model_dir = tmp_path / "models--mlx-community--MiniMax-M2" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        config = {"model_type": "minimax"}
        (model_dir / "config.json").write_text(json.dumps(config))

        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_model_detection_info("mlx-community/MiniMax-M2")

            assert result["model_family"] == "minimax"
            assert result["recommended_options"] == MODEL_PARSER_CONFIGS["minimax"]
            assert result["is_downloaded"] is True
            assert result["available_parsers"] == AVAILABLE_PARSERS

    def test_returns_info_for_not_downloaded(self, tmp_path):
        """Test returns info for model not downloaded."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_model_detection_info("mlx-community/MiniMax-M2.1-3bit")

            assert result["model_family"] == "minimax"
            assert result["is_downloaded"] is False
            assert result["available_parsers"] == AVAILABLE_PARSERS

    def test_returns_info_for_unknown_model(self, tmp_path):
        """Test returns info for unknown model family."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_model_detection_info("mlx-community/Unknown-Model")

            assert result["model_family"] is None
            assert result["recommended_options"] == {}
            assert result["is_downloaded"] is False

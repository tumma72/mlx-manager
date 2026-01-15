"""Tests for the model detection utility."""

import json
from unittest.mock import patch

from mlx_manager.utils.model_detection import (
    MODEL_FAMILY_MIN_VERSIONS,
    check_mlx_lm_support,
    detect_model_family,
    get_local_model_path,
    get_mlx_lm_version,
    get_model_detection_info,
    get_parser_options,
    parse_version,
    read_model_config,
)


class TestFuzzyParserOptions:
    """Tests for fuzzy parser options matching (replaces static MODEL_PARSER_CONFIGS)."""

    def test_minimax_m2_gets_all_options(self):
        """Test MiniMax M2 gets all parser options via fuzzy matching."""
        options = get_parser_options("mlx-community/MiniMax-M2.1-3bit")
        assert options.get("tool_call_parser") == "minimax_m2"
        assert options.get("reasoning_parser") == "minimax_m2"
        assert options.get("message_converter") == "minimax_m2"

    def test_qwen3_base_gets_tool_and_reasoning(self):
        """Test base Qwen3 gets tool/reasoning but NOT message_converter."""
        options = get_parser_options("mlx-community/Qwen3-8B-4bit")
        assert options.get("tool_call_parser") == "qwen3"
        assert options.get("reasoning_parser") == "qwen3"
        # Base Qwen3 should NOT get message_converter (no "coder" in name)
        assert "message_converter" not in options

    def test_qwen3_coder_gets_all_options(self):
        """Test Qwen3 Coder gets all options including message_converter."""
        options = get_parser_options("mlx-community/Qwen3-Coder-7B-4bit")
        assert options.get("tool_call_parser") == "qwen3_coder"
        assert options.get("message_converter") == "qwen3_coder"

    def test_glm_gets_options_via_single_option_fallback(self):
        """Test GLM gets parser options via single-option family fallback."""
        options = get_parser_options("mlx-community/GLM-4-9B")
        assert options.get("tool_call_parser") == "glm4_moe"
        assert options.get("reasoning_parser") == "glm4_moe"
        # Message converter not matched (no "moe" variant in model name)
        assert "message_converter" not in options

    def test_nemotron_gets_options_via_single_option_fallback(self):
        """Test Nemotron gets parser options via single-option family fallback."""
        options = get_parser_options("mlx-community/Nemotron-3-8B")
        assert options.get("tool_call_parser") == "nemotron3_nano"
        assert options.get("reasoning_parser") == "nemotron3_nano"
        # Message converter not matched (no "nano" variant in model name)
        assert "message_converter" not in options

    def test_unknown_model_returns_empty(self):
        """Test unknown model family returns empty dict."""
        options = get_parser_options("mlx-community/Llama-3.1-70B-4bit")
        assert options == {}


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
            # Note: qwen2 contains "qwen" so it should match qwen3 family
            result = detect_model_family("mlx-community/Qwen3-8B")
            assert result == "qwen3"

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
            assert result == "qwen3"

    def test_detects_qwen_coder_from_path(self, tmp_path):
        """Test detects Qwen Coder from model path."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = detect_model_family("mlx-community/Qwen2.5-Coder-32B-4bit")
            assert result == "qwen3_coder"

    def test_returns_none_for_unknown_model(self, tmp_path):
        """Test returns None for unknown model family."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = detect_model_family("mlx-community/Llama-3-8B-4bit")
            assert result is None


class TestGetParserOptions:
    """Tests for get_parser_options function (uses fuzzy matching)."""

    def test_returns_minimax_options(self, tmp_path):
        """Test returns MiniMax parser options via fuzzy matching."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_parser_options("mlx-community/MiniMax-M2.1-3bit")
            assert result.get("tool_call_parser") == "minimax_m2"
            assert result.get("reasoning_parser") == "minimax_m2"
            assert result.get("message_converter") == "minimax_m2"

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
            # get_parser_options filters out None values
            assert result["recommended_options"]["tool_call_parser"] == "minimax_m2"
            assert result["is_downloaded"] is True

    def test_returns_info_for_not_downloaded(self, tmp_path):
        """Test returns info for model not downloaded."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_model_detection_info("mlx-community/MiniMax-M2.1-3bit")

            assert result["model_family"] == "minimax"
            assert result["is_downloaded"] is False

    def test_returns_info_for_unknown_model(self, tmp_path):
        """Test returns info for unknown model family."""
        with patch("mlx_manager.utils.model_detection.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            result = get_model_detection_info("mlx-community/Unknown-Model")

            assert result["model_family"] is None
            assert result["recommended_options"] == {}
            assert result["is_downloaded"] is False


class TestVersionParsing:
    """Tests for version parsing utilities."""

    def test_parse_version_simple(self):
        """Test parsing simple version string."""
        assert parse_version("0.28.4") == (0, 28, 4)

    def test_parse_version_two_parts(self):
        """Test parsing two-part version string."""
        assert parse_version("1.0") == (1, 0)

    def test_parse_version_single(self):
        """Test parsing single version number."""
        assert parse_version("2") == (2,)

    def test_parse_version_invalid(self):
        """Test parsing invalid version returns (0,)."""
        assert parse_version("invalid") == (0,)


class TestGetMlxLmVersion:
    """Tests for get_mlx_lm_version function."""

    def test_returns_version_when_installed(self):
        """Test returns version string when mlx-lm is installed."""
        with patch("mlx_manager.utils.model_detection.version") as mock_version:
            mock_version.return_value = "0.30.2"
            result = get_mlx_lm_version()
            assert result == "0.30.2"
            mock_version.assert_called_once_with("mlx-lm")

    def test_returns_none_when_not_installed(self):
        """Test returns None when mlx-lm is not installed."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            "mlx_manager.utils.model_detection.version",
            side_effect=PackageNotFoundError(),
        ):
            result = get_mlx_lm_version()
            assert result is None


class TestCheckMlxLmSupport:
    """Tests for check_mlx_lm_support function."""

    def test_minimax_supported_with_new_version(self):
        """Test MiniMax is supported with mlx-lm >= 0.28.4."""
        with patch(
            "mlx_manager.utils.model_detection.get_mlx_lm_version",
            return_value="0.30.2",
        ):
            result = check_mlx_lm_support("minimax")
            assert result["supported"] is True
            assert result["installed_version"] == "0.30.2"
            assert result["required_version"] == "0.28.4"

    def test_minimax_not_supported_with_old_version(self):
        """Test MiniMax is not supported with old mlx-lm."""
        with patch(
            "mlx_manager.utils.model_detection.get_mlx_lm_version",
            return_value="0.28.0",
        ):
            result = check_mlx_lm_support("minimax")
            assert result["supported"] is False
            assert result["installed_version"] == "0.28.0"
            assert result["required_version"] == "0.28.4"
            assert "error" in result
            assert "0.28.4" in result["error"]
            assert "upgrade_command" in result

    def test_minimax_exact_version_supported(self):
        """Test MiniMax is supported with exact minimum version."""
        with patch(
            "mlx_manager.utils.model_detection.get_mlx_lm_version",
            return_value="0.28.4",
        ):
            result = check_mlx_lm_support("minimax")
            assert result["supported"] is True

    def test_not_installed_returns_error(self):
        """Test returns error when mlx-lm is not installed."""
        with patch(
            "mlx_manager.utils.model_detection.get_mlx_lm_version",
            return_value=None,
        ):
            result = check_mlx_lm_support("minimax")
            assert result["supported"] is False
            assert result["installed_version"] is None
            assert "error" in result
            assert "not installed" in result["error"]

    def test_unknown_family_always_supported(self):
        """Test unknown model families have no version requirement."""
        with patch(
            "mlx_manager.utils.model_detection.get_mlx_lm_version",
            return_value="0.20.0",
        ):
            result = check_mlx_lm_support("unknown_family")
            assert result["supported"] is True
            assert result["required_version"] is None

    def test_qwen_no_version_requirement(self):
        """Test Qwen has no minimum version requirement."""
        with patch(
            "mlx_manager.utils.model_detection.get_mlx_lm_version",
            return_value="0.20.0",
        ):
            result = check_mlx_lm_support("qwen")
            assert result["supported"] is True


class TestModelFamilyMinVersions:
    """Tests for MODEL_FAMILY_MIN_VERSIONS constant."""

    def test_minimax_has_version_requirement(self):
        """Test MiniMax has a minimum version requirement."""
        assert "minimax" in MODEL_FAMILY_MIN_VERSIONS
        assert MODEL_FAMILY_MIN_VERSIONS["minimax"] == "0.28.4"

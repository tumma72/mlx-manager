"""Tests for audio model type detection.

Verifies that detect_model_type() correctly identifies audio models
via config fields, architecture strings, model_type values, and name patterns.
"""

import pytest

from mlx_manager.mlx_server.models.detection import detect_model_type
from mlx_manager.mlx_server.models.types import ModelType

# ── Config-based detection (highest priority) ──────────────────────────


@pytest.mark.parametrize(
    "config_key",
    [
        "audio_config",
        "tts_config",
        "stt_config",
        "vocoder_config",
        "codec_config",
    ],
)
def test_audio_detected_from_config_fields(config_key: str) -> None:
    """Audio models with specific config keys should be detected as AUDIO."""
    config = {config_key: {"some": "value"}}
    result = detect_model_type("some-org/some-model", config=config)
    assert result == ModelType.AUDIO


# ── Architecture-based detection ───────────────────────────────────────


@pytest.mark.parametrize(
    "architecture",
    [
        "KokoroModel",
        "WhisperForConditionalGeneration",
        "BarkModel",
        "SpeechT5ForTextToSpeech",
        "ParlerTTSForConditionalGeneration",
        "SesameModel",
        "SparkTTSModel",
        "DiaModel",
        "OuteTTSModel",
        "ChatterboxModel",
        "ParakeetCTCModel",
        "VoxtralSTTModel",
        "VibevoiceTTSModel",
        "VoxCPMModel",
        "SopranoModel",
    ],
)
def test_audio_detected_from_architecture(architecture: str) -> None:
    """Audio architectures should be detected as AUDIO."""
    config = {"architectures": [architecture]}
    result = detect_model_type("some-org/some-model", config=config)
    assert result == ModelType.AUDIO


# ── model_type field detection ─────────────────────────────────────────


@pytest.mark.parametrize(
    "model_type_value",
    [
        "kokoro",
        "whisper",
        "bark",
        "speecht5",
        "parler",
        "dia",
        "outetts",
        "spark",
        "chatterbox",
        "soprano",
        "parakeet",
        "qwen3_tts",
        "qwen3_asr",
        "glm4_voice",
    ],
)
def test_audio_detected_from_model_type_field(model_type_value: str) -> None:
    """Audio model_type values should be detected as AUDIO."""
    config = {"model_type": model_type_value}
    result = detect_model_type("some-org/some-model", config=config)
    assert result == ModelType.AUDIO


# ── Name-based fallback detection ──────────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    [
        "mlx-community/Kokoro-82M-4bit",
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/bark-small",
        "mlx-community/speecht5-tts",
        "mlx-community/parler-tts-large",
        "some-org/my-tts-model",
        "some-org/my-stt-model",
        "some-org/speech-to-text-v1",
        "mlx-community/chatterbox-tts-0.1",
        "mlx-community/dia-1.6B-4bit",
        "mlx-community/outetts-0.3",
        "mlx-community/spark-tts-0.5B",
        "mlx-community/parakeet-ctc-0.6b",
        "mlx-community/voxtral-mini",
        "mlx-community/vibevoice-v1",
        "mlx-community/voxcpm-v1",
        "mlx-community/soprano-v1",
        "mlx-community/descript-audio-codec",
        "mlx-community/snac_24khz",
        "mlx-community/vocos-mel-24khz",
    ],
)
def test_audio_detected_from_name_pattern(model_id: str) -> None:
    """Audio model names should be detected as AUDIO (no config available)."""
    result = detect_model_type(model_id, config={})
    assert result == ModelType.AUDIO


# ── Negative cases: should NOT be detected as AUDIO ───────────────────


@pytest.mark.parametrize(
    "model_id,config,expected",
    [
        # Text-gen model
        (
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            {"architectures": ["LlamaForCausalLM"]},
            ModelType.TEXT_GEN,
        ),
        # Vision model
        (
            "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            {"vision_config": {}, "architectures": ["Qwen2VLForConditionalGeneration"]},
            ModelType.VISION,
        ),
        # Embeddings model
        (
            "mlx-community/all-MiniLM-L6-v2-4bit",
            {"architectures": ["BertModel"]},
            ModelType.EMBEDDINGS,
        ),
        # Generic unknown model defaults to text-gen
        (
            "some-org/custom-model-v1",
            {},
            ModelType.TEXT_GEN,
        ),
    ],
)
def test_non_audio_models_not_detected_as_audio(
    model_id: str, config: dict, expected: ModelType
) -> None:
    """Non-audio models should not be incorrectly classified as AUDIO."""
    result = detect_model_type(model_id, config=config)
    assert result == expected


# ── Real-world model IDs ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id,config,expected",
    [
        # Kokoro TTS (primary reference model)
        (
            "mlx-community/Kokoro-82M-4bit",
            {"model_type": "kokoro"},
            ModelType.AUDIO,
        ),
        # Whisper STT
        (
            "mlx-community/whisper-large-v3-turbo-asr-fp16",
            {"model_type": "whisper"},
            ModelType.AUDIO,
        ),
        # Text model must not be audio
        (
            "mlx-community/Qwen3-0.6B-4bit-DWQ",
            {"architectures": ["Qwen2ForCausalLM"]},
            ModelType.TEXT_GEN,
        ),
    ],
)
def test_real_world_models(model_id: str, config: dict, expected: ModelType) -> None:
    """Real-world model IDs should be classified correctly."""
    result = detect_model_type(model_id, config=config)
    assert result == expected


# ── Config loading error handling ─────────────────────────────────────


class TestConfigLoadingFallback:
    """Tests for config loading error handling and fallback paths."""

    def test_config_loading_exception_falls_back_to_name(self) -> None:
        """When config loading raises, fall back to name-based detection."""
        from unittest.mock import patch

        with patch(
            "mlx_manager.utils.model_detection.read_model_config",
            side_effect=RuntimeError("Read error"),
        ):
            # Audio model name should still be detected
            result = detect_model_type("mlx-community/Kokoro-82M-4bit")
            assert result == ModelType.AUDIO

    def test_config_none_falls_through_to_name_detection(self) -> None:
        """When config is None (not downloaded), name patterns are used."""
        from unittest.mock import patch

        with patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value=None,
        ):
            result = detect_model_type("mlx-community/all-MiniLM-L6-v2-4bit")
            assert result == ModelType.EMBEDDINGS


# ── Config-based embeddings detection (model_type field) ─────────────


@pytest.mark.parametrize(
    "model_type_value",
    ["embedding", "sentence-transformers", "bert"],
)
def test_embeddings_detected_from_model_type_field(model_type_value: str) -> None:
    """Config model_type with embedding indicators -> EMBEDDINGS."""
    config = {"model_type": model_type_value}
    result = detect_model_type("some-org/some-model", config=config)
    assert result == ModelType.EMBEDDINGS


# ── Name-based vision detection ───────────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    [
        "mlx-community/some-vlm-model",
        "mlx-community/llava-1.5-7b-4bit",
        "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        "mlx-community/pixtral-12b-4bit",
        "mlx-community/gemma-3-27b-it-4bit",
    ],
)
def test_vision_detected_from_name_pattern(model_id: str) -> None:
    """Vision model names should be detected as VISION (no config)."""
    result = detect_model_type(model_id, config={})
    assert result == ModelType.VISION


# ── Name-based embeddings detection ──────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    [
        "mlx-community/all-MiniLM-L6-v2-4bit",
        "mlx-community/e5-small-v2",
        "mlx-community/bge-small-en-v1.5",
        "mlx-community/gte-small-4bit",
        "mlx-community/sentence-transformers-test",
    ],
)
def test_embeddings_detected_from_name_pattern(model_id: str) -> None:
    """Embeddings model names should be detected as EMBEDDINGS (no config)."""
    result = detect_model_type(model_id, config={})
    assert result == ModelType.EMBEDDINGS


# ── Regression tests: false positive fixes ────────────────────────────


@pytest.mark.parametrize(
    "model_id,config,expected",
    [
        # GLM-4.7-Flash regression: should be TEXT_GEN, not AUDIO
        # Config has model_type "glm4_moe_lite" which should not trigger audio detection
        (
            "mlx-community/GLM-4.7-Flash-4bit",
            {"model_type": "glm4_moe_lite", "architectures": ["Glm4MoeLiteForCausalLM"]},
            ModelType.TEXT_GEN,
        ),
        # Nemotron regression: empty config + "nvidia" in name (contains "dia")
        # Should be TEXT_GEN, not AUDIO
        (
            "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
            {},
            ModelType.TEXT_GEN,
        ),
    ],
)
def test_regression_false_positive_fixes(model_id: str, config: dict, expected: ModelType) -> None:
    """Regression tests for models that were incorrectly detected as AUDIO.

    - GLM-4.7-Flash has model_type "glm4_moe_lite" (not "glm4_voice"), should be TEXT_GEN
    - Nemotron contains "nvidia" which contains "dia" substring, but should be TEXT_GEN
    """
    result = detect_model_type(model_id, config=config)
    assert result == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "mlx-community/descript-audio-codec",
        "mlx-community/snac_24khz",
        "mlx-community/vocos-mel-24khz",
    ],
)
def test_audio_codecs_detected_from_name(model_id: str) -> None:
    """Audio codec models should be detected as AUDIO from name patterns.

    These are specialized audio processing models (codecs/vocoders) that should
    be classified as AUDIO even without config information.
    """
    result = detect_model_type(model_id, config={})
    assert result == ModelType.AUDIO

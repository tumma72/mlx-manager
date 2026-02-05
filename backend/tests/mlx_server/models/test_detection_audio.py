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
        "glm",
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
def test_real_world_models(
    model_id: str, config: dict, expected: ModelType
) -> None:
    """Real-world model IDs should be classified correctly."""
    result = detect_model_type(model_id, config=config)
    assert result == expected

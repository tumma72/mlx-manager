"""Tests for mlx_server/models/detection.py uncovered lines.

Targets: lines 213-214, 224, 234, 238-266, 287, 290, 319, 332
"""

from unittest.mock import patch

from mlx_manager.mlx_server.models.detection import (
    TypeDetectionResult,
    detect_model_type,
    detect_model_type_detailed,
)
from mlx_manager.mlx_server.models.types import ModelType


class TestDetectModelTypeDetailed:
    """Tests for detect_model_type_detailed() function covering uncovered lines."""

    def test_returns_vision_from_config_field(self):
        """Vision detected via config_field when multimodal=vision (line 224)."""
        config = {"vision_config": {"image_size": 224}, "model_type": "llava"}
        result = detect_model_type_detailed("test/vision-model", config=config)
        assert result.model_type == ModelType.VISION
        assert result.detection_method == "config_field"

    def test_returns_audio_from_config_field_audio_config(self):
        """Audio detected via config_field audio_config key (line 234)."""
        config = {"audio_config": {"sample_rate": 24000}}
        result = detect_model_type_detailed("test/audio-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "config_field"

    def test_returns_audio_from_config_field_tts_config(self):
        """Audio detected via config_field tts_config key (line 234)."""
        config = {"tts_config": {"voice": "en"}}
        result = detect_model_type_detailed("test/tts-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "config_field"

    def test_returns_audio_from_config_field_stt_config(self):
        """Audio detected via config_field stt_config key (line 234)."""
        config = {"stt_config": {"language": "en"}}
        result = detect_model_type_detailed("test/stt-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "config_field"

    def test_returns_audio_from_architecture_kokoro(self):
        """Audio detected via architecture for kokoro (lines 238-257)."""
        config = {"architectures": ["KokoroTTSModel"]}
        result = detect_model_type_detailed("test/kokoro-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "architecture"
        assert result.architecture == "KokoroTTSModel"

    def test_returns_audio_from_architecture_whisper(self):
        """Audio detected via architecture for whisper (lines 238-257)."""
        config = {"architectures": ["WhisperForConditionalGeneration"]}
        result = detect_model_type_detailed("test/whisper-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "architecture"

    def test_returns_embeddings_from_architecture(self):
        """Embeddings detected via architecture (lines 259-261)."""
        config = {"architectures": ["BertForSentenceEmbedding"]}
        result = detect_model_type_detailed("test/bert-model", config=config)
        assert result.model_type == ModelType.EMBEDDINGS
        assert result.detection_method == "architecture"

    def test_returns_text_gen_from_causal_lm_architecture(self):
        """Text-gen detected via CausalLM architecture (lines 263-266)."""
        config = {"architectures": ["LlamaForCausalLM"]}
        result = detect_model_type_detailed("test/llama-model", config=config)
        assert result.model_type == ModelType.TEXT_GEN
        assert result.detection_method == "architecture"

    def test_returns_text_gen_from_conditional_generation_architecture(self):
        """Text-gen detected via ForConditionalGeneration architecture (line 265)."""
        config = {"architectures": ["T5ForConditionalGeneration"]}
        result = detect_model_type_detailed("test/t5-model", config=config)
        assert result.model_type == ModelType.TEXT_GEN
        assert result.detection_method == "architecture"

    def test_returns_audio_from_model_type_field(self):
        """Audio detected via model_type field (line 287)."""
        config = {"model_type": "kokoro"}
        result = detect_model_type_detailed("test/kokoro-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "config_field"

    def test_returns_audio_from_model_type_whisper(self):
        """Audio detected via model_type=whisper (line 287)."""
        config = {"model_type": "whisper"}
        result = detect_model_type_detailed("test/whisper-base", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "config_field"

    def test_returns_embeddings_from_model_type_field(self):
        """Embeddings detected via model_type containing 'bert' (line 290)."""
        config = {"model_type": "bert"}
        result = detect_model_type_detailed("test/bert-base", config=config)
        assert result.model_type == ModelType.EMBEDDINGS
        assert result.detection_method == "config_field"

    def test_returns_embeddings_from_model_type_sentence(self):
        """Embeddings detected via model_type containing 'sentence' (line 290)."""
        config = {"model_type": "sentence_transformers"}
        result = detect_model_type_detailed("test/sentence-model", config=config)
        assert result.model_type == ModelType.EMBEDDINGS
        assert result.detection_method == "config_field"

    def test_returns_audio_from_name_pattern_kokoro(self):
        """Audio detected via name pattern when no config (line 319)."""
        result = detect_model_type_detailed("mlx-community/Kokoro-82M-4bit")
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "name_pattern"

    def test_returns_audio_from_name_pattern_tts(self):
        """Audio detected via 'tts' name pattern (line 319)."""
        result = detect_model_type_detailed("some/whisper-tts-model")
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "name_pattern"

    def test_returns_vision_from_name_pattern(self):
        """Vision detected via name pattern when no config (line 332)."""
        result = detect_model_type_detailed("mlx-community/Qwen2-VL-2B-4bit")
        assert result.model_type == ModelType.VISION
        assert result.detection_method == "name_pattern"

    def test_returns_vision_from_vlm_pattern(self):
        """Vision detected via 'vlm' name pattern (line 332)."""
        result = detect_model_type_detailed("test/some-vlm-model")
        assert result.model_type == ModelType.VISION
        assert result.detection_method == "name_pattern"

    def test_returns_embeddings_from_name_pattern(self):
        """Embeddings detected via 'minilm' name pattern when no config available."""
        # MiniLM has architecture "BertModel" which triggers architecture detection
        # Use a model name without a downloaded config to force name-pattern detection
        result = detect_model_type_detailed("test/all-minilm-embed-model")
        assert result.model_type == ModelType.EMBEDDINGS
        assert result.detection_method == "name_pattern"

    def test_defaults_to_text_gen(self):
        """Defaults to TEXT_GEN when no pattern matches."""
        result = detect_model_type_detailed("mlx-community/some-regular-llm")
        assert result.model_type == ModelType.TEXT_GEN
        assert result.detection_method == "default"

    def test_handles_exception_loading_config(self):
        """Config load exception results in None config (lines 213-214)."""
        with patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed.__wrapped__"
            if hasattr(detect_model_type_detailed, "__wrapped__")
            else "mlx_manager.mlx_server.models.detection.detect_model_type_detailed"
        ):
            # Patch read_model_config to raise an exception
            with patch(
                "mlx_manager.utils.model_detection.read_model_config",
                side_effect=OSError("disk error"),
            ):
                # Should fall back to name-based detection since config fails
                result = detect_model_type_detailed("mlx-community/Kokoro-82M-4bit")
                assert result.model_type == ModelType.AUDIO
                assert result.detection_method == "name_pattern"

    def test_returns_architecture_string_from_config(self):
        """Architecture string is populated from config.json architectures."""
        config = {"architectures": ["QwenForCausalLM"], "model_type": "qwen2"}
        result = detect_model_type_detailed("test/qwen-model", config=config)
        assert result.architecture == "QwenForCausalLM"

    def test_architecture_empty_when_no_config(self):
        """Architecture is empty string when no config available."""
        result = detect_model_type_detailed("mlx-community/some-llm-model-7b")
        assert result.architecture == ""

    def test_architecture_empty_when_no_architectures_key(self):
        """Architecture is empty string when architectures key missing."""
        config = {"model_type": "llama"}
        result = detect_model_type_detailed("test/model", config=config)
        assert result.architecture == ""

    def test_vocoder_config_detected_as_audio(self):
        """vocoder_config triggers audio detection (line 234)."""
        config = {"vocoder_config": {"vocoder_type": "hifigan"}}
        result = detect_model_type_detailed("test/vocoder-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "config_field"

    def test_codec_config_detected_as_audio(self):
        """codec_config triggers audio detection (line 234)."""
        config = {"codec_config": {"codec_type": "dac"}}
        result = detect_model_type_detailed("test/codec-model", config=config)
        assert result.model_type == ModelType.AUDIO
        assert result.detection_method == "config_field"

    def test_embed_name_pattern_returns_embeddings(self):
        """'embed' in name returns EMBEDDINGS via name_pattern."""
        result = detect_model_type_detailed("test/text-embed-model")
        assert result.model_type == ModelType.EMBEDDINGS
        assert result.detection_method == "name_pattern"


class TestDetectModelTypeFunction:
    """Tests for detect_model_type() (the simpler function) covering edge cases."""

    def test_detects_vision_from_config(self):
        """detect_model_type detects VISION from config."""
        config = {"vision_config": {"image_size": 224}, "model_type": "llava"}
        result = detect_model_type("test/vision-model", config=config)
        assert result == ModelType.VISION

    def test_detects_audio_from_config_audio_config(self):
        """detect_model_type detects AUDIO from audio_config key."""
        config = {"audio_config": {"sample_rate": 24000}}
        result = detect_model_type("test/audio-model", config=config)
        assert result == ModelType.AUDIO

    def test_detects_audio_from_architecture(self):
        """detect_model_type detects AUDIO from architecture."""
        config = {"architectures": ["KokoroModel"]}
        result = detect_model_type("test/model", config=config)
        assert result == ModelType.AUDIO

    def test_detects_embeddings_from_architecture(self):
        """detect_model_type detects EMBEDDINGS from architecture."""
        config = {"architectures": ["BertEmbeddingModel"]}
        result = detect_model_type("test/model", config=config)
        assert result == ModelType.EMBEDDINGS

    def test_detects_audio_from_model_type(self):
        """detect_model_type detects AUDIO from model_type field."""
        config = {"model_type": "whisper"}
        result = detect_model_type("test/whisper-model", config=config)
        assert result == ModelType.AUDIO

    def test_detects_embeddings_from_model_type(self):
        """detect_model_type detects EMBEDDINGS from model_type field."""
        config = {"model_type": "sentence_transformers"}
        result = detect_model_type("test/model", config=config)
        assert result == ModelType.EMBEDDINGS

    def test_detects_audio_from_name_pattern(self):
        """detect_model_type detects AUDIO from name pattern."""
        result = detect_model_type("mlx-community/Kokoro-82M-bf16")
        assert result == ModelType.AUDIO

    def test_detects_vision_from_name_pattern(self):
        """detect_model_type detects VISION from name pattern (gemma-3)."""
        result = detect_model_type("mlx-community/gemma-3-27b-it-4bit")
        assert result == ModelType.VISION

    def test_detects_embeddings_from_name_pattern(self):
        """detect_model_type detects EMBEDDINGS from name pattern."""
        result = detect_model_type("mlx-community/all-MiniLM-L6-v2-4bit")
        assert result == ModelType.EMBEDDINGS

    def test_defaults_to_text_gen(self):
        """detect_model_type defaults to TEXT_GEN."""
        result = detect_model_type("mlx-community/Qwen3-0.6B-4bit")
        assert result == ModelType.TEXT_GEN

    def test_exception_in_config_load_falls_back_to_name(self):
        """Exception loading config falls back to name-pattern detection."""
        with patch(
            "mlx_manager.utils.model_detection.read_model_config",
            side_effect=OSError("disk error"),
        ):
            result = detect_model_type("mlx-community/Kokoro-82M-4bit")
            assert result == ModelType.AUDIO


class TestTypeDetectionResult:
    """Tests for TypeDetectionResult NamedTuple."""

    def test_can_create_result(self):
        """TypeDetectionResult can be instantiated."""
        result = TypeDetectionResult(ModelType.TEXT_GEN, "default", "")
        assert result.model_type == ModelType.TEXT_GEN
        assert result.detection_method == "default"
        assert result.architecture == ""

    def test_can_access_by_index(self):
        """TypeDetectionResult fields accessible by index (NamedTuple)."""
        result = TypeDetectionResult(ModelType.VISION, "config_field", "VisionModel")
        assert result[0] == ModelType.VISION
        assert result[1] == "config_field"
        assert result[2] == "VisionModel"

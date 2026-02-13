"""Tests for the probe package.

Tests the strategy pattern implementation, registry, orchestrator,
and all type-specific probe strategies.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.types import ModelType

# ============================================================================
# Strategy Registry Tests
# ============================================================================


def test_all_strategies_registered():
    """Test that all 4 model types have registered strategies."""
    from mlx_manager.services.probe import registered_model_types

    types = registered_model_types()
    assert ModelType.TEXT_GEN in types
    assert ModelType.VISION in types
    assert ModelType.EMBEDDINGS in types
    assert ModelType.AUDIO in types
    assert len(types) == 4


def test_get_probe_strategy_text_gen():
    """Test get_probe_strategy returns TextGenProbe."""
    from mlx_manager.services.probe import get_probe_strategy
    from mlx_manager.services.probe.text_gen import TextGenProbe

    strategy = get_probe_strategy(ModelType.TEXT_GEN)
    assert strategy is not None
    assert isinstance(strategy, TextGenProbe)
    assert strategy.model_type == ModelType.TEXT_GEN


def test_get_probe_strategy_vision():
    """Test get_probe_strategy returns VisionProbe."""
    from mlx_manager.services.probe import get_probe_strategy
    from mlx_manager.services.probe.vision import VisionProbe

    strategy = get_probe_strategy(ModelType.VISION)
    assert strategy is not None
    assert isinstance(strategy, VisionProbe)
    assert strategy.model_type == ModelType.VISION


def test_get_probe_strategy_embeddings():
    """Test get_probe_strategy returns EmbeddingsProbe."""
    from mlx_manager.services.probe import get_probe_strategy
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    strategy = get_probe_strategy(ModelType.EMBEDDINGS)
    assert strategy is not None
    assert isinstance(strategy, EmbeddingsProbe)
    assert strategy.model_type == ModelType.EMBEDDINGS


def test_get_probe_strategy_audio():
    """Test get_probe_strategy returns AudioProbe."""
    from mlx_manager.services.probe import get_probe_strategy
    from mlx_manager.services.probe.audio import AudioProbe

    strategy = get_probe_strategy(ModelType.AUDIO)
    assert strategy is not None
    assert isinstance(strategy, AudioProbe)
    assert strategy.model_type == ModelType.AUDIO


def test_has_probe_strategy():
    """Test has_probe_strategy returns True for all registered types."""
    from mlx_manager.services.probe import has_probe_strategy

    assert has_probe_strategy(ModelType.TEXT_GEN)
    assert has_probe_strategy(ModelType.VISION)
    assert has_probe_strategy(ModelType.EMBEDDINGS)
    assert has_probe_strategy(ModelType.AUDIO)


# ============================================================================
# ProbeStep/ProbeResult Tests
# ============================================================================


def test_probe_step_to_sse():
    """Test ProbeStep SSE serialization for backward compatibility."""
    from mlx_manager.services.probe import ProbeStep

    step = ProbeStep(
        step="test_tools",
        status="completed",
        capability="supports_native_tools",
        value=True,
    )
    sse = step.to_sse()

    assert sse.startswith("data: ")
    assert sse.endswith("\n\n")

    data = json.loads(sse[6:-2])  # Strip "data: " prefix and "\n\n" suffix
    assert data["step"] == "test_tools"
    assert data["status"] == "completed"
    assert data["capability"] == "supports_native_tools"
    assert data["value"] is True


def test_probe_step_with_error():
    """Test ProbeStep SSE serialization with error field."""
    from mlx_manager.services.probe import ProbeStep

    step = ProbeStep(
        step="load_model",
        status="failed",
        error="Model not found",
    )
    sse = step.to_sse()

    data = json.loads(sse[6:-2])
    assert data["step"] == "load_model"
    assert data["status"] == "failed"
    assert data["error"] == "Model not found"
    assert "capability" not in data
    assert "value" not in data


def test_probe_result_defaults():
    """Test ProbeResult initializes with all fields as None."""
    from mlx_manager.services.probe import ProbeResult

    result = ProbeResult()

    # Text-gen
    assert result.supports_native_tools is None
    assert result.supports_thinking is None
    assert result.practical_max_tokens is None

    # Vision
    assert result.supports_multi_image is None
    assert result.supports_video is None

    # Embeddings
    assert result.embedding_dimensions is None
    assert result.max_sequence_length is None
    assert result.is_normalized is None

    # Audio
    assert result.supports_tts is None
    assert result.supports_stt is None

    # Metadata
    assert result.model_type is None
    assert result.errors == []


def test_probe_result_text_gen_fields():
    """Test ProbeResult can be populated with text-gen capabilities."""
    from mlx_manager.services.probe import ProbeResult

    result = ProbeResult(
        model_type="text-gen",
        supports_native_tools=True,
        supports_thinking=False,
        practical_max_tokens=8192,
    )

    assert result.model_type == "text-gen"
    assert result.supports_native_tools is True
    assert result.supports_thinking is False
    assert result.practical_max_tokens == 8192


def test_probe_result_vision_fields():
    """Test ProbeResult can be populated with vision capabilities."""
    from mlx_manager.services.probe import ProbeResult

    result = ProbeResult(
        model_type="vision",
        supports_multi_image=True,
        supports_video=False,
        practical_max_tokens=4096,
    )

    assert result.model_type == "vision"
    assert result.supports_multi_image is True
    assert result.supports_video is False


def test_probe_result_embeddings_fields():
    """Test ProbeResult can be populated with embeddings capabilities."""
    from mlx_manager.services.probe import ProbeResult

    result = ProbeResult(
        model_type="embeddings",
        embedding_dimensions=384,
        max_sequence_length=512,
        is_normalized=True,
    )

    assert result.model_type == "embeddings"
    assert result.embedding_dimensions == 384
    assert result.max_sequence_length == 512
    assert result.is_normalized is True


def test_probe_result_audio_fields():
    """Test ProbeResult can be populated with audio capabilities."""
    from mlx_manager.services.probe import ProbeResult

    result = ProbeResult(
        model_type="audio",
        supports_tts=True,
        supports_stt=False,
    )

    assert result.model_type == "audio"
    assert result.supports_tts is True
    assert result.supports_stt is False


# ============================================================================
# Orchestrator (service.py) Tests
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_full_flow_text_gen():
    """Test full orchestrator flow with text-gen model."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/text-model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 2.0
    mock_loaded.config = {"max_position_embeddings": 8192}

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}
    mock_pool.unload_model = AsyncMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=ModelType.TEXT_GEN,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ) as mock_save,
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/text-model"):
            steps.append(step)

        # Verify step sequence
        step_names = [s.step for s in steps]
        assert "detect_type" in step_names
        assert "load_model" in step_names
        assert "check_context" in step_names
        assert "test_thinking" in step_names
        assert "test_tools" in step_names
        assert "save_results" in step_names
        assert "cleanup" in step_names

        # Model should be unloaded
        mock_pool.unload_model.assert_called_once_with("test/text-model")

        # Capabilities should be saved
        mock_save.assert_called_once()
        saved_result = mock_save.call_args[0][1]
        assert saved_result.model_type == ModelType.TEXT_GEN.value


@pytest.mark.asyncio
async def test_orchestrator_full_flow_embeddings():
    """Test full orchestrator flow with embeddings model."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/embed-model"
    mock_loaded.tokenizer = MagicMock()

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}
    mock_pool.unload_model = AsyncMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=ModelType.EMBEDDINGS,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ) as mock_save,
        patch(
            "mlx_manager.mlx_server.services.embeddings.generate_embeddings",
            new_callable=AsyncMock,
            return_value=([[0.5] * 384], None),
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"max_position_embeddings": 512},
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/embed-model"):
            steps.append(step)

        step_names = [s.step for s in steps]
        assert "detect_type" in step_names
        assert "test_encode" in step_names
        assert "check_normalization" in step_names
        assert "check_max_length" in step_names
        assert "test_similarity" in step_names

        mock_save.assert_called_once()
        saved_result = mock_save.call_args[0][1]
        assert saved_result.model_type == ModelType.EMBEDDINGS.value


@pytest.mark.asyncio
async def test_orchestrator_type_detection_failure():
    """Test orchestrator handles type detection failure."""
    mock_pool = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            side_effect=Exception("Unknown model type"),
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/unknown"):
            steps.append(step)

        # Should only have detect_type step
        assert len(steps) == 2  # running + failed
        assert steps[0].step == "detect_type"
        assert steps[0].status == "running"
        assert steps[1].step == "detect_type"
        assert steps[1].status == "failed"
        assert "Unknown model type" in steps[1].error


@pytest.mark.asyncio
async def test_orchestrator_model_load_failure():
    """Test orchestrator handles model load failure gracefully."""
    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(side_effect=Exception("Model not found"))
    mock_pool._models = {}

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=ModelType.TEXT_GEN,
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/missing"):
            steps.append(step)

        # Should have detect_type (running + completed) and load_model (running + failed)
        step_names = [s.step for s in steps]
        assert "detect_type" in step_names
        assert "load_model" in step_names

        load_steps = [s for s in steps if s.step == "load_model"]
        assert any(s.status == "failed" for s in load_steps)
        failed_step = next(s for s in load_steps if s.status == "failed")
        assert "Model not found" in failed_step.error


@pytest.mark.asyncio
async def test_orchestrator_preloaded_model_not_unloaded():
    """Test orchestrator doesn't unload pre-loaded models."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/preloaded"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 1.0
    mock_loaded.config = {}
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {"test/preloaded": mock_loaded}  # Already loaded
    mock_pool.unload_model = AsyncMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=ModelType.TEXT_GEN,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/preloaded"):
            steps.append(step)

        # Should NOT unload
        mock_pool.unload_model.assert_not_called()

        # Cleanup step should be skipped
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert len(cleanup_steps) == 1
        assert cleanup_steps[0].status == "skipped"


@pytest.mark.asyncio
async def test_orchestrator_save_failure_doesnt_crash():
    """Test orchestrator handles save failure gracefully."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.config = {}

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}
    mock_pool.unload_model = AsyncMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=ModelType.TEXT_GEN,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
            side_effect=Exception("Database error"),
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should still complete with save_results failed
        save_steps = [s for s in steps if s.step == "save_results"]
        assert any(s.status == "failed" for s in save_steps)

        # Cleanup should still run
        assert any(s.step == "cleanup" for s in steps)


# ============================================================================
# TextGenProbe Tests
# ============================================================================


@pytest.mark.asyncio
async def test_text_gen_probe_happy_path():
    """Test TextGenProbe with all capabilities enabled."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/text-model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.size_gb = 2.0
    mock_loaded.config = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "max_position_embeddings": 32768,
    }

    result = ProbeResult()

    with (
        patch.object(
            TextGenProbe,
            "_verify_thinking_support",
            return_value=(True, "think_tag"),
        ),
        patch.object(
            TextGenProbe,
            "_verify_tool_support",
            return_value=("template", "hermes_json"),
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=16.0),
    ):
        # Mock config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_loaded.config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/text-model", mock_loaded, result):
            steps.append(step)

        # Verify capabilities populated
        assert result.supports_thinking is True
        assert result.supports_native_tools is True
        assert result.tool_format == "template"
        assert result.practical_max_tokens is not None
        assert isinstance(result.practical_max_tokens, int)
        assert result.practical_max_tokens > 0

        # Verify steps
        step_names = [s.step for s in steps]
        assert "check_context" in step_names
        assert "test_thinking" in step_names
        assert "test_tools" in step_names


@pytest.mark.asyncio
async def test_text_gen_probe_tokenizer_none():
    """Test TextGenProbe skips thinking/tools when tokenizer is None."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/text-model"
    mock_loaded.tokenizer = None  # No tokenizer
    mock_loaded.config = {}

    result = ProbeResult()

    probe = TextGenProbe()
    steps = []
    async for step in probe.probe("test/text-model", mock_loaded, result):
        steps.append(step)

    # Thinking and tools should be skipped
    thinking_steps = [s for s in steps if s.step == "test_thinking"]
    assert len(thinking_steps) == 1
    assert thinking_steps[0].status == "skipped"

    tools_steps = [s for s in steps if s.step == "test_tools"]
    assert len(tools_steps) == 1
    assert tools_steps[0].status == "skipped"

    # Result should not have thinking/tools set
    assert result.supports_thinking is None
    assert result.supports_native_tools is None


# ============================================================================
# VisionProbe Tests
# ============================================================================


@pytest.mark.asyncio
async def test_vision_probe_happy_path():
    """Test VisionProbe with all capabilities enabled."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.size_gb = 4.0
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={
                "image_token_id": 123,
                "video_token_id": 456,
                "text_config": {
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "max_position_embeddings": 8192,
                },
            },
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=32.0),
    ):
        # Mock config file
        import tempfile

        config = {
            "text_config": {
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 8192,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Verify capabilities
        assert result.supports_multi_image is True
        assert result.supports_video is True
        assert result.practical_max_tokens is not None

        # Verify steps
        step_names = [s.step for s in steps]
        assert "check_processor" in step_names
        assert "check_multi_image" in step_names
        assert "check_video" in step_names
        assert "check_context" in step_names


@pytest.mark.asyncio
async def test_vision_probe_processor_check():
    """Test VisionProbe processor validation."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.config = {}

    result = ProbeResult()

    with patch(
        "mlx_manager.utils.model_detection.read_model_config",
        return_value={},
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        processor_steps = [s for s in steps if s.step == "check_processor"]
        # Should have running + completed
        assert any(s.status == "completed" for s in processor_steps)


# ============================================================================
# EmbeddingsProbe Tests
# ============================================================================


@pytest.mark.asyncio
async def test_embeddings_probe_happy_path():
    """Test EmbeddingsProbe with all capabilities."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/embed-model"

    result = ProbeResult()

    # Create normalized embedding (384-dim, L2 norm = 1.0)
    import math

    dim = 384
    embedding = [1.0 / math.sqrt(dim)] * dim

    with (
        patch(
            "mlx_manager.mlx_server.services.embeddings.generate_embeddings",
            new_callable=AsyncMock,
            side_effect=[
                # First call: test_encode
                ([embedding], None),
                # Second call: test_similarity
                (
                    [
                        [0.5] * dim,  # cat
                        [0.6] * dim,  # kitten (more similar)
                        [0.1] * dim,  # airplane (less similar)
                    ],
                    None,
                ),
            ],
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"max_position_embeddings": 512},
        ),
    ):
        probe = EmbeddingsProbe()
        steps = []
        async for step in probe.probe("test/embed-model", mock_loaded, result):
            steps.append(step)

        # Verify capabilities
        assert result.embedding_dimensions == 384
        assert result.is_normalized is True
        assert result.max_sequence_length == 512

        # Verify steps
        step_names = [s.step for s in steps]
        assert "test_encode" in step_names
        assert "check_normalization" in step_names
        assert "check_max_length" in step_names
        assert "test_similarity" in step_names


@pytest.mark.asyncio
async def test_embeddings_probe_encode_failure():
    """Test EmbeddingsProbe handles encode failure and stops early."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/embed-model"

    result = ProbeResult()

    with patch(
        "mlx_manager.mlx_server.services.embeddings.generate_embeddings",
        new_callable=AsyncMock,
        side_effect=Exception("Encoding failed"),
    ):
        probe = EmbeddingsProbe()
        steps = []
        async for step in probe.probe("test/embed-model", mock_loaded, result):
            steps.append(step)

        # Should fail at test_encode and not continue
        encode_steps = [s for s in steps if s.step == "test_encode"]
        assert any(s.status == "failed" for s in encode_steps)

        # Other steps should not run
        step_names = [s.step for s in steps]
        assert "check_normalization" not in step_names
        assert "test_similarity" not in step_names


# ============================================================================
# AudioProbe Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_probe_tts_detection():
    """Test AudioProbe detects TTS models (Kokoro pattern)."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/kokoro-model"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={
                "architectures": ["KokoroForCausalLM"],
                "model_type": "kokoro",
            },
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.generate_speech",
            new_callable=AsyncMock,
            return_value=(b"fake_audio_data", 24000),
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/kokoro-model", mock_loaded, result):
            steps.append(step)

        # Verify TTS detected
        assert result.supports_tts is True
        assert result.supports_stt is False
        assert result.model_family == "kokoro"

        # Verify TTS test ran and STT skipped
        step_names = [s.step for s in steps]
        assert "detect_audio_type" in step_names
        assert "test_tts" in step_names

        tts_steps = [s for s in steps if s.step == "test_tts"]
        assert any(s.status == "completed" for s in tts_steps)

        stt_steps = [s for s in steps if s.step == "test_stt"]
        assert any(s.status == "skipped" for s in stt_steps)


@pytest.mark.asyncio
async def test_audio_probe_stt_detection():
    """Test AudioProbe detects STT models (Whisper pattern)."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/whisper-model"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={
                "architectures": ["WhisperForConditionalGeneration"],
                "model_type": "whisper",
            },
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.transcribe_audio",
            new_callable=AsyncMock,
            return_value={"text": "Hello world"},
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/whisper-model", mock_loaded, result):
            steps.append(step)

        # Verify STT detected
        assert result.supports_stt is True
        assert result.supports_tts is False
        assert result.model_family == "whisper"

        # Verify STT test ran and TTS skipped
        stt_steps = [s for s in steps if s.step == "test_stt"]
        assert any(s.status == "completed" for s in stt_steps)

        tts_steps = [s for s in steps if s.step == "test_tts"]
        assert any(s.status == "skipped" for s in tts_steps)


@pytest.mark.asyncio
async def test_audio_probe_name_based_fallback():
    """Test AudioProbe uses name-based detection when config is unavailable."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "mlx-community/kokoro-82m-4bit"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},  # Empty config
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.generate_speech",
            new_callable=AsyncMock,
            return_value=(b"audio", 24000),
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("mlx-community/kokoro-82m-4bit", mock_loaded, result):
            steps.append(step)

        # Should still detect TTS from model name
        assert result.supports_tts is True
        assert result.model_family == "kokoro"


# ============================================================================
# Additional Coverage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_strategy_not_found():
    """Test orchestrator handles missing strategy gracefully."""
    # Create a custom model type that has no strategy
    mock_pool = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=MagicMock(value="unknown_type"),
        ),
        patch(
            "mlx_manager.services.probe.strategy.get_probe_strategy",
            return_value=None,  # No strategy found
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/unknown-type"):
            steps.append(step)

        # Should have detect_type (completed) and find_strategy (failed)
        step_names = [s.step for s in steps]
        assert "detect_type" in step_names
        assert "find_strategy" in step_names

        find_strategy_steps = [s for s in steps if s.step == "find_strategy"]
        assert any(s.status == "failed" for s in find_strategy_steps)


@pytest.mark.asyncio
async def test_orchestrator_cleanup_failure():
    """Test orchestrator handles cleanup failure gracefully."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.config = {}

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}
    mock_pool.unload_model = AsyncMock(side_effect=Exception("Unload failed"))

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type",
            return_value=ModelType.TEXT_GEN,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
    ):
        from mlx_manager.services.probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Cleanup should fail but not crash
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert any(s.status == "failed" for s in cleanup_steps)


@pytest.mark.asyncio
async def test_text_gen_probe_context_check_failure():
    """Test TextGenProbe handles context check failure and continues."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=Exception("Config not found"),
        ),
    ):
        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # Context check catches exception and logs warning, returns None
        # So it won't fail, just have practical_max_tokens as None
        assert result.practical_max_tokens is None


@pytest.mark.asyncio
async def test_text_gen_probe_thinking_failure():
    """Test TextGenProbe handles thinking check failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.config = {}

    result = ProbeResult()

    with patch(
        "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
        side_effect=Exception("Thinking check failed"),
    ):
        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # Thinking test should fail
        thinking_steps = [s for s in steps if s.step == "test_thinking"]
        assert any(s.status == "failed" for s in thinking_steps)


@pytest.mark.asyncio
async def test_text_gen_probe_tools_failure():
    """Test TextGenProbe handles tool verification failure gracefully."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.adapter.supports_native_tools.return_value = False
    mock_loaded.adapter.format_tools_for_prompt.return_value = ""
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=Exception("Config not found"),
        ),
    ):
        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # Tools should complete with no support detected
        tools_steps = [s for s in steps if s.step == "test_tools"]
        assert any(s.status == "completed" for s in tools_steps)
        assert result.tool_format is None
        assert result.supports_native_tools is False


@pytest.mark.asyncio
async def test_vision_probe_processor_failure():
    """Test VisionProbe handles processor check failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = None  # No processor
    mock_loaded.config = {}

    result = ProbeResult()

    with patch(
        "mlx_manager.utils.model_detection.read_model_config",
        return_value={},
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Processor check should fail
        processor_steps = [s for s in steps if s.step == "check_processor"]
        assert any(s.status == "failed" for s in processor_steps)


@pytest.mark.asyncio
async def test_vision_probe_multi_image_failure():
    """Test VisionProbe handles multi-image check failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.tokenizer.image_processor = MagicMock()
    mock_loaded.config = {}

    result = ProbeResult()

    with patch(
        "mlx_manager.utils.model_detection.read_model_config",
        side_effect=Exception("Config read failed"),
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Multi-image check should fail
        multi_image_steps = [s for s in steps if s.step == "check_multi_image"]
        assert any(s.status == "failed" for s in multi_image_steps)


@pytest.mark.asyncio
async def test_vision_probe_video_failure():
    """Test VisionProbe handles video check failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.tokenizer.image_processor = MagicMock()
    mock_loaded.config = {}

    result = ProbeResult()

    # First call succeeds (processor check), second call fails (multi-image), third fails (video)
    call_count = 0

    def read_config_side_effect(*args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {}  # multi-image check
        else:
            raise Exception("Config read failed")  # video check

    with patch(
        "mlx_manager.utils.model_detection.read_model_config",
        side_effect=read_config_side_effect,
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Video check should fail
        video_steps = [s for s in steps if s.step == "check_video"]
        assert any(s.status == "failed" for s in video_steps)


@pytest.mark.asyncio
async def test_vision_probe_context_failure():
    """Test VisionProbe handles context check failure and continues."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.tokenizer.image_processor = MagicMock()
    mock_loaded.config = {}
    mock_loaded.size_gb = 4.0

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=Exception("Config download failed"),
        ),
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Context check catches exception and logs warning, returns None
        assert result.practical_max_tokens is None


@pytest.mark.asyncio
async def test_embeddings_probe_normalization_failure():
    """Test EmbeddingsProbe handles normalization check failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/embed-model"

    result = ProbeResult()

    # Return None for embedding which will cause normalization check to skip
    with (
        patch(
            "mlx_manager.mlx_server.services.embeddings.generate_embeddings",
            new_callable=AsyncMock,
            return_value=([None], None),
        ),
    ):
        probe = EmbeddingsProbe()
        steps = []
        async for step in probe.probe("test/embed-model", mock_loaded, result):
            steps.append(step)

        # Encode should fail because embedding is None
        encode_steps = [s for s in steps if s.step == "test_encode"]
        assert any(s.status == "failed" for s in encode_steps)


@pytest.mark.asyncio
async def test_embeddings_probe_max_length_failure():
    """Test EmbeddingsProbe handles max_length check failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/embed-model"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.services.embeddings.generate_embeddings",
            new_callable=AsyncMock,
            return_value=([[0.5] * 384], None),
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            side_effect=Exception("Config read failed"),
        ),
    ):
        probe = EmbeddingsProbe()
        steps = []
        async for step in probe.probe("test/embed-model", mock_loaded, result):
            steps.append(step)

        # Max length check should fail
        max_len_steps = [s for s in steps if s.step == "check_max_length"]
        assert any(s.status == "failed" for s in max_len_steps)


@pytest.mark.asyncio
async def test_embeddings_probe_similarity_failure():
    """Test EmbeddingsProbe handles similarity check failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/embed-model"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.services.embeddings.generate_embeddings",
            new_callable=AsyncMock,
            side_effect=[
                # First call: test_encode
                ([[0.5] * 384], None),
                # Second call: test_similarity - fails
                Exception("Embedding generation failed"),
            ],
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"max_position_embeddings": 512},
        ),
    ):
        probe = EmbeddingsProbe()
        steps = []
        async for step in probe.probe("test/embed-model", mock_loaded, result):
            steps.append(step)

        # Similarity test should fail
        similarity_steps = [s for s in steps if s.step == "test_similarity"]
        assert any(s.status == "failed" for s in similarity_steps)


@pytest.mark.asyncio
async def test_audio_probe_detection_failure():
    """Test AudioProbe handles detection failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/audio-model"

    result = ProbeResult()

    with patch(
        "mlx_manager.utils.model_detection.read_model_config",
        side_effect=Exception("Config read failed"),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/audio-model", mock_loaded, result):
            steps.append(step)

        # Detection should fail
        detection_steps = [s for s in steps if s.step == "detect_audio_type"]
        assert any(s.status == "failed" for s in detection_steps)


@pytest.mark.asyncio
async def test_audio_probe_tts_test_failure():
    """Test AudioProbe handles TTS test failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/kokoro-model"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"architectures": ["KokoroForCausalLM"]},
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.generate_speech",
            new_callable=AsyncMock,
            side_effect=Exception("TTS generation failed"),
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/kokoro-model", mock_loaded, result):
            steps.append(step)

        # TTS test should fail
        tts_steps = [s for s in steps if s.step == "test_tts"]
        assert any(s.status == "failed" for s in tts_steps)


@pytest.mark.asyncio
async def test_audio_probe_stt_test_failure():
    """Test AudioProbe handles STT test failure."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/whisper-model"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"architectures": ["WhisperForConditionalGeneration"]},
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.transcribe_audio",
            new_callable=AsyncMock,
            side_effect=Exception("STT transcription failed"),
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/whisper-model", mock_loaded, result):
            steps.append(step)

        # STT test should fail
        stt_steps = [s for s in steps if s.step == "test_stt"]
        assert any(s.status == "failed" for s in stt_steps)


# ============================================================================
# Service._save_capabilities and _apply_result_to_caps Tests
# ============================================================================


@pytest.mark.asyncio
async def test_save_capabilities_creates_new():
    """Test _save_capabilities creates new Model record."""

    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.service import _save_capabilities

    result = ProbeResult(
        model_type="text-gen",
        supports_native_tools=True,
        supports_thinking=False,
        practical_max_tokens=8192,
    )

    # Mock update_model_capabilities since _save_capabilities now uses it
    with patch("mlx_manager.services.model_registry.update_model_capabilities") as mock_update_caps:
        mock_update_caps.return_value = AsyncMock()
        await _save_capabilities("test/model", result)

        # Verify update_model_capabilities was called with correct arguments
        assert mock_update_caps.called
        call_args = mock_update_caps.call_args
        assert call_args[0][0] == "test/model"
        kwargs = call_args[1]
        assert kwargs.get("model_type") == "text-gen"
        assert kwargs.get("supports_native_tools") is True
        assert kwargs.get("supports_thinking") is False
        assert kwargs.get("practical_max_tokens") == 8192
        assert kwargs.get("probe_version") == 2


@pytest.mark.asyncio
async def test_save_capabilities_updates_existing():
    """Test _save_capabilities updates existing Model record."""

    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.service import _save_capabilities

    result = ProbeResult(
        model_type="vision",
        supports_multi_image=True,
        supports_video=False,
        practical_max_tokens=4096,
    )

    # Mock update_model_capabilities since _save_capabilities now uses it
    with patch("mlx_manager.services.model_registry.update_model_capabilities") as mock_update_caps:
        mock_update_caps.return_value = AsyncMock()
        await _save_capabilities("test/vision-model", result)

        # Verify update_model_capabilities was called with correct arguments
        assert mock_update_caps.called
        call_args = mock_update_caps.call_args
        assert call_args[0][0] == "test/vision-model"
        kwargs = call_args[1]
        assert kwargs.get("model_type") == "vision"
        assert kwargs.get("supports_multi_image") is True
        assert kwargs.get("supports_video") is False
        assert kwargs.get("practical_max_tokens") == 4096
        assert kwargs.get("probe_version") == 2


@pytest.mark.asyncio
async def test_save_capabilities_all_fields():
    """Test _save_capabilities passes all field types to update_model_capabilities."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.service import _save_capabilities

    result = ProbeResult(
        model_type="embeddings",
        # Text-gen
        supports_native_tools=True,
        supports_thinking=True,
        practical_max_tokens=2048,
        # Vision
        supports_multi_image=True,
        supports_video=True,
        # Embeddings
        embedding_dimensions=768,
        max_sequence_length=512,
        is_normalized=True,
        # Audio
        supports_tts=True,
        supports_stt=False,
    )

    with patch(
        "mlx_manager.services.model_registry.update_model_capabilities",
        new_callable=AsyncMock,
    ) as mock_update_caps:
        await _save_capabilities("test/all-model", result)

        assert mock_update_caps.called
        call_args = mock_update_caps.call_args
        assert call_args[0][0] == "test/all-model"
        kwargs = call_args[1]
        assert kwargs["model_type"] == "embeddings"
        assert kwargs["supports_native_tools"] is True
        assert kwargs["supports_thinking"] is True
        assert kwargs["practical_max_tokens"] == 2048
        assert kwargs["supports_multi_image"] is True
        assert kwargs["supports_video"] is True
        assert kwargs["embedding_dimensions"] == 768
        assert kwargs["max_sequence_length"] == 512
        assert kwargs["is_normalized"] is True
        assert kwargs["supports_tts"] is True
        assert kwargs["supports_stt"] is False
        assert kwargs["probe_version"] == 2


# ============================================================================
# Edge Case Coverage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_text_gen_probe_max_tokens_no_max_pos():
    """Test TextGenProbe when config has no max_position_embeddings."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 2.0

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
    ):
        # Mock config without max_position_embeddings
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"hidden_size": 768}, f)  # No max_pos
            config_path = f.name
        mock_download.return_value = config_path

        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # practical_max_tokens should be None
        assert result.practical_max_tokens is None


@pytest.mark.asyncio
async def test_text_gen_probe_max_tokens_missing_params():
    """Test TextGenProbe when config is missing key params."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 2.0

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
    ):
        # Mock config with max_pos but missing layer/head info
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"max_position_embeddings": 8192}, f)  # No layer info
            config_path = f.name
        mock_download.return_value = config_path

        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # Should return max_pos as-is when params missing
        assert result.practical_max_tokens == 8192


@pytest.mark.asyncio
async def test_text_gen_probe_max_tokens_low_memory():
    """Test TextGenProbe with low available memory."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 15.0  # Large model

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=16.0),
    ):
        import tempfile

        config = {
            "max_position_embeddings": 32768,
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # Should return min(max_pos, 2048) when memory is low
        assert result.practical_max_tokens is not None
        assert result.practical_max_tokens <= 2048


@pytest.mark.asyncio
async def test_vision_probe_no_multi_image_support():
    """Test VisionProbe detects lack of multi-image support."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.config = {}
    mock_loaded.size_gb = 4.0

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},  # No image_token_id
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=32.0),
    ):
        import tempfile

        config = {"text_config": {"max_position_embeddings": 4096}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Should detect no multi-image support
        assert result.supports_multi_image is False
        assert result.supports_video is False


@pytest.mark.asyncio
async def test_vision_probe_vision_max_tokens_missing_params():
    """Test VisionProbe handles missing text_config params."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.config = {}
    mock_loaded.size_gb = 4.0

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
    ):
        import tempfile

        # Config with incomplete text_config
        config = {"text_config": {"max_position_embeddings": 4096}}  # Missing layer info
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Should return max_pos when params missing
        assert result.practical_max_tokens == 4096


@pytest.mark.asyncio
async def test_audio_probe_both_tts_and_stt():
    """Test AudioProbe can detect both TTS and STT support."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/multimodal-audio"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={
                "tts_config": {},
                "stt_config": {},
            },
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.generate_speech",
            new_callable=AsyncMock,
            return_value=(b"audio", 24000),
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.transcribe_audio",
            new_callable=AsyncMock,
            return_value={"text": "test"},
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/multimodal-audio", mock_loaded, result):
            steps.append(step)

        # Both should be detected
        assert result.supports_tts is True
        assert result.supports_stt is True


@pytest.mark.asyncio
async def test_text_gen_probe_head_dim_calculation():
    """Test TextGenProbe calculates head_dim from hidden_size and num_heads."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 2.0

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=16.0),
    ):
        import tempfile

        # Config without head_dim but with hidden_size and num_attention_heads
        config = {
            "max_position_embeddings": 8192,
            "num_hidden_layers": 24,
            "num_key_value_heads": 8,
            "hidden_size": 1024,
            "num_attention_heads": 16,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # Should calculate practical_max_tokens using derived head_dim
        assert result.practical_max_tokens is not None
        assert result.practical_max_tokens > 0


@pytest.mark.asyncio
async def test_vision_probe_vision_max_tokens_low_memory():
    """Test VisionProbe with low memory returns minimum context."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.config = {}
    mock_loaded.size_gb = 15.0  # Large model

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=16.0),
    ):
        import tempfile

        config = {
            "text_config": {
                "max_position_embeddings": 32768,
                "num_hidden_layers": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Should return min(max_pos, 2048) when memory is low
        assert result.practical_max_tokens is not None
        assert result.practical_max_tokens <= 2048


@pytest.mark.asyncio
async def test_vision_probe_head_dim_calculation():
    """Test VisionProbe calculates head_dim from hidden_size."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.config = {}
    mock_loaded.size_gb = 4.0

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=32.0),
    ):
        import tempfile

        config = {
            "text_config": {
                "max_position_embeddings": 8192,
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "hidden_size": 1024,
                "num_attention_heads": 16,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Should calculate practical_max_tokens using derived head_dim
        assert result.practical_max_tokens is not None
        assert result.practical_max_tokens > 0


@pytest.mark.asyncio
async def test_audio_probe_config_based_tts_detection():
    """Test AudioProbe detects TTS via config fields."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/tts-model"

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={
                "vocoder_config": {},  # TTS indicator
            },
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.generate_speech",
            new_callable=AsyncMock,
            return_value=(b"audio", 24000),
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/tts-model", mock_loaded, result):
            steps.append(step)

        assert result.supports_tts is True
        assert result.supports_stt is False


@pytest.mark.asyncio
async def test_audio_probe_name_based_stt_detection():
    """Test AudioProbe detects STT from model name."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.audio import AudioProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/asr-model"  # ASR is STT

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},  # No config hints
        ),
        patch(
            "mlx_manager.mlx_server.services.audio.transcribe_audio",
            new_callable=AsyncMock,
            return_value={"text": "test"},
        ),
    ):
        probe = AudioProbe()
        steps = []
        async for step in probe.probe("test/asr-model", mock_loaded, result):
            steps.append(step)

        assert result.supports_stt is True
        assert result.supports_tts is False


@pytest.mark.asyncio
async def test_vision_probe_processor_exception():
    """Test VisionProbe handles processor exception during check."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    # Mock processor that raises an exception during image processing
    mock_processor = MagicMock()
    mock_image_processor = MagicMock()
    mock_image_processor.side_effect = Exception("Image processing failed")
    mock_processor.image_processor = mock_image_processor

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.config = {}

    result = ProbeResult()

    with patch(
        "mlx_manager.utils.model_detection.read_model_config",
        return_value={},
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Processor check should fail
        processor_steps = [s for s in steps if s.step == "check_processor"]
        assert any(s.status == "failed" for s in processor_steps)


# ============================================================================
# TextGen Probe: Tool & Thinking Verification Tests
# ============================================================================


@pytest.mark.asyncio
async def test_text_gen_probe_template_delivery():
    """Test Attempt 1: template delivery returns ('template', parser_id)."""
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = True
    mock_adapter.tool_parser.parser_id = "hermes_json"
    mock_adapter.tool_parser.validates.return_value = True

    with (
        patch(
            "mlx_manager.services.probe.text_gen.TextGenProbe._generate",
            new_callable=AsyncMock,
            return_value=(
                '<tool_call>\n{"name": "get_weather",'
                ' "arguments": {"location": "Tokyo"}}\n</tool_call>'
            ),
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        probe = TextGenProbe()
        tool_format, parser_id = await probe._verify_tool_support(mock_loaded, mock_adapter)
        assert tool_format == "template"
        assert parser_id == "hermes_json"


@pytest.mark.asyncio
async def test_text_gen_probe_adapter_delivery():
    """Test Attempt 2: adapter delivery returns ('adapter', parser_id)."""
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False
    mock_adapter.tool_parser.parser_id = "hermes_json"
    mock_adapter.tool_parser.validates.return_value = True
    mock_adapter.format_tools_for_prompt.return_value = "<tools>...</tools>"

    with (
        patch(
            "mlx_manager.services.probe.text_gen.TextGenProbe._generate",
            new_callable=AsyncMock,
            return_value=(
                '<tool_call>\n{"name": "get_weather",'
                ' "arguments": {"location": "Tokyo"}}\n</tool_call>'
            ),
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        probe = TextGenProbe()
        tool_format, parser_id = await probe._verify_tool_support(mock_loaded, mock_adapter)
        assert tool_format == "adapter"
        assert parser_id == "hermes_json"


@pytest.mark.asyncio
async def test_text_gen_probe_fallback_parser_sweep():
    """Test fallback: adapter parser fails, sweep finds different parser."""
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = True
    mock_adapter.tool_parser.parser_id = "glm4_native"
    mock_adapter.tool_parser.validates.return_value = False  # Adapter parser fails

    output = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>'
    )

    with (
        patch(
            "mlx_manager.services.probe.text_gen.TextGenProbe._generate",
            new_callable=AsyncMock,
            return_value=output,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        probe = TextGenProbe()
        tool_format, parser_id = await probe._verify_tool_support(mock_loaded, mock_adapter)
        assert tool_format == "template"
        # The hermes_json parser should match the <tool_call> format
        assert parser_id == "hermes_json"


def test_detect_unknown_xml_tags():
    """Test _detect_unknown_xml_tags finds unknown tags but skips known ones."""
    from mlx_manager.services.probe.text_gen import _detect_unknown_xml_tags

    # Known tags should be excluded
    output_known = "<think>some thinking</think><tool_call>call</tool_call>"
    assert _detect_unknown_xml_tags(output_known) == set()

    # Unknown tags should be detected
    output_unknown = "<custom_tool>data</custom_tool><think>ok</think>"
    assert "custom_tool" in _detect_unknown_xml_tags(output_unknown)

    # No tags at all
    assert _detect_unknown_xml_tags("plain text response") == set()


@pytest.mark.asyncio
async def test_text_gen_probe_no_adapter_skips():
    """Test TextGenProbe skips thinking/tools when adapter is None."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.adapter = None  # No adapter
    mock_loaded.config = {}

    result = ProbeResult()

    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=Exception("Config not found"),
    ):
        probe = TextGenProbe()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

        # Both should be skipped
        thinking_steps = [s for s in steps if s.step == "test_thinking"]
        assert len(thinking_steps) == 1
        assert thinking_steps[0].status == "skipped"

        tools_steps = [s for s in steps if s.step == "test_tools"]
        assert len(tools_steps) == 1
        assert tools_steps[0].status == "skipped"


@pytest.mark.asyncio
async def test_text_gen_probe_thinking_generation_based():
    """Test generation-based thinking verification via _verify_thinking_support."""
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.thinking_parser.parser_id = "think_tag"
    mock_adapter.thinking_parser.extract.return_value = "Some thinking content"

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.services.probe.text_gen.TextGenProbe._generate",
            new_callable=AsyncMock,
            return_value="<think>Some thinking content</think>The answer is 4.",
        ),
    ):
        probe = TextGenProbe()
        supports, parser_id = await probe._verify_thinking_support(mock_loaded, mock_adapter)
        assert supports is True
        assert parser_id == "think_tag"


@pytest.mark.asyncio
async def test_text_gen_probe_thinking_template_authoritative():
    """Test thinking: template support is authoritative even without tags in output."""
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.thinking_parser.parser_id = "think_tag"
    mock_adapter.thinking_parser.extract.return_value = None  # No tags in output

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.services.probe.text_gen.TextGenProbe._generate",
            new_callable=AsyncMock,
            return_value="The answer is 4.",  # No thinking tags
        ),
    ):
        probe = TextGenProbe()
        supports, parser_id = await probe._verify_thinking_support(mock_loaded, mock_adapter)
        assert supports is True  # Template is authoritative
        assert parser_id == "think_tag"


@pytest.mark.asyncio
async def test_text_gen_probe_no_tool_support_detected():
    """Test probe reports no tool support when both attempts fail."""
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False
    mock_adapter.format_tools_for_prompt.return_value = ""  # No adapter delivery either

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.text_gen.TextGenProbe._generate",
            new_callable=AsyncMock,
            return_value="I don't know how to call tools.",
        ),
    ):
        probe = TextGenProbe()
        tool_format, parser_id = await probe._verify_tool_support(mock_loaded, mock_adapter)
        assert tool_format is None
        assert parser_id is None


# ============================================================================
# ProbeStep Details Field Tests
# ============================================================================


def test_probe_step_details_field():
    """Test ProbeStep supports optional details dict."""
    from mlx_manager.services.probe import ProbeStep

    step = ProbeStep(
        step="test_tools",
        status="completed",
        capability="tool_format",
        value="template",
        details={"raw_output": "some output", "parsers_tried": ["hermes_json"]},
    )
    sse = step.to_sse()
    data = json.loads(sse[6:-2])
    assert data["details"]["raw_output"] == "some output"
    assert data["details"]["parsers_tried"] == ["hermes_json"]


def test_probe_step_details_omitted_when_none():
    """Test ProbeStep omits details from SSE when None."""
    from mlx_manager.services.probe import ProbeStep

    step = ProbeStep(step="test_tools", status="running")
    sse = step.to_sse()
    data = json.loads(sse[6:-2])
    assert "details" not in data


# ============================================================================
# Tool Output Validation Tests
# ============================================================================


def test_validate_tool_output_adapter_parser_first():
    """Test _validate_tool_output tries adapter parser first."""
    from mlx_manager.services.probe.text_gen import _validate_tool_output

    mock_adapter = MagicMock()
    mock_adapter.tool_parser.parser_id = "hermes_json"
    mock_adapter.tool_parser.validates.return_value = True

    result = _validate_tool_output("some output", "get_weather", mock_adapter)
    assert result == "hermes_json"
    mock_adapter.tool_parser.validates.assert_called_once_with("some output", "get_weather")


def test_validate_tool_output_falls_through_to_sweep():
    """Test _validate_tool_output sweeps all parsers when adapter parser fails."""
    from mlx_manager.services.probe.text_gen import _validate_tool_output

    mock_adapter = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"

    # Should fall through to _find_matching_parser sweep
    with patch(
        "mlx_manager.services.probe.base._find_matching_parser",
        return_value="hermes_json",
    ) as mock_sweep:
        result = _validate_tool_output("some output", "get_weather", mock_adapter)
        assert result == "hermes_json"
        mock_sweep.assert_called_once_with("some output", "get_weather", exclude_parser_id=None)


# ============================================================================
# Probe Class Hierarchy Tests
# ============================================================================


def test_text_gen_probe_is_generative_probe():
    """TextGenProbe should be a GenerativeProbe."""
    from mlx_manager.services.probe.base import GenerativeProbe
    from mlx_manager.services.probe.text_gen import TextGenProbe

    assert isinstance(TextGenProbe(), GenerativeProbe)


def test_vision_probe_is_generative_probe():
    """VisionProbe should be a GenerativeProbe."""
    from mlx_manager.services.probe.base import GenerativeProbe
    from mlx_manager.services.probe.vision import VisionProbe

    assert isinstance(VisionProbe(), GenerativeProbe)


def test_text_gen_probe_is_base_probe():
    """TextGenProbe should be a BaseProbe."""
    from mlx_manager.services.probe.base import BaseProbe
    from mlx_manager.services.probe.text_gen import TextGenProbe

    assert isinstance(TextGenProbe(), BaseProbe)


def test_vision_probe_is_base_probe():
    """VisionProbe should be a BaseProbe."""
    from mlx_manager.services.probe.base import BaseProbe
    from mlx_manager.services.probe.vision import VisionProbe

    assert isinstance(VisionProbe(), BaseProbe)


def test_embeddings_probe_is_base_probe():
    """EmbeddingsProbe should be a BaseProbe."""
    from mlx_manager.services.probe.base import BaseProbe
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    assert isinstance(EmbeddingsProbe(), BaseProbe)


def test_audio_probe_is_base_probe():
    """AudioProbe should be a BaseProbe."""
    from mlx_manager.services.probe.audio import AudioProbe
    from mlx_manager.services.probe.base import BaseProbe

    assert isinstance(AudioProbe(), BaseProbe)


def test_embeddings_probe_is_not_generative():
    """EmbeddingsProbe should NOT be a GenerativeProbe."""
    from mlx_manager.services.probe.base import GenerativeProbe
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    assert not isinstance(EmbeddingsProbe(), GenerativeProbe)


def test_audio_probe_is_not_generative():
    """AudioProbe should NOT be a GenerativeProbe."""
    from mlx_manager.services.probe.audio import AudioProbe
    from mlx_manager.services.probe.base import GenerativeProbe

    assert not isinstance(AudioProbe(), GenerativeProbe)


# ============================================================================
# VisionProbe Generative Capability Tests
# ============================================================================


@pytest.mark.asyncio
async def test_vision_probe_discovers_thinking():
    """VisionProbe should discover thinking support when adapter is present."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()
    mock_adapter = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-thinking-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.adapter = mock_adapter
    mock_loaded.size_gb = 4.0
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"image_token_id": 123, "video_token_id": 456},
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=32.0),
        patch.object(
            VisionProbe,
            "_verify_thinking_support",
            return_value=(True, "think_tag"),
        ),
        patch.object(
            VisionProbe,
            "_verify_tool_support",
            return_value=("template", "hermes_json"),
        ),
    ):
        import tempfile

        config = {
            "text_config": {
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 8192,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-thinking-model", mock_loaded, result):
            steps.append(step)

        # Verify vision-specific capabilities
        assert result.supports_multi_image is True
        assert result.supports_video is True
        assert result.practical_max_tokens is not None

        # Verify generative capabilities also detected
        assert result.supports_thinking is True
        assert result.thinking_parser_id == "think_tag"
        assert result.supports_native_tools is True
        assert result.tool_format == "template"
        assert result.tool_parser_id == "hermes_json"

        # Verify all steps present
        step_names = [s.step for s in steps]
        assert "check_processor" in step_names
        assert "check_multi_image" in step_names
        assert "check_video" in step_names
        assert "check_context" in step_names
        assert "test_thinking" in step_names
        assert "test_tools" in step_names


@pytest.mark.asyncio
async def test_vision_probe_no_adapter_skips_generative():
    """VisionProbe should skip thinking/tools when adapter is None."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.adapter = None  # No adapter
    mock_loaded.size_gb = 4.0
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=32.0),
    ):
        import tempfile

        config = {"text_config": {"max_position_embeddings": 4096}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Vision-specific steps should complete
        step_names = [s.step for s in steps]
        assert "check_processor" in step_names
        assert "check_multi_image" in step_names
        assert "check_video" in step_names
        assert "check_context" in step_names

        # Thinking and tools should be skipped
        thinking_steps = [s for s in steps if s.step == "test_thinking"]
        assert len(thinking_steps) == 1
        assert thinking_steps[0].status == "skipped"

        tools_steps = [s for s in steps if s.step == "test_tools"]
        assert len(tools_steps) == 1
        assert tools_steps[0].status == "skipped"

        # Generative results should remain None
        assert result.supports_thinking is None
        assert result.supports_native_tools is None


@pytest.mark.asyncio
async def test_vision_probe_generative_failure_doesnt_crash():
    """VisionProbe should handle generative probe failure gracefully."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()
    mock_adapter = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.adapter = mock_adapter
    mock_loaded.size_gb = 4.0
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=32.0),
        patch.object(
            VisionProbe,
            "_verify_thinking_support",
            side_effect=Exception("Generation failed"),
        ),
        patch.object(
            VisionProbe,
            "_verify_tool_support",
            side_effect=Exception("Generation failed"),
        ),
    ):
        import tempfile

        config = {"text_config": {"max_position_embeddings": 4096}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Thinking and tools should fail but not crash the probe
        thinking_steps = [s for s in steps if s.step == "test_thinking"]
        assert any(s.status == "failed" for s in thinking_steps)

        tools_steps = [s for s in steps if s.step == "test_tools"]
        assert any(s.status == "failed" for s in tools_steps)

        # Vision-specific steps should still have completed
        assert "check_processor" in [s.step for s in steps]


# ============================================================================
# Additional Vision Coverage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_vision_probe_generate_with_tools_and_thinking():
    """Test VisionProbe._generate with tools and enable_thinking kwargs."""
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.model = mock_model
    mock_loaded.config = {}

    # Mock VLM dependencies
    mock_vlm_result = MagicMock()
    mock_vlm_result.text = "Generated response"

    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    messages = [{"role": "user", "content": "Test message"}]

    with (
        patch("mlx_vlm.generate", return_value=mock_vlm_result),
        patch(
            "mlx_vlm.prompt_utils.apply_chat_template", return_value="formatted"
        ) as mock_template,
        patch("mlx_vlm.utils.load_config", return_value={}),
    ):
        probe = VisionProbe()
        result = await probe._generate(mock_loaded, messages, tools=tools, enable_thinking=True)

        assert result == "Generated response"
        # Verify tools and enable_thinking were passed through
        mock_template.assert_called_once()
        call_kwargs = mock_template.call_args[1]
        assert call_kwargs.get("tools") == tools
        assert call_kwargs.get("enable_thinking") is True


@pytest.mark.asyncio
async def test_vision_probe_messages_to_text_with_list_content():
    """Test VisionProbe._generate with structured list content in messages."""
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.model = mock_model
    mock_loaded.config = {}

    mock_vlm_result = MagicMock()
    mock_vlm_result.text = "Response"

    # Messages with structured content (list format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "image", "source": "data:image/png;base64,..."},
                {"type": "text", "text": "Second part"},
            ],
        }
    ]

    with (
        patch("mlx_vlm.generate", return_value=mock_vlm_result),
        patch(
            "mlx_vlm.prompt_utils.apply_chat_template", return_value="formatted"
        ) as mock_template,
        patch("mlx_vlm.utils.load_config", return_value={}),
    ):
        probe = VisionProbe()
        result = await probe._generate(mock_loaded, messages)

        assert result == "Response"
        # The first argument to apply_chat_template should be the text prompt
        # which should have extracted text parts joined
        call_args = mock_template.call_args[0]
        text_prompt = call_args[2]  # Third positional arg is the text prompt
        assert "First part" in text_prompt
        assert "Second part" in text_prompt


@pytest.mark.asyncio
async def test_vision_probe_processor_without_image_processor():
    """Test _check_processor when processor exists but has no image_processor attribute."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    # Mock processor without image_processor attribute
    mock_processor = MagicMock()
    del mock_processor.image_processor  # Remove the attribute

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.config = {}

    result = ProbeResult()

    with patch(
        "mlx_manager.utils.model_detection.read_model_config",
        return_value={},
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        processor_steps = [s for s in steps if s.step == "check_processor"]
        # Should still complete successfully (fallback return True)
        assert any(s.status == "completed" for s in processor_steps)


@pytest.mark.asyncio
async def test_vision_probe_context_check_exception():
    """Test probe handles exception during context check gracefully."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.size_gb = 4.0
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch(
            "mlx_manager.services.probe.vision._estimate_vision_max_tokens",
            side_effect=ValueError("Config parsing failed"),
        ),
    ):
        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        context_steps = [s for s in steps if s.step == "check_context"]
        # Should have failed status with error message
        failed_steps = [s for s in context_steps if s.status == "failed"]
        assert len(failed_steps) > 0
        assert "Config parsing failed" in failed_steps[0].error


@pytest.mark.asyncio
async def test_vision_probe_max_tokens_missing_max_pos():
    """Test _estimate_vision_max_tokens when max_position_embeddings is missing."""
    from mlx_manager.services.probe import ProbeResult
    from mlx_manager.services.probe.vision import VisionProbe

    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/vision-model"
    mock_loaded.tokenizer = mock_processor
    mock_loaded.size_gb = 4.0
    mock_loaded.config = {}

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={},
        ),
        patch("huggingface_hub.hf_hub_download") as mock_download,
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=32.0),
    ):
        import tempfile

        # Config without max_position_embeddings or max_sequence_length
        config = {
            "text_config": {
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                # Missing max_position_embeddings
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        mock_download.return_value = config_path

        probe = VisionProbe()
        steps = []
        async for step in probe.probe("test/vision-model", mock_loaded, result):
            steps.append(step)

        # Should complete but practical_max_tokens should be None
        context_steps = [s for s in steps if s.step == "check_context"]
        completed = [s for s in context_steps if s.status == "completed"]
        assert len(completed) > 0
        assert result.practical_max_tokens is None

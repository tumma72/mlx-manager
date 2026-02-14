"""Additional coverage tests for probe/service.py uncovered lines."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.detection import TypeDetectionResult


@pytest.mark.asyncio
async def test_audio_codec_rejection():
    """Test that audio codec models are rejected during pre-validation.

    Covers lines 69-82: Audio pre-validation for unsupported codec models.
    """
    from mlx_manager.mlx_server.models.types import ModelType

    mock_pool = MagicMock()
    mock_pool._models = {}

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.AUDIO, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=MagicMock(),  # Strategy exists for AUDIO
        ),
        patch(
            "mlx_manager.services.probe.audio._detect_audio_capabilities",
            return_value=(False, False, None),  # Not TTS, not STT (codec!)
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/audio-codec"):
            steps.append(step)

        # Should fail at load_model step with codec error
        load_steps = [s for s in steps if s.step == "load_model"]
        assert len(load_steps) > 0

        failed_load = next((s for s in load_steps if s.status == "failed"), None)
        assert failed_load is not None
        assert "Unsupported audio model subtype" in failed_load.error
        assert "audio codec" in failed_load.error.lower()

        # Should not proceed to actual model loading
        assert not mock_pool.get_model.called


@pytest.mark.asyncio
async def test_strategy_probe_exception():
    """Test that exceptions during strategy.probe() are caught and yielded.

    Covers lines 98-100: Exception handling in strategy probe execution.
    """
    from mlx_manager.mlx_server.models.types import ModelType

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool.unload_model = AsyncMock()

    # Create a strategy that raises during probe
    async def failing_probe(*args, **kwargs):
        if False:  # pragma: no cover
            yield None
        raise RuntimeError("Strategy probe internal error")

    mock_strategy = MagicMock()
    mock_strategy.probe = failing_probe

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=mock_strategy,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should have a strategy_error step
        error_steps = [s for s in steps if s.step == "strategy_error"]
        assert len(error_steps) == 1
        assert error_steps[0].status == "failed"
        assert "Strategy probe internal error" in error_steps[0].error

        # Should still attempt to save results and cleanup
        save_steps = [s for s in steps if s.step == "save_results"]
        assert len(save_steps) > 0


@pytest.mark.asyncio
async def test_load_model_exception():
    """Test that model loading failures are caught and yielded.

    Covers lines 89-92: Exception during model loading.
    """
    from mlx_manager.mlx_server.models.types import ModelType

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool.get_model = AsyncMock(side_effect=RuntimeError("Model load failed"))

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=MagicMock(),
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should fail at load_model step
        load_steps = [s for s in steps if s.step == "load_model"]
        failed_load = next((s for s in load_steps if s.status == "failed"), None)
        assert failed_load is not None
        assert "Model load failed" in failed_load.error

        # Should not proceed further
        save_steps = [s for s in steps if s.step == "save_results"]
        assert len(save_steps) == 0


@pytest.mark.asyncio
async def test_preloaded_model_no_db_record():
    """Test cleanup path when preloaded model has no DB record.

    Covers lines 138-140: Model record not found in DB during capability update.
    """
    from mlx_manager.mlx_server.models.types import ModelType

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/preloaded"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool._models = {"test/preloaded": mock_loaded}  # Already loaded
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)

    # Mock strategy that completes successfully
    async def successful_probe(*args, **kwargs):
        from mlx_manager.services.probe.steps import ProbeStep

        yield ProbeStep(step="strategy_step", status="completed")

    mock_strategy = MagicMock()
    mock_strategy.probe = successful_probe

    # Mock DB to return no model record
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)  # No record!
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=mock_strategy,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch(
            "mlx_manager.database.get_session",
            return_value=mock_session,
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/preloaded"):
            steps.append(step)

        # Should skip cleanup (model was preloaded)
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert len(cleanup_steps) == 1
        assert cleanup_steps[0].status == "skipped"

        # Loaded model capabilities should not be updated (no DB record)
        assert mock_loaded.capabilities is None


@pytest.mark.asyncio
async def test_preloaded_model_no_capabilities():
    """Test cleanup path when preloaded model DB record exists but has no capabilities.

    Covers line 139-140: Model record found but capabilities is None.
    """
    from mlx_manager.mlx_server.models.types import ModelType

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/preloaded"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool._models = {"test/preloaded": mock_loaded}
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)

    async def successful_probe(*args, **kwargs):
        if False:  # pragma: no cover
            yield None

    mock_strategy = MagicMock()
    mock_strategy.probe = successful_probe

    # Mock DB to return model record WITHOUT capabilities
    mock_model_record = MagicMock()
    mock_model_record.capabilities = None  # No capabilities!

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=mock_model_record)
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=mock_strategy,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch(
            "mlx_manager.database.get_session",
            return_value=mock_session,
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/preloaded"):
            steps.append(step)

        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert len(cleanup_steps) == 1
        assert cleanup_steps[0].status == "skipped"

        # Capabilities should still be None (record had no capabilities)
        assert mock_loaded.capabilities is None


@pytest.mark.asyncio
async def test_preloaded_model_with_capabilities():
    """Test cleanup path when preloaded model DB record has capabilities.

    Covers line 140: Update capabilities on loaded model.
    """
    from mlx_manager.mlx_server.models.types import ModelType

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/preloaded"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool._models = {"test/preloaded": mock_loaded}
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)

    async def successful_probe(*args, **kwargs):
        from mlx_manager.services.probe.steps import ProbeStep

        yield ProbeStep(step="strategy_step", status="completed")

    mock_strategy = MagicMock()
    mock_strategy.probe = successful_probe

    # Mock DB to return model record WITH capabilities
    mock_capabilities = MagicMock()
    mock_capabilities.supports_native_tools = True

    mock_model_record = MagicMock()
    mock_model_record.capabilities = mock_capabilities

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=mock_model_record)
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=mock_strategy,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch(
            "mlx_manager.database.get_session",
            return_value=mock_session,
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/preloaded"):
            steps.append(step)

        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert len(cleanup_steps) == 1
        assert cleanup_steps[0].status == "skipped"

        # Capabilities should be updated from DB
        assert mock_loaded.capabilities is mock_capabilities


@pytest.mark.asyncio
async def test_preloaded_model_db_exception():
    """Test cleanup path when DB query raises exception.

    Covers lines 141-142: Exception during capability update is caught.
    """
    from mlx_manager.mlx_server.models.types import ModelType

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/preloaded"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool._models = {"test/preloaded": mock_loaded}
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)

    async def successful_probe(*args, **kwargs):
        from mlx_manager.services.probe.steps import ProbeStep

        yield ProbeStep(step="strategy_step", status="completed")

    mock_strategy = MagicMock()
    mock_strategy.probe = successful_probe

    # Mock get_session to raise an exception
    def failing_session():
        raise RuntimeError("Database connection failed")

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=mock_strategy,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch(
            "mlx_manager.database.get_session",
            side_effect=failing_session,
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/preloaded"):
            steps.append(step)

        # Should still complete with skipped cleanup
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert len(cleanup_steps) == 1
        assert cleanup_steps[0].status == "skipped"

        # Exception should be caught, capabilities remain None
        assert mock_loaded.capabilities is None


@pytest.mark.asyncio
async def test_save_capabilities_all_fields():
    """Test _save_capabilities with all optional fields set.

    Covers lines 160, 164, 166, 168: All optional capability fields.
    """
    from mlx_manager.services.probe.service import _save_capabilities
    from mlx_manager.services.probe.steps import ProbeResult

    result = ProbeResult(
        model_type="text-gen",
        supports_native_tools=True,
        supports_thinking=True,
        tool_format="template",
        practical_max_tokens=32768,
        model_family="qwen",
        tool_parser_id="qwen_tool_tag",
        thinking_parser_id="think_tag",
        supports_multi_image=False,
        supports_video=False,
        embedding_dimensions=768,
        max_sequence_length=2048,
        is_normalized=True,
        supports_tts=False,
        supports_stt=False,
    )

    with patch(
        "mlx_manager.services.model_registry.update_model_capabilities",
        new_callable=AsyncMock,
    ) as mock_update:
        await _save_capabilities("test/model", result)

        # Verify all fields were passed
        mock_update.assert_called_once()
        call_kwargs = mock_update.call_args[1]

        assert call_kwargs["model_type"] == "text-gen"
        assert call_kwargs["supports_native_tools"] is True
        assert call_kwargs["supports_thinking"] is True
        assert call_kwargs["tool_format"] == "template"
        assert call_kwargs["practical_max_tokens"] == 32768
        assert call_kwargs["model_family"] == "qwen"
        assert call_kwargs["tool_parser_id"] == "qwen_tool_tag"
        assert call_kwargs["thinking_parser_id"] == "think_tag"
        assert call_kwargs["supports_multi_image"] is False
        assert call_kwargs["supports_video"] is False
        assert call_kwargs["embedding_dimensions"] == 768
        assert call_kwargs["max_sequence_length"] == 2048
        assert call_kwargs["is_normalized"] is True
        assert call_kwargs["supports_tts"] is False
        assert call_kwargs["supports_stt"] is False
        assert call_kwargs["probe_version"] == 2


@pytest.mark.asyncio
async def test_save_capabilities_minimal_fields():
    """Test _save_capabilities with minimal fields (only model_type).

    Ensures that None fields are NOT included in the update dict.
    """
    from mlx_manager.services.probe.service import _save_capabilities
    from mlx_manager.services.probe.steps import ProbeResult

    result = ProbeResult(
        model_type="embeddings",
        # All other fields remain None
    )

    with patch(
        "mlx_manager.services.model_registry.update_model_capabilities",
        new_callable=AsyncMock,
    ) as mock_update:
        await _save_capabilities("test/embedding-model", result)

        mock_update.assert_called_once()
        call_kwargs = mock_update.call_args[1]

        # Only model_type and probe_version should be set
        assert call_kwargs["model_type"] == "embeddings"
        assert call_kwargs["probe_version"] == 2

        # None fields should not be in the dict
        assert "supports_native_tools" not in call_kwargs
        assert "supports_thinking" not in call_kwargs
        assert "tool_format" not in call_kwargs
        assert "practical_max_tokens" not in call_kwargs
        assert "model_family" not in call_kwargs
        assert "tool_parser_id" not in call_kwargs
        assert "thinking_parser_id" not in call_kwargs


@pytest.mark.asyncio
async def test_detect_type_exception():
    """Test that detect_type failures are caught and yielded properly."""
    mock_pool = MagicMock()
    mock_pool._models = {}

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            side_effect=ValueError("Cannot determine model type"),
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/invalid-model"):
            steps.append(step)

        # Should fail at detect_type
        detect_steps = [s for s in steps if s.step == "detect_type"]
        assert len(detect_steps) == 2  # running + failed

        failed_step = next(s for s in detect_steps if s.status == "failed")
        assert "Cannot determine model type" in failed_step.error

        # Should not proceed further
        load_steps = [s for s in steps if s.step == "load_model"]
        assert len(load_steps) == 0


@pytest.mark.asyncio
async def test_no_strategy_for_model_type():
    """Test graceful handling when no probe strategy exists for model type."""
    from mlx_manager.mlx_server.models.types import ModelType

    mock_pool = MagicMock()
    mock_pool._models = {}

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=None,  # No strategy registered!
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should fail at find_strategy
        strategy_steps = [s for s in steps if s.step == "find_strategy"]
        assert len(strategy_steps) == 1
        assert strategy_steps[0].status == "failed"
        assert "No probe strategy registered" in strategy_steps[0].error

        # Should not proceed to loading
        load_steps = [s for s in steps if s.step == "load_model"]
        assert len(load_steps) == 0


@pytest.mark.asyncio
async def test_save_results_exception():
    """Test that save_results failures are caught and yielded."""
    from mlx_manager.mlx_server.models.types import ModelType

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool.unload_model = AsyncMock()

    async def successful_probe(*args, **kwargs):
        if False:  # pragma: no cover
            yield None

    mock_strategy = MagicMock()
    mock_strategy.probe = successful_probe

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=mock_strategy,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            side_effect=Exception("Database connection lost"),
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should have failed save_results step
        save_steps = [s for s in steps if s.step == "save_results"]
        assert any(s.status == "failed" for s in save_steps)

        failed_save = next(s for s in save_steps if s.status == "failed")
        assert "Database connection lost" in failed_save.error

        # Should still attempt cleanup
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert len(cleanup_steps) > 0


@pytest.mark.asyncio
async def test_cleanup_exception():
    """Test that cleanup failures are caught and yielded."""
    from mlx_manager.mlx_server.models.types import ModelType

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()

    mock_pool = MagicMock()
    mock_pool._models = {}  # Not preloaded
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool.unload_model = AsyncMock(side_effect=Exception("Unload failed"))

    async def successful_probe(*args, **kwargs):
        if False:  # pragma: no cover
            yield None

    mock_strategy = MagicMock()
    mock_strategy.probe = successful_probe

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=TypeDetectionResult(ModelType.TEXT_GEN, "config_field", "TestArch"),
        ),
        patch(
            "mlx_manager.services.probe.service.get_probe_strategy",
            return_value=mock_strategy,
        ),
        patch(
            "mlx_manager.services.probe.service._save_capabilities",
            new_callable=AsyncMock,
        ),
    ):
        from mlx_manager.services.probe.service import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should have failed cleanup step
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        failed_cleanup = next((s for s in cleanup_steps if s.status == "failed"), None)
        assert failed_cleanup is not None
        assert "Unload failed" in failed_cleanup.error

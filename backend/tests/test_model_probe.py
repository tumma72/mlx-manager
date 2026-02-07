"""Tests for model probe service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.services.model_probe import ProbeStep


@pytest.mark.asyncio
async def test_probe_model_full_flow():
    """Test full probe flow with mocked dependencies."""
    # Mock model pool
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.config = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "max_position_embeddings": 32768,
    }
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 2.0

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}  # model not pre-loaded
    mock_pool.unload_model = AsyncMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.services.model_probe._save_capabilities",
            new_callable=AsyncMock,
        ),
    ):
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should have steps for: load, context, thinking, tools, save, cleanup
        assert len(steps) >= 6

        # Verify step types
        step_names = [s.step for s in steps]
        assert "load_model" in step_names
        assert "check_context" in step_names
        assert "test_thinking" in step_names
        assert "test_tools" in step_names
        assert "save_results" in step_names
        assert "cleanup" in step_names

        # Model should be unloaded since it wasn't pre-loaded
        mock_pool.unload_model.assert_called_once_with("test/model")


@pytest.mark.asyncio
async def test_probe_model_already_loaded():
    """Test that pre-loaded models are NOT unloaded after probe."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.config = {
        "num_hidden_layers": 12,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "max_position_embeddings": 8192,
    }
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 1.0
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {"test/model": mock_loaded}  # Already loaded!
    mock_pool.unload_model = AsyncMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.model_probe._save_capabilities",
            new_callable=AsyncMock,
        ),
    ):
        steps = []
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Should NOT unload since model was already loaded
        mock_pool.unload_model.assert_not_called()

        # Cleanup step should be skipped
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert len(cleanup_steps) == 1
        assert cleanup_steps[0].status == "skipped"


@pytest.mark.asyncio
async def test_probe_model_load_failure():
    """Test probe handles model load failure gracefully."""
    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(side_effect=Exception("Model not found"))
    mock_pool._models = {}

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool",
        return_value=mock_pool,
    ):
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("nonexistent/model"):
            steps.append(step)

        # Should still produce steps, but load_model step should fail
        assert len(steps) >= 1
        load_steps = [s for s in steps if s.step == "load_model"]
        failed_step = next(s for s in load_steps if s.status == "failed")
        assert "Model not found" in failed_step.error


@pytest.mark.asyncio
async def test_probe_model_tokenizer_none():
    """Test probe when tokenizer is None (skips tool/thinking tests)."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.config = {
        "num_hidden_layers": 12,
        "max_position_embeddings": 8192,
    }
    mock_loaded.tokenizer = None  # No tokenizer
    mock_loaded.size_gb = 1.0

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
            "mlx_manager.services.model_probe._save_capabilities",
            new_callable=AsyncMock,
        ),
    ):
        steps = []
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Thinking and tools tests should be skipped
        thinking_step = next(s for s in steps if s.step == "test_thinking")
        assert thinking_step.status == "skipped"

        tools_step = next(s for s in steps if s.step == "test_tools")
        assert tools_step.status == "skipped"


@pytest.mark.asyncio
async def test_probe_step_to_sse():
    """Test ProbeStep SSE serialization."""
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


@pytest.mark.asyncio
async def test_probe_step_with_error():
    """Test ProbeStep SSE serialization with error."""
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


@pytest.mark.asyncio
async def test_estimate_practical_max_tokens():
    """Test practical max tokens estimation."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.config = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "max_position_embeddings": 32768,
    }
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 2.0

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}
    mock_pool.unload_model = AsyncMock()

    # Mock HF download to return config
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_loaded.config, f)
        config_path = f.name

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch("huggingface_hub.hf_hub_download", return_value=config_path),
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=16.0),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.model_probe._save_capabilities",
            new_callable=AsyncMock,
        ),
    ):
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Check context step - find the completed one
        context_steps = [s for s in steps if s.step == "check_context"]
        completed_context = next(s for s in context_steps if s.status == "completed")
        assert completed_context.capability == "practical_max_tokens"
        assert completed_context.value is not None
        assert isinstance(completed_context.value, int)
        assert completed_context.value > 0


@pytest.mark.asyncio
async def test_probe_saves_capabilities():
    """Test that probe saves capabilities to database."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.config = {
        "num_hidden_layers": 12,
        "max_position_embeddings": 8192,
    }
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 1.0

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}
    mock_pool.unload_model = AsyncMock()

    mock_save = AsyncMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.model_probe._save_capabilities",
            mock_save,
        ),
    ):
        steps = []
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Verify save was called
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == "test/model"
        # Verify ProbeResult was passed
        probe_result = call_args[0][1]
        assert probe_result.supports_native_tools is True
        assert probe_result.supports_thinking is False

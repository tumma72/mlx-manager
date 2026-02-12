"""Additional tests for model probe service to increase coverage."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_probe_context_check_failure():
    """Test probe handles context check failure gracefully."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
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
            "mlx_manager.services.model_probe._estimate_practical_max_tokens",
            side_effect=Exception("Context estimation failed"),
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
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Check that context step failed
        context_steps = [s for s in steps if s.step == "check_context"]
        failed_steps = [s for s in context_steps if s.status == "failed"]
        assert len(failed_steps) == 1
        assert "Context estimation failed" in failed_steps[0].error


@pytest.mark.asyncio
async def test_probe_thinking_test_failure():
    """Test probe handles thinking test failure gracefully."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
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
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            side_effect=Exception("Thinking test failed"),
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
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

        # Check that thinking step failed
        thinking_steps = [s for s in steps if s.step == "test_thinking"]
        failed_steps = [s for s in thinking_steps if s.status == "failed"]
        assert len(failed_steps) == 1
        assert "Thinking test failed" in failed_steps[0].error


@pytest.mark.asyncio
async def test_probe_tools_test_failure():
    """Test probe handles tools test failure gracefully."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
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
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            side_effect=Exception("Tool test failed"),
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

        # Check that tools step failed
        tools_steps = [s for s in steps if s.step == "test_tools"]
        failed_steps = [s for s in tools_steps if s.status == "failed"]
        assert len(failed_steps) == 1
        assert "Tool test failed" in failed_steps[0].error


@pytest.mark.asyncio
async def test_probe_save_failure():
    """Test probe handles save failure gracefully."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
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
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.model_probe._save_capabilities",
            side_effect=Exception("Database error"),
        ),
    ):
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Check that save step failed
        save_steps = [s for s in steps if s.step == "save_results"]
        failed_steps = [s for s in save_steps if s.status == "failed"]
        assert len(failed_steps) == 1
        assert "Database error" in failed_steps[0].error


@pytest.mark.asyncio
async def test_probe_cleanup_failure():
    """Test probe handles cleanup failure gracefully."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 1.0

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {}  # Not pre-loaded
    mock_pool.unload_model = AsyncMock(side_effect=Exception("Unload failed"))

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
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

        # Check that cleanup step failed
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        failed_steps = [s for s in cleanup_steps if s.status == "failed"]
        assert len(failed_steps) == 1
        assert "Unload failed" in failed_steps[0].error


@pytest.mark.asyncio
async def test_probe_preloaded_model_updates_capabilities():
    """Test that pre-loaded models get their capabilities updated."""
    from datetime import UTC, datetime

    from mlx_manager.models import Model

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 1.0
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {"test/model": mock_loaded}  # Pre-loaded

    # Mock DB capabilities
    mock_caps = Model(
        repo_id="test/model",
        supports_native_tools=True,
        supports_thinking=False,
        practical_max_tokens=4096,
        downloaded_at=datetime.now(tz=UTC),
    )

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_caps
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = AsyncMock()

    mock_get_session = MagicMock(return_value=mock_session)

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.services.model_probe._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch("mlx_manager.database.get_session", mock_get_session),
    ):
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Verify cleanup was skipped
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert cleanup_steps[0].status == "skipped"

        # Verify capabilities were loaded from DB
        assert mock_loaded.capabilities == mock_caps


@pytest.mark.asyncio
async def test_probe_preloaded_model_db_fetch_failure():
    """Test that DB fetch failure in pre-loaded model path is handled silently."""
    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/model"
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.size_gb = 1.0
    mock_loaded.capabilities = None

    mock_pool = MagicMock()
    mock_pool.get_model = AsyncMock(return_value=mock_loaded)
    mock_pool._models = {"test/model": mock_loaded}  # Pre-loaded

    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("DB error")
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = AsyncMock()

    mock_get_session = MagicMock(return_value=mock_session)

    with (
        patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.model_probe._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch("mlx_manager.database.get_session", mock_get_session),
    ):
        from mlx_manager.services.model_probe import probe_model

        steps = []
        async for step in probe_model("test/model"):
            steps.append(step)

        # Verify cleanup was skipped (error was silently handled)
        cleanup_steps = [s for s in steps if s.step == "cleanup"]
        assert cleanup_steps[0].status == "skipped"


@pytest.mark.asyncio
async def test_estimate_practical_max_tokens_no_max_pos():
    """Test _estimate_practical_max_tokens when max_position_embeddings is missing."""
    import tempfile

    mock_loaded = MagicMock()
    mock_loaded.size_gb = 2.0

    config = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 64,
        # No max_position_embeddings
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    with patch("huggingface_hub.hf_hub_download", return_value=config_path):
        from mlx_manager.services.model_probe import _estimate_practical_max_tokens

        result = _estimate_practical_max_tokens("test/model", mock_loaded)
        assert result is None


@pytest.mark.asyncio
async def test_estimate_practical_max_tokens_missing_params():
    """Test _estimate_practical_max_tokens when critical params are missing."""
    import tempfile

    mock_loaded = MagicMock()
    mock_loaded.size_gb = 2.0

    config = {
        "max_position_embeddings": 32768,
        # Missing num_hidden_layers, num_key_value_heads, head_dim
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    with patch("huggingface_hub.hf_hub_download", return_value=config_path):
        from mlx_manager.services.model_probe import _estimate_practical_max_tokens

        result = _estimate_practical_max_tokens("test/model", mock_loaded)
        # Should return max_pos as int when params missing
        assert result == 32768


@pytest.mark.asyncio
async def test_estimate_practical_max_tokens_computed_head_dim():
    """Test _estimate_practical_max_tokens when head_dim must be computed."""
    import tempfile

    mock_loaded = MagicMock()
    mock_loaded.size_gb = 2.0

    config = {
        "max_position_embeddings": 32768,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        # No head_dim, but has hidden_size and num_attention_heads
        "hidden_size": 2048,
        "num_attention_heads": 32,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    with (
        patch("huggingface_hub.hf_hub_download", return_value=config_path),
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=16.0),
    ):
        from mlx_manager.services.model_probe import _estimate_practical_max_tokens

        result = _estimate_practical_max_tokens("test/model", mock_loaded)
        # Should compute head_dim = 2048 / 32 = 64 and return a value
        assert result is not None
        assert isinstance(result, int)
        assert result >= 512


@pytest.mark.asyncio
async def test_estimate_practical_max_tokens_low_memory():
    """Test _estimate_practical_max_tokens when available memory is too low."""
    import tempfile

    mock_loaded = MagicMock()
    mock_loaded.size_gb = 15.0  # Very large model

    config = {
        "max_position_embeddings": 32768,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 64,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    with (
        patch("huggingface_hub.hf_hub_download", return_value=config_path),
        patch("mlx_manager.mlx_server.utils.memory.get_device_memory_gb", return_value=16.0),
    ):
        from mlx_manager.services.model_probe import _estimate_practical_max_tokens

        result = _estimate_practical_max_tokens("test/model", mock_loaded)
        # Should return minimum of max_pos and 2048
        assert result == 2048


@pytest.mark.asyncio
async def test_estimate_practical_max_tokens_exception():
    """Test _estimate_practical_max_tokens when an exception occurs."""
    mock_loaded = MagicMock()
    mock_loaded.size_gb = 2.0

    with patch("huggingface_hub.hf_hub_download", side_effect=Exception("HF error")):
        from mlx_manager.services.model_probe import _estimate_practical_max_tokens

        result = _estimate_practical_max_tokens("test/model", mock_loaded)
        assert result is None


@pytest.mark.asyncio
async def test_save_capabilities_create_new():
    """Test _save_capabilities creates new Model record when none exists."""
    from mlx_manager.services.model_probe import ProbeResult, _save_capabilities

    result = ProbeResult(
        supports_native_tools=True, supports_thinking=False, practical_max_tokens=4096
    )

    # Need to mock update_model_capabilities since _save_capabilities now uses it
    with patch(
        "mlx_manager.services.model_registry.update_model_capabilities"
    ) as mock_update_caps:
        mock_update_caps.return_value = AsyncMock()
        await _save_capabilities("test/model", result)

        # Verify update_model_capabilities was called with correct arguments
        assert mock_update_caps.called
        call_args = mock_update_caps.call_args
        assert call_args[0][0] == "test/model"
        kwargs = call_args[1]
        assert kwargs.get("supports_native_tools") is True
        assert kwargs.get("supports_thinking") is False
        assert kwargs.get("practical_max_tokens") == 4096
        assert kwargs.get("probe_version") == 2


@pytest.mark.asyncio
async def test_save_capabilities_update_existing():
    """Test _save_capabilities updates existing Model record."""

    from mlx_manager.services.model_probe import ProbeResult, _save_capabilities

    result = ProbeResult(
        supports_native_tools=True, supports_thinking=True, practical_max_tokens=8192
    )

    # Need to mock update_model_capabilities since _save_capabilities now uses it
    with patch(
        "mlx_manager.services.model_registry.update_model_capabilities"
    ) as mock_update_caps:
        mock_update_caps.return_value = AsyncMock()
        await _save_capabilities("test/model", result)

        # Verify update_model_capabilities was called with correct arguments
        assert mock_update_caps.called
        call_args = mock_update_caps.call_args
        assert call_args[0][0] == "test/model"
        kwargs = call_args[1]
        assert kwargs.get("supports_native_tools") is True
        assert kwargs.get("supports_thinking") is True
        assert kwargs.get("practical_max_tokens") == 8192
        assert kwargs.get("probe_version") == 2

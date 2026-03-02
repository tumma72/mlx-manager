"""Additional tests for ModelPoolManager covering uncovered lines.

Targets:
- pool.py lines: 111-153, 161-167, 246-267, 432-435, 468-469,
  506-510, 532-557, 576-581, 587-592, 618-619, 907, 1026-1027
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.pool import (
    LoadedModel,
    ModelPoolManager,
    ProfileSettings,
)
from mlx_manager.mlx_server.models.types import ModelType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pool() -> ModelPoolManager:
    """Create a fresh ModelPoolManager for each test."""
    return ModelPoolManager(max_memory_gb=48.0, max_models=4)


@pytest.fixture
def mock_adapter():
    """Create a mock ModelAdapter."""
    adapter = MagicMock()
    adapter.configure = MagicMock()
    adapter.reset_to_defaults = MagicMock()
    return adapter


@pytest.fixture
def loaded_model_with_adapter(mock_adapter):
    """Create a LoadedModel with a mock adapter."""
    model = LoadedModel(
        model_id="test-model",
        model=MagicMock(),
        tokenizer=MagicMock(),
        size_gb=4.0,
    )
    model.adapter = mock_adapter
    return model


# ============================================================================
# TestProfileSettings - register/unregister (lines 111-167)
# ============================================================================


class TestRegisterProfileSettings:
    """Tests for register_profile_settings() method (lines 111-153)."""

    def test_register_adds_to_profile_settings(self, pool):
        """Registers profile settings in the internal dict (line 111-112)."""
        settings = ProfileSettings(system_prompt="Hello!")
        pool.register_profile_settings("test/model", settings)
        assert "test/model" in pool._profile_settings
        assert pool._profile_settings["test/model"].system_prompt == "Hello!"

    def test_register_no_loaded_model_no_reconfigure(self, pool):
        """If model not loaded, no adapter reconfiguration happens."""
        settings = ProfileSettings(system_prompt="Hello!")
        # No loaded model - should not raise
        pool.register_profile_settings("test/model", settings)
        assert "test/model" in pool._profile_settings

    def test_register_with_loaded_model_reconfigures_adapter(self, pool, loaded_model_with_adapter):
        """If model is loaded with adapter, adapter.configure() is called (lines 115-153)."""
        pool._models["test-model"] = loaded_model_with_adapter

        settings = ProfileSettings(
            system_prompt="System prompt",
            enable_tool_injection=True,
            template_options={"thinking": True},
        )
        pool.register_profile_settings("test-model", settings)

        loaded_model_with_adapter.adapter.configure.assert_called_once()
        call_kwargs = loaded_model_with_adapter.adapter.configure.call_args[1]
        assert call_kwargs["system_prompt"] == "System prompt"
        assert call_kwargs["enable_tool_injection"] is True

    def test_register_with_loaded_model_no_adapter_no_reconfigure(self, pool):
        """If model is loaded but has no adapter, no reconfiguration happens."""
        model = LoadedModel(
            model_id="test-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        # adapter is None by default
        pool._models["test-model"] = model

        settings = ProfileSettings(system_prompt="Hello!")
        # Should not raise even though adapter is None
        pool.register_profile_settings("test-model", settings)

    def test_register_with_tool_parser_id_resolves_parser(self, pool, loaded_model_with_adapter):
        """tool_parser_id triggers parser resolution (lines 120-130)."""
        pool._models["test-model"] = loaded_model_with_adapter

        mock_parser = MagicMock()
        with patch(
            "mlx_manager.mlx_server.parsers.resolve_tool_parser", return_value=mock_parser
        ) as mock_resolve:
            settings = ProfileSettings(tool_parser_id="hermes")
            pool.register_profile_settings("test-model", settings)
            mock_resolve.assert_called_once_with("hermes")

        # Verify parser was passed to configure
        call_kwargs = loaded_model_with_adapter.adapter.configure.call_args[1]
        assert call_kwargs.get("tool_parser") == mock_parser

    def test_register_with_unknown_tool_parser_id_logs_warning(
        self, pool, loaded_model_with_adapter
    ):
        """Unknown tool_parser_id logs warning, doesn't raise (lines 125-130)."""
        pool._models["test-model"] = loaded_model_with_adapter

        with patch(
            "mlx_manager.mlx_server.parsers.resolve_tool_parser",
            side_effect=KeyError("unknown_parser"),
        ):
            settings = ProfileSettings(tool_parser_id="unknown_parser")
            # Should not raise
            pool.register_profile_settings("test-model", settings)

        # configure() is still called but without tool_parser
        call_kwargs = loaded_model_with_adapter.adapter.configure.call_args[1]
        assert "tool_parser" not in call_kwargs

    def test_register_with_thinking_parser_id_resolves_parser(
        self, pool, loaded_model_with_adapter
    ):
        """thinking_parser_id triggers parser resolution (lines 131-141)."""
        pool._models["test-model"] = loaded_model_with_adapter

        mock_parser = MagicMock()
        with patch(
            "mlx_manager.mlx_server.parsers.resolve_thinking_parser", return_value=mock_parser
        ) as mock_resolve:
            settings = ProfileSettings(thinking_parser_id="think_tag")
            pool.register_profile_settings("test-model", settings)
            mock_resolve.assert_called_once_with("think_tag")

        call_kwargs = loaded_model_with_adapter.adapter.configure.call_args[1]
        assert call_kwargs.get("thinking_parser") == mock_parser

    def test_register_with_unknown_thinking_parser_id_logs_warning(
        self, pool, loaded_model_with_adapter
    ):
        """Unknown thinking_parser_id logs warning, doesn't raise (lines 136-141)."""
        pool._models["test-model"] = loaded_model_with_adapter

        with patch(
            "mlx_manager.mlx_server.parsers.resolve_thinking_parser",
            side_effect=KeyError("unknown_parser"),
        ):
            settings = ProfileSettings(thinking_parser_id="unknown_parser")
            # Should not raise
            pool.register_profile_settings("test-model", settings)

        # configure() is still called without thinking_parser
        call_kwargs = loaded_model_with_adapter.adapter.configure.call_args[1]
        assert "thinking_parser" not in call_kwargs


class TestUnregisterProfileSettings:
    """Tests for unregister_profile_settings() method (lines 155-167)."""

    def test_unregister_removes_from_profile_settings(self, pool):
        """Unregistering removes settings from dict (lines 161-162)."""
        settings = ProfileSettings(system_prompt="Hello!")
        pool._profile_settings["test/model"] = settings

        pool.unregister_profile_settings("test/model")

        assert "test/model" not in pool._profile_settings

    def test_unregister_nonexistent_model_no_error(self, pool):
        """Unregistering non-existent model does not raise."""
        pool.unregister_profile_settings("nonexistent/model")  # Should not raise

    def test_unregister_with_loaded_model_resets_adapter(self, pool, loaded_model_with_adapter):
        """If model is loaded with adapter, adapter.reset_to_defaults() is called
        (lines 165-167)."""
        pool._models["test-model"] = loaded_model_with_adapter
        pool._profile_settings["test-model"] = ProfileSettings()

        pool.unregister_profile_settings("test-model")

        loaded_model_with_adapter.adapter.reset_to_defaults.assert_called_once()

    def test_unregister_with_loaded_model_no_adapter_no_reset(self, pool):
        """If model has no adapter, reset_to_defaults is not called."""
        model = LoadedModel(
            model_id="test-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        pool._models["test-model"] = model
        pool._profile_settings["test-model"] = ProfileSettings()

        # Should not raise even though adapter is None
        pool.unregister_profile_settings("test-model")


# ============================================================================
# TestModelSizeFromDisk - estimate from HF cache (lines 246-267)
# ============================================================================


class TestModelSizeFromDisk:
    """Tests for _estimate_model_size() reading from HF cache (lines 246-266)."""

    def test_estimate_from_actual_disk_files(self, pool, tmp_path):
        """Estimates size from actual safetensors files in HF cache (lines 246-265)."""
        # Create a fake HF cache structure
        model_id = "mlx-community/test-model-7b"
        cache_name = "models--mlx-community--test-model-7b"
        snapshot_dir = tmp_path / cache_name / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Create fake safetensors files
        weight_file = snapshot_dir / "model.safetensors"
        weight_file.write_bytes(b"x" * (4 * 1024**3))  # 4GB

        with patch("mlx_manager.config.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            size = pool._estimate_model_size(model_id)

        # Should be ~4.2GB (4GB * 1.05 overhead)
        assert size == pytest.approx(4.0 * 1.05, rel=0.01)

    def test_estimate_ignores_non_weight_files(self, pool, tmp_path):
        """Only counts .safetensors, .bin, .gguf files (lines 254-256)."""
        model_id = "mlx-community/test-model"
        cache_name = "models--mlx-community--test-model"
        snapshot_dir = tmp_path / cache_name / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Create weight files
        (snapshot_dir / "model.safetensors").write_bytes(b"x" * (1 * 1024**3))

        # Create non-weight files (should be ignored)
        (snapshot_dir / "config.json").write_bytes(b"{}") * 1000
        (snapshot_dir / "tokenizer.json").write_bytes(b"{}") * 5000

        with patch("mlx_manager.config.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            size = pool._estimate_model_size(model_id)

        # Only the 1GB safetensors file should count
        assert size == pytest.approx(1.0 * 1.05, rel=0.01)

    def test_estimate_falls_back_when_no_weight_files(self, pool, tmp_path):
        """Falls back to name-pattern when cache has no weight files (line 257-266)."""
        model_id = "mlx-community/test-7b-model"
        cache_name = "models--mlx-community--test-7b-model"
        snapshot_dir = tmp_path / cache_name / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Create only non-weight files
        (snapshot_dir / "config.json").write_text("{}")

        with patch("mlx_manager.config.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            size = pool._estimate_model_size(model_id)

        # Falls back to 7B pattern: 4.0
        assert size == 4.0

    def test_estimate_falls_back_on_exception(self, pool, tmp_path):
        """Falls back to name-pattern on exception reading cache (lines 266-267)."""
        model_id = "mlx-community/test-7b-model"

        # Cause an exception inside the try block by making sorted() raise
        with patch("mlx_manager.config.settings") as mock_settings:
            # Point to a non-existent path to trigger the exception path
            mock_settings.hf_cache_path = MagicMock()
            mock_settings.hf_cache_path.__truediv__ = MagicMock(
                side_effect=OSError("permission denied")
            )
            size = pool._estimate_model_size(model_id)

        # Falls back to 7B name-pattern: 4.0
        assert size == 4.0

    def test_estimate_uses_latest_snapshot(self, pool, tmp_path):
        """Uses the most recent snapshot when multiple snapshots exist."""
        import time

        model_id = "mlx-community/test-model"
        cache_name = "models--mlx-community--test-model"
        snapshots_dir = tmp_path / cache_name / "snapshots"
        snapshots_dir.mkdir(parents=True)

        # Old snapshot with small file
        old_snap = snapshots_dir / "old123"
        old_snap.mkdir()
        (old_snap / "model.safetensors").write_bytes(b"x" * (1 * 1024**3))  # 1GB

        time.sleep(0.01)  # Ensure different mtime

        # New snapshot with larger file
        new_snap = snapshots_dir / "new456"
        new_snap.mkdir()
        (new_snap / "model.safetensors").write_bytes(b"x" * (2 * 1024**3))  # 2GB

        with patch("mlx_manager.config.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            size = pool._estimate_model_size(model_id)

        # Should use the new snapshot (2GB + 5% overhead)
        assert size == pytest.approx(2.0 * 1.05, rel=0.01)


# ============================================================================
# TestLoadModelDBCaching - DB-cached model type (lines 432-435)
# ============================================================================


class TestLoadModelDBCachedType:
    """Tests for DB-cached model type lookup in _load_model (lines 432-435)."""

    @pytest.mark.asyncio
    async def test_load_model_uses_db_cached_type(self, pool):
        """When DB has cached model type, it is used instead of detection (lines 432-435)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.tool_parser.parser_id = "null"
        mock_adapter.thinking_parser.parser_id = "null"
        mock_adapter.post_load_configure = AsyncMock()

        with (
            patch("mlx_manager.mlx_server.models.pool.detect_model_type") as mock_detect,
            patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
        ):
            # Mock the DB session to return a cached model type
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = "text-gen"

            # Mock get_session as a context manager
            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(return_value=mock_result)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                result = await pool._load_model("test/model")

        # Detection should NOT have been called since DB had cached type
        mock_detect.assert_not_called()
        assert result.model_type == "text-gen"

    @pytest.mark.asyncio
    async def test_load_model_falls_back_to_detection_when_no_db_type(self, pool):
        """Falls back to detect_model_type when DB has no cached type (line 442-443)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.tool_parser.parser_id = "null"
        mock_adapter.thinking_parser.parser_id = "null"
        mock_adapter.post_load_configure = AsyncMock()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.TEXT_GEN,
            ) as mock_detect,
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
        ):
            # Mock DB returning None (no cached type)
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(return_value=mock_result)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                await pool._load_model("test/model")

        # detect_model_type should be called as fallback
        mock_detect.assert_called_once_with("test/model")


# ============================================================================
# TestLoadModelAudioError - RuntimeError wrapping (lines 468-469)
# ============================================================================


class TestLoadModelAudioRuntimeError:
    """Tests for audio model load error wrapping (lines 468-469)."""

    @pytest.mark.asyncio
    async def test_audio_load_value_error_wrapped(self, pool):
        """ValueError from mlx_audio.load_model is wrapped in RuntimeError (line 469)."""
        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.AUDIO,
            ),
            patch(
                "mlx_manager.database.get_session",
                side_effect=Exception("no db"),
            ),
        ):
            # The exception from get_session causes fallback to detection
            # Now mock mlx_audio
            with patch(
                "mlx_audio.utils.load_model",
                side_effect=ValueError("Unsupported audio architecture"),
            ):
                with patch(
                    "asyncio.to_thread",
                    side_effect=ValueError("Unsupported audio architecture"),
                ):
                    pool._loading["test/audio-model"] = asyncio.Event()
                    with pytest.raises(RuntimeError, match="Failed to load model"):
                        try:
                            await pool._load_model("test/audio-model")
                        finally:
                            pool._loading.pop("test/audio-model", None)


# ============================================================================
# TestLoadModelCapabilities - DB capabilities loading (lines 506-510)
# ============================================================================


class TestLoadModelCapabilities:
    """Tests for loading DB capabilities and parsers (lines 532-557)."""

    @pytest.mark.asyncio
    async def test_capabilities_with_parsers_are_resolved(self, pool):
        """Tool and thinking parsers from capabilities are resolved (lines 532-557)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.tool_parser.parser_id = "hermes"
        mock_adapter.thinking_parser.parser_id = "think_tag"
        mock_adapter.post_load_configure = AsyncMock()

        # Create mock capabilities with parser IDs
        mock_caps = MagicMock()
        mock_caps.capability_type = "inference"
        mock_caps.model_family = "hermes"
        mock_caps.tool_parser_id = "hermes"
        mock_caps.thinking_parser_id = "think_tag"

        mock_model_record = MagicMock()
        mock_model_record.capabilities = mock_caps

        mock_tool_parser = MagicMock()
        mock_thinking_parser = MagicMock()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.TEXT_GEN,
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                return_value=mock_adapter,
            ) as mock_create_adapter,
            patch(
                "mlx_manager.mlx_server.parsers.resolve_tool_parser",
                return_value=mock_tool_parser,
            ),
            patch(
                "mlx_manager.mlx_server.parsers.resolve_thinking_parser",
                return_value=mock_thinking_parser,
            ),
        ):
            # Mock DB: first call (model type) returns None,
            # second call (capabilities) returns model
            type_result = MagicMock()
            type_result.scalar_one_or_none.return_value = None

            caps_result = MagicMock()
            caps_result.scalar_one_or_none.return_value = mock_model_record

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(side_effect=[type_result, caps_result])

            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                await pool._load_model("test/hermes-model")

        # Verify create_adapter was called with tool_parser and thinking_parser
        create_adapter_kwargs = mock_create_adapter.call_args[1]
        assert create_adapter_kwargs.get("tool_parser") == mock_tool_parser
        assert create_adapter_kwargs.get("thinking_parser") == mock_thinking_parser

    @pytest.mark.asyncio
    async def test_unknown_tool_parser_in_capabilities_logs_warning(self, pool):
        """Unknown tool parser ID from capabilities logs warning (lines 547-552)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.tool_parser.parser_id = "null"
        mock_adapter.thinking_parser.parser_id = "null"
        mock_adapter.post_load_configure = AsyncMock()

        mock_caps = MagicMock()
        mock_caps.capability_type = "inference"
        mock_caps.model_family = "default"
        mock_caps.tool_parser_id = "unknown_parser"
        mock_caps.thinking_parser_id = None

        mock_model_record = MagicMock()
        mock_model_record.capabilities = mock_caps

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.TEXT_GEN,
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "mlx_manager.mlx_server.parsers.resolve_tool_parser",
                side_effect=KeyError("unknown_parser"),
            ),
        ):
            type_result = MagicMock()
            type_result.scalar_one_or_none.return_value = None

            caps_result = MagicMock()
            caps_result.scalar_one_or_none.return_value = mock_model_record

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(side_effect=[type_result, caps_result])

            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                # Should not raise, just log warning
                result = await pool._load_model("test/model")

        assert result is not None


# ============================================================================
# TestLoadModelProfileSettings - profile-level parser overrides (lines 576-592)
# ============================================================================


class TestLoadModelProfileSettings:
    """Tests for profile settings parser overrides during load (lines 576-592)."""

    @pytest.mark.asyncio
    async def test_profile_tool_parser_applied_when_no_caps_parser(self, pool):
        """Profile's tool_parser_id is used when capabilities have no parser (lines 576-581)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.tool_parser.parser_id = "hermes"
        mock_adapter.thinking_parser.parser_id = "null"
        mock_adapter.post_load_configure = AsyncMock()

        mock_tool_parser = MagicMock()

        # Register profile settings with a tool_parser_id
        pool._profile_settings["test/model"] = ProfileSettings(
            tool_parser_id="hermes",
            system_prompt="Custom prompt",
        )

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.TEXT_GEN,
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                return_value=mock_adapter,
            ) as mock_create_adapter,
            patch(
                "mlx_manager.mlx_server.parsers.resolve_tool_parser",
                return_value=mock_tool_parser,
            ),
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
        ):
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(return_value=mock_result)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                await pool._load_model("test/model")

        # Verify profile tool_parser was passed to create_adapter
        create_adapter_kwargs = mock_create_adapter.call_args[1]
        assert create_adapter_kwargs.get("tool_parser") == mock_tool_parser
        assert create_adapter_kwargs.get("system_prompt") == "Custom prompt"

    @pytest.mark.asyncio
    async def test_profile_unknown_tool_parser_logs_warning(self, pool):
        """Unknown profile tool_parser_id logs warning (lines 580-581)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.tool_parser.parser_id = "null"
        mock_adapter.thinking_parser.parser_id = "null"
        mock_adapter.post_load_configure = AsyncMock()

        pool._profile_settings["test/model"] = ProfileSettings(
            tool_parser_id="nonexistent_parser",
        )

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.TEXT_GEN,
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "mlx_manager.mlx_server.parsers.resolve_tool_parser",
                side_effect=KeyError("nonexistent_parser"),
            ),
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
        ):
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(return_value=mock_result)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                # Should not raise
                result = await pool._load_model("test/model")

        assert result is not None

    @pytest.mark.asyncio
    async def test_profile_thinking_parser_applied(self, pool):
        """Profile's thinking_parser_id is applied (lines 586-592)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.tool_parser.parser_id = "null"
        mock_adapter.thinking_parser.parser_id = "think_tag"
        mock_adapter.post_load_configure = AsyncMock()

        mock_thinking_parser = MagicMock()

        pool._profile_settings["test/model"] = ProfileSettings(
            thinking_parser_id="think_tag",
        )

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.TEXT_GEN,
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                return_value=mock_adapter,
            ) as mock_create_adapter,
            patch(
                "mlx_manager.mlx_server.parsers.resolve_thinking_parser",
                return_value=mock_thinking_parser,
            ),
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
        ):
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(return_value=mock_result)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                await pool._load_model("test/model")

        create_adapter_kwargs = mock_create_adapter.call_args[1]
        assert create_adapter_kwargs.get("thinking_parser") == mock_thinking_parser


# ============================================================================
# TestLoadModelAdapterFailure - adapter creation failure (lines 618-619)
# ============================================================================


class TestLoadModelAdapterCreationFailure:
    """Tests for graceful adapter creation failure (lines 618-619)."""

    @pytest.mark.asyncio
    async def test_adapter_creation_failure_is_logged_not_raised(self, pool):
        """If create_adapter fails, model is still loaded (lines 618-619)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.detect_model_type",
                return_value=ModelType.TEXT_GEN,
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                side_effect=RuntimeError("create_adapter failed"),
            ),
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
        ):
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session_ctx = MagicMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(return_value=mock_result)

            with patch("mlx_manager.database.get_session", return_value=mock_session_ctx):
                # Model loading should succeed despite adapter creation failure
                result = await pool._load_model("test/model")

        assert result is not None
        assert result.adapter is None  # Adapter not set when creation fails


# ============================================================================
# TestGetLoadedModel - get_loaded_model (line 907)
# ============================================================================


class TestGetLoadedModel:
    """Tests for get_loaded_model() method (line 907)."""

    def test_returns_model_when_loaded(self, pool):
        """Returns LoadedModel when model is in pool (line 907)."""
        model = LoadedModel(
            model_id="test-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        pool._models["test-model"] = model

        result = pool.get_loaded_model("test-model")
        assert result is model

    def test_returns_none_when_not_loaded(self, pool):
        """Returns None when model is not in pool (line 907)."""
        result = pool.get_loaded_model("nonexistent-model")
        assert result is None


# ============================================================================
# TestLoadModelAsType - adapter creation failure (lines 1026-1027)
# ============================================================================


class TestLoadModelAsTypeAdapterFailure:
    """Tests for adapter creation failure in _load_model_as_type (lines 1026-1027)."""

    @pytest.mark.asyncio
    async def test_adapter_creation_failure_is_logged(self, pool):
        """If create_adapter fails in _load_model_as_type, model still loads (lines 1026-1027)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0},
            ),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter",
                side_effect=RuntimeError("create_adapter failed"),
            ),
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
        ):
            result = await pool._load_model_as_type("test/model", ModelType.TEXT_GEN)

        assert result is not None
        assert result.adapter is None


# ============================================================================
# TestProfileSettings Dataclass
# ============================================================================


class TestProfileSettingsModel:
    """Tests for ProfileSettings Pydantic model."""

    def test_defaults_are_correct(self):
        """Default values are None/False."""
        settings = ProfileSettings()
        assert settings.system_prompt is None
        assert settings.enable_tool_injection is False
        assert settings.template_options is None
        assert settings.tool_parser_id is None
        assert settings.thinking_parser_id is None

    def test_custom_values(self):
        """Custom values are stored correctly."""
        settings = ProfileSettings(
            system_prompt="You are a helpful assistant.",
            enable_tool_injection=True,
            template_options={"thinking": True},
            tool_parser_id="hermes",
            thinking_parser_id="think_tag",
        )
        assert settings.system_prompt == "You are a helpful assistant."
        assert settings.enable_tool_injection is True
        assert settings.template_options == {"thinking": True}
        assert settings.tool_parser_id == "hermes"
        assert settings.thinking_parser_id == "think_tag"

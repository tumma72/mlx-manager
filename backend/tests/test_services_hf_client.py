"""Tests for the HuggingFace client service."""

from unittest.mock import MagicMock, patch

import pytest

from mlx_manager.services.hf_api import ModelInfo


@pytest.fixture
def hf_client_instance(tmp_path):
    """Create a HuggingFaceClient instance with mocked dependencies."""
    with patch("mlx_manager.services.hf_client.settings") as mock_settings:
        mock_settings.hf_cache_path = tmp_path
        mock_settings.hf_organization = "mlx-community"
        mock_settings.offline_mode = False

        # Import after patching
        from mlx_manager.services.hf_client import HuggingFaceClient

        client = HuggingFaceClient()
        client.cache_dir = tmp_path
        yield client


class TestHuggingFaceClientIsDownloaded:
    """Tests for the _is_downloaded method."""

    def test_not_downloaded_no_cache(self, hf_client_instance, tmp_path):
        """Test model not downloaded when cache doesn't exist."""
        result = hf_client_instance._is_downloaded("mlx-community/test-model")
        assert result is False

    def test_not_downloaded_empty_snapshots(self, hf_client_instance, tmp_path):
        """Test model not downloaded when snapshots dir is empty."""
        model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots"
        model_dir.mkdir(parents=True)

        result = hf_client_instance._is_downloaded("mlx-community/test-model")
        assert result is False

    def test_downloaded_with_snapshot(self, hf_client_instance, tmp_path):
        """Test model is downloaded when snapshot exists."""
        model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots"
        model_dir.mkdir(parents=True)
        (model_dir / "abc123").mkdir()

        result = hf_client_instance._is_downloaded("mlx-community/test-model")
        assert result is True


class TestHuggingFaceClientGetLocalPath:
    """Tests for the get_local_path method."""

    def test_no_local_path_when_not_downloaded(self, hf_client_instance, tmp_path):
        """Test returns None when model not downloaded."""
        result = hf_client_instance.get_local_path("mlx-community/test-model")
        assert result is None

    def test_returns_local_path_when_downloaded(self, hf_client_instance, tmp_path):
        """Test returns path when model is downloaded."""
        model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots"
        model_dir.mkdir(parents=True)
        snapshot_dir = model_dir / "abc123"
        snapshot_dir.mkdir()

        result = hf_client_instance.get_local_path("mlx-community/test-model")
        assert result == str(snapshot_dir)


class TestHuggingFaceClientListLocalModels:
    """Tests for the list_local_models method."""

    def test_empty_when_no_cache(self, hf_client_instance, tmp_path):
        """Test returns empty list when cache dir doesn't exist."""
        hf_client_instance.cache_dir = tmp_path / "nonexistent"
        result = hf_client_instance.list_local_models()
        assert result == []

    def test_lists_downloaded_models(self, hf_client_instance, tmp_path):
        """Test lists locally downloaded models."""
        # Create a mock downloaded model
        model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots" / "abc123"
        model_dir.mkdir(parents=True)

        # Create a file to simulate model content
        model_file = model_dir / "model.safetensors"
        model_file.write_bytes(b"x" * 1000)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            result = hf_client_instance.list_local_models()

        assert len(result) == 1
        assert result[0]["model_id"] == "mlx-community/test-model"
        assert result[0]["size_bytes"] == 1000


class TestHuggingFaceClientDeleteModel:
    """Tests for the delete_model method."""

    @pytest.mark.asyncio
    async def test_delete_model_success(self, hf_client_instance, tmp_path):
        """Test deleting an existing model."""
        model_dir = tmp_path / "models--mlx-community--test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "file.txt").write_text("test")

        result = await hf_client_instance.delete_model("mlx-community/test-model")

        assert result is True
        assert not model_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, hf_client_instance, tmp_path):
        """Test deleting a non-existent model."""
        result = await hf_client_instance.delete_model("mlx-community/nonexistent")
        assert result is False


class TestHuggingFaceClientSearchModels:
    """Tests for the search_mlx_models method."""

    @pytest.mark.asyncio
    async def test_search_models(self, hf_client_instance):
        """Test searching for models."""
        mock_models = [
            ModelInfo(
                model_id="mlx-community/test-model",
                author="mlx-community",
                downloads=1000,
                likes=50,
                tags=["mlx", "4bit"],
                last_modified=None,
                size_bytes=5_000_000_000,  # 5 billion bytes
            )
        ]

        with (
            patch("mlx_manager.services.hf_client.search_models") as mock_search,
            patch("mlx_manager.services.hf_client.settings") as mock_settings,
        ):
            mock_search.return_value = mock_models
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            results = await hf_client_instance.search_mlx_models("test", limit=10)

        assert len(results) == 1
        assert results[0]["model_id"] == "mlx-community/test-model"
        assert results[0]["downloads"] == 1000
        assert results[0]["likes"] == 50
        # 5_000_000_000 / 1024^3 = 4.66 GiB
        assert results[0]["estimated_size_gb"] == 4.66

    @pytest.mark.asyncio
    async def test_search_models_with_size_filter(self, hf_client_instance):
        """Test searching with size filter."""
        mock_models = [
            ModelInfo(
                model_id="mlx-community/large-model",
                author="mlx-community",
                downloads=500,
                likes=25,
                tags=[],
                last_modified=None,
                size_bytes=100_000_000_000,  # 100GB
            )
        ]

        with (
            patch("mlx_manager.services.hf_client.search_models") as mock_search,
            patch("mlx_manager.services.hf_client.settings") as mock_settings,
        ):
            mock_search.return_value = mock_models
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            # Filter for models <= 50GB
            results = await hf_client_instance.search_mlx_models("test", max_size_gb=50)

        # Should filter out the large model
        assert len(results) == 0


class TestHuggingFaceClientDownloadModel:
    """Tests for the download_model method."""

    @pytest.mark.asyncio
    async def test_download_model_success(self, hf_client_instance, tmp_path):
        """Test successful model download."""

        def mock_snapshot_download(repo_id, **kwargs):
            # Handle dry_run call
            if kwargs.get("dry_run"):
                # Return mock DryRunFileInfo objects
                mock_info = MagicMock()
                mock_info.file_size = 1_000_000_000  # 1GB
                return [mock_info]
            # Return actual download path
            return str(tmp_path / "downloaded")

        with patch("mlx_manager.services.hf_client.snapshot_download", mock_snapshot_download):
            events = []
            async for event in hf_client_instance.download_model("mlx-community/Qwen3-8B-4bit"):
                events.append(event)

        # First event is immediate starting (without size), second has size after dry_run
        assert events[0]["status"] == "starting"
        assert events[0]["total_bytes"] == 0  # Immediate yield before dry_run
        # Second event has size info after dry_run completes
        assert events[1]["status"] == "starting"
        assert events[1]["total_bytes"] == 1_000_000_000
        assert events[-1]["status"] == "completed"
        assert events[-1]["progress"] == 100

    @pytest.mark.asyncio
    async def test_download_model_failure(self, hf_client_instance):
        """Test model download failure."""

        def mock_snapshot_download(repo_id, **kwargs):
            # Handle dry_run call successfully
            if kwargs.get("dry_run"):
                mock_info = MagicMock()
                mock_info.file_size = 1_000_000_000
                return [mock_info]
            # Actual download fails
            raise Exception("Download failed")

        with patch("mlx_manager.services.hf_client.snapshot_download", mock_snapshot_download):
            events = []
            async for event in hf_client_instance.download_model("mlx-community/Qwen3-8B-4bit"):
                events.append(event)

        assert events[0]["status"] == "starting"
        assert events[-1]["status"] == "failed"
        assert "Download failed" in events[-1]["error"]

    @pytest.mark.asyncio
    async def test_download_model_dry_run_failure_falls_back_to_estimation(
        self, hf_client_instance, tmp_path
    ):
        """Test that dry_run failure falls back to name-based size estimation."""

        def mock_snapshot_download(repo_id, **kwargs):
            if kwargs.get("dry_run"):
                # dry_run fails
                raise Exception("API error")
            return str(tmp_path / "downloaded")

        with patch("mlx_manager.services.hf_client.snapshot_download", mock_snapshot_download):
            events = []
            async for event in hf_client_instance.download_model("mlx-community/Qwen3-8B-4bit"):
                events.append(event)

        # First event is immediate starting (without size)
        assert events[0]["status"] == "starting"
        assert events[0]["total_bytes"] == 0  # Immediate yield before dry_run
        # Second event has estimated size after dry_run failure falls back to estimation
        # Size estimated from name: 8B * 0.5 bytes * 1.1 ≈ 4.1 GiB
        assert events[1]["status"] == "starting"
        assert events[1]["total_size_gb"] > 3.5
        assert events[-1]["status"] == "completed"


class TestHuggingFaceClientOfflineMode:
    """Tests for offline mode behavior."""

    @pytest.mark.asyncio
    async def test_search_returns_empty_in_offline_mode(self, tmp_path):
        """Test search returns empty list in offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = True

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            result = await client.search_mlx_models("test")

        assert result == []

    @pytest.mark.asyncio
    async def test_download_fails_in_offline_mode(self, tmp_path):
        """Test download yields failure in offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = True

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            events = []
            async for event in client.download_model("mlx-community/test"):
                events.append(event)

        assert len(events) == 1
        assert events[0]["status"] == "failed"
        assert "Offline mode" in events[0]["error"]

    def test_is_downloaded_returns_false_in_offline_mode(self, tmp_path):
        """Test _is_downloaded returns False in offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = True

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            result = client._is_downloaded("mlx-community/test")

        assert result is False

    def test_get_local_path_returns_none_in_offline_mode(self, tmp_path):
        """Test get_local_path returns None in offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = True

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            result = client.get_local_path("mlx-community/test")

        assert result is None

    def test_list_local_models_returns_empty_in_offline_mode(self, tmp_path):
        """Test list_local_models returns empty in offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = True

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            result = client.list_local_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_delete_model_returns_false_in_offline_mode(self, tmp_path):
        """Test delete_model returns False in offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = True

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            result = await client.delete_model("mlx-community/test")

        assert result is False


class TestHuggingFaceClientIsDownloadedExceptions:
    """Tests for _is_downloaded exception handling."""

    def test_is_downloaded_handles_iteration_error(self, hf_client_instance, tmp_path):
        """Test _is_downloaded handles errors when iterating snapshots."""
        model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots"
        model_dir.mkdir(parents=True)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.offline_mode = False

            # Mock iterdir to raise an error
            with patch.object(model_dir.__class__, "iterdir", side_effect=PermissionError):
                result = hf_client_instance._is_downloaded("mlx-community/test-model")

        assert result is False


class TestHuggingFaceClientGetLocalPathExceptions:
    """Tests for get_local_path exception handling."""

    def test_get_local_path_handles_stat_error(self, hf_client_instance, tmp_path):
        """Test get_local_path handles errors when getting file stats."""
        model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots"
        model_dir.mkdir(parents=True)
        snapshot_dir = model_dir / "abc123"
        snapshot_dir.mkdir()

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.offline_mode = False

            def failing_iterdir():
                raise PermissionError("Access denied")

            with patch.object(model_dir.__class__, "iterdir", failing_iterdir):
                result = hf_client_instance.get_local_path("mlx-community/test-model")

        assert result is None


class TestHuggingFaceClientListLocalModelsExceptions:
    """Tests for list_local_models exception handling."""

    def test_list_local_models_handles_iterdir_error(self, hf_client_instance, tmp_path):
        """Test list_local_models handles errors when iterating cache dir."""
        hf_client_instance.cache_dir = tmp_path
        (tmp_path / "models--mlx-community--test").mkdir()

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            with patch.object(hf_client_instance, "get_local_path", return_value=None):
                result = hf_client_instance.list_local_models()

        assert result == []

    def test_list_local_models_handles_general_exception(self, hf_client_instance, tmp_path):
        """Test list_local_models handles general exceptions gracefully."""
        hf_client_instance.cache_dir = tmp_path

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            def failing_iterdir():
                raise Exception("Unexpected error")

            with patch.object(tmp_path.__class__, "iterdir", failing_iterdir):
                result = hf_client_instance.list_local_models()

        assert result == []


class TestHuggingFaceClientSearchModelsEdgeCases:
    """Tests for search_mlx_models edge cases."""

    @pytest.mark.asyncio
    async def test_search_handles_model_with_none_values(self, hf_client_instance):
        """Test search handles models with None values."""
        mock_models = [
            ModelInfo(
                model_id="mlx-community/test-model",
                author=None,
                downloads=0,
                likes=0,
                tags=[],
                last_modified=None,
                size_bytes=None,  # No size info
            )
        ]

        with (
            patch("mlx_manager.services.hf_client.search_models") as mock_search,
            patch("mlx_manager.services.hf_client.settings") as mock_settings,
        ):
            mock_search.return_value = mock_models
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            results = await hf_client_instance.search_mlx_models("test")

        assert len(results) == 1
        assert results[0]["model_id"] == "mlx-community/test-model"
        assert results[0]["author"] == "mlx-community"  # Defaults to org
        assert results[0]["downloads"] == 0
        assert results[0]["likes"] == 0
        assert results[0]["tags"] == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, hf_client_instance):
        """Test search stops after reaching limit."""
        mock_models = [
            ModelInfo(
                model_id=f"mlx-community/model-{i}",
                author="mlx-community",
                downloads=100 - i,
                likes=50 - i,
                tags=[],
                last_modified=None,
                size_bytes=1_000_000_000,
            )
            for i in range(30)
        ]

        with (
            patch("mlx_manager.services.hf_client.search_models") as mock_search,
            patch("mlx_manager.services.hf_client.settings") as mock_settings,
        ):
            mock_search.return_value = mock_models
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            results = await hf_client_instance.search_mlx_models("test", limit=5)

        assert len(results) == 5


# ============================================================================
# Tests for SilentProgress class (lines 30-31)
# ============================================================================


class TestSilentProgress:
    """Tests for the SilentProgress class."""

    def test_silent_progress_disables_output(self):
        """Test that SilentProgress disables console output."""
        from mlx_manager.services.hf_client import SilentProgress

        # Create instance - should not raise
        progress = SilentProgress(total=100)

        # Verify disable is set to True
        assert progress.disable is True

        # Clean up
        progress.close()

    def test_silent_progress_accepts_args(self):
        """Test that SilentProgress accepts standard tqdm arguments."""
        from mlx_manager.services.hf_client import SilentProgress

        # Should accept common tqdm arguments without raising
        progress = SilentProgress(total=100, desc="Test", unit="B", unit_scale=True)
        assert progress.disable is True
        progress.close()


class TestCancellableProgress:
    """Tests for the make_cancellable_progress factory and DownloadCancelledError."""

    def test_cancellable_progress_disables_output(self):
        """CancellableProgress suppresses console output like SilentProgress."""
        from mlx_manager.services.hf_client import make_cancellable_progress

        cls = make_cancellable_progress(cancel_event=None)
        progress = cls(total=100)
        assert progress.disable is True
        progress.close()

    def test_cancellable_progress_update_without_cancel(self):
        """update() completes without error when no cancel event is set."""
        from mlx_manager.services.hf_client import make_cancellable_progress

        cls = make_cancellable_progress(cancel_event=None)
        progress = cls(total=100)
        # Should not raise
        progress.update(10)
        progress.update(20)
        progress.close()

    def test_cancellable_progress_raises_on_cancel(self):
        """update() raises DownloadCancelledError when cancel event is set."""
        import threading

        from mlx_manager.services.hf_client import (
            DownloadCancelledError,
            make_cancellable_progress,
        )

        event = threading.Event()
        cls = make_cancellable_progress(cancel_event=event)
        progress = cls(total=100)

        # First update works fine (no cancel)
        progress.update(10)

        # Set the cancel event
        event.set()

        # Next update should raise
        with pytest.raises(DownloadCancelledError, match="cancelled by user"):
            progress.update(10)

        progress.close()

    def test_cancellable_progress_concurrent_independence(self):
        """Each factory call creates an independent class with its own cancel event."""
        import threading

        from mlx_manager.services.hf_client import (
            DownloadCancelledError,
            make_cancellable_progress,
        )

        event_a = threading.Event()
        event_b = threading.Event()
        cls_a = make_cancellable_progress(cancel_event=event_a)
        cls_b = make_cancellable_progress(cancel_event=event_b)
        progress_a = cls_a(total=100)
        progress_b = cls_b(total=100)

        # Cancel only event_a
        event_a.set()

        # progress_a should raise
        with pytest.raises(DownloadCancelledError):
            progress_a.update(10)

        # progress_b should still work fine (no exception)
        progress_b.update(10)

        progress_a.close()
        progress_b.close()


# ============================================================================
# Tests for _get_directory_size exception handling (lines 232-235)
# ============================================================================


class TestGetDirectorySize:
    """Tests for the _get_directory_size method."""

    def test_get_directory_size_nonexistent(self, hf_client_instance, tmp_path):
        """Test returns 0 for non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        result = hf_client_instance._get_directory_size(nonexistent)
        assert result == 0

    def test_get_directory_size_success(self, hf_client_instance, tmp_path):
        """Test returns correct size for directory with files."""
        (tmp_path / "file1.txt").write_bytes(b"x" * 1000)
        (tmp_path / "file2.txt").write_bytes(b"y" * 500)

        result = hf_client_instance._get_directory_size(tmp_path)
        assert result == 1500

    def test_get_directory_size_exception(self, hf_client_instance, tmp_path):
        """Test returns 0 when exception occurs (line 234-235)."""
        # Create directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # Mock rglob to raise an exception
        with patch.object(test_dir.__class__, "rglob", side_effect=PermissionError("No access")):
            result = hf_client_instance._get_directory_size(test_dir)

        assert result == 0


# ============================================================================
# Tests for list_local_models non-organization filtering (lines 264-275)
# ============================================================================


class TestListLocalModelsNoOrganization:
    """Tests for list_local_models when no organization filter is set."""

    def test_list_local_models_mlx_community_model(self, tmp_path):
        """Test lists mlx-community models when no org filter."""
        # Create model with mlx-community in name
        model_dir = tmp_path / "models--mlx-community--Qwen-7B" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 1000)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = None  # No org filter
            mock_settings.offline_mode = False

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            client.cache_dir = tmp_path

            result = client.list_local_models()

        assert len(result) == 1
        assert result[0]["model_id"] == "mlx-community/Qwen-7B"

    def test_list_local_models_lmstudio_community_model(self, tmp_path):
        """Test lists lmstudio-community models when no org filter."""
        model_dir = tmp_path / "models--lmstudio-community--Llama-3-8B" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 1000)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = None  # No org filter
            mock_settings.offline_mode = False

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            client.cache_dir = tmp_path

            result = client.list_local_models()

        assert len(result) == 1
        assert result[0]["model_id"] == "lmstudio-community/Llama-3-8B"

    def test_list_local_models_mlx_in_name(self, tmp_path):
        """Test lists models with 'mlx' in name when no org filter."""
        model_dir = tmp_path / "models--someorg--model-mlx-optimized" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 1000)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = None
            mock_settings.offline_mode = False

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            client.cache_dir = tmp_path

            result = client.list_local_models()

        assert len(result) == 1
        assert result[0]["model_id"] == "someorg/model-mlx-optimized"

    def test_list_local_models_excludes_non_mlx(self, tmp_path):
        """Test excludes non-MLX models when no org filter (line 274-275)."""
        # Create a non-MLX model
        model_dir = tmp_path / "models--someorg--regular-model" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 1000)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = None  # No org filter
            mock_settings.offline_mode = False

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            client.cache_dir = tmp_path

            result = client.list_local_models()

        # Should be filtered out - no mlx in name
        assert len(result) == 0

    def test_list_local_models_filters_by_organization(self, tmp_path):
        """Test filters by organization when org is set (line 263-264)."""
        # Create models from different orgs
        mlx_dir = tmp_path / "models--mlx-community--Model-A" / "snapshots" / "abc"
        mlx_dir.mkdir(parents=True)
        (mlx_dir / "model.safetensors").write_bytes(b"x" * 1000)

        other_dir = tmp_path / "models--other-org--Model-B" / "snapshots" / "abc"
        other_dir.mkdir(parents=True)
        (other_dir / "model.safetensors").write_bytes(b"x" * 1000)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"  # Filter by org
            mock_settings.offline_mode = False

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            client.cache_dir = tmp_path

            result = client.list_local_models()

        # Should only include mlx-community model
        assert len(result) == 1
        assert result[0]["model_id"] == "mlx-community/Model-A"

    def test_list_local_models_skips_non_model_dirs(self, tmp_path):
        """Test skips directories that don't start with models--."""
        # Create a non-model directory
        (tmp_path / "some-other-dir").mkdir()

        # Create valid model
        model_dir = tmp_path / "models--mlx-community--Test" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 1000)

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            client.cache_dir = tmp_path

            result = client.list_local_models()

        assert len(result) == 1

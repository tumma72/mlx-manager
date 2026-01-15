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
            async for event in hf_client_instance.download_model(
                "mlx-community/Qwen3-8B-4bit"
            ):
                events.append(event)

        # First event is starting, then completed (progress events may or may not appear)
        assert events[0]["status"] == "starting"
        assert events[0]["total_bytes"] == 1_000_000_000
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
            async for event in hf_client_instance.download_model(
                "mlx-community/Qwen3-8B-4bit"
            ):
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
            async for event in hf_client_instance.download_model(
                "mlx-community/Qwen3-8B-4bit"
            ):
                events.append(event)

        assert events[0]["status"] == "starting"
        # Size estimated from name: 8B * 0.5 bytes * 1.1 ≈ 4.1 GiB
        assert events[0]["total_size_gb"] > 3.5
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

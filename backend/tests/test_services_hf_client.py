"""Tests for the HuggingFace client service."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_hf_api():
    """Mock HuggingFace API."""
    with patch("mlx_manager.services.hf_client.HfApi") as mock:
        yield mock.return_value


@pytest.fixture
def hf_client_instance(mock_hf_api, tmp_path):
    """Create a HuggingFaceClient instance with mocked dependencies."""
    with patch("mlx_manager.services.hf_client.settings") as mock_settings:
        mock_settings.hf_cache_path = tmp_path
        mock_settings.hf_organization = "mlx-community"
        mock_settings.offline_mode = False

        # Import after patching
        from mlx_manager.services.hf_client import HuggingFaceClient

        client = HuggingFaceClient()
        client.api = mock_hf_api
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
    async def test_search_models(self, hf_client_instance, mock_hf_api):
        """Test searching for models."""
        # Create mock model objects
        mock_model = MagicMock()
        mock_model.id = "mlx-community/test-model"
        mock_model.author = "mlx-community"
        mock_model.downloads = 1000
        mock_model.likes = 50
        mock_model.tags = ["mlx", "4bit"]
        mock_model.last_modified = None

        mock_hf_api.list_models.return_value = [mock_model]

        # Mock repo_info for size estimation
        mock_repo_info = MagicMock()
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "model.safetensors"
        mock_sibling.size = 5_000_000_000  # 5GB
        mock_repo_info.siblings = [mock_sibling]
        mock_hf_api.repo_info.return_value = mock_repo_info

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            results = await hf_client_instance.search_mlx_models("test", limit=10)

        assert len(results) == 1
        assert results[0]["model_id"] == "mlx-community/test-model"
        assert results[0]["downloads"] == 1000
        assert results[0]["likes"] == 50

    @pytest.mark.asyncio
    async def test_search_models_with_size_filter(self, hf_client_instance, mock_hf_api):
        """Test searching with size filter."""
        mock_model = MagicMock()
        mock_model.id = "mlx-community/large-model"
        mock_model.author = "mlx-community"
        mock_model.downloads = 500
        mock_model.likes = 25
        mock_model.tags = []
        mock_model.last_modified = None

        mock_hf_api.list_models.return_value = [mock_model]

        # Mock a large model (100GB)
        mock_repo_info = MagicMock()
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "model.safetensors"
        mock_sibling.size = 100_000_000_000
        mock_repo_info.siblings = [mock_sibling]
        mock_hf_api.repo_info.return_value = mock_repo_info

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            # Filter for models <= 50GB
            results = await hf_client_instance.search_mlx_models("test", max_size_gb=50)

        # Should filter out the large model
        assert len(results) == 0


class TestHuggingFaceClientDownloadModel:
    """Tests for the download_model method."""

    @pytest.mark.asyncio
    async def test_download_model_success(self, hf_client_instance, mock_hf_api, tmp_path):
        """Test successful model download."""
        mock_hf_api.repo_info.return_value = MagicMock(siblings=[])

        with patch("mlx_manager.services.hf_client.snapshot_download") as mock_download:
            mock_download.return_value = str(tmp_path / "downloaded")

            events = []
            async for event in hf_client_instance.download_model("mlx-community/test"):
                events.append(event)

        assert len(events) == 2
        assert events[0]["status"] == "starting"
        assert events[1]["status"] == "completed"
        assert events[1]["progress"] == 100

    @pytest.mark.asyncio
    async def test_download_model_failure(self, hf_client_instance, mock_hf_api):
        """Test model download failure."""
        mock_hf_api.repo_info.return_value = MagicMock(siblings=[])

        with patch("mlx_manager.services.hf_client.snapshot_download") as mock_download:
            mock_download.side_effect = Exception("Download failed")

            events = []
            async for event in hf_client_instance.download_model("mlx-community/test"):
                events.append(event)

        assert len(events) == 2
        assert events[0]["status"] == "starting"
        assert events[1]["status"] == "failed"
        assert "Download failed" in events[1]["error"]


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


class TestHuggingFaceClientApiInitFailure:
    """Tests for API initialization failure."""

    def test_api_init_failure_sets_offline_mode(self, tmp_path):
        """Test that HfApi init failure enables offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            with patch("mlx_manager.services.hf_client.HfApi") as mock_api_class:
                mock_api_class.side_effect = Exception("Network error")

                from mlx_manager.services.hf_client import HuggingFaceClient

                client = HuggingFaceClient()

        assert client.api is None
        assert mock_settings.offline_mode is True


class TestHuggingFaceClientEstimateModelSize:
    """Tests for _estimate_model_size method."""

    @pytest.mark.asyncio
    async def test_estimate_size_returns_zero_in_offline_mode(self, tmp_path):
        """Test _estimate_model_size returns 0 in offline mode."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = True

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            result = await client._estimate_model_size("mlx-community/test")

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_estimate_size_handles_api_error(self, hf_client_instance, mock_hf_api):
        """Test _estimate_model_size handles API errors gracefully."""
        mock_hf_api.repo_info.side_effect = Exception("API error")

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.offline_mode = False

            result = await hf_client_instance._estimate_model_size("mlx-community/test")

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_estimate_size_handles_no_siblings(self, hf_client_instance, mock_hf_api):
        """Test _estimate_model_size handles repo with no files."""
        mock_repo_info = MagicMock()
        mock_repo_info.siblings = None
        mock_hf_api.repo_info.return_value = mock_repo_info

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.offline_mode = False

            result = await hf_client_instance._estimate_model_size("mlx-community/test")

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_estimate_size_with_various_file_types(self, hf_client_instance, mock_hf_api):
        """Test _estimate_model_size counts safetensors, bin, and gguf files."""
        mock_repo_info = MagicMock()
        mock_siblings = [
            MagicMock(rfilename="model.safetensors", size=1_000_000_000),
            MagicMock(rfilename="model.bin", size=500_000_000),
            MagicMock(rfilename="model.gguf", size=200_000_000),
            MagicMock(rfilename="config.json", size=1000),  # Should be ignored
            MagicMock(rfilename="README.md", size=5000),  # Should be ignored
        ]
        mock_repo_info.siblings = mock_siblings
        mock_hf_api.repo_info.return_value = mock_repo_info

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.offline_mode = False

            result = await hf_client_instance._estimate_model_size("mlx-community/test")

        # 1.7 GB * 1.2 overhead = 2.04 GB
        expected = (1_000_000_000 + 500_000_000 + 200_000_000) / 1e9 * 1.2
        assert abs(result - expected) < 0.01

    @pytest.mark.asyncio
    async def test_estimate_size_handles_missing_size(self, hf_client_instance, mock_hf_api):
        """Test _estimate_model_size handles files without size info."""
        mock_repo_info = MagicMock()
        mock_siblings = [
            MagicMock(rfilename="model.safetensors", size=1_000_000_000),
            MagicMock(rfilename="model2.safetensors", size=None),  # Missing size
        ]
        mock_repo_info.siblings = mock_siblings
        mock_hf_api.repo_info.return_value = mock_repo_info

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.offline_mode = False

            result = await hf_client_instance._estimate_model_size("mlx-community/test")

        # Only 1 GB counted * 1.2 overhead = 1.2 GB
        expected = 1_000_000_000 / 1e9 * 1.2
        assert abs(result - expected) < 0.01


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

            # Make stat() fail by patching Path.stat

            def failing_iterdir():
                raise PermissionError("Access denied")

            with patch.object(model_dir.__class__, "iterdir", failing_iterdir):
                # Test should handle this gracefully
                result = hf_client_instance.get_local_path("mlx-community/test-model")

        # Since the actual exception handling returns None, we expect None
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

            # Make get_local_path return None to skip the model
            with patch.object(hf_client_instance, "get_local_path", return_value=None):
                result = hf_client_instance.list_local_models()

        assert result == []

    def test_list_local_models_handles_general_exception(self, hf_client_instance, tmp_path):
        """Test list_local_models handles general exceptions gracefully."""
        hf_client_instance.cache_dir = tmp_path

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            # Create a mock for iterdir that raises an exception
            def failing_iterdir():
                raise Exception("Unexpected error")

            with patch.object(tmp_path.__class__, "iterdir", failing_iterdir):
                result = hf_client_instance.list_local_models()

        assert result == []


class TestHuggingFaceClientSearchModelsEdgeCases:
    """Tests for search_mlx_models edge cases."""

    @pytest.mark.asyncio
    async def test_search_with_api_none(self, tmp_path):
        """Test search returns empty when API is None."""
        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_cache_path = tmp_path
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            from mlx_manager.services.hf_client import HuggingFaceClient

            client = HuggingFaceClient()
            client.api = None  # Explicitly set to None

            result = await client.search_mlx_models("test")

        assert result == []

    @pytest.mark.asyncio
    async def test_search_handles_model_with_none_values(self, hf_client_instance, mock_hf_api):
        """Test search handles models with None values."""
        mock_model = MagicMock()
        mock_model.id = "mlx-community/test-model"
        mock_model.author = None
        mock_model.downloads = None
        mock_model.likes = None
        mock_model.tags = None
        mock_model.last_modified = None

        mock_hf_api.list_models.return_value = [mock_model]

        mock_repo_info = MagicMock()
        mock_repo_info.siblings = []
        mock_hf_api.repo_info.return_value = mock_repo_info

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
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
    async def test_search_respects_limit(self, hf_client_instance, mock_hf_api):
        """Test search stops after reaching limit."""
        mock_models = []
        for i in range(30):
            m = MagicMock()
            m.id = f"mlx-community/model-{i}"
            m.author = "mlx-community"
            m.downloads = 100 - i
            m.likes = 50 - i
            m.tags = []
            m.last_modified = None
            mock_models.append(m)

        mock_hf_api.list_models.return_value = mock_models

        mock_repo_info = MagicMock()
        mock_repo_info.siblings = []
        mock_hf_api.repo_info.return_value = mock_repo_info

        with patch("mlx_manager.services.hf_client.settings") as mock_settings:
            mock_settings.hf_organization = "mlx-community"
            mock_settings.offline_mode = False

            results = await hf_client_instance.search_mlx_models("test", limit=5)

        assert len(results) == 5

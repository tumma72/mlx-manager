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

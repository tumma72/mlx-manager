"""HuggingFace Hub client service."""

import asyncio
import logging
import shutil
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm as tqdm_auto

from mlx_manager.config import settings
from mlx_manager.services.hf_api import (
    estimate_size_from_name,
    get_model_size_gb,
    search_models,
)
from mlx_manager.types import DownloadStatus, LocalModelInfo, ModelSearchResult

# Suppress huggingface_hub warnings at module level
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class SilentProgress(tqdm_auto):
    """tqdm subclass that suppresses console output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True  # Disable console output
        super().__init__(*args, **kwargs)


class HuggingFaceClient:
    """Service for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        self.cache_dir = settings.hf_cache_path

    async def search_mlx_models(
        self,
        query: str,
        max_size_gb: float | None = None,
        limit: int = 20,
    ) -> list[ModelSearchResult]:
        """Search for MLX models in mlx-community organization.

        Uses the HuggingFace REST API directly with expand=safetensors
        to get model sizes in a single request (no N+1 API calls).
        """
        if settings.offline_mode:
            return []

        # Fetch extra for filtering, but cap at reasonable limit
        fetch_limit = min(limit * 2, 100) if max_size_gb else limit

        models = await search_models(
            query=query,
            author=settings.hf_organization,
            sort="downloads",
            limit=fetch_limit,
        )

        results: list[ModelSearchResult] = []
        for model in models:
            size_gb = get_model_size_gb(model)

            # Filter by size if specified
            if max_size_gb and size_gb > max_size_gb:
                continue

            # Extract author from model_id if not provided (format: author/model-name)
            author = model.author
            if not author and "/" in model.model_id:
                author = model.model_id.split("/")[0]

            results.append(
                ModelSearchResult(
                    model_id=model.model_id,
                    author=author,
                    downloads=model.downloads,
                    likes=model.likes,
                    estimated_size_gb=size_gb,
                    tags=model.tags,
                    is_downloaded=self._is_downloaded(model.model_id),
                    last_modified=model.last_modified,
                )
            )

            if len(results) >= limit:
                break

        return results

    def _is_downloaded(self, model_id: str) -> bool:
        """Check if model is in local cache."""
        if settings.offline_mode:
            return False

        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.cache_dir / cache_name

        # Check for snapshots directory which indicates complete download
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists():
            try:
                return any(snapshots_dir.iterdir())
            except Exception:
                pass
        return False

    def get_local_path(self, model_id: str) -> str | None:
        """Get the local path for a downloaded model."""
        if settings.offline_mode:
            return None

        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.cache_dir / cache_name / "snapshots"

        if model_path.exists():
            # Get the latest snapshot
            try:
                snapshots = sorted(
                    model_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
                )
                if snapshots:
                    return str(snapshots[0])
            except Exception:
                pass
        return None

    async def download_model(
        self,
        model_id: str,
    ) -> AsyncGenerator[DownloadStatus, None]:
        """Download a model with progress updates."""
        if settings.offline_mode:
            yield DownloadStatus(
                status="failed",
                model_id=model_id,
                error="Offline mode - cannot download models",
            )
            return

        loop = asyncio.get_event_loop()

        # Suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Get total size via dry_run first
            try:
                dry_run_result = await loop.run_in_executor(
                    None,
                    lambda: snapshot_download(
                        repo_id=model_id, dry_run=True, tqdm_class=SilentProgress
                    ),
                )
                total_bytes = sum(f.file_size for f in dry_run_result if f.file_size)
            except Exception:
                # Fall back to estimation if dry_run fails
                estimated_gb = estimate_size_from_name(model_id) or 0.0
                total_bytes = int(estimated_gb * 1024**3)

        # Estimate size for backward compatibility
        total_size_gb = total_bytes / (1024**3) if total_bytes else 0.0

        yield DownloadStatus(
            status="starting",
            model_id=model_id,
            total_size_gb=total_size_gb,
            total_bytes=total_bytes,
            downloaded_bytes=0,
            progress=0,
        )

        # Start download as a background task
        download_task = loop.run_in_executor(
            None,
            lambda: self._download_with_progress(model_id),
        )

        # Calculate directory path for progress polling
        cache_name = f"models--{model_id.replace('/', '--')}"
        download_dir = self.cache_dir / cache_name

        # Poll directory size while downloading
        try:
            while not download_task.done():
                await asyncio.sleep(1.0)  # Poll every second

                current_bytes = self._get_directory_size(download_dir)
                progress = int((current_bytes / total_bytes) * 100) if total_bytes > 0 else 0

                yield DownloadStatus(
                    status="downloading",
                    model_id=model_id,
                    total_size_gb=total_size_gb,
                    total_bytes=total_bytes,
                    downloaded_bytes=current_bytes,
                    progress=min(progress, 99),  # Cap at 99 until actually done
                )

            # Get the result (may raise exception)
            local_dir = await download_task

            yield DownloadStatus(
                status="completed",
                model_id=model_id,
                local_path=local_dir,
                total_bytes=total_bytes,
                downloaded_bytes=total_bytes,
                progress=100,
            )

        except Exception as e:
            yield DownloadStatus(status="failed", model_id=model_id, error=str(e))

    def _download_with_progress(self, model_id: str) -> str:
        """Perform download (runs in executor)."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            return snapshot_download(repo_id=model_id, tqdm_class=SilentProgress)

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of files in directory."""
        if not path.exists():
            return 0
        try:
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except Exception:
            return 0

    def list_local_models(self) -> list[LocalModelInfo]:
        """List all locally downloaded MLX models."""
        if settings.offline_mode:
            return []

        models: list[LocalModelInfo] = []

        if not self.cache_dir.exists():
            return models

        try:
            for item in self.cache_dir.iterdir():
                if item.name.startswith(f"models--{settings.hf_organization}--"):
                    model_id = item.name.replace("models--", "").replace("--", "/")
                    local_path = self.get_local_path(model_id)

                    if local_path:
                        size_bytes = sum(
                            f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()
                        )

                        models.append(
                            LocalModelInfo(
                                model_id=model_id,
                                local_path=local_path,
                                size_bytes=size_bytes,
                                size_gb=round(size_bytes / (1024**3), 2),
                            )
                        )
        except Exception:
            pass

        return models

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model from local cache."""
        if settings.offline_mode:
            return False

        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.cache_dir / cache_name

        if model_path.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: shutil.rmtree(model_path))
            return True
        return False


# Singleton instance
hf_client = HuggingFaceClient()

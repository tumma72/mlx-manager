"""HuggingFace Hub client service."""

import asyncio
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from mlx_manager.config import settings
from mlx_manager.types import DownloadStatus, LocalModelInfo, ModelSearchResult


class HuggingFaceClient:
    """Service for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        self.api = None
        self.cache_dir = settings.hf_cache_path

        # Only initialize HuggingFace API if not in offline mode
        if not settings.offline_mode:
            try:
                self.api = HfApi()
            except Exception:
                # If HuggingFace API fails, fall back to offline mode
                self.api = None
                settings.offline_mode = True

    async def search_mlx_models(
        self,
        query: str,
        max_size_gb: float | None = None,
        limit: int = 20,
    ) -> list[ModelSearchResult]:
        """Search for MLX models in mlx-community organization."""
        if settings.offline_mode or self.api is None:
            # Return empty list in offline mode or when API unavailable
            return []

        loop = asyncio.get_event_loop()

        # Run in executor since huggingface_hub is sync
        models = await loop.run_in_executor(
            None,
            lambda: list(
                self.api.list_models(
                    search=query,
                    author=settings.hf_organization,
                    sort="downloads",
                    direction=-1,
                    limit=limit * 2,  # Fetch extra for filtering
                )
            )
            if self.api is not None
            else [],
        )

        results: list[ModelSearchResult] = []
        for model in models:
            estimated_size = await self._estimate_model_size(model.id)

            # Filter by size if specified
            if max_size_gb and estimated_size > max_size_gb:
                continue

            results.append(
                ModelSearchResult(
                    model_id=model.id,
                    author=model.author or settings.hf_organization,
                    downloads=model.downloads or 0,
                    likes=model.likes or 0,
                    estimated_size_gb=round(estimated_size, 2),
                    tags=list(model.tags) if model.tags else [],
                    is_downloaded=self._is_downloaded(model.id),
                    last_modified=(
                        model.last_modified.isoformat() if model.last_modified else None
                    ),
                )
            )

            if len(results) >= limit:
                break

        return results

    async def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in GB based on safetensors files."""
        if settings.offline_mode or self.api is None:
            return 0.0

        try:
            loop = asyncio.get_event_loop()
            repo_info = await loop.run_in_executor(
                None,
                lambda: self.api.repo_info(model_id, files_metadata=True),  # type: ignore
            )

            total_bytes = 0
            if repo_info.siblings:
                for sibling in repo_info.siblings:
                    if sibling.rfilename.endswith((".safetensors", ".bin", ".gguf")):
                        if sibling.size:
                            total_bytes += sibling.size

            # Add 20% overhead for KV cache and runtime
            return (total_bytes / 1e9) * 1.2
        except Exception:
            return 0.0

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

        # Get total size first
        total_size = await self._estimate_model_size(model_id)

        yield DownloadStatus(status="starting", model_id=model_id, total_size_gb=total_size)

        try:
            # Use snapshot_download for full model
            local_dir = await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=model_id,
                    local_dir_use_symlinks=True,
                    resume_download=True,
                ),
            )

            yield DownloadStatus(
                status="completed",
                model_id=model_id,
                local_path=local_dir,
                progress=100,
            )

        except Exception as e:
            yield DownloadStatus(status="failed", model_id=model_id, error=str(e))

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
                                size_gb=round(size_bytes / 1e9, 2),
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

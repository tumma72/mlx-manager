"""HuggingFace Hub client service."""

import asyncio
import logging
import shutil
import threading
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from loguru import logger
from tqdm.auto import tqdm

from mlx_manager.config import settings
from mlx_manager.models.dto.models import DownloadStatus, LocalModel, ModelSearchResult
from mlx_manager.models.enums import DownloadStatusEnum
from mlx_manager.services.hf_api import (
    estimate_size_from_name,
    get_model_size_gb,
    search_models,
)


async def get_hf_token() -> str | None:
    """Retrieve decrypted HuggingFace token from settings, if configured."""
    from sqlmodel import select

    from mlx_manager.database import async_session
    from mlx_manager.models import Setting
    from mlx_manager.services.encryption_service import DecryptionError, decrypt_api_key

    try:
        async with async_session() as session:
            result = await session.execute(
                select(Setting).where(Setting.key == "huggingface_token")
            )
            setting = result.scalar_one_or_none()
            if setting and setting.value:
                return decrypt_api_key(setting.value)
    except DecryptionError:
        logger.warning("Failed to decrypt HuggingFace token - please re-enter in Settings")
    except Exception as e:
        logger.debug("No HuggingFace token configured: {}", e)
    return None


# Suppress huggingface_hub warnings at module level
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ============================================================================
# Download Cancellation Infrastructure
# ============================================================================

# Track cancellation events per download (keyed by download_id as string)
_cancel_events: dict[str, threading.Event] = {}


def register_cancel_event(download_id: str) -> threading.Event:
    """Register a cancellation event for a download.

    Returns the event so the download loop can check it.
    """
    event = threading.Event()
    _cancel_events[download_id] = event
    return event


def request_cancel(download_id: str) -> bool:
    """Signal a download to cancel.

    Returns True if the event was found and set.
    """
    event = _cancel_events.get(download_id)
    if event:
        event.set()
        return True
    return False


def cleanup_cancel_event(download_id: str) -> None:
    """Remove cancel event after download completes/cancels."""
    _cancel_events.pop(download_id, None)


def cleanup_partial_download(model_id: str) -> bool:
    """Remove all partial download files for a model from HF cache.

    Uses huggingface_hub's cache management to safely delete partial data.
    Falls back to manual directory deletion if the cache API fails.
    """
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                delete_strategy = cache.delete_revisions(
                    *(rev.commit_hash for rev in repo.revisions)
                )
                delete_strategy.execute()
                logger.info(
                    "Cleaned up {} for {}", delete_strategy.expected_freed_size_str, model_id
                )
                return True
        # Model not found in cache via API - try manual cleanup
        return _manual_cleanup(model_id)
    except Exception as e:
        logger.error("Cache API cleanup failed for {}: {}", model_id, e)
        return _manual_cleanup(model_id)


def _manual_cleanup(model_id: str) -> bool:
    """Fallback: manually delete the HF cache directory for a model."""
    cache_dir = (
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("Manually cleaned up {}", cache_dir)
        return True
    return False


class SilentProgress(tqdm):  # type: ignore[unsupported-base]
    """tqdm subclass that suppresses console output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True  # Disable console output
        super().__init__(*args, **kwargs)


class DownloadCancelledError(Exception):
    """Raised when a download is cancelled via cancel event."""


def make_cancellable_progress(
    cancel_event: threading.Event | None = None,
) -> type[tqdm]:
    """Create a tqdm subclass that checks for cancellation on each chunk update.

    Each call creates a unique class with its own cancel_event closure,
    so concurrent downloads each get independent cancellation.
    """

    class CancellableProgress(tqdm):  # type: ignore[unsupported-base]
        """tqdm subclass that suppresses output and supports cancellation."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> bool | None:
            result: bool | None = super().update(n)
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelledError("Download cancelled by user")
            return result

    return CancellableProgress


class HuggingFaceClient:
    """Service for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        self.cache_dir = settings.hf_cache_path

    def _model_cache_dir(self, model_id: str) -> Path:
        """Return the HF cache directory for a model (models--author--name)."""
        cache_name = f"models--{model_id.replace('/', '--')}"
        return self.cache_dir / cache_name

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

        token = await get_hf_token()
        models = await search_models(
            query=query,
            author=settings.hf_organization,
            sort="downloads",
            limit=fetch_limit,
            token=token,
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
            author = author or "unknown"

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

    def has_incomplete_blobs(self, model_id: str) -> bool:
        """Check if a model has .incomplete blob files in the HF cache.

        HuggingFace Hub creates .incomplete files for blobs that are still
        being downloaded. Their presence means the model is not fully downloaded.
        """
        blobs_dir = self._model_cache_dir(model_id) / "blobs"
        if not blobs_dir.exists():
            return False
        try:
            return any(f.suffix == ".incomplete" for f in blobs_dir.iterdir())
        except Exception:
            return False

    def _is_downloaded(self, model_id: str) -> bool:
        """Check if model is fully downloaded in local cache.

        A model is considered downloaded only if it has a snapshots directory
        AND has no .incomplete blob files (which indicate a partial download).
        """
        if settings.offline_mode:
            return False

        # Check for snapshots directory which indicates download was started
        snapshots_dir = self._model_cache_dir(model_id) / "snapshots"
        if snapshots_dir.exists():
            try:
                has_snapshots = any(snapshots_dir.iterdir())
            except Exception as e:
                logger.error("Failed to check snapshots directory for {}: {}", model_id, e)
                return False

            if not has_snapshots:
                return False

            # Reject models with incomplete blob files
            if self.has_incomplete_blobs(model_id):
                logger.debug("Model {} has incomplete blob files — not fully downloaded", model_id)
                return False

            return True
        return False

    def get_local_path(self, model_id: str) -> str | None:
        """Get the local path for a downloaded model."""
        model_path = self._model_cache_dir(model_id) / "snapshots"

        if model_path.exists():
            # Get the latest snapshot
            try:
                snapshots = sorted(
                    model_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
                )
                if snapshots:
                    return str(snapshots[0])
            except Exception as e:
                logger.debug("Failed to get local path for {}: {}", model_id, e)
        return None

    async def download_model(
        self,
        model_id: str,
        cancel_event: threading.Event | None = None,
    ) -> AsyncGenerator[DownloadStatus, None]:
        """Download a model with progress updates.

        Args:
            model_id: HuggingFace model ID to download.
            cancel_event: Optional threading.Event to signal cancellation.
                          When set, the download thread is interrupted via
                          DownloadCancelledError raised inside the tqdm callback.
        """
        logger.info("Starting download process for {}", model_id)

        if settings.offline_mode:
            logger.warning("Download blocked for {} - offline mode enabled", model_id)
            yield DownloadStatus(
                status=DownloadStatusEnum.FAILED,
                model_id=model_id,
                error="Offline mode - cannot download models",
            )
            return

        # Yield immediate status so SSE connection gets a response before dry_run
        # This prevents the frontend from showing a hung connection
        yield DownloadStatus(
            status=DownloadStatusEnum.STARTING,
            model_id=model_id,
            total_bytes=0,
            downloaded_bytes=0,
            progress=0,
        )

        loop = asyncio.get_event_loop()
        token = await get_hf_token()

        # Suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Get total size via dry_run first
            logger.info("Getting total size for {} via dry_run", model_id)
            try:
                dry_run_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=model_id,
                            dry_run=True,
                            tqdm_class=SilentProgress,
                            token=token,
                        ),
                    ),
                    timeout=30.0,  # 30 second timeout for size check
                )
                total_bytes = sum(f.file_size for f in dry_run_result if f.file_size)
                logger.info("Dry run complete for {}: {} bytes", model_id, total_bytes)
            except TimeoutError:
                logger.warning(
                    f"Dry run timed out for {model_id}, proceeding without size estimate"
                )
                total_bytes = 0
            except Exception as e:
                # Fall back to estimation if dry_run fails
                logger.warning("Dry run failed for {}: {}", model_id, e)
                estimated_gb = estimate_size_from_name(model_id) or 0.0
                total_bytes = int(estimated_gb * 1024**3)
                logger.info("Using estimated size for {}: {} bytes", model_id, total_bytes)

        # Estimate size for backward compatibility
        total_size_gb = total_bytes / (1024**3) if total_bytes else 0.0

        # Yield status with size information (this is the second yield after dry_run)
        logger.info("Size check complete for {}: {} GB", model_id, "total_size_gb:.2f")
        yield DownloadStatus(
            status=DownloadStatusEnum.STARTING,
            model_id=model_id,
            total_size_gb=total_size_gb,
            total_bytes=total_bytes,
            downloaded_bytes=0,
            progress=0,
        )

        # Check if already cancelled before starting download
        if cancel_event and cancel_event.is_set():
            logger.info("Download already cancelled before start for {}", model_id)
            yield DownloadStatus(
                status=DownloadStatusEnum.CANCELLED,
                model_id=model_id,
                total_bytes=total_bytes,
                downloaded_bytes=0,
                progress=0,
            )
            return

        # Start download as a background task, passing cancel_event so the
        # tqdm progress callback can interrupt snapshot_download mid-chunk.
        logger.info("Starting actual download for {}", model_id)
        download_task = loop.run_in_executor(
            None,
            lambda: self._download_with_progress(model_id, cancel_event, token),
        )

        # Calculate directory path for progress polling
        download_dir = self._model_cache_dir(model_id)
        logger.debug("Polling download directory: {}", download_dir)

        # Poll directory size while downloading
        poll_count = 0
        try:
            while not download_task.done():
                await asyncio.sleep(1.0)  # Poll every second
                poll_count += 1

                # Check for cancellation — the tqdm callback will interrupt
                # the thread, but we also check here so we can yield the
                # cancelled status promptly once the task finishes.
                if cancel_event and cancel_event.is_set():
                    logger.info("Download cancelled for {}", model_id)
                    try:
                        await download_task
                    except (DownloadCancelledError, asyncio.CancelledError, Exception):
                        pass
                    yield DownloadStatus(
                        status=DownloadStatusEnum.CANCELLED,
                        model_id=model_id,
                        total_bytes=total_bytes,
                        downloaded_bytes=self._get_directory_size(download_dir),
                        progress=0,
                    )
                    return

                current_bytes = self._get_directory_size(download_dir)
                progress = int((current_bytes / total_bytes) * 100) if total_bytes > 0 else 0

                # Log progress every 10 polls (10 seconds)
                if poll_count % 10 == 0:
                    logger.debug(
                        f"Download progress for {model_id}: "
                        f"{current_bytes}/{total_bytes} bytes ({progress}%)"
                    )

                yield DownloadStatus(
                    status=DownloadStatusEnum.DOWNLOADING,
                    model_id=model_id,
                    total_size_gb=total_size_gb,
                    total_bytes=total_bytes,
                    downloaded_bytes=current_bytes,
                    progress=min(progress, 99),  # Cap at 99 until actually done
                )

            # Get the result (may raise DownloadCancelledError or other exception)
            local_dir = await download_task
            logger.info("Download completed for {}: {}", model_id, local_dir)

            yield DownloadStatus(
                status=DownloadStatusEnum.COMPLETED,
                model_id=model_id,
                local_path=local_dir,
                total_bytes=total_bytes,
                downloaded_bytes=total_bytes,
                progress=100,
            )

        except DownloadCancelledError:
            # The tqdm callback interrupted snapshot_download before the
            # polling loop detected the cancel event.
            logger.info("Download interrupted by cancellation for {}", model_id)
            yield DownloadStatus(
                status=DownloadStatusEnum.CANCELLED,
                model_id=model_id,
                total_bytes=total_bytes,
                downloaded_bytes=self._get_directory_size(download_dir),
                progress=0,
            )
        except Exception as e:
            logger.exception("Download failed for {}: {}", model_id, e)
            yield DownloadStatus(status=DownloadStatusEnum.FAILED, model_id=model_id, error=str(e))

    def _download_with_progress(
        self,
        model_id: str,
        cancel_event: threading.Event | None = None,
        token: str | None = None,
    ) -> str:
        """Perform download (runs in executor).

        Args:
            model_id: HuggingFace model ID to download.
            cancel_event: Optional threading.Event for cancellation.
                When set, the tqdm progress callback will raise
                DownloadCancelledError, interrupting snapshot_download.
            token: Optional HuggingFace API token for authenticated downloads.
        """
        progress_class = make_cancellable_progress(cancel_event)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            return snapshot_download(repo_id=model_id, tqdm_class=progress_class, token=token)

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of downloaded data in HF cache directory.

        Only counts files in blobs/ to avoid double-counting symlinks
        in snapshots/ that point back to blobs/.
        """
        blobs_dir = path / "blobs"
        if not blobs_dir.exists():
            # Fallback for non-standard cache layouts
            if not path.exists():
                return 0
            try:
                return sum(
                    f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()
                )
            except Exception:
                return 0
        try:
            return sum(f.stat().st_size for f in blobs_dir.rglob("*") if f.is_file())
        except Exception as e:
            logger.exception("Failed to calculate directory size for {}: {}", path, e)
            return 0

    def list_local_models(self) -> list[LocalModel]:
        """List all locally downloaded MLX models.

        If hf_organization is set, only lists models from that organization.
        If hf_organization is None, lists all models with 'mlx' in the path
        (common convention for MLX-optimized models).

        Includes model characteristics extracted from config.json when available.
        """
        from mlx_manager.utils.model_detection import extract_characteristics_from_model

        models: list[LocalModel] = []

        if not self.cache_dir.exists():
            logger.warning("No HuggingFace local cache found at: {}", self.cache_dir.absolute)
            return models

        try:
            for item in self.cache_dir.iterdir():
                # Only consider model directories (models--author--name format)
                if not item.name.startswith("models--"):
                    continue

                # Extract model_id from directory name
                model_id: str = item.name.replace("models--", "").replace("--", "/")
                logger.debug("Found cached model: {}", model_id)

                # Filter by organization if specified
                if settings.hf_organization:
                    if not model_id.startswith(f"{settings.hf_organization}/"):
                        continue
                else:
                    # When no organization filter, show models that appear to be MLX models
                    # Common patterns: mlx-community/, lmstudio-community/, or 'mlx' in name
                    model_name_lower: str = model_id.lower()
                    is_mlx_model: bool = (
                        "lmstudio-community/" in model_name_lower or "mlx" in model_name_lower
                    )
                    if not is_mlx_model:
                        continue

                local_path: str | None = self.get_local_path(model_id)

                if local_path:
                    size_bytes = sum(
                        f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()
                    )

                    # Extract model characteristics from config.json
                    characteristics = extract_characteristics_from_model(model_id)

                    models.append(
                        LocalModel(
                            model_id=model_id,
                            local_path=local_path,
                            size_bytes=size_bytes,
                            size_gb=round(size_bytes / (1024**3), 2),
                            characteristics=characteristics,
                        )
                    )
                else:
                    logger.warning("Invalid model path {}", local_path)

        except Exception as e:
            logger.exception("Failed to list local models: {}", e)

        return models

    def list_incomplete_models(self) -> list[tuple[str, int]]:
        """Scan HF cache for models with incomplete downloads.

        Returns a list of (model_id, downloaded_bytes) for models that have
        .incomplete blob files — indicating a partial download that was started
        outside of MLX Manager or interrupted without cleanup.

        downloaded_bytes is the sum of all complete blob sizes (the known-good
        portion). The true total is unknown without querying the HF API, so
        total_bytes is left for the download_model() dry-run to determine.
        """
        results: list[tuple[str, int]] = []

        if not self.cache_dir.exists():
            return results

        for item in self.cache_dir.iterdir():
            if not item.name.startswith("models--"):
                continue

            model_id = item.name.replace("models--", "").replace("--", "/")

            # Filter by organization (same logic as list_local_models)
            if settings.hf_organization:
                if not model_id.startswith(f"{settings.hf_organization}/"):
                    continue
            else:
                model_name_lower = model_id.lower()
                if not ("lmstudio-community/" in model_name_lower or "mlx" in model_name_lower):
                    continue

            if not self.has_incomplete_blobs(model_id):
                continue

            # Sum only complete blob sizes (known-good data)
            blobs_dir = item / "blobs"
            downloaded_bytes = 0
            try:
                for blob in blobs_dir.iterdir():
                    if blob.is_file() and blob.suffix != ".incomplete":
                        downloaded_bytes += blob.stat().st_size
            except Exception:
                pass

            results.append((model_id, downloaded_bytes))
            logger.info("Detected incomplete download in HF cache: {}", model_id)

        return results

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model from local cache."""
        if settings.offline_mode:
            return False

        model_path = self._model_cache_dir(model_id)
        if model_path.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: shutil.rmtree(model_path))
            return True
        return False


# Singleton instance
hf_client = HuggingFaceClient()

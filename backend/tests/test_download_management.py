"""Tests for download management (pause, resume, cancel)."""

import threading
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import select

from mlx_manager.models import Download
from mlx_manager.services.hf_client import (
    _cancel_events,
    cleanup_cancel_event,
    cleanup_partial_download,
    register_cancel_event,
    request_cancel,
)

# ============================================================================
# Cancel Event Mechanism Tests
# ============================================================================


class TestCancelEventMechanism:
    """Tests for the threading.Event-based cancellation infrastructure."""

    def setup_method(self) -> None:
        """Clear cancel events before each test."""
        _cancel_events.clear()

    def test_register_cancel_event_creates_event(self) -> None:
        event = register_cancel_event("download-1")
        assert isinstance(event, threading.Event)
        assert not event.is_set()
        assert "download-1" in _cancel_events

    def test_request_cancel_sets_event(self) -> None:
        event = register_cancel_event("download-2")
        result = request_cancel("download-2")
        assert result is True
        assert event.is_set()

    def test_request_cancel_unknown_id_returns_false(self) -> None:
        result = request_cancel("nonexistent-id")
        assert result is False

    def test_cleanup_cancel_event_removes_it(self) -> None:
        register_cancel_event("download-3")
        assert "download-3" in _cancel_events
        cleanup_cancel_event("download-3")
        assert "download-3" not in _cancel_events

    def test_cleanup_nonexistent_event_no_error(self) -> None:
        # Should not raise any error
        cleanup_cancel_event("does-not-exist")

    def test_multiple_events_independent(self) -> None:
        event_a = register_cancel_event("a")
        event_b = register_cancel_event("b")
        request_cancel("a")
        assert event_a.is_set()
        assert not event_b.is_set()


# ============================================================================
# Cleanup Partial Download Tests
# ============================================================================


class TestCleanupPartialDownload:
    """Tests for HF cache cleanup."""

    @patch("huggingface_hub.scan_cache_dir")
    def test_cleanup_uses_hf_cache_api(self, mock_scan: MagicMock) -> None:
        """Cache API finds and deletes model revisions."""
        mock_revision = MagicMock()
        mock_revision.commit_hash = "abc123"

        mock_repo = MagicMock()
        mock_repo.repo_id = "mlx-community/test-model"
        mock_repo.revisions = [mock_revision]

        mock_cache = MagicMock()
        mock_cache.repos = [mock_repo]

        mock_strategy = MagicMock()
        mock_strategy.expected_freed_size_str = "2.5 GB"
        mock_cache.delete_revisions.return_value = mock_strategy

        mock_scan.return_value = mock_cache

        result = cleanup_partial_download("mlx-community/test-model")
        assert result is True
        mock_cache.delete_revisions.assert_called_once_with("abc123")
        mock_strategy.execute.assert_called_once()

    @patch("mlx_manager.services.hf_client._manual_cleanup", return_value=True)
    @patch("huggingface_hub.scan_cache_dir", side_effect=RuntimeError("Cache API error"))
    def test_cleanup_falls_back_to_manual_on_error(
        self,
        mock_scan: MagicMock,
        mock_manual: MagicMock,
    ) -> None:
        """Falls back to manual deletion when cache API fails."""
        result = cleanup_partial_download("mlx-community/test-model")
        assert result is True
        mock_manual.assert_called_once_with("mlx-community/test-model")

    @patch("mlx_manager.services.hf_client._manual_cleanup", return_value=False)
    @patch("huggingface_hub.scan_cache_dir")
    def test_cleanup_model_not_in_cache(self, mock_scan: MagicMock, mock_manual: MagicMock) -> None:
        """Returns False when model not found in cache."""
        mock_cache = MagicMock()
        mock_cache.repos = []
        mock_scan.return_value = mock_cache

        result = cleanup_partial_download("mlx-community/unknown-model")
        assert result is False


# ============================================================================
# Pause Endpoint Tests
# ============================================================================


class TestPauseEndpoint:
    """Tests for POST /api/models/download/{id}/pause."""

    @pytest.mark.anyio
    async def test_pause_downloading_sets_paused(self, auth_client, test_session) -> None:
        """Pausing an actively downloading model sets status to paused."""
        download = Download(
            model_id="mlx-community/test-model",
            status="downloading",
            started_at=datetime.now(tz=UTC),
            total_bytes=1000000,
            downloaded_bytes=500000,
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        with patch("mlx_manager.routers.models.request_cancel") as mock_cancel:
            response = await auth_client.post(f"/api/models/download/{download.id}/pause")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"
        assert data["model_id"] == "mlx-community/test-model"
        mock_cancel.assert_called_once_with(str(download.id))

    @pytest.mark.anyio
    async def test_pause_non_downloading_returns_409(self, auth_client, test_session) -> None:
        """Cannot pause a download that is not in 'downloading' state."""
        download = Download(
            model_id="mlx-community/test-model",
            status="completed",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        response = await auth_client.post(f"/api/models/download/{download.id}/pause")
        assert response.status_code == 409

    @pytest.mark.anyio
    async def test_pause_paused_returns_409(self, auth_client, test_session) -> None:
        """Cannot pause a download that is already paused."""
        download = Download(
            model_id="mlx-community/test-model",
            status="paused",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        response = await auth_client.post(f"/api/models/download/{download.id}/pause")
        assert response.status_code == 409

    @pytest.mark.anyio
    async def test_pause_unknown_id_returns_404(self, auth_client) -> None:
        """Pausing a nonexistent download returns 404."""
        response = await auth_client.post("/api/models/download/99999/pause")
        assert response.status_code == 404


# ============================================================================
# Resume Endpoint Tests
# ============================================================================


class TestResumeEndpoint:
    """Tests for POST /api/models/download/{id}/resume."""

    @pytest.mark.anyio
    async def test_resume_paused_sets_downloading(self, auth_client, test_session) -> None:
        """Resuming a paused download sets status to downloading and returns task_id."""
        download = Download(
            model_id="mlx-community/test-model",
            status="paused",
            started_at=datetime.now(tz=UTC),
            total_bytes=1000000,
            downloaded_bytes=500000,
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        response = await auth_client.post(f"/api/models/download/{download.id}/resume")
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["model_id"] == "mlx-community/test-model"
        assert data["download_id"] == download.id

    @pytest.mark.anyio
    async def test_resume_non_paused_returns_409(self, auth_client, test_session) -> None:
        """Cannot resume a download that is not paused."""
        download = Download(
            model_id="mlx-community/test-model",
            status="downloading",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        response = await auth_client.post(f"/api/models/download/{download.id}/resume")
        assert response.status_code == 409

    @pytest.mark.anyio
    async def test_resume_unknown_id_returns_404(self, auth_client) -> None:
        """Resuming a nonexistent download returns 404."""
        response = await auth_client.post("/api/models/download/99999/resume")
        assert response.status_code == 404


# ============================================================================
# Cancel Endpoint Tests
# ============================================================================


class TestCancelEndpoint:
    """Tests for POST /api/models/download/{id}/cancel."""

    @pytest.mark.anyio
    async def test_cancel_downloading_sets_cancelled(self, auth_client, test_session) -> None:
        """Cancelling an active download sets status to cancelled and cleans up."""
        download = Download(
            model_id="mlx-community/test-model",
            status="downloading",
            started_at=datetime.now(tz=UTC),
            total_bytes=1000000,
            downloaded_bytes=500000,
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        with (
            patch("mlx_manager.routers.models.request_cancel") as mock_cancel,
            patch(
                "mlx_manager.routers.models.cleanup_partial_download",
                return_value=True,
            ) as mock_cleanup,
        ):
            response = await auth_client.post(f"/api/models/download/{download.id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
        assert data["cleanup_success"] is True
        mock_cancel.assert_called_once_with(str(download.id))
        mock_cleanup.assert_called_once_with("mlx-community/test-model")

    @pytest.mark.anyio
    async def test_cancel_paused_sets_cancelled(self, auth_client, test_session) -> None:
        """Cancelling a paused download also works (no cancel signal needed)."""
        download = Download(
            model_id="mlx-community/test-model",
            status="paused",
            started_at=datetime.now(tz=UTC),
            total_bytes=1000000,
            downloaded_bytes=500000,
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        with patch(
            "mlx_manager.routers.models.cleanup_partial_download",
            return_value=True,
        ) as mock_cleanup:
            response = await auth_client.post(f"/api/models/download/{download.id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
        mock_cleanup.assert_called_once()

    @pytest.mark.anyio
    async def test_cancel_completed_returns_409(self, auth_client, test_session) -> None:
        """Cannot cancel a download that is already completed."""
        download = Download(
            model_id="mlx-community/test-model",
            status="completed",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        response = await auth_client.post(f"/api/models/download/{download.id}/cancel")
        assert response.status_code == 409

    @pytest.mark.anyio
    async def test_cancel_already_cancelled_returns_409(self, auth_client, test_session) -> None:
        """Cannot cancel a download that is already cancelled."""
        download = Download(
            model_id="mlx-community/test-model",
            status="cancelled",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        response = await auth_client.post(f"/api/models/download/{download.id}/cancel")
        assert response.status_code == 409

    @pytest.mark.anyio
    async def test_cancel_unknown_id_returns_404(self, auth_client) -> None:
        """Cancelling a nonexistent download returns 404."""
        response = await auth_client.post("/api/models/download/99999/cancel")
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_cancel_cleanup_failure_still_cancels(self, auth_client, test_session) -> None:
        """Download is still cancelled even if cleanup fails."""
        download = Download(
            model_id="mlx-community/test-model",
            status="downloading",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()
        await test_session.refresh(download)

        with (
            patch("mlx_manager.routers.models.request_cancel"),
            patch(
                "mlx_manager.routers.models.cleanup_partial_download",
                side_effect=RuntimeError("Cleanup failed"),
            ),
        ):
            response = await auth_client.post(f"/api/models/download/{download.id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
        assert data["cleanup_success"] is False


# ============================================================================
# Active Downloads Listing Tests
# ============================================================================


class TestActiveDownloadsWithPaused:
    """Tests that paused downloads appear in active downloads listing."""

    def setup_method(self) -> None:
        """Clear in-memory download tasks before each test."""
        from mlx_manager.routers.models import download_tasks

        download_tasks.clear()

    @pytest.mark.anyio
    async def test_paused_downloads_in_active_list(self, auth_client, test_session) -> None:
        """Paused downloads are included in the active downloads response."""
        download = Download(
            model_id="mlx-community/paused-model",
            status="paused",
            started_at=datetime.now(tz=UTC),
            total_bytes=2000000,
            downloaded_bytes=1000000,
        )
        test_session.add(download)
        await test_session.commit()

        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200
        data = response.json()
        # Filter to our specific model in case of other test data
        paused = [d for d in data if d["model_id"] == "mlx-community/paused-model"]
        assert len(paused) == 1
        assert paused[0]["status"] == "paused"
        assert paused[0]["download_id"] == download.id

    @pytest.mark.anyio
    async def test_cancelled_downloads_not_in_active_list(self, auth_client, test_session) -> None:
        """Cancelled downloads are not included in active downloads."""
        download = Download(
            model_id="mlx-community/cancelled-model",
            status="cancelled",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()

        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200
        data = response.json()
        cancelled = [d for d in data if d["model_id"] == "mlx-community/cancelled-model"]
        assert len(cancelled) == 0


# ============================================================================
# Recovery on Restart Tests
# ============================================================================


class TestDownloadRecovery:
    """Tests for download recovery behavior on server restart."""

    @pytest.mark.anyio
    async def test_downloading_recovers_as_pending(self, test_session) -> None:
        """Downloads in 'downloading' state are marked pending for auto-resume."""
        download = Download(
            model_id="mlx-community/test-model",
            status="downloading",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()
        download_id = download.id

        # Mock get_session to use test_session
        with patch("mlx_manager.database.get_session") as mock_get_session:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def mock_session_ctx():
                yield test_session

            mock_get_session.return_value = mock_session_ctx()

            from mlx_manager.database import recover_incomplete_downloads

            pending = await recover_incomplete_downloads()

        # The download should be in the pending list
        assert len(pending) == 1
        assert pending[0] == (download_id, "mlx-community/test-model")

    @pytest.mark.anyio
    async def test_paused_stays_paused_on_restart(self, test_session) -> None:
        """Downloads in 'paused' state remain paused after restart."""
        download = Download(
            model_id="mlx-community/paused-model",
            status="paused",
            started_at=datetime.now(tz=UTC),
            total_bytes=2000000,
            downloaded_bytes=1000000,
        )
        test_session.add(download)
        await test_session.commit()

        with patch("mlx_manager.database.get_session") as mock_get_session:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def mock_session_ctx():
                yield test_session

            mock_get_session.return_value = mock_session_ctx()

            from mlx_manager.database import recover_incomplete_downloads

            pending = await recover_incomplete_downloads()

        # Paused downloads should NOT be in the auto-resume list
        assert len(pending) == 0

        # Verify the download is still "paused" in the database
        result = await test_session.execute(select(Download).where(Download.id == download.id))
        db_download = result.scalars().first()
        assert db_download is not None
        assert db_download.status == "paused"

    @pytest.mark.anyio
    async def test_cancelled_stays_cancelled_on_restart(self, test_session) -> None:
        """Cancelled downloads are not recovered."""
        download = Download(
            model_id="mlx-community/cancelled-model",
            status="cancelled",
            started_at=datetime.now(tz=UTC),
        )
        test_session.add(download)
        await test_session.commit()

        with patch("mlx_manager.database.get_session") as mock_get_session:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def mock_session_ctx():
                yield test_session

            mock_get_session.return_value = mock_session_ctx()

            from mlx_manager.database import recover_incomplete_downloads

            pending = await recover_incomplete_downloads()

        # Cancelled downloads should NOT be in the auto-resume list
        assert len(pending) == 0

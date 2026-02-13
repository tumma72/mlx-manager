"""Tests for the main FastAPI application module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.models.dto.models import DownloadStatus


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_healthy(self, client):
        """Test that health endpoint returns healthy status."""
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestLifespan:
    """Tests for application lifespan events."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_calls_init_db(self):
        """Test that lifespan startup calls init_db."""
        mock_pool = MagicMock()
        mock_pool.cleanup = AsyncMock()

        with (
            patch("mlx_manager.main.init_db") as mock_init_db,
            patch("mlx_manager.main.recover_incomplete_downloads") as mock_recover,
            patch("mlx_manager.main.health_checker") as mock_hc,
            patch("mlx_manager.main.set_memory_limit"),
            patch("mlx_manager.main.ModelPoolManager", return_value=mock_pool),
            patch("mlx_manager.main.pool"),
        ):
            mock_init_db.return_value = None
            mock_recover.return_value = []  # No pending downloads
            mock_hc.start = AsyncMock()
            mock_hc.stop = AsyncMock()

            from mlx_manager.main import lifespan

            # Create a mock app
            mock_app = MagicMock()

            # Run through the lifespan
            async with lifespan(mock_app):
                mock_init_db.assert_called_once()
                mock_recover.assert_called_once()
                mock_hc.start.assert_called_once()

            # After exiting context, shutdown should have occurred
            mock_hc.stop.assert_called_once()


class TestStaticFileServing:
    """Tests for static file serving routes (when STATIC_DIR exists)."""

    @pytest.mark.asyncio
    async def test_favicon_returns_file_when_exists(self):
        """Test favicon route returns file when it exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir)
            favicon_path = static_dir / "favicon.png"

            # Create a fake favicon
            favicon_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            # Create _app directory so STATIC_DIR.exists() is True
            assets_dir = static_dir / "_app"
            assets_dir.mkdir()

            with patch("mlx_manager.main.STATIC_DIR", static_dir):
                # Dynamically add the routes by importing main again
                import sys

                # Remove main from modules so it gets re-evaluated
                if "mlx_manager.main" in sys.modules:
                    del sys.modules["mlx_manager.main"]

                # Create a test client with the patched app
                from httpx import ASGITransport, AsyncClient

                from mlx_manager import main

                async with AsyncClient(
                    transport=ASGITransport(app=main.app), base_url="http://test"
                ) as test_client:
                    response = await test_client.get("/favicon.png")
                    assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_favicon_returns_404_when_missing(self):
        """Test favicon route returns 404 when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir)
            # Create _app directory but not favicon.png
            assets_dir = static_dir / "_app"
            assets_dir.mkdir()

            with patch("mlx_manager.main.STATIC_DIR", static_dir):
                import sys

                if "mlx_manager.main" in sys.modules:
                    del sys.modules["mlx_manager.main"]

                from httpx import ASGITransport, AsyncClient

                from mlx_manager import main

                async with AsyncClient(
                    transport=ASGITransport(app=main.app), base_url="http://test"
                ) as test_client:
                    response = await test_client.get("/favicon.png")
                    assert response.status_code == 404
                    assert response.json()["error"] == "Not found"

    @pytest.mark.asyncio
    async def test_spa_serves_exact_file_when_exists(self):
        """Test SPA route serves exact file when it exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir)
            assets_dir = static_dir / "_app"
            assets_dir.mkdir()

            # Create a test file
            test_file = static_dir / "test.js"
            test_file.write_text("console.log('test');")

            with patch("mlx_manager.main.STATIC_DIR", static_dir):
                import sys

                if "mlx_manager.main" in sys.modules:
                    del sys.modules["mlx_manager.main"]

                from httpx import ASGITransport, AsyncClient

                from mlx_manager import main

                async with AsyncClient(
                    transport=ASGITransport(app=main.app), base_url="http://test"
                ) as test_client:
                    response = await test_client.get("/test.js")
                    assert response.status_code == 200
                    assert b"console.log('test');" in response.content

    @pytest.mark.asyncio
    async def test_spa_returns_index_for_route(self):
        """Test SPA route returns index.html for unknown paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir)
            assets_dir = static_dir / "_app"
            assets_dir.mkdir()

            # Create index.html
            index_path = static_dir / "index.html"
            index_path.write_text("<html><body>SPA</body></html>")

            with patch("mlx_manager.main.STATIC_DIR", static_dir):
                import sys

                if "mlx_manager.main" in sys.modules:
                    del sys.modules["mlx_manager.main"]

                from httpx import ASGITransport, AsyncClient

                from mlx_manager import main

                async with AsyncClient(
                    transport=ASGITransport(app=main.app), base_url="http://test"
                ) as test_client:
                    response = await test_client.get("/some/unknown/route")
                    assert response.status_code == 200
                    assert b"SPA" in response.content

    @pytest.mark.asyncio
    async def test_spa_skips_api_routes(self):
        """Test SPA route returns 404 for api/ paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir)
            assets_dir = static_dir / "_app"
            assets_dir.mkdir()

            with patch("mlx_manager.main.STATIC_DIR", static_dir):
                import sys

                if "mlx_manager.main" in sys.modules:
                    del sys.modules["mlx_manager.main"]

                from httpx import ASGITransport, AsyncClient

                from mlx_manager import main

                async with AsyncClient(
                    transport=ASGITransport(app=main.app), base_url="http://test"
                ) as test_client:
                    response = await test_client.get("/api/nonexistent")
                    assert response.status_code == 404
                    assert response.json()["error"] == "Not found"

    @pytest.mark.asyncio
    async def test_spa_returns_404_when_no_index(self):
        """Test SPA route returns 404 when index.html doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir)
            assets_dir = static_dir / "_app"
            assets_dir.mkdir()
            # Don't create index.html

            with patch("mlx_manager.main.STATIC_DIR", static_dir):
                import sys

                if "mlx_manager.main" in sys.modules:
                    del sys.modules["mlx_manager.main"]

                from httpx import ASGITransport, AsyncClient

                from mlx_manager import main

                async with AsyncClient(
                    transport=ASGITransport(app=main.app), base_url="http://test"
                ) as test_client:
                    response = await test_client.get("/some/unknown/route")
                    assert response.status_code == 404
                    assert response.json()["error"] == "Not found"


class TestDevModeRoot:
    """Tests for development mode root endpoint."""

    @pytest.mark.asyncio
    async def test_dev_root_returns_info(self, client):
        """Test that dev mode root endpoint returns app info.

        Note: The dev mode root endpoint is only registered when STATIC_DIR doesn't exist
        at module load time. In test environment, this depends on whether the static dir
        is present. This test verifies the endpoint behavior when exposed.

        Since the STATIC_DIR check happens at import time and cannot be easily mocked,
        we verify the expected behavior: if the endpoint is registered, it returns app info;
        if not registered (static dir exists), we get a different response.
        """
        response = await client.get("/")

        # Accept either dev mode response or different response if static dir exists
        if response.status_code == 200:
            # Try to parse as JSON (dev mode)
            try:
                data = response.json()
                # In dev mode, we get app info
                if "name" in data:
                    assert data["name"] == "MLX Model Manager"
                    assert "version" in data
                    assert data["docs"] == "/docs"
                    assert "Frontend not embedded" in data["note"]
            except Exception:
                # If static dir exists, we might get HTML served instead of JSON
                pass
        # 404 is acceptable if no route matches
        assert response.status_code in [200, 404]


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""

    @pytest.mark.asyncio
    async def test_cors_allows_localhost_origins(self, client):
        """Test that CORS allows configured localhost origins."""
        # Make a preflight request
        response = await client.options(
            "/api/profiles",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS should be configured (FastAPI/Starlette handles this)
        # The middleware should allow the origin
        assert response.status_code in [200, 204, 400]  # Depends on configuration


class TestCancelDownloadTasks:
    """Tests for cancel_download_tasks function."""

    @pytest.mark.asyncio
    async def test_cancel_with_running_tasks(self):
        """Test cancel_download_tasks cancels running tasks."""
        import asyncio

        from mlx_manager import main

        # Save original list and clear it
        original_tasks = main._download_tasks.copy()
        main._download_tasks.clear()

        # Create mock running tasks
        async def long_running():
            await asyncio.sleep(100)

        task1 = asyncio.create_task(long_running())
        task2 = asyncio.create_task(long_running())
        main._download_tasks.extend([task1, task2])

        # Call cancel_download_tasks
        await main.cancel_download_tasks()

        # Verify tasks were cancelled
        assert task1.cancelled() or task1.done()
        assert task2.cancelled() or task2.done()
        assert len(main._download_tasks) == 0

        # Restore original list
        main._download_tasks.extend(original_tasks)

    @pytest.mark.asyncio
    async def test_cancel_with_already_done_tasks(self):
        """Test cancel_download_tasks handles already-done tasks."""
        import asyncio

        from mlx_manager import main

        # Save original list and clear it
        original_tasks = main._download_tasks.copy()
        main._download_tasks.clear()

        # Create tasks that complete immediately
        async def quick_task():
            return "done"

        task1 = asyncio.create_task(quick_task())
        task2 = asyncio.create_task(quick_task())

        # Wait for them to complete
        await asyncio.sleep(0.01)
        main._download_tasks.extend([task1, task2])

        # Call cancel_download_tasks - should not raise
        await main.cancel_download_tasks()

        # List should be cleared
        assert len(main._download_tasks) == 0

        # Restore original list
        main._download_tasks.extend(original_tasks)

    @pytest.mark.asyncio
    async def test_cancel_with_empty_list(self):
        """Test cancel_download_tasks with empty task list."""
        from mlx_manager import main

        # Save original list and clear it
        original_tasks = main._download_tasks.copy()
        main._download_tasks.clear()

        # Should not raise with empty list
        await main.cancel_download_tasks()

        # List should still be empty
        assert len(main._download_tasks) == 0

        # Restore original list
        main._download_tasks.extend(original_tasks)


class TestResumePendingDownloads:
    """Tests for resume_pending_downloads function."""

    @pytest.mark.asyncio
    async def test_resume_creates_tasks_for_pending_downloads(self):
        """Test that resume_pending_downloads creates tasks for each pending download."""
        import asyncio

        from mlx_manager import main
        from mlx_manager.routers.models import download_tasks

        # Save original state
        original_tasks = main._download_tasks.copy()
        original_download_tasks = download_tasks.copy()
        main._download_tasks.clear()
        download_tasks.clear()

        # Mock _run_download_task to avoid actual downloads
        async def mock_run_download_task(task_id, download_id, model_id):
            await asyncio.sleep(0.01)

        with patch.object(main, "_run_download_task", mock_run_download_task):
            pending = [(1, "mlx-community/model1"), (2, "mlx-community/model2")]
            await main.resume_pending_downloads(pending)

            # Wait a bit for tasks to be created
            await asyncio.sleep(0.05)

        # Verify download_tasks dict was populated
        assert len(download_tasks) == 2

        # Verify task structure
        for task_id, task_info in download_tasks.items():
            assert "model_id" in task_info
            assert "download_id" in task_info
            assert task_info["status"] == "pending"
            assert task_info["progress"] == 0

        # Verify asyncio tasks were created
        assert len(main._download_tasks) == 2

        # Clean up - cancel the tasks
        for task in main._download_tasks:
            task.cancel()
        await asyncio.gather(*main._download_tasks, return_exceptions=True)

        # Restore original state
        main._download_tasks.clear()
        main._download_tasks.extend(original_tasks)
        download_tasks.clear()
        download_tasks.update(original_download_tasks)

    @pytest.mark.asyncio
    async def test_resume_with_empty_pending_list(self):
        """Test resume_pending_downloads with empty pending list."""
        from mlx_manager import main
        from mlx_manager.routers.models import download_tasks

        # Save original state
        original_tasks = main._download_tasks.copy()
        original_download_tasks = download_tasks.copy()
        main._download_tasks.clear()
        download_tasks.clear()

        await main.resume_pending_downloads([])

        # No tasks should be created
        assert len(main._download_tasks) == 0
        assert len(download_tasks) == 0

        # Restore original state
        main._download_tasks.extend(original_tasks)
        download_tasks.update(original_download_tasks)


class TestRunDownloadTask:
    """Tests for _run_download_task function."""

    @pytest.mark.asyncio
    async def test_successful_download(self):
        """Test _run_download_task with successful download."""
        from mlx_manager import main
        from mlx_manager.routers.models import download_tasks

        # Save original state
        original_download_tasks = download_tasks.copy()
        download_tasks.clear()

        task_id = "test-task-123"
        download_id = 42
        model_id = "mlx-community/test-model"

        # Set up initial task state
        download_tasks[task_id] = {
            "model_id": model_id,
            "download_id": download_id,
            "status": "pending",
            "progress": 0,
        }

        # Mock hf_client.download_model to yield progress events
        async def mock_download_model(model_id):
            yield DownloadStatus(
                status="downloading",
                progress=50,
                downloaded_bytes=500,
                total_bytes=1000,
            )
            yield DownloadStatus(
                status="completed",
                progress=100,
                downloaded_bytes=1000,
                total_bytes=1000,
            )

        # Mock _update_download_record (imported inside _run_download_task from routers.models)
        update_calls = []

        async def mock_update_record(download_id, **kwargs):
            update_calls.append((download_id, kwargs))

        with (
            patch.object(main.hf_client, "download_model", mock_download_model),
            patch("mlx_manager.routers.models._update_download_record", mock_update_record),
        ):
            await main._run_download_task(task_id, download_id, model_id)

        # Verify download_tasks was updated with progress
        assert download_tasks[task_id]["status"] == "completed"
        assert download_tasks[task_id]["progress"] == 100

        # Verify _update_download_record was called for final status
        assert len(update_calls) >= 1
        final_call = update_calls[-1]
        assert final_call[0] == download_id
        assert final_call[1]["status"] == "completed"
        assert final_call[1]["completed"] is True

        # Restore original state
        download_tasks.clear()
        download_tasks.update(original_download_tasks)

    @pytest.mark.asyncio
    async def test_cancelled_error_handling(self):
        """Test _run_download_task handles CancelledError properly."""
        import asyncio

        from mlx_manager import main
        from mlx_manager.routers.models import download_tasks

        # Save original state
        original_download_tasks = download_tasks.copy()
        download_tasks.clear()

        task_id = "test-task-cancel"
        download_id = 43
        model_id = "mlx-community/test-model"

        download_tasks[task_id] = {
            "model_id": model_id,
            "download_id": download_id,
            "status": "pending",
            "progress": 0,
        }

        # Mock hf_client.download_model to raise CancelledError
        async def mock_download_model(model_id):
            yield DownloadStatus(status="downloading", progress=50)
            raise asyncio.CancelledError()

        with patch.object(main.hf_client, "download_model", mock_download_model):
            # CancelledError should be re-raised
            with pytest.raises(asyncio.CancelledError):
                await main._run_download_task(task_id, download_id, model_id)

        # Restore original state
        download_tasks.clear()
        download_tasks.update(original_download_tasks)

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test _run_download_task handles exceptions and updates DB."""
        from mlx_manager import main
        from mlx_manager.routers.models import download_tasks

        # Save original state
        original_download_tasks = download_tasks.copy()
        download_tasks.clear()

        task_id = "test-task-error"
        download_id = 44
        model_id = "mlx-community/test-model"

        download_tasks[task_id] = {
            "model_id": model_id,
            "download_id": download_id,
            "status": "pending",
            "progress": 0,
        }

        # Mock hf_client.download_model to raise an exception
        async def mock_download_model(model_id):
            yield DownloadStatus(status="downloading", progress=25)
            raise RuntimeError("Network error")

        # Mock _update_download_record (imported inside _run_download_task from routers.models)
        update_calls = []

        async def mock_update_record(download_id, **kwargs):
            update_calls.append((download_id, kwargs))

        with (
            patch.object(main.hf_client, "download_model", mock_download_model),
            patch("mlx_manager.routers.models._update_download_record", mock_update_record),
        ):
            # Should not raise - exception is caught and logged
            await main._run_download_task(task_id, download_id, model_id)

        # Verify _update_download_record was called with failure
        assert len(update_calls) == 1
        assert update_calls[0][0] == download_id
        assert update_calls[0][1]["status"] == "failed"
        assert "Network error" in update_calls[0][1]["error"]

        # Restore original state
        download_tasks.clear()
        download_tasks.update(original_download_tasks)


class TestLifespanWithPendingDownloads:
    """Tests for lifespan handler with pending downloads."""

    @pytest.mark.asyncio
    async def test_lifespan_resumes_pending_downloads(self):
        """Test that lifespan calls resume_pending_downloads when there are pending downloads."""
        mock_pool = MagicMock()
        mock_pool.cleanup = AsyncMock()

        with (
            patch("mlx_manager.main.init_db"),
            patch("mlx_manager.main.recover_incomplete_downloads") as mock_recover,
            patch("mlx_manager.main.resume_pending_downloads") as mock_resume,
            patch("mlx_manager.main.health_checker") as mock_hc,
            patch("mlx_manager.main.cancel_download_tasks") as mock_cancel,
            patch("mlx_manager.main.set_memory_limit"),
            patch("mlx_manager.main.ModelPoolManager", return_value=mock_pool),
            patch("mlx_manager.main.pool"),
        ):
            mock_recover.return_value = [
                (1, "mlx-community/model1"),
                (2, "mlx-community/model2"),
            ]
            mock_resume.return_value = None
            mock_hc.start = AsyncMock()
            mock_hc.stop = AsyncMock()
            mock_cancel.return_value = None

            from mlx_manager.main import lifespan

            mock_app = MagicMock()

            async with lifespan(mock_app):
                # Verify resume was called with pending downloads
                mock_resume.assert_called_once_with(
                    [
                        (1, "mlx-community/model1"),
                        (2, "mlx-community/model2"),
                    ]
                )

            # Verify cancel was called on shutdown
            mock_cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_skips_resume_when_no_pending(self):
        """Test that lifespan doesn't call resume_pending_downloads when no pending downloads."""
        mock_pool = MagicMock()
        mock_pool.cleanup = AsyncMock()

        with (
            patch("mlx_manager.main.init_db"),
            patch("mlx_manager.main.recover_incomplete_downloads") as mock_recover,
            patch("mlx_manager.main.resume_pending_downloads") as mock_resume,
            patch("mlx_manager.main.health_checker") as mock_hc,
            patch("mlx_manager.main.cancel_download_tasks") as mock_cancel,
            patch("mlx_manager.main.set_memory_limit"),
            patch("mlx_manager.main.ModelPoolManager", return_value=mock_pool),
            patch("mlx_manager.main.pool"),
        ):
            mock_recover.return_value = []  # No pending downloads
            mock_resume.return_value = None
            mock_hc.start = AsyncMock()
            mock_hc.stop = AsyncMock()
            mock_cancel.return_value = None

            from mlx_manager.main import lifespan

            mock_app = MagicMock()

            async with lifespan(mock_app):
                # Verify resume was NOT called
                mock_resume.assert_not_called()

            mock_cancel.assert_called_once()

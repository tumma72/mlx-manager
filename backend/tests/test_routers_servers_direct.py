"""Direct unit tests for servers router functions.

These tests call router functions directly with mock sessions to ensure
coverage is properly tracked (avoiding ASGI transport coverage issues).
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.models import RunningInstance, ServerProfile
from mlx_manager.routers.servers import (
    list_running_servers,
    restart_server,
    start_server,
    stop_server,
)


class TestListRunningServersDirect:
    """Direct tests for list_running_servers function."""

    @pytest.mark.asyncio
    async def test_list_running_servers_empty(self):
        """Test list_running_servers returns empty list when no servers."""
        mock_session = AsyncMock()

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.get_all_running.return_value = []

            result = await list_running_servers(session=mock_session)

        assert result == []

    @pytest.mark.asyncio
    async def test_list_running_servers_with_data(self):
        """Test list_running_servers returns server data."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1
        mock_profile.name = "Test Server"
        mock_profile.port = 10240

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_profile]
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.get_all_running.return_value = [
                {
                    "profile_id": 1,
                    "pid": 12345,
                    "status": "running",
                    "create_time": time.time() - 100,
                    "memory_mb": 500,
                }
            ]

            result = await list_running_servers(session=mock_session)

        assert len(result) == 1
        assert result[0].profile_id == 1
        assert result[0].profile_name == "Test Server"
        assert result[0].pid == 12345

    @pytest.mark.asyncio
    async def test_list_running_servers_skips_missing_profiles(self):
        """Test list_running_servers skips servers without matching profiles."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []  # No profiles found
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.get_all_running.return_value = [
                {"profile_id": 999, "pid": 12345}  # Profile doesn't exist
            ]

            result = await list_running_servers(session=mock_session)

        assert result == []


class TestStartServerDirect:
    """Direct tests for start_server function."""

    @pytest.mark.asyncio
    async def test_start_server_success(self):
        """Test start_server starts and records server."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1
        mock_profile.port = 10240

        mock_session = AsyncMock()
        # Query for stale instance - none
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.start_server = AsyncMock(return_value=12345)

            result = await start_server(profile=mock_profile, session=mock_session)

        assert result == {"pid": 12345, "port": 10240}
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_server_cleans_stale_instance(self):
        """Test start_server cleans up stale DB record."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1
        mock_profile.port = 10240

        stale_instance = MagicMock(spec=RunningInstance)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = stale_instance
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.is_running.return_value = False  # Process not running
            mock_sm.start_server = AsyncMock(return_value=12345)

            result = await start_server(profile=mock_profile, session=mock_session)

        assert result == {"pid": 12345, "port": 10240}
        mock_session.delete.assert_called_with(stale_instance)

    @pytest.mark.asyncio
    async def test_start_server_runtime_error(self):
        """Test start_server raises 409 on runtime error."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.start_server = AsyncMock(side_effect=RuntimeError("Already running"))

            with pytest.raises(HTTPException) as exc_info:
                await start_server(profile=mock_profile, session=mock_session)

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_start_server_generic_error(self):
        """Test start_server raises 500 on generic error."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.start_server = AsyncMock(side_effect=Exception("Unknown error"))

            with pytest.raises(HTTPException) as exc_info:
                await start_server(profile=mock_profile, session=mock_session)

        assert exc_info.value.status_code == 500


class TestStopServerDirect:
    """Direct tests for stop_server function."""

    @pytest.mark.asyncio
    async def test_stop_server_success(self):
        """Test stop_server stops and removes record."""
        mock_instance = MagicMock(spec=RunningInstance)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_instance
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.stop_server = AsyncMock(return_value=True)

            result = await stop_server(profile_id=1, session=mock_session)

        assert result == {"stopped": True}
        mock_session.delete.assert_called_with(mock_instance)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_server_not_running(self):
        """Test stop_server raises 404 when server not running."""
        mock_session = AsyncMock()

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.stop_server = AsyncMock(return_value=False)

            with pytest.raises(HTTPException) as exc_info:
                await stop_server(profile_id=1, session=mock_session)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_server_no_db_record(self):
        """Test stop_server succeeds even without DB record."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # No record
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.stop_server = AsyncMock(return_value=True)

            result = await stop_server(profile_id=1, session=mock_session)

        assert result == {"stopped": True}
        mock_session.delete.assert_not_called()


class TestRestartServerDirect:
    """Direct tests for restart_server function."""

    @pytest.mark.asyncio
    async def test_restart_server_updates_existing_instance(self):
        """Test restart_server updates existing DB record."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1

        mock_instance = MagicMock(spec=RunningInstance)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_instance
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.stop_server = AsyncMock(return_value=True)
            mock_sm.start_server = AsyncMock(return_value=54321)

            with patch("mlx_manager.routers.servers.asyncio.sleep", new_callable=AsyncMock):
                result = await restart_server(profile=mock_profile, session=mock_session)

        assert result == {"pid": 54321}
        assert mock_instance.pid == 54321
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_server_creates_new_instance(self):
        """Test restart_server creates new record when none exists."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # No existing record
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.stop_server = AsyncMock(return_value=True)
            mock_sm.start_server = AsyncMock(return_value=54321)

            with patch("mlx_manager.routers.servers.asyncio.sleep", new_callable=AsyncMock):
                result = await restart_server(profile=mock_profile, session=mock_session)

        assert result == {"pid": 54321}
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_server_error(self):
        """Test restart_server raises 500 on error."""
        mock_profile = MagicMock(spec=ServerProfile)
        mock_profile.id = 1

        mock_session = AsyncMock()

        with patch("mlx_manager.routers.servers.server_manager") as mock_sm:
            mock_sm.stop_server = AsyncMock(return_value=True)
            mock_sm.start_server = AsyncMock(side_effect=Exception("Failed to start"))

            with patch("mlx_manager.routers.servers.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(HTTPException) as exc_info:
                    await restart_server(profile=mock_profile, session=mock_session)

        assert exc_info.value.status_code == 500

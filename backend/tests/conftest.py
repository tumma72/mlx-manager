"""Pytest fixtures for backend tests."""

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

# Set test environment before importing app modules
os.environ["MLX_MANAGER_DATABASE_PATH"] = ":memory:"
os.environ["MLX_MANAGER_DEBUG"] = "false"

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user
from mlx_manager.main import app
from mlx_manager.models import User, UserStatus
from mlx_manager.services.auth_service import create_access_token, hash_password


@pytest.fixture(autouse=True)
def mock_find_mlx_openai_server(request):
    """Mock find_mlx_openai_server globally since it's not available on Linux CI.

    Skipped for test_utils_command_builder.py which tests the function directly.
    """
    # Skip for tests that specifically test find_mlx_openai_server
    if "test_utils_command_builder" in request.fspath.basename:
        yield
        return

    with patch(
        "mlx_manager.utils.command_builder.find_mlx_openai_server",
        return_value="/usr/local/bin/mlx-openai-server",
    ):
        yield


@pytest.fixture(scope="function")
async def test_engine():
    """Create a test database engine with in-memory SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture(scope="function")
async def test_session(test_engine):
    """Create a test database session."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
async def client(test_engine):
    """Create an async test client with test database."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    # Mock the health checker to prevent it from running during tests
    with patch("mlx_manager.main.health_checker") as mock_health_checker:
        mock_health_checker.start = AsyncMock()
        mock_health_checker.stop = AsyncMock()

        # Mock the server manager
        with patch("mlx_manager.main.server_manager") as mock_server_manager:
            mock_server_manager.cleanup = AsyncMock()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    return {
        "name": "Test Profile",
        "description": "A test profile",
        "model_path": "mlx-community/test-model-4bit",
        "model_type": "lm",
        "port": 10240,
        "host": "127.0.0.1",
        "max_concurrency": 1,
        "queue_timeout": 300,
        "queue_size": 100,
        "log_level": "INFO",
    }


@pytest.fixture
def sample_profile_data_alt():
    """Alternative sample profile data for testing."""
    return {
        "name": "Another Profile",
        "description": "Another test profile",
        "model_path": "mlx-community/another-model-4bit",
        "model_type": "lm",
        "port": 10241,
        "host": "127.0.0.1",
        "max_concurrency": 2,
        "queue_timeout": 600,
        "queue_size": 200,
        "log_level": "DEBUG",
    }


@pytest.fixture
def mock_hf_client():
    """Mock HuggingFace client for testing."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.search_mlx_models = AsyncMock(
            return_value=[
                {
                    "model_id": "mlx-community/test-model",
                    "author": "mlx-community",
                    "downloads": 1000,
                    "likes": 50,
                    "estimated_size_gb": 5.0,
                    "tags": ["test", "4bit"],
                    "is_downloaded": False,
                    "last_modified": "2024-01-01T00:00:00Z",
                }
            ]
        )
        mock.list_local_models = MagicMock(
            return_value=[
                {
                    "model_id": "mlx-community/local-model",
                    "local_path": "/path/to/model",
                    "size_bytes": 5000000000,
                    "size_gb": 5.0,
                }
            ]
        )
        mock.download_model = AsyncMock()
        mock.delete_model = AsyncMock(return_value=True)
        yield mock


@pytest.fixture
def mock_server_manager():
    """Mock server manager for testing."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.start_server = AsyncMock(return_value=12345)
        mock.stop_server = AsyncMock(return_value=True)
        mock.check_health = AsyncMock(
            return_value={
                "status": "healthy",
                "response_time_ms": 45.0,
                "model_loaded": True,
            }
        )
        mock.get_server_stats = MagicMock(
            return_value={
                "pid": 12345,
                "memory_mb": 1024.0,
                "cpu_percent": 10.0,
                "status": "running",
                "create_time": 1704067200.0,
            }
        )
        mock.get_all_running = MagicMock(return_value=[])
        mock.processes = {}
        yield mock


@pytest.fixture
def mock_launchd_manager():
    """Mock launchd manager for testing."""
    with patch("mlx_manager.routers.system.launchd_manager") as mock:
        mock.install = MagicMock(return_value="/path/to/plist")
        mock.uninstall = MagicMock(return_value=True)
        mock.is_installed = MagicMock(return_value=False)
        mock.is_running = MagicMock(return_value=False)
        mock.get_label = MagicMock(return_value="com.mlx-manager.test")
        mock.get_status = MagicMock(
            return_value={
                "installed": False,
                "running": False,
                "label": "com.mlx-manager.test",
            }
        )
        yield mock


# ============================================================================
# Authentication fixtures
# ============================================================================


@pytest.fixture
def test_user_data():
    """Test user data for authentication tests."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
    }


@pytest.fixture
def test_admin_user_data():
    """Test admin user data."""
    return {
        "email": "admin@example.com",
        "password": "adminpassword123",
    }


@pytest.fixture
async def test_user(test_session, test_user_data):
    """Create a test user in the database and return it."""
    user = User(
        email=test_user_data["email"],
        hashed_password=hash_password(test_user_data["password"]),
        is_admin=False,
        status=UserStatus.APPROVED,
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    return user


@pytest.fixture
async def test_admin_user(test_session, test_admin_user_data):
    """Create a test admin user in the database and return it."""
    user = User(
        email=test_admin_user_data["email"],
        hashed_password=hash_password(test_admin_user_data["password"]),
        is_admin=True,
        status=UserStatus.APPROVED,
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    return user


@pytest.fixture
def auth_token(test_user_data):
    """Generate a valid JWT token for the test user."""
    return create_access_token(data={"sub": test_user_data["email"]})


@pytest.fixture
def admin_auth_token(test_admin_user_data):
    """Generate a valid JWT token for the test admin user."""
    return create_access_token(data={"sub": test_admin_user_data["email"]})


@pytest.fixture
def auth_headers(auth_token):
    """HTTP headers with Bearer token for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def admin_auth_headers(admin_auth_token):
    """HTTP headers with Bearer token for admin requests."""
    return {"Authorization": f"Bearer {admin_auth_token}"}


@pytest.fixture(scope="function")
async def auth_client(test_engine, test_user_data):
    """Create an async test client with authentication.

    This fixture creates a test user and provides a client with auth headers.
    Use this for tests that require an authenticated user.
    """
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create test user in database
    async with async_session() as session:
        user = User(
            email=test_user_data["email"],
            hashed_password=hash_password(test_user_data["password"]),
            is_admin=False,
            status=UserStatus.APPROVED,
        )
        session.add(user)
        await session.commit()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    # Generate auth token for the test user
    token = create_access_token(data={"sub": test_user_data["email"]})

    # Mock the health checker to prevent it from running during tests
    with patch("mlx_manager.main.health_checker") as mock_health_checker:
        mock_health_checker.start = AsyncMock()
        mock_health_checker.stop = AsyncMock()

        # Mock the server manager
        with patch("mlx_manager.main.server_manager") as mock_server_manager:
            mock_server_manager.cleanup = AsyncMock()

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={"Authorization": f"Bearer {token}"},
            ) as client:
                yield client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def admin_client(test_engine, test_admin_user_data):
    """Create an async test client with admin authentication.

    This fixture creates an admin test user and provides a client with auth headers.
    Use this for tests that require admin privileges.
    """
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create admin test user in database
    async with async_session() as session:
        user = User(
            email=test_admin_user_data["email"],
            hashed_password=hash_password(test_admin_user_data["password"]),
            is_admin=True,
            status=UserStatus.APPROVED,
        )
        session.add(user)
        await session.commit()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    # Generate auth token for the admin user
    token = create_access_token(data={"sub": test_admin_user_data["email"]})

    # Mock the health checker to prevent it from running during tests
    with patch("mlx_manager.main.health_checker") as mock_health_checker:
        mock_health_checker.start = AsyncMock()
        mock_health_checker.stop = AsyncMock()

        # Mock the server manager
        with patch("mlx_manager.main.server_manager") as mock_server_manager:
            mock_server_manager.cleanup = AsyncMock()

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={"Authorization": f"Bearer {token}"},
            ) as client:
                yield client

    app.dependency_overrides.clear()

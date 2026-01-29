"""Tests for BackendRouter with pattern matching and failover."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.models import BackendMapping, BackendType, CloudCredential
from mlx_manager.mlx_server.services.cloud.router import (
    BackendRouter,
    get_router,
    reset_router,
)


class TestPatternMatching:
    """Tests for pattern matching logic."""

    def test_exact_match(self) -> None:
        """Exact model name matches exactly."""
        router = BackendRouter()
        assert router._pattern_matches("gpt-4", "gpt-4") is True

    def test_exact_match_different_model(self) -> None:
        """Different model name doesn't match."""
        router = BackendRouter()
        assert router._pattern_matches("gpt-4", "gpt-3.5") is False

    def test_wildcard_asterisk_suffix(self) -> None:
        """Wildcard pattern with asterisk suffix matches."""
        router = BackendRouter()
        assert router._pattern_matches("gpt-*", "gpt-4") is True
        assert router._pattern_matches("gpt-*", "gpt-4-turbo") is True
        assert router._pattern_matches("gpt-*", "gpt-3.5-turbo") is True

    def test_wildcard_asterisk_no_match(self) -> None:
        """Wildcard pattern doesn't match unrelated models."""
        router = BackendRouter()
        assert router._pattern_matches("gpt-*", "claude-3") is False
        assert router._pattern_matches("gpt-*", "llama-2") is False

    def test_wildcard_middle_pattern(self) -> None:
        """Wildcard pattern with asterisk in middle matches."""
        router = BackendRouter()
        assert router._pattern_matches("claude-*-opus", "claude-3-opus") is True
        assert router._pattern_matches("claude-*-opus", "claude-3.5-opus") is True
        assert router._pattern_matches("claude-*-opus", "claude-3-sonnet") is False

    def test_fnmatch_question_mark(self) -> None:
        """Question mark matches single character."""
        router = BackendRouter()
        assert router._pattern_matches("gpt-?", "gpt-4") is True
        assert router._pattern_matches("gpt-?", "gpt-4-turbo") is False


class TestFindMapping:
    """Tests for _find_mapping method."""

    @pytest.fixture
    def router(self) -> BackendRouter:
        """Create a router instance."""
        return BackendRouter()

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    async def test_returns_none_when_no_mappings(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Returns None when no mappings exist."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        mapping = await router._find_mapping(mock_db, "gpt-4")

        assert mapping is None

    async def test_returns_exact_match(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Returns mapping with exact model match."""
        exact_mapping = MagicMock(spec=BackendMapping)
        exact_mapping.model_pattern = "gpt-4"
        exact_mapping.backend_type = BackendType.OPENAI

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [exact_mapping]
        mock_db.execute.return_value = mock_result

        mapping = await router._find_mapping(mock_db, "gpt-4")

        assert mapping == exact_mapping

    async def test_respects_priority_order(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Higher priority mappings are checked first."""
        # Lower priority wildcard
        low_priority = MagicMock(spec=BackendMapping)
        low_priority.model_pattern = "*"
        low_priority.backend_type = BackendType.LOCAL

        # Higher priority exact match (comes first from DB)
        high_priority = MagicMock(spec=BackendMapping)
        high_priority.model_pattern = "gpt-4"
        high_priority.backend_type = BackendType.OPENAI

        mock_result = MagicMock()
        # DB returns in priority order (desc)
        mock_result.scalars.return_value.all.return_value = [high_priority, low_priority]
        mock_db.execute.return_value = mock_result

        mapping = await router._find_mapping(mock_db, "gpt-4")

        assert mapping == high_priority
        assert mapping.backend_type == BackendType.OPENAI

    async def test_only_returns_enabled_mappings(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Only enabled mappings are considered (filtered by query)."""
        # This is enforced by the SQL query - just verify query is constructed correctly
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        await router._find_mapping(mock_db, "gpt-4")

        # Verify execute was called (query includes enabled filter)
        mock_db.execute.assert_called_once()

    async def test_wildcard_matches_when_no_exact(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Wildcard pattern matches when no exact match exists."""
        wildcard = MagicMock(spec=BackendMapping)
        wildcard.model_pattern = "gpt-*"
        wildcard.backend_type = BackendType.OPENAI

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [wildcard]
        mock_db.execute.return_value = mock_result

        mapping = await router._find_mapping(mock_db, "gpt-4-turbo")

        assert mapping == wildcard


class TestRouteRequest:
    """Tests for route_request method."""

    @pytest.fixture
    def router(self) -> BackendRouter:
        """Create a router instance."""
        return BackendRouter()

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    async def test_no_mapping_defaults_to_local(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """No mapping routes to local inference."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        expected_response = {"id": "local-123", "choices": []}

        with patch.object(router, "_route_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = expected_response

            result = await router.route_request(
                model="some-model",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                db=mock_db,
            )

            assert result == expected_response
            mock_local.assert_called_once()

    async def test_local_backend_routes_to_local(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """LOCAL backend type routes to local inference."""
        local_mapping = MagicMock(spec=BackendMapping)
        local_mapping.model_pattern = "local-model"
        local_mapping.backend_type = BackendType.LOCAL
        local_mapping.fallback_backend = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [local_mapping]
        mock_db.execute.return_value = mock_result

        expected_response = {"id": "local-456", "choices": []}

        with patch.object(router, "_route_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = expected_response

            result = await router.route_request(
                model="local-model",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                db=mock_db,
            )

            assert result == expected_response

    async def test_openai_backend_routes_to_cloud(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """OPENAI backend type routes to OpenAI cloud."""
        openai_mapping = MagicMock(spec=BackendMapping)
        openai_mapping.model_pattern = "gpt-*"
        openai_mapping.backend_type = BackendType.OPENAI
        openai_mapping.backend_model = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [openai_mapping]
        mock_db.execute.return_value = mock_result

        expected_response = {"id": "chatcmpl-123", "choices": []}

        with patch.object(router, "_route_cloud", new_callable=AsyncMock) as mock_cloud:
            mock_cloud.return_value = expected_response

            result = await router.route_request(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                db=mock_db,
            )

            assert result == expected_response
            mock_cloud.assert_called_once()
            # Verify backend_type passed
            call_args = mock_cloud.call_args
            assert call_args[0][1] == BackendType.OPENAI  # backend_type arg

    async def test_anthropic_backend_routes_to_cloud(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """ANTHROPIC backend type routes to Anthropic cloud."""
        anthropic_mapping = MagicMock(spec=BackendMapping)
        anthropic_mapping.model_pattern = "claude-*"
        anthropic_mapping.backend_type = BackendType.ANTHROPIC
        anthropic_mapping.backend_model = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [anthropic_mapping]
        mock_db.execute.return_value = mock_result

        expected_response = {"id": "msg_123", "choices": []}

        with patch.object(router, "_route_cloud", new_callable=AsyncMock) as mock_cloud:
            mock_cloud.return_value = expected_response

            result = await router.route_request(
                model="claude-3-opus",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                db=mock_db,
            )

            assert result == expected_response
            call_args = mock_cloud.call_args
            assert call_args[0][1] == BackendType.ANTHROPIC


class TestFailover:
    """Tests for failover behavior."""

    @pytest.fixture
    def router(self) -> BackendRouter:
        """Create a router instance."""
        return BackendRouter()

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    async def test_local_failure_with_fallback_routes_to_cloud(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Local failure with fallback configured routes to cloud."""
        mapping = MagicMock(spec=BackendMapping)
        mapping.model_pattern = "my-model"
        mapping.backend_type = BackendType.LOCAL
        mapping.fallback_backend = BackendType.OPENAI
        mapping.backend_model = "gpt-4"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mapping]
        mock_db.execute.return_value = mock_result

        cloud_response = {"id": "cloud-fallback", "choices": []}

        with (
            patch.object(router, "_route_local", new_callable=AsyncMock) as mock_local,
            patch.object(router, "_route_cloud", new_callable=AsyncMock) as mock_cloud,
        ):
            mock_local.side_effect = RuntimeError("Local inference failed")
            mock_cloud.return_value = cloud_response

            result = await router.route_request(
                model="my-model",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                db=mock_db,
            )

            assert result == cloud_response
            mock_local.assert_called_once()
            mock_cloud.assert_called_once()

    async def test_local_failure_without_fallback_raises(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Local failure without fallback raises the exception."""
        mapping = MagicMock(spec=BackendMapping)
        mapping.model_pattern = "my-model"
        mapping.backend_type = BackendType.LOCAL
        mapping.fallback_backend = None  # No fallback

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mapping]
        mock_db.execute.return_value = mock_result

        with patch.object(router, "_route_local", new_callable=AsyncMock) as mock_local:
            mock_local.side_effect = RuntimeError("Local inference failed")

            with pytest.raises(RuntimeError, match="Local inference failed"):
                await router.route_request(
                    model="my-model",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=100,
                    db=mock_db,
                )

    async def test_fallback_uses_backend_model_override(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Fallback uses backend_model if specified."""
        mapping = MagicMock(spec=BackendMapping)
        mapping.model_pattern = "my-local-model"
        mapping.backend_type = BackendType.LOCAL
        mapping.fallback_backend = BackendType.OPENAI
        mapping.backend_model = "gpt-4-turbo"  # Different from request model

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mapping]
        mock_db.execute.return_value = mock_result

        with (
            patch.object(router, "_route_local", new_callable=AsyncMock) as mock_local,
            patch.object(router, "_route_cloud", new_callable=AsyncMock) as mock_cloud,
        ):
            mock_local.side_effect = RuntimeError("Local failed")
            mock_cloud.return_value = {"id": "fallback", "choices": []}

            await router.route_request(
                model="my-local-model",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                db=mock_db,
            )

            # Verify cloud was called with backend_model, not original model
            call_args = mock_cloud.call_args
            assert call_args[0][2] == "gpt-4-turbo"  # model arg


class TestGetCloudBackend:
    """Tests for _get_cloud_backend method."""

    @pytest.fixture
    def router(self) -> BackendRouter:
        """Create a router instance."""
        return BackendRouter()

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    async def test_creates_openai_backend(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Creates OpenAI backend from credentials."""
        credential = MagicMock(spec=CloudCredential)
        credential.encrypted_api_key = "sk-test-key"
        credential.base_url = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = credential
        mock_db.execute.return_value = mock_result

        backend = await router._get_cloud_backend(mock_db, BackendType.OPENAI)

        assert backend._api_key == "sk-test-key"
        assert backend.base_url == "https://api.openai.com"

    async def test_creates_anthropic_backend(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Creates Anthropic backend from credentials."""
        credential = MagicMock(spec=CloudCredential)
        credential.encrypted_api_key = "sk-ant-test"
        credential.base_url = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = credential
        mock_db.execute.return_value = mock_result

        backend = await router._get_cloud_backend(mock_db, BackendType.ANTHROPIC)

        assert backend._api_key == "sk-ant-test"
        assert backend.base_url == "https://api.anthropic.com"

    async def test_uses_custom_base_url(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Uses custom base_url from credentials if set."""
        credential = MagicMock(spec=CloudCredential)
        credential.encrypted_api_key = "sk-test"
        credential.base_url = "https://api.azure.openai.com"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = credential
        mock_db.execute.return_value = mock_result

        backend = await router._get_cloud_backend(mock_db, BackendType.OPENAI)

        assert backend.base_url == "https://api.azure.openai.com"

    async def test_caches_backend_for_reuse(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Backend is cached and reused on subsequent calls."""
        credential = MagicMock(spec=CloudCredential)
        credential.encrypted_api_key = "sk-test"
        credential.base_url = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = credential
        mock_db.execute.return_value = mock_result

        # First call creates backend
        backend1 = await router._get_cloud_backend(mock_db, BackendType.OPENAI)

        # Second call returns same instance
        backend2 = await router._get_cloud_backend(mock_db, BackendType.OPENAI)

        assert backend1 is backend2
        # Only one DB query
        assert mock_db.execute.call_count == 1

    async def test_raises_when_no_credentials(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Raises ValueError when no credentials configured."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(ValueError, match="No credentials configured for openai"):
            await router._get_cloud_backend(mock_db, BackendType.OPENAI)

    async def test_raises_for_local_backend_type(
        self, router: BackendRouter, mock_db: AsyncMock
    ) -> None:
        """Raises ValueError for LOCAL backend type (not a cloud backend)."""
        credential = MagicMock(spec=CloudCredential)
        credential.encrypted_api_key = "sk-test"
        credential.base_url = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = credential
        mock_db.execute.return_value = mock_result

        with pytest.raises(ValueError, match="Unknown backend type"):
            await router._get_cloud_backend(mock_db, BackendType.LOCAL)


class TestSingleton:
    """Tests for singleton pattern."""

    async def test_get_router_returns_same_instance(self) -> None:
        """get_router returns the same instance on repeated calls."""
        await reset_router()  # Start clean

        router1 = get_router()
        router2 = get_router()

        assert router1 is router2

    async def test_reset_router_clears_instance(self) -> None:
        """reset_router clears the singleton instance."""
        await reset_router()

        router1 = get_router()
        await reset_router()
        router2 = get_router()

        assert router1 is not router2

    async def test_reset_router_closes_backends(self) -> None:
        """reset_router closes cloud backends before clearing."""
        await reset_router()

        router = get_router()
        # Add a mock backend
        mock_backend = AsyncMock()
        router._cloud_backends[BackendType.OPENAI] = mock_backend

        await reset_router()

        mock_backend.close.assert_called_once()

    async def test_get_router_creates_new_after_reset(self) -> None:
        """get_router creates fresh instance after reset."""
        await reset_router()

        router1 = get_router()
        router1._cloud_backends[BackendType.OPENAI] = AsyncMock()  # Use mock with close()

        await reset_router()
        router2 = get_router()

        # New instance should have empty cache
        assert len(router2._cloud_backends) == 0


class TestClose:
    """Tests for close method."""

    async def test_close_closes_all_backends(self) -> None:
        """close() closes all cached cloud backends."""
        router = BackendRouter()

        mock_openai = AsyncMock()
        mock_anthropic = AsyncMock()
        router._cloud_backends[BackendType.OPENAI] = mock_openai
        router._cloud_backends[BackendType.ANTHROPIC] = mock_anthropic

        await router.close()

        mock_openai.close.assert_called_once()
        mock_anthropic.close.assert_called_once()
        assert len(router._cloud_backends) == 0

    async def test_close_clears_cache(self) -> None:
        """close() clears the backend cache."""
        router = BackendRouter()
        router._cloud_backends[BackendType.OPENAI] = AsyncMock()

        await router.close()

        assert len(router._cloud_backends) == 0

    async def test_close_handles_empty_cache(self) -> None:
        """close() handles empty backend cache gracefully."""
        router = BackendRouter()

        # Should not raise
        await router.close()

        assert len(router._cloud_backends) == 0

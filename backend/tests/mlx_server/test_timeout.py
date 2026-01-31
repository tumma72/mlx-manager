"""Tests for endpoint timeout functionality."""

import asyncio

import pytest

from mlx_manager.mlx_server.errors import TimeoutHTTPException
from mlx_manager.mlx_server.middleware.timeout import get_timeout_for_endpoint, with_timeout


class TestWithTimeoutDecorator:
    """Test the with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_fast_function_succeeds(self) -> None:
        """Function completing before timeout returns normally."""

        @with_timeout(1.0)
        async def fast_function() -> str:
            await asyncio.sleep(0.01)
            return "success"

        result = await fast_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_slow_function_raises_timeout(self) -> None:
        """Function exceeding timeout raises TimeoutHTTPException."""

        @with_timeout(0.05)
        async def slow_function() -> str:
            await asyncio.sleep(1.0)
            return "never returned"

        with pytest.raises(TimeoutHTTPException) as exc_info:
            await slow_function()

        assert exc_info.value.status_code == 408
        assert exc_info.value.timeout_seconds == 0.05
        assert "timed out" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Decorated function preserves name and docstring."""

        @with_timeout(1.0)
        async def documented_function() -> str:
            """This is a docstring."""
            return "ok"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a docstring."

    @pytest.mark.asyncio
    async def test_exception_propagates(self) -> None:
        """Non-timeout exceptions propagate normally."""

        @with_timeout(1.0)
        async def raising_function() -> str:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await raising_function()

    @pytest.mark.asyncio
    async def test_with_arguments(self) -> None:
        """Decorated function handles arguments correctly."""

        @with_timeout(1.0)
        async def add_function(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        result = await add_function(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_with_kwargs(self) -> None:
        """Decorated function handles keyword arguments correctly."""

        @with_timeout(1.0)
        async def greet(name: str, greeting: str = "Hello") -> str:
            await asyncio.sleep(0.01)
            return f"{greeting}, {name}!"

        result = await greet("World", greeting="Hi")
        assert result == "Hi, World!"


class TestTimeoutSettings:
    """Test timeout configuration."""

    def test_default_timeouts(self) -> None:
        """Default timeouts match CONTEXT.md decisions."""
        from mlx_manager.mlx_server.config import get_settings

        # Clear cache to get fresh settings
        get_settings.cache_clear()
        settings = get_settings()

        assert settings.timeout_chat_seconds == 900.0  # 15 minutes
        assert settings.timeout_completions_seconds == 600.0  # 10 minutes
        assert settings.timeout_embeddings_seconds == 120.0  # 2 minutes

    def test_get_timeout_for_chat_endpoint(self) -> None:
        """get_timeout_for_endpoint returns chat timeout for chat endpoints."""
        assert get_timeout_for_endpoint("/v1/chat/completions") == 900.0

    def test_get_timeout_for_completions_endpoint(self) -> None:
        """get_timeout_for_endpoint returns completions timeout for completions endpoint."""
        assert get_timeout_for_endpoint("/v1/completions") == 600.0

    def test_get_timeout_for_embeddings_endpoint(self) -> None:
        """get_timeout_for_endpoint returns embeddings timeout for embeddings endpoint."""
        assert get_timeout_for_endpoint("/v1/embeddings") == 120.0

    def test_get_timeout_for_unknown_endpoint(self) -> None:
        """get_timeout_for_endpoint returns chat timeout for unknown endpoints."""
        assert get_timeout_for_endpoint("/v1/unknown") == 900.0


class TestTimeoutHTTPException:
    """Test TimeoutHTTPException properties."""

    def test_status_code(self) -> None:
        """TimeoutHTTPException has status code 408."""
        exc = TimeoutHTTPException(timeout_seconds=60.0)
        assert exc.status_code == 408

    def test_timeout_seconds_stored(self) -> None:
        """TimeoutHTTPException stores timeout value."""
        exc = TimeoutHTTPException(timeout_seconds=123.5)
        assert exc.timeout_seconds == 123.5

    def test_custom_detail(self) -> None:
        """TimeoutHTTPException accepts custom detail."""
        exc = TimeoutHTTPException(
            timeout_seconds=60.0,
            detail="Custom timeout message",
        )
        assert exc.detail == "Custom timeout message"

    def test_default_detail(self) -> None:
        """TimeoutHTTPException generates default detail if not provided."""
        exc = TimeoutHTTPException(timeout_seconds=60.0)
        assert "60" in str(exc.detail)
        assert "timed out" in str(exc.detail).lower()

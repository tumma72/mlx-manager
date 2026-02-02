"""Backend router with failover logic."""

import fnmatch
import logging
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.database import get_db
from mlx_manager.mlx_server.services.cloud.anthropic import AnthropicCloudBackend
from mlx_manager.mlx_server.services.cloud.openai import OpenAICloudBackend
from mlx_manager.models import (
    API_TYPE_FOR_BACKEND,
    DEFAULT_BASE_URLS,
    ApiType,
    BackendMapping,
    BackendType,
    CloudCredential,
)

logger = logging.getLogger(__name__)


class BackendRouter:
    """Routes requests to appropriate backend with failover support.

    Looks up backend mapping for model, routes to local or cloud,
    and handles automatic failover on local failure.
    """

    def __init__(self) -> None:
        """Initialize router."""
        # Cache by credential ID for multiple providers of same type
        self._cloud_backends: dict[int, OpenAICloudBackend | AnthropicCloudBackend] = {}

    async def route_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        stream: bool = False,
        db: AsyncSession | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None] | dict:
        """Route request to appropriate backend.

        Args:
            model: Model ID to route
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, return async generator
            db: Database session (optional, will create if needed)
            **kwargs: Additional parameters

        Returns:
            Response from selected backend
        """
        # Get or create database session
        if db is None:
            async for session in get_db():
                return await self._route_with_session(
                    session, model, messages, max_tokens, temperature, stream, **kwargs
                )
            raise RuntimeError("Failed to get database session")
        else:
            return await self._route_with_session(
                db, model, messages, max_tokens, temperature, stream, **kwargs
            )

    async def _route_with_session(
        self,
        db: AsyncSession,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None] | dict:
        """Route with database session."""
        # Find matching backend mapping
        mapping = await self._find_mapping(db, model)

        if mapping is None:
            # No mapping - default to local
            logger.info(f"No mapping for {model}, using local")
            return await self._route_local(
                model, messages, max_tokens, temperature, stream, **kwargs
            )

        logger.info(f"Routing {model} to {mapping.backend_type.value}")

        # Route based on backend type
        if mapping.backend_type == BackendType.LOCAL:
            try:
                return await self._route_local(
                    model, messages, max_tokens, temperature, stream, **kwargs
                )
            except Exception as e:
                if mapping.fallback_backend:
                    logger.warning(f"Local failed, falling back to {mapping.fallback_backend}: {e}")
                    return await self._route_cloud(
                        db,
                        mapping.fallback_backend,
                        mapping.backend_model or model,
                        messages,
                        max_tokens,
                        temperature,
                        stream,
                        **kwargs,
                    )
                raise
        else:
            # Cloud backend
            return await self._route_cloud(
                db,
                mapping.backend_type,
                mapping.backend_model or model,
                messages,
                max_tokens,
                temperature,
                stream,
                **kwargs,
            )

    async def _find_mapping(
        self,
        db: AsyncSession,
        model: str,
    ) -> BackendMapping | None:
        """Find backend mapping for model.

        Matches in priority order:
        1. Exact model name match
        2. Wildcard pattern match (e.g., "gpt-*")
        """
        # Get all enabled mappings ordered by priority (higher first)
        result = await db.execute(
            select(BackendMapping)
            .where(BackendMapping.enabled == True)  # type: ignore[arg-type]  # noqa: E712
            .order_by(BackendMapping.priority.desc())  # type: ignore[arg-type, attr-defined]
        )
        mappings = result.scalars().all()

        # Find first matching pattern
        for mapping in mappings:
            if self._pattern_matches(mapping.model_pattern, model):
                return mapping  # type: ignore[return-value]

        return None

    def _pattern_matches(self, pattern: str, model: str) -> bool:
        """Check if pattern matches model name.

        Supports:
        - Exact match: "gpt-4" matches "gpt-4"
        - Wildcard: "gpt-*" matches "gpt-4", "gpt-4-turbo"
        - Fnmatch patterns: "claude-3-*" matches "claude-3-opus"
        """
        # Exact match first
        if pattern == model:
            return True
        # Wildcard pattern
        return fnmatch.fnmatch(model, pattern)

    async def _route_local(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None] | dict:
        """Route to local MLX inference."""
        from mlx_manager.mlx_server.services.inference import generate_chat_completion

        return await generate_chat_completion(
            model_id=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )

    async def _route_cloud(
        self,
        db: AsyncSession,
        backend_type: BackendType,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None] | dict:
        """Route to cloud backend."""
        # Get or create cloud backend client
        backend = await self._get_cloud_backend(db, backend_type)

        return await backend.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )

    async def _get_cloud_backend(
        self,
        db: AsyncSession,
        backend_type: BackendType,
    ) -> OpenAICloudBackend | AnthropicCloudBackend:
        """Get or create cloud backend client based on API type.

        Uses credential's api_type to determine which client to create.
        Falls back to API_TYPE_FOR_BACKEND mapping for backwards compatibility.
        """
        # Load credentials from database
        result = await db.execute(
            select(CloudCredential).where(
                CloudCredential.backend_type == backend_type  # type: ignore[arg-type]
            )
        )
        credential = result.scalar_one_or_none()

        if credential is None:
            raise ValueError(f"No credentials configured for {backend_type.value}")

        # Check cache by credential ID
        credential_id = credential.id
        if credential_id is not None and credential_id in self._cloud_backends:
            return self._cloud_backends[credential_id]

        # Determine API type from credential or fall back to mapping
        api_type = credential.api_type
        if api_type is None:
            # Backwards compatibility: use mapping for older credentials
            api_type = API_TYPE_FOR_BACKEND.get(backend_type, ApiType.OPENAI)

        # Determine base URL from credential or fall back to default
        base_url = credential.base_url
        if not base_url:
            base_url = DEFAULT_BASE_URLS.get(backend_type, "https://api.openai.com")

        # Get API key (already decrypted by encryption_service)
        api_key = credential.encrypted_api_key

        # Create appropriate client based on API type
        if api_type == ApiType.ANTHROPIC:
            backend: OpenAICloudBackend | AnthropicCloudBackend = AnthropicCloudBackend(
                api_key=api_key,
                base_url=base_url,
            )
        else:  # OPENAI or default
            backend = OpenAICloudBackend(
                api_key=api_key,
                base_url=base_url,
            )

        # Cache by credential ID
        if credential_id is not None:
            self._cloud_backends[credential_id] = backend
        return backend

    async def refresh_rules(self) -> None:
        """Reload routing rules from database. Call after rule/credential updates.

        Clears cached cloud backends so they will be recreated with fresh credentials.
        """
        # Clear cached cloud backends (credentials may have changed)
        for backend in self._cloud_backends.values():
            await backend.close()
        self._cloud_backends.clear()

        logger.info("Routing rules refreshed: cloud backends cleared for reload")

    async def close(self) -> None:
        """Close all cloud backend clients."""
        for backend in self._cloud_backends.values():
            await backend.close()
        self._cloud_backends.clear()


# Module-level singleton
_router: BackendRouter | None = None


def get_router() -> BackendRouter:
    """Get the backend router singleton."""
    global _router
    if _router is None:
        _router = BackendRouter()
    return _router


async def reset_router() -> None:
    """Reset the router (for testing)."""
    global _router
    if _router:
        await _router.close()
    _router = None

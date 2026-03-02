"""Backend router with failover logic."""

import re

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.database import get_db
from mlx_manager.mlx_server.models.ir import InternalRequest, RoutingOutcome
from mlx_manager.mlx_server.services.cloud.anthropic import AnthropicCloudBackend
from mlx_manager.mlx_server.services.cloud.openai import OpenAICloudBackend
from mlx_manager.models.enums import ApiType, BackendType, PatternType
from mlx_manager.shared import (
    API_TYPE_FOR_BACKEND,
    DEFAULT_BASE_URLS,
    BackendMapping,
    CloudCredential,
)


class BackendRouter:
    """Routes requests to appropriate backend with failover support.

    Accepts protocol-neutral InternalRequest, looks up backend mapping,
    routes to local inference or cloud, and returns RoutingOutcome.
    """

    def __init__(self) -> None:
        """Initialize router."""
        # Cache by credential ID for multiple providers of same type
        self._cloud_backends: dict[int, OpenAICloudBackend | AnthropicCloudBackend] = {}

    async def route_request(
        self,
        ir: InternalRequest,
        db: AsyncSession | None = None,
    ) -> RoutingOutcome:
        """Route request to appropriate backend.

        Args:
            ir: Protocol-neutral internal request
            db: Database session (optional, will create if needed)

        Returns:
            RoutingOutcome indicating how to deliver the response
        """
        if db is None:
            async for session in get_db():
                return await self._route_with_session(session, ir)
            raise RuntimeError("Failed to get database session")
        else:
            return await self._route_with_session(db, ir)

    async def _route_with_session(
        self,
        db: AsyncSession,
        ir: InternalRequest,
    ) -> RoutingOutcome:
        """Route with database session."""
        mapping = await self._find_mapping(db, ir.model)

        if mapping is None:
            logger.info(f"No mapping for {ir.model}, using local")
            return await self._route_local(ir)

        logger.info(f"Routing {ir.model} to {mapping.backend_type.value}")

        # Apply model override: profile takes precedence over backend_model
        if mapping.profile_id:
            repo_id = await self._resolve_profile_model(db, mapping.profile_id)
            if repo_id:
                ir = ir.model_copy(update={"model": repo_id})
            else:
                logger.warning(f"Profile {mapping.profile_id} not found, skipping override")
        elif mapping.backend_model:
            ir = ir.model_copy(update={"model": mapping.backend_model})

        if mapping.backend_type == BackendType.LOCAL:
            try:
                return await self._route_local(ir)
            except Exception as e:
                if mapping.fallback_backend:
                    logger.warning(f"Local failed, falling back to {mapping.fallback_backend}: {e}")
                    return await self._route_cloud(db, mapping.fallback_backend, ir)
                raise
        else:
            return await self._route_cloud(db, mapping.backend_type, ir)

    async def _find_mapping(
        self,
        db: AsyncSession,
        model: str,
    ) -> BackendMapping | None:
        """Find backend mapping for model.

        Matches in priority order using the mapping's pattern_type:
        - EXACT: exact string match
        - PREFIX: model starts with pattern
        - REGEX: full regex match
        """
        result = await db.execute(
            select(BackendMapping)
            .where(BackendMapping.enabled == True)  # type: ignore[arg-type]  # noqa: E712
            .order_by(BackendMapping.priority.desc())  # type: ignore[arg-type, attr-defined]
        )
        mappings = result.scalars().all()

        for mapping in mappings:
            if self._pattern_matches(mapping, model):
                return mapping  # type: ignore[return-value]

        return None

    @staticmethod
    def _pattern_matches(mapping: BackendMapping, model: str) -> bool:
        """Check if mapping pattern matches model name using pattern_type."""
        pattern = mapping.model_pattern
        pattern_type = mapping.pattern_type

        if pattern_type == PatternType.EXACT:
            return pattern == model
        elif pattern_type == PatternType.PREFIX:
            return model.startswith(pattern)
        elif pattern_type == PatternType.REGEX:
            return re.fullmatch(pattern, model) is not None
        else:
            # Fallback for unknown pattern types
            return pattern == model

    async def _resolve_profile_model(self, db: AsyncSession, profile_id: int) -> str | None:
        """Resolve a profile ID to its model's repo_id.

        Uses lazy imports to avoid circular dependencies.
        """
        from mlx_manager.models.profiles import ExecutionProfile

        result = await db.execute(
            select(ExecutionProfile).where(ExecutionProfile.id == profile_id)  # type: ignore[arg-type]
        )
        profile = result.scalar_one_or_none()
        if profile is None or profile.model_id is None:
            return None

        from mlx_manager.models.entities import Model

        model_result = await db.execute(select(Model).where(Model.id == profile.model_id))  # type: ignore[arg-type]
        model = model_result.scalar_one_or_none()
        return model.repo_id if model is not None else None

    async def _route_local(self, ir: InternalRequest) -> RoutingOutcome:
        """Route to local MLX inference."""
        from mlx_manager.mlx_server.services.inference import (
            generate_chat_complete_response,
            generate_chat_stream,
        )

        if ir.stream:
            stream = await generate_chat_stream(
                model_id=ir.model,
                messages=ir.messages,
                max_tokens=ir.params.max_tokens or 4096,
                temperature=ir.params.temperature or 1.0,
                top_p=ir.params.top_p or 1.0,
                stop=ir.stop,
                tools=ir.tools,
                images=ir.images,
            )
            return RoutingOutcome(ir_stream=stream)
        else:
            result = await generate_chat_complete_response(
                model_id=ir.model,
                messages=ir.messages,
                max_tokens=ir.params.max_tokens or 4096,
                temperature=ir.params.temperature or 1.0,
                top_p=ir.params.top_p or 1.0,
                stop=ir.stop,
                tools=ir.tools,
                images=ir.images,
            )
            return RoutingOutcome(ir_result=result)

    async def _route_cloud(
        self,
        db: AsyncSession,
        backend_type: BackendType,
        ir: InternalRequest,
    ) -> RoutingOutcome:
        """Route to cloud backend."""
        backend = await self._get_cloud_backend(db, backend_type)

        return await backend.forward_request(ir)

    async def _get_cloud_backend(
        self,
        db: AsyncSession,
        backend_type: BackendType,
    ) -> OpenAICloudBackend | AnthropicCloudBackend:
        """Get or create cloud backend client based on API type.

        Uses credential's api_type to determine which client to create.
        Falls back to API_TYPE_FOR_BACKEND mapping for backwards compatibility.
        """
        result = await db.execute(
            select(CloudCredential).where(
                CloudCredential.backend_type == backend_type  # type: ignore[arg-type]
            )
        )
        credential = result.scalar_one_or_none()

        if credential is None:
            raise ValueError(f"No credentials configured for {backend_type.value}")

        credential_id = credential.id
        if credential_id is not None and credential_id in self._cloud_backends:
            return self._cloud_backends[credential_id]

        api_type = credential.api_type
        if api_type is None:
            api_type = API_TYPE_FOR_BACKEND.get(backend_type, ApiType.OPENAI)

        base_url = credential.base_url
        if not base_url:
            base_url = DEFAULT_BASE_URLS.get(backend_type, "https://api.openai.com")

        api_key = credential.encrypted_api_key

        if api_type == ApiType.ANTHROPIC:
            backend: OpenAICloudBackend | AnthropicCloudBackend = AnthropicCloudBackend(
                api_key=api_key,
                base_url=base_url,
            )
        else:
            backend = OpenAICloudBackend(
                api_key=api_key,
                base_url=base_url,
            )

        if credential_id is not None:
            self._cloud_backends[credential_id] = backend
        return backend

    async def refresh_rules(self) -> None:
        """Reload routing rules from database. Call after rule/credential updates.

        Clears cached cloud backends so they will be recreated with fresh credentials.
        """
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

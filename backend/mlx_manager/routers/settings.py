"""Settings API router for providers, routing rules, and model pool configuration."""

import json
import re
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user
from mlx_manager.models import (
    BackendMapping,
    BackendMappingCreate,
    BackendMappingResponse,
    BackendMappingUpdate,
    BackendType,
    CloudCredential,
    CloudCredentialCreate,
    CloudCredentialResponse,
    RuleMatchResult,
    RulePriorityUpdate,
    ServerConfig,
    ServerConfigResponse,
    ServerConfigUpdate,
    User,
)
from mlx_manager.services.encryption_service import decrypt_api_key, encrypt_api_key

router = APIRouter(prefix="/api/settings", tags=["settings"])


# ============================================================================
# Provider Endpoints
# ============================================================================


@router.get("/providers", response_model=list[CloudCredentialResponse])
async def list_providers(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """List all configured cloud providers (without API keys)."""
    result = await session.execute(select(CloudCredential))
    return list(result.scalars().all())


@router.post("/providers", response_model=CloudCredentialResponse)
async def create_or_update_provider(
    current_user: Annotated[User, Depends(get_current_user)],
    data: CloudCredentialCreate,
    session: AsyncSession = Depends(get_db),
):
    """Create or update cloud provider credentials (upsert by backend_type)."""
    # Check for existing credential for this backend type
    result = await session.execute(
        select(CloudCredential).where(CloudCredential.backend_type == data.backend_type)
    )
    credential = result.scalar_one_or_none()

    # Encrypt the API key before storage
    encrypted_key = encrypt_api_key(data.api_key)

    if credential:
        # Update existing
        credential.encrypted_api_key = encrypted_key
        credential.base_url = data.base_url
        credential.updated_at = datetime.now(tz=UTC)
    else:
        # Create new
        credential = CloudCredential(
            backend_type=data.backend_type,
            encrypted_api_key=encrypted_key,
            base_url=data.base_url,
        )
        session.add(credential)

    await session.commit()
    await session.refresh(credential)
    return credential


@router.delete("/providers/{backend_type}", status_code=204)
async def delete_provider(
    current_user: Annotated[User, Depends(get_current_user)],
    backend_type: BackendType,
    session: AsyncSession = Depends(get_db),
):
    """Delete cloud provider credentials."""
    result = await session.execute(
        select(CloudCredential).where(CloudCredential.backend_type == backend_type)
    )
    credential = result.scalar_one_or_none()

    if not credential:
        raise HTTPException(status_code=404, detail="Provider not found")

    await session.delete(credential)
    await session.commit()


@router.post("/providers/{backend_type}/test")
async def test_provider_connection(
    current_user: Annotated[User, Depends(get_current_user)],
    backend_type: BackendType,
    session: AsyncSession = Depends(get_db),
):
    """Test connection to cloud provider API.

    For OpenAI: GET /v1/models with Bearer token
    For Anthropic: GET /v1/models with x-api-key header
    """
    result = await session.execute(
        select(CloudCredential).where(CloudCredential.backend_type == backend_type)
    )
    credential = result.scalar_one_or_none()

    if not credential:
        raise HTTPException(status_code=404, detail="Provider not configured")

    # Decrypt the API key for testing
    api_key = decrypt_api_key(credential.encrypted_api_key)

    # Determine base URL and headers based on backend type
    if backend_type == BackendType.OPENAI:
        base_url = credential.base_url or "https://api.openai.com"
        headers = {"Authorization": f"Bearer {api_key}"}
    elif backend_type == BackendType.ANTHROPIC:
        base_url = credential.base_url or "https://api.anthropic.com"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported backend type for testing")

    # Test the connection
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/v1/models", headers=headers)

            if response.status_code == 200:
                return {"success": True}
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid API key - authentication failed",
                )
            elif response.status_code == 403:
                raise HTTPException(
                    status_code=400,
                    detail="API key does not have permission to access models",
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider returned status {response.status_code}",
                )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=400,
            detail=f"Could not connect to {base_url}",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=400,
            detail=f"Connection to {base_url} timed out",
        )


# ============================================================================
# Routing Rules Endpoints
# ============================================================================


@router.get("/rules", response_model=list[BackendMappingResponse])
async def list_rules(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """List all routing rules ordered by priority (highest first)."""
    result = await session.execute(
        select(BackendMapping).order_by(BackendMapping.priority.desc())  # type: ignore[attr-defined]
    )
    return list(result.scalars().all())


@router.post("/rules", response_model=BackendMappingResponse, status_code=201)
async def create_rule(
    current_user: Annotated[User, Depends(get_current_user)],
    data: BackendMappingCreate,
    session: AsyncSession = Depends(get_db),
):
    """Create a new routing rule."""
    # Validate pattern type
    if data.pattern_type not in ("exact", "prefix", "regex"):
        raise HTTPException(
            status_code=400,
            detail="pattern_type must be 'exact', 'prefix', or 'regex'",
        )

    # Validate regex pattern if provided
    if data.pattern_type == "regex":
        try:
            re.compile(data.model_pattern)
        except re.error as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regex pattern: {e}",
            )

    rule = BackendMapping(
        model_pattern=data.model_pattern,
        pattern_type=data.pattern_type,
        backend_type=data.backend_type,
        backend_model=data.backend_model,
        fallback_backend=data.fallback_backend,
        priority=data.priority,
    )
    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    return rule


# NOTE: Static routes (/rules/priorities, /rules/test) must come BEFORE
# dynamic routes (/rules/{rule_id}) to avoid path parameter matching


@router.put("/rules/priorities")
async def update_rule_priorities(
    current_user: Annotated[User, Depends(get_current_user)],
    updates: list[RulePriorityUpdate],
    session: AsyncSession = Depends(get_db),
):
    """Batch update rule priorities (for drag-drop reordering)."""
    for update in updates:
        result = await session.execute(
            select(BackendMapping).where(BackendMapping.id == update.id)
        )
        rule = result.scalar_one_or_none()

        if rule:
            rule.priority = update.priority
            rule.updated_at = datetime.now(tz=UTC)
            session.add(rule)

    await session.commit()
    return {"success": True, "updated": len(updates)}


@router.post("/rules/test", response_model=RuleMatchResult)
async def test_rule_match(
    current_user: Annotated[User, Depends(get_current_user)],
    model_name: str,
    session: AsyncSession = Depends(get_db),
):
    """Test which rule matches a given model name."""
    # Get all enabled rules ordered by priority
    result = await session.execute(
        select(BackendMapping)
        .where(BackendMapping.enabled == True)  # noqa: E712
        .order_by(BackendMapping.priority.desc())  # type: ignore[attr-defined]
    )
    rules = result.scalars().all()

    for rule in rules:
        if _matches_pattern(model_name, rule.model_pattern, rule.pattern_type):
            return RuleMatchResult(
                matched_rule_id=rule.id,
                backend_type=rule.backend_type,
                pattern_matched=rule.model_pattern,
            )

    # No match - defaults to local
    return RuleMatchResult(
        matched_rule_id=None,
        backend_type=BackendType.LOCAL,
        pattern_matched=None,
    )


@router.put("/rules/{rule_id}", response_model=BackendMappingResponse)
async def update_rule(
    current_user: Annotated[User, Depends(get_current_user)],
    rule_id: int,
    data: BackendMappingUpdate,
    session: AsyncSession = Depends(get_db),
):
    """Update a routing rule."""
    result = await session.execute(select(BackendMapping).where(BackendMapping.id == rule_id))
    rule = result.scalar_one_or_none()

    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    # Validate pattern type if provided
    if data.pattern_type is not None and data.pattern_type not in ("exact", "prefix", "regex"):
        raise HTTPException(
            status_code=400,
            detail="pattern_type must be 'exact', 'prefix', or 'regex'",
        )

    # Validate regex pattern if updating to regex
    pattern_type = data.pattern_type if data.pattern_type is not None else rule.pattern_type
    pattern = data.model_pattern if data.model_pattern is not None else rule.model_pattern
    if pattern_type == "regex":
        try:
            re.compile(pattern)
        except re.error as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regex pattern: {e}",
            )

    # Update fields
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(rule, key, value)

    rule.updated_at = datetime.now(tz=UTC)
    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(
    current_user: Annotated[User, Depends(get_current_user)],
    rule_id: int,
    session: AsyncSession = Depends(get_db),
):
    """Delete a routing rule."""
    result = await session.execute(select(BackendMapping).where(BackendMapping.id == rule_id))
    rule = result.scalar_one_or_none()

    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    await session.delete(rule)
    await session.commit()


def _matches_pattern(model_name: str, pattern: str, pattern_type: str) -> bool:
    """Check if a model name matches a pattern based on pattern type."""
    if pattern_type == "exact":
        return model_name == pattern
    elif pattern_type == "prefix":
        return model_name.startswith(pattern)
    elif pattern_type == "regex":
        try:
            return bool(re.match(pattern, model_name))
        except re.error:
            return False
    return False


# ============================================================================
# Model Pool Configuration Endpoints
# ============================================================================


@router.get("/pool", response_model=ServerConfigResponse)
async def get_pool_config(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """Get current model pool configuration.

    Creates default config if it doesn't exist (singleton pattern).
    """
    result = await session.execute(select(ServerConfig).where(ServerConfig.id == 1))
    config = result.scalar_one_or_none()

    if not config:
        # Create default config
        config = ServerConfig(id=1)
        session.add(config)
        await session.commit()
        await session.refresh(config)

    # Parse preload_models from JSON
    try:
        preload_models = json.loads(config.preload_models)
    except json.JSONDecodeError:
        preload_models = []

    return ServerConfigResponse(
        memory_limit_mode=config.memory_limit_mode,
        memory_limit_value=config.memory_limit_value,
        eviction_policy=config.eviction_policy,
        preload_models=preload_models,
    )


@router.put("/pool", response_model=ServerConfigResponse)
async def update_pool_config(
    current_user: Annotated[User, Depends(get_current_user)],
    data: ServerConfigUpdate,
    session: AsyncSession = Depends(get_db),
):
    """Update model pool configuration."""
    result = await session.execute(select(ServerConfig).where(ServerConfig.id == 1))
    config = result.scalar_one_or_none()

    if not config:
        # Create default config first
        config = ServerConfig(id=1)
        session.add(config)

    # Validate memory_limit_mode
    if data.memory_limit_mode is not None and data.memory_limit_mode not in ("percent", "gb"):
        raise HTTPException(
            status_code=400,
            detail="memory_limit_mode must be 'percent' or 'gb'",
        )

    # Validate eviction_policy
    if data.eviction_policy is not None and data.eviction_policy not in ("lru", "lfu", "ttl"):
        raise HTTPException(
            status_code=400,
            detail="eviction_policy must be 'lru', 'lfu', or 'ttl'",
        )

    # Update fields
    if data.memory_limit_mode is not None:
        config.memory_limit_mode = data.memory_limit_mode
    if data.memory_limit_value is not None:
        config.memory_limit_value = data.memory_limit_value
    if data.eviction_policy is not None:
        config.eviction_policy = data.eviction_policy
    if data.preload_models is not None:
        config.preload_models = json.dumps(data.preload_models)

    config.updated_at = datetime.now(tz=UTC)
    session.add(config)
    await session.commit()
    await session.refresh(config)

    # Parse preload_models for response
    try:
        preload_models = json.loads(config.preload_models)
    except json.JSONDecodeError:
        preload_models = []

    return ServerConfigResponse(
        memory_limit_mode=config.memory_limit_mode,
        memory_limit_value=config.memory_limit_value,
        eviction_policy=config.eviction_policy,
        preload_models=preload_models,
    )

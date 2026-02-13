"""Model registry service for managing the unified Model entity."""

import logging
from datetime import UTC, datetime

from sqlmodel import select

logger = logging.getLogger(__name__)


async def sync_models_from_cache() -> int:
    """Scan HuggingFace cache and create Model records for downloaded models.

    Called at startup to populate the models table from the local HF cache.

    Returns:
        Number of new models registered.
    """
    from mlx_manager.database import get_session
    from mlx_manager.models import Model
    from mlx_manager.services.hf_client import hf_client

    local_models = hf_client.list_local_models()
    new_count = 0

    async with get_session() as session:
        for lm in local_models:
            # Check if already registered
            result = await session.execute(select(Model).where(Model.repo_id == lm["model_id"]))
            if result.scalar_one_or_none():
                continue

            # Detect model type
            from mlx_manager.mlx_server.models.detection import detect_model_type

            model_type = detect_model_type(lm["model_id"])

            model = Model(
                repo_id=lm["model_id"],
                model_type=model_type.value,
                local_path=lm["local_path"],
                size_bytes=lm["size_bytes"],
                downloaded_at=datetime.now(tz=UTC),
            )

            session.add(model)
            new_count += 1
            logger.info(f"Registered model: {lm['model_id']} (type={model_type.value})")

        if new_count > 0:
            await session.commit()
            logger.info(f"Synced {new_count} new models from HuggingFace cache")

    return new_count


async def register_model_from_download(
    repo_id: str, local_path: str, size_bytes: int | None = None
) -> None:
    """Register a model after download completion.

    Creates a Model record if one doesn't already exist, or updates
    the existing record with download info.
    """
    from mlx_manager.database import get_session
    from mlx_manager.mlx_server.models.detection import detect_model_type
    from mlx_manager.models import Model

    async with get_session() as session:
        result = await session.execute(select(Model).where(Model.repo_id == repo_id))
        model = result.scalar_one_or_none()

        if model:
            model.local_path = local_path
            if size_bytes is not None:
                model.size_bytes = size_bytes
            model.downloaded_at = datetime.now(tz=UTC)
        else:
            model_type = detect_model_type(repo_id)
            model = Model(
                repo_id=repo_id,
                model_type=model_type.value,
                local_path=local_path,
                size_bytes=size_bytes,
                downloaded_at=datetime.now(tz=UTC),
            )

        session.add(model)
        await session.commit()
        logger.info(f"Registered downloaded model: {repo_id}")


async def update_model_capabilities(repo_id: str, **caps: object) -> None:
    """Update capability fields for a model.

    Creates a ModelCapabilities record (upsert pattern: delete + insert).

    Args:
        repo_id: HuggingFace model ID
        **caps: Capability fields (e.g., supports_native_tools=True, model_type="text-gen")
    """
    from sqlalchemy import delete
    from sqlmodel import select

    from mlx_manager.database import get_session
    from mlx_manager.models import Model
    from mlx_manager.models.capabilities import ModelCapabilities

    async with get_session() as session:
        result = await session.execute(select(Model).where(Model.repo_id == repo_id))
        model = result.scalar_one_or_none()

        if not model:
            model = Model(repo_id=repo_id)
            session.add(model)
            await session.flush()  # Get the ID

        # Update model_type on Model if provided
        model_type = caps.pop("model_type", None) or model.model_type or "text-gen"
        if isinstance(model_type, str):
            model.model_type = model_type

        # Delete existing capabilities (upsert)
        await session.execute(
            delete(ModelCapabilities).where(
                ModelCapabilities.model_id == model.id  # type: ignore[arg-type]
            )
        )
        await session.flush()

        # Build kwargs for ModelCapabilities
        cap_kwargs: dict[str, object] = {
            "model_id": model.id,
            "capability_type": str(model_type),
            "probed_at": datetime.now(tz=UTC),
            "probe_version": caps.pop("probe_version", 2),
            "model_family": caps.pop("model_family", None),
        }

        # Add remaining capability fields that exist on the model
        for key, value in caps.items():
            if hasattr(ModelCapabilities, key):
                cap_kwargs[key] = value

        capability = ModelCapabilities(**cap_kwargs)
        session.add(capability)
        await session.commit()
        logger.info(f"Updated capabilities for {repo_id} (type={model_type})")

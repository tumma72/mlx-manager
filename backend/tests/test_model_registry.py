"""Tests for model registry service."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import select

from mlx_manager.models import Model
from mlx_manager.services.model_registry import (
    register_model_from_download,
    sync_models_from_cache,
    update_model_capabilities,
)


def _create_session_mock(test_engine):
    """Helper to create a mock get_session context manager."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    @asynccontextmanager
    async def mock_get_session():
        async with async_session() as session:
            yield session

    return mock_get_session


@pytest.mark.asyncio
async def test_sync_models_from_cache_empty(test_engine):
    """Test sync_models_from_cache when no local models exist."""
    with (
        patch("mlx_manager.database.get_session", _create_session_mock(test_engine)),
        patch("mlx_manager.services.hf_client.hf_client") as mock_hf,
    ):
        mock_hf.list_local_models.return_value = []

        count = await sync_models_from_cache()
        assert count == 0


@pytest.mark.asyncio
async def test_sync_models_from_cache_new_models(test_engine):
    """Test sync_models_from_cache with new models to register."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    # Mock local models
    mock_local_model = MagicMock()
    mock_local_model.model_id = "mlx-community/test-sync-model"
    mock_local_model.local_path = "/path/to/model"
    mock_local_model.size_bytes = 1000000

    with (
        patch("mlx_manager.database.get_session", _create_session_mock(test_engine)),
        patch("mlx_manager.services.hf_client.hf_client") as mock_hf,
        patch("mlx_manager.mlx_server.models.detection.detect_model_type") as mock_detect,
    ):
        mock_hf.list_local_models.return_value = [mock_local_model]
        from mlx_manager.mlx_server.models.types import ModelType

        mock_detect.return_value = ModelType.TEXT_GEN

        count = await sync_models_from_cache()
        assert count == 1

        # Verify model was created in DB
        async_session = sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/test-sync-model")
            )
            model = result.scalar_one_or_none()
            assert model is not None
            assert model.model_type == "text-gen"
            assert model.local_path == "/path/to/model"
            assert model.size_bytes == 1000000
            assert model.downloaded_at is not None


@pytest.mark.asyncio
async def test_sync_models_from_cache_skip_existing(test_engine):
    """Test sync_models_from_cache skips already registered models."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create existing model in DB
    async with async_session() as session:
        existing = Model(
            repo_id="mlx-community/existing-sync-model",
            model_type="text-gen",
            local_path="/old/path",
            downloaded_at=datetime.now(tz=UTC),
        )
        session.add(existing)
        await session.commit()

    # Mock local models with same model
    mock_local_model = MagicMock()
    mock_local_model.model_id = "mlx-community/existing-sync-model"
    mock_local_model.local_path = "/new/path"
    mock_local_model.size_bytes = 2000000

    with (
        patch("mlx_manager.database.get_session", _create_session_mock(test_engine)),
        patch("mlx_manager.services.hf_client.hf_client") as mock_hf,
    ):
        mock_hf.list_local_models.return_value = [mock_local_model]

        count = await sync_models_from_cache()
        assert count == 0  # Should skip existing

        # Verify model was not updated
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/existing-sync-model")
            )
            model = result.scalar_one_or_none()
            assert model.local_path == "/old/path"  # Still old path


@pytest.mark.asyncio
async def test_register_model_from_download_new_model(test_engine):
    """Test register_model_from_download creates a new model."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    with (
        patch("mlx_manager.database.get_session", _create_session_mock(test_engine)),
        patch("mlx_manager.mlx_server.models.detection.detect_model_type") as mock_detect,
    ):
        from mlx_manager.mlx_server.models.types import ModelType

        mock_detect.return_value = ModelType.VISION

        await register_model_from_download(
            repo_id="mlx-community/new-download-model",
            local_path="/download/path",
            size_bytes=5000000,
        )

        # Verify model was created
        async_session = sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/new-download-model")
            )
            model = result.scalar_one_or_none()
            assert model is not None
            assert model.model_type == "vision"
            assert model.local_path == "/download/path"
            assert model.size_bytes == 5000000
            assert model.downloaded_at is not None


@pytest.mark.asyncio
async def test_register_model_from_download_update_existing(test_engine):
    """Test register_model_from_download updates existing model."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create existing model
    async with async_session() as session:
        existing = Model(
            repo_id="mlx-community/update-download-model",
            model_type="text-gen",
            local_path="/old/download/path",
            size_bytes=1000000,
            downloaded_at=datetime.now(tz=UTC),
        )
        session.add(existing)
        await session.commit()

    with patch("mlx_manager.database.get_session", _create_session_mock(test_engine)):
        await register_model_from_download(
            repo_id="mlx-community/update-download-model",
            local_path="/new/download/path",
            size_bytes=2000000,
        )

        # Verify model was updated
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/update-download-model")
            )
            model = result.scalar_one_or_none()
            assert model is not None
            assert model.local_path == "/new/download/path"
            assert model.size_bytes == 2000000


@pytest.mark.asyncio
async def test_register_model_from_download_no_size(test_engine):
    """Test register_model_from_download without size_bytes."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create existing model with size
    async with async_session() as session:
        existing = Model(
            repo_id="mlx-community/no-size-model",
            model_type="text-gen",
            local_path="/old/path",
            size_bytes=1000000,
            downloaded_at=datetime.now(tz=UTC),
        )
        session.add(existing)
        await session.commit()

    with patch("mlx_manager.database.get_session", _create_session_mock(test_engine)):
        # Update without size_bytes
        await register_model_from_download(
            repo_id="mlx-community/no-size-model",
            local_path="/new/path",
            size_bytes=None,
        )

        # Verify model was updated but size unchanged
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/no-size-model")
            )
            model = result.scalar_one_or_none()
            assert model is not None
            assert model.local_path == "/new/path"
            assert model.size_bytes == 1000000  # Unchanged


@pytest.mark.asyncio
async def test_update_model_capabilities_new_model(test_engine):
    """Test update_model_capabilities creates model and capabilities."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models.capabilities import ModelCapabilities

    with patch("mlx_manager.database.get_session", _create_session_mock(test_engine)):
        await update_model_capabilities(
            repo_id="mlx-community/caps-new-model",
            model_type="text-gen",
            supports_native_tools=True,
            supports_thinking=False,
            model_family="qwen",
            tool_parser_id="qwen3",
            thinking_parser_id="null",
            practical_max_tokens=4096,
            probe_version=2,
        )

        # Verify model and capabilities were created
        async_session = sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/caps-new-model")
            )
            model = result.scalar_one_or_none()
            assert model is not None
            assert model.model_type == "text-gen"

            # Verify capabilities
            caps_result = await session.execute(
                select(ModelCapabilities).where(ModelCapabilities.model_id == model.id)
            )
            caps = caps_result.scalar_one_or_none()
            assert caps is not None
            assert caps.supports_native_tools is True
            assert caps.supports_thinking is False
            assert caps.model_family == "qwen"
            assert caps.tool_parser_id == "qwen3"
            assert caps.thinking_parser_id == "null"
            assert caps.practical_max_tokens == 4096
            assert caps.probe_version == 2


@pytest.mark.asyncio
async def test_update_model_capabilities_update_existing(test_engine):
    """Test update_model_capabilities updates existing capabilities (upsert)."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models.capabilities import ModelCapabilities

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create existing model with capabilities
    async with async_session() as session:
        model = Model(
            repo_id="mlx-community/caps-update-model",
            model_type="text-gen",
            downloaded_at=datetime.now(tz=UTC),
        )
        session.add(model)
        await session.flush()

        old_caps = ModelCapabilities(
            model_id=model.id,
            capability_type="text-gen",
            supports_native_tools=False,
            supports_thinking=False,
            probe_version=1,
        )
        session.add(old_caps)
        await session.commit()

    with patch("mlx_manager.database.get_session", _create_session_mock(test_engine)):
        # Update capabilities
        await update_model_capabilities(
            repo_id="mlx-community/caps-update-model",
            model_type="text-gen",
            supports_native_tools=True,
            supports_thinking=True,
            model_family="glm4",
            probe_version=2,
        )

        # Verify capabilities were updated
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/caps-update-model")
            )
            model = result.scalar_one_or_none()
            assert model is not None

            caps_result = await session.execute(
                select(ModelCapabilities).where(ModelCapabilities.model_id == model.id)
            )
            caps = caps_result.scalar_one_or_none()
            assert caps is not None
            assert caps.supports_native_tools is True
            assert caps.supports_thinking is True
            assert caps.model_family == "glm4"
            assert caps.probe_version == 2


@pytest.mark.asyncio
async def test_update_model_capabilities_with_model_type_enum(test_engine):
    """Test update_model_capabilities handles ModelType enum for model_type."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.mlx_server.models.types import ModelType
    from mlx_manager.models.capabilities import ModelCapabilities

    with patch("mlx_manager.database.get_session", _create_session_mock(test_engine)):
        # Pass ModelType enum instead of string
        await update_model_capabilities(
            repo_id="mlx-community/caps-enum-model",
            model_type=ModelType.VISION,  # Enum, not string
            supports_multi_image=True,
            supports_video=False,
        )

        # Verify model was created with correct type
        async_session = sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/caps-enum-model")
            )
            model = result.scalar_one_or_none()
            assert model is not None
            assert model.model_type == "vision"

            caps_result = await session.execute(
                select(ModelCapabilities).where(ModelCapabilities.model_id == model.id)
            )
            caps = caps_result.scalar_one_or_none()
            assert caps is not None
            assert caps.capability_type == "vision"


@pytest.mark.asyncio
async def test_update_model_capabilities_unknown_fields_ignored(test_engine):
    """Test update_model_capabilities ignores unknown capability fields."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models.capabilities import ModelCapabilities

    with patch("mlx_manager.database.get_session", _create_session_mock(test_engine)):
        # Pass some unknown fields
        await update_model_capabilities(
            repo_id="mlx-community/caps-unknown-fields",
            model_type="text-gen",
            supports_native_tools=True,
            unknown_field_1="should be ignored",
            unknown_field_2=123,
        )

        # Verify capabilities were created without unknown fields
        async_session = sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/caps-unknown-fields")
            )
            model = result.scalar_one_or_none()
            assert model is not None

            caps_result = await session.execute(
                select(ModelCapabilities).where(ModelCapabilities.model_id == model.id)
            )
            caps = caps_result.scalar_one_or_none()
            assert caps is not None
            assert caps.supports_native_tools is True
            # Unknown fields should not cause errors


@pytest.mark.asyncio
async def test_update_model_capabilities_all_capability_fields(test_engine):
    """Test update_model_capabilities with all possible capability fields."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models.capabilities import ModelCapabilities

    with patch("mlx_manager.database.get_session", _create_session_mock(test_engine)):
        # Pass all capability fields
        await update_model_capabilities(
            repo_id="mlx-community/caps-all-fields",
            model_type="vision",
            supports_native_tools=True,
            supports_thinking=True,
            tool_format="template",
            tool_parser_id="qwen3",
            thinking_parser_id="think_tag",
            practical_max_tokens=8192,
            model_family="qwen",
            supports_multi_image=True,
            supports_video=False,
            embedding_dimensions=768,
            max_sequence_length=512,
            is_normalized=True,
            supports_tts=False,
            supports_stt=False,
            probe_version=2,
        )

        # Verify all fields were saved
        async_session = sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.repo_id == "mlx-community/caps-all-fields")
            )
            model = result.scalar_one_or_none()
            assert model is not None

            caps_result = await session.execute(
                select(ModelCapabilities).where(ModelCapabilities.model_id == model.id)
            )
            caps = caps_result.scalar_one_or_none()
            assert caps is not None
            assert caps.supports_native_tools is True
            assert caps.supports_thinking is True
            assert caps.tool_format == "template"
            assert caps.tool_parser_id == "qwen3"
            assert caps.thinking_parser_id == "think_tag"
            assert caps.practical_max_tokens == 8192
            assert caps.model_family == "qwen"
            assert caps.supports_multi_image is True
            assert caps.supports_video is False
            assert caps.embedding_dimensions == 768
            assert caps.max_sequence_length == 512
            assert caps.is_normalized is True
            assert caps.supports_tts is False
            assert caps.supports_stt is False
            assert caps.probe_version == 2

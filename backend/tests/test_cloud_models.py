"""Tests for cloud backend routing database models."""

import pytest
from sqlmodel import SQLModel

from mlx_manager.models import (
    BackendMapping,
    BackendMappingCreate,
    BackendMappingResponse,
    BackendType,
    CloudCredential,
    CloudCredentialCreate,
    CloudCredentialResponse,
)


class TestBackendType:
    """Tests for BackendType enum."""

    def test_backend_type_values(self) -> None:
        """All expected backend types exist."""
        assert BackendType.LOCAL.value == "local"
        assert BackendType.OPENAI.value == "openai"
        assert BackendType.ANTHROPIC.value == "anthropic"

    def test_backend_type_count(self) -> None:
        """Ensure we have exactly 3 backend types."""
        assert len(BackendType) == 3

    def test_backend_type_is_string_enum(self) -> None:
        """BackendType is both str and Enum."""
        assert isinstance(BackendType.LOCAL, str)
        assert BackendType.LOCAL == "local"


class TestBackendMapping:
    """Tests for BackendMapping model."""

    def test_create_with_required_fields(self) -> None:
        """Create mapping with only required fields."""
        mapping = BackendMapping(
            model_pattern="gpt-*",
            backend_type=BackendType.OPENAI,
        )
        assert mapping.model_pattern == "gpt-*"
        assert mapping.backend_type == BackendType.OPENAI

    def test_optional_fields_default_correctly(self) -> None:
        """Optional fields have correct defaults."""
        mapping = BackendMapping(
            model_pattern="llama-*",
            backend_type=BackendType.LOCAL,
        )
        assert mapping.enabled is True
        assert mapping.priority == 0
        assert mapping.backend_model is None
        assert mapping.fallback_backend is None
        assert mapping.id is None

    def test_timestamps_auto_generated(self) -> None:
        """Timestamps are auto-generated on creation."""
        mapping = BackendMapping(
            model_pattern="claude-*",
            backend_type=BackendType.ANTHROPIC,
        )
        assert mapping.created_at is not None
        assert mapping.updated_at is not None

    def test_pattern_matching_examples(self) -> None:
        """Pattern examples for documentation."""
        # Exact model name
        exact = BackendMapping(
            model_pattern="gpt-4-turbo",
            backend_type=BackendType.OPENAI,
        )
        assert exact.model_pattern == "gpt-4-turbo"

        # Wildcard pattern
        wildcard = BackendMapping(
            model_pattern="mlx-community/*",
            backend_type=BackendType.LOCAL,
        )
        assert wildcard.model_pattern == "mlx-community/*"

    def test_fallback_configuration(self) -> None:
        """Fallback backend can be configured."""
        mapping = BackendMapping(
            model_pattern="local-model",
            backend_type=BackendType.LOCAL,
            fallback_backend=BackendType.OPENAI,
        )
        assert mapping.fallback_backend == BackendType.OPENAI

    def test_backend_model_override(self) -> None:
        """Backend model can override the requested model name."""
        mapping = BackendMapping(
            model_pattern="fast",  # Alias
            backend_type=BackendType.OPENAI,
            backend_model="gpt-4o-mini",  # Actual model
        )
        assert mapping.backend_model == "gpt-4o-mini"

    def test_priority_for_pattern_matching(self) -> None:
        """Higher priority mappings are checked first."""
        high_priority = BackendMapping(
            model_pattern="gpt-4*",
            backend_type=BackendType.OPENAI,
            priority=100,
        )
        low_priority = BackendMapping(
            model_pattern="gpt-*",
            backend_type=BackendType.OPENAI,
            priority=0,
        )
        assert high_priority.priority > low_priority.priority


class TestCloudCredential:
    """Tests for CloudCredential model."""

    def test_create_with_required_fields(self) -> None:
        """Create credential with required fields."""
        cred = CloudCredential(
            backend_type=BackendType.OPENAI,
            encrypted_api_key="encrypted_key_here",
        )
        assert cred.backend_type == BackendType.OPENAI
        assert cred.encrypted_api_key == "encrypted_key_here"

    def test_base_url_is_optional(self) -> None:
        """Base URL is optional for custom endpoints."""
        cred = CloudCredential(
            backend_type=BackendType.OPENAI,
            encrypted_api_key="key",
        )
        assert cred.base_url is None

        cred_with_url = CloudCredential(
            backend_type=BackendType.OPENAI,
            encrypted_api_key="key",
            base_url="https://custom.openai.azure.com/",
        )
        assert cred_with_url.base_url == "https://custom.openai.azure.com/"

    def test_timestamps_auto_generated(self) -> None:
        """Timestamps are auto-generated on creation."""
        cred = CloudCredential(
            backend_type=BackendType.ANTHROPIC,
            encrypted_api_key="key",
        )
        assert cred.created_at is not None
        assert cred.updated_at is not None


class TestBackendMappingCreate:
    """Tests for BackendMappingCreate schema."""

    def test_accepts_valid_data(self) -> None:
        """Schema accepts valid creation data."""
        create = BackendMappingCreate(
            model_pattern="gpt-*",
            backend_type=BackendType.OPENAI,
        )
        assert create.model_pattern == "gpt-*"
        assert create.backend_type == BackendType.OPENAI

    def test_optional_fields_have_defaults(self) -> None:
        """Optional fields default correctly."""
        create = BackendMappingCreate(
            model_pattern="test",
            backend_type=BackendType.LOCAL,
        )
        assert create.backend_model is None
        assert create.fallback_backend is None
        assert create.priority == 0

    def test_accepts_all_optional_fields(self) -> None:
        """Schema accepts all optional fields."""
        create = BackendMappingCreate(
            model_pattern="local-llama",
            backend_type=BackendType.LOCAL,
            backend_model="llama-3-8b",
            fallback_backend=BackendType.OPENAI,
            priority=50,
        )
        assert create.backend_model == "llama-3-8b"
        assert create.fallback_backend == BackendType.OPENAI
        assert create.priority == 50


class TestBackendMappingResponse:
    """Tests for BackendMappingResponse schema."""

    def test_response_model_structure(self) -> None:
        """Response model has expected fields."""
        response = BackendMappingResponse(
            id=1,
            model_pattern="gpt-*",
            backend_type=BackendType.OPENAI,
            backend_model=None,
            fallback_backend=None,
            priority=0,
            enabled=True,
        )
        assert response.id == 1
        assert response.enabled is True


class TestCloudCredentialCreate:
    """Tests for CloudCredentialCreate schema."""

    def test_accepts_valid_data(self) -> None:
        """Schema accepts valid creation data."""
        create = CloudCredentialCreate(
            backend_type=BackendType.OPENAI,
            api_key="sk-test-key-12345",
        )
        assert create.backend_type == BackendType.OPENAI
        assert create.api_key == "sk-test-key-12345"

    def test_api_key_is_plain_text(self) -> None:
        """API key field accepts plain text (encryption happens on storage)."""
        create = CloudCredentialCreate(
            backend_type=BackendType.ANTHROPIC,
            api_key="anthropic-key-here",
            base_url=None,
        )
        # Key is plain text - will be encrypted by service before storage
        assert "anthropic-key-here" in create.api_key

    def test_base_url_optional(self) -> None:
        """Base URL is optional."""
        create = CloudCredentialCreate(
            backend_type=BackendType.OPENAI,
            api_key="key",
        )
        assert create.base_url is None


class TestCloudCredentialResponse:
    """Tests for CloudCredentialResponse schema."""

    def test_excludes_api_key(self) -> None:
        """Response model does NOT expose API key."""
        from datetime import UTC, datetime

        response = CloudCredentialResponse(
            id=1,
            backend_type=BackendType.OPENAI,
            base_url=None,
            created_at=datetime.now(tz=UTC),
        )
        # Verify no api_key or encrypted_api_key field
        assert not hasattr(response, "api_key")
        assert not hasattr(response, "encrypted_api_key")
        # Verify expected fields exist
        assert response.id == 1
        assert response.backend_type == BackendType.OPENAI


class TestTableRegistration:
    """Tests for SQLModel table registration."""

    def test_tables_registered_with_metadata(self) -> None:
        """New tables are registered with SQLModel metadata."""
        tables = SQLModel.metadata.tables.keys()
        assert "backend_mappings" in tables
        assert "cloud_credentials" in tables

    def test_backend_mapping_is_table(self) -> None:
        """BackendMapping is a database table."""
        assert BackendMapping.__tablename__ == "backend_mappings"

    def test_cloud_credential_is_table(self) -> None:
        """CloudCredential is a database table."""
        assert CloudCredential.__tablename__ == "cloud_credentials"

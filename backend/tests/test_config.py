"""Tests for config module — JWT secret auto-generation."""

import os
import stat

import pytest

from mlx_manager.config import _resolve_jwt_secret


class TestResolveJwtSecret:
    """Tests for _resolve_jwt_secret()."""

    def test_generates_secret_when_no_file(self, tmp_path, monkeypatch):
        """First run: generates a new secret and writes it to disk."""
        secret_path = tmp_path / ".jwt_secret"
        monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", secret_path)

        secret = _resolve_jwt_secret()

        assert len(secret) > 0
        assert secret_path.exists()
        assert secret_path.read_text() == secret

    def test_file_has_600_permissions(self, tmp_path, monkeypatch):
        """Generated secret file should only be readable by owner."""
        secret_path = tmp_path / ".jwt_secret"
        monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", secret_path)

        _resolve_jwt_secret()

        file_mode = stat.S_IMODE(os.stat(secret_path).st_mode)
        assert file_mode == 0o600

    def test_reads_existing_file(self, tmp_path, monkeypatch):
        """Subsequent runs: reads the persisted secret from disk."""
        secret_path = tmp_path / ".jwt_secret"
        secret_path.write_text("my-persisted-secret")
        monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", secret_path)

        assert _resolve_jwt_secret() == "my-persisted-secret"

    def test_strips_whitespace_from_file(self, tmp_path, monkeypatch):
        """Strips trailing newlines/whitespace from the secret file."""
        secret_path = tmp_path / ".jwt_secret"
        secret_path.write_text("  secret-with-whitespace  \n")
        monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", secret_path)

        assert _resolve_jwt_secret() == "secret-with-whitespace"

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        """Creates parent directories if they don't exist."""
        secret_path = tmp_path / "nested" / "dir" / ".jwt_secret"
        monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", secret_path)

        _resolve_jwt_secret()

        assert secret_path.exists()

    def test_generates_unique_secrets(self, tmp_path, monkeypatch):
        """Each generation produces a different secret."""
        path_a = tmp_path / "a" / ".jwt_secret"
        path_b = tmp_path / "b" / ".jwt_secret"

        monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", path_a)
        secret_a = _resolve_jwt_secret()

        monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", path_b)
        secret_b = _resolve_jwt_secret()

        assert secret_a != secret_b


@pytest.mark.parametrize("env_value", ["my-env-secret", "override-value"])
def test_env_var_overrides_default_factory(env_value, tmp_path, monkeypatch):
    """MLX_MANAGER_JWT_SECRET env var takes priority over file-based secret."""
    from mlx_manager.config import Settings

    secret_path = tmp_path / ".jwt_secret"
    secret_path.write_text("file-based-secret")
    monkeypatch.setattr("mlx_manager.config._JWT_SECRET_PATH", secret_path)
    monkeypatch.setenv("MLX_MANAGER_JWT_SECRET", env_value)

    s = Settings()
    assert s.jwt_secret == env_value

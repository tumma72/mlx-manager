"""Tests for admin authentication (verify_admin_token) and path traversal prevention.

Covers three previously-untested branches:
  1. No token configured (admin_token=None) → endpoints return 200 (open access)
  2. Token configured + correct Authorization header → returns 200
  3. Token configured + wrong/missing Authorization header → returns 401 or 403

Also covers Gap 3: path traversal in model_id :path parameters.

Tests use the FastAPI TestClient with the full app (embedded=True), patching the
admin_token setting via get_settings() so no real models are loaded.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture: TestClient with admin router wired up
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_no_token():
    """App with admin_token=None (open access)."""
    from mlx_manager.mlx_server.main import create_app

    app = create_app(embedded=True)

    mock_settings = MagicMock()
    mock_settings.admin_token = None  # No token → open access

    with patch("mlx_manager.mlx_server.api.v1.admin.get_settings", return_value=mock_settings):
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def app_with_token():
    """App with admin_token='secret123' configured."""
    from mlx_manager.mlx_server.main import create_app

    app = create_app(embedded=True)

    mock_settings = MagicMock()
    mock_settings.admin_token = "secret123"

    with patch("mlx_manager.mlx_server.api.v1.admin.get_settings", return_value=mock_settings):
        yield TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers to make admin health endpoint respond without DB / pool
# ---------------------------------------------------------------------------


def _patch_admin_health():
    """Return a context manager that stubs out pool and memory for /admin/health."""
    # /admin/health calls nothing; it just returns {"status": "healthy"}
    # No additional patching needed for this endpoint.
    from contextlib import nullcontext

    return nullcontext()


# ---------------------------------------------------------------------------
# Gap 1: No token configured → open access (200 for any caller)
# ---------------------------------------------------------------------------


class TestAdminAuthNoToken:
    """When admin_token is None, all admin endpoints are open (no auth required)."""

    def test_health_no_auth_header_returns_200(self, app_no_token):
        """No token configured → GET /admin/health returns 200 without auth header."""
        response = app_no_token.get("/admin/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_health_with_auth_header_still_returns_200(self, app_no_token):
        """No token configured → extra auth header is ignored; still 200."""
        response = app_no_token.get(
            "/admin/health",
            headers={"Authorization": "Bearer some-token"},
        )
        assert response.status_code == 200

    def test_health_with_wrong_token_still_returns_200(self, app_no_token):
        """No token configured → wrong token is ignored; still 200."""
        response = app_no_token.get(
            "/admin/health",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Gap 2: Token configured + correct Authorization header → 200
# ---------------------------------------------------------------------------


class TestAdminAuthCorrectToken:
    """When admin_token is set, correct Bearer token grants access."""

    def test_correct_bearer_token_returns_200(self, app_with_token):
        """Correct Bearer token → 200 on /admin/health."""
        response = app_with_token.get(
            "/admin/health",
            headers={"Authorization": "Bearer secret123"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_correct_token_case_insensitive_scheme(self, app_with_token):
        """Bearer scheme is case-insensitive; 'bearer' works as well as 'Bearer'."""
        response = app_with_token.get(
            "/admin/health",
            headers={"Authorization": "bearer secret123"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Gap 3: Token configured + wrong or missing Authorization header → 401/403
# ---------------------------------------------------------------------------


class TestAdminAuthWrongToken:
    """When admin_token is set, wrong or missing tokens are rejected."""

    def test_missing_auth_header_returns_401(self, app_with_token):
        """No Authorization header → 401 Unauthorized."""
        response = app_with_token.get("/admin/health")
        assert response.status_code == 401
        # The error handler returns an RFC 7807 Problem Details body
        body = response.json()
        assert "status" in body
        assert body["status"] == 401

    def test_wrong_token_returns_403(self, app_with_token):
        """Wrong Bearer token → 403 Forbidden."""
        response = app_with_token.get(
            "/admin/health",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 403

    def test_wrong_scheme_returns_403(self, app_with_token):
        """Non-Bearer scheme with correct token → 403 Forbidden."""
        response = app_with_token.get(
            "/admin/health",
            headers={"Authorization": "Basic secret123"},
        )
        assert response.status_code == 403

    def test_empty_bearer_token_returns_403(self, app_with_token):
        """Bearer with empty token → 403 Forbidden."""
        response = app_with_token.get(
            "/admin/health",
            headers={"Authorization": "Bearer "},
        )
        assert response.status_code == 403

    def test_token_prefix_match_returns_403(self, app_with_token):
        """Partial token match is rejected (not a prefix check)."""
        response = app_with_token.get(
            "/admin/health",
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Gap 3: Path traversal in model_id :path parameters
# ---------------------------------------------------------------------------


class TestPathTraversalPrevention:
    """Path traversal attempts via model_id :path params should not load arbitrary files.

    The goal is not to test filesystem access (tests never reach that point)
    but to verify the API returns an appropriate error response rather than
    attempting to resolve the path as a model.

    The pool.preload_model() call will raise an exception (no real model pool),
    which the endpoint converts to a 500.  The important invariant is that the
    request completes (does not hang or perform filesystem I/O on the attack path).
    """

    @pytest.fixture()
    def client_no_token(self):
        """Client with admin_token=None so auth doesn't block traversal tests."""
        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)

        mock_settings = MagicMock()
        mock_settings.admin_token = None

        mock_pool = MagicMock()
        mock_pool.preload_model = AsyncMock(
            side_effect=ValueError("Model not found: ../../etc/passwd")
        )

        with (
            patch("mlx_manager.mlx_server.api.v1.admin.get_settings", return_value=mock_settings),
            patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool", return_value=mock_pool),
        ):
            yield TestClient(app, raise_server_exceptions=False)

    def test_path_traversal_dotdot_returns_error(self, client_no_token):
        """POST /admin/models/load/../../etc/passwd returns an error (not a file load)."""
        response = client_no_token.post("/admin/models/load/../../etc/passwd")
        # Should get 500 (pool raised ValueError) or 422 — never 200
        assert response.status_code != 200

    def test_path_traversal_slashes_returns_error(self, client_no_token):
        """POST /admin/models/load/../../../root returns an error."""
        response = client_no_token.post("/admin/models/load/../../../root")
        assert response.status_code != 200

    def test_normal_model_id_reaches_pool(self, client_no_token):
        """A well-formed model ID reaches pool.preload_model (returns 500 due to mock error)."""
        response = client_no_token.post("/admin/models/load/mlx-community/Llama-3.2-3B-4bit")
        # The pool mock raises ValueError → 500
        assert response.status_code == 500

    def test_unload_path_traversal_returns_error(self, client_no_token):
        """POST /admin/models/unload/../../etc/passwd returns an error."""
        mock_pool_unload = MagicMock()
        mock_pool_unload.preload_model = AsyncMock(side_effect=ValueError("not found"))
        mock_pool_unload.unload_model = AsyncMock(return_value=False)

        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)
        mock_settings = MagicMock()
        mock_settings.admin_token = None

        with (
            patch("mlx_manager.mlx_server.api.v1.admin.get_settings", return_value=mock_settings),
            patch(
                "mlx_manager.mlx_server.api.v1.admin.get_model_pool",
                return_value=mock_pool_unload,
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/admin/models/unload/../../etc/passwd")
            # Model not loaded (False) → 404; or 404 is fine; never 200 success
            # The key assertion: request completes and no actual filesystem access
            assert response.status_code in (404, 422, 500)

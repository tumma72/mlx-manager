"""Tests for Request ID middleware propagation."""

import re

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlx_manager.mlx_server.errors import register_error_handlers
from mlx_manager.mlx_server.middleware.request_id import RequestIDMiddleware

# Pattern for generated request IDs: req_ followed by exactly 12 hex characters
REQUEST_ID_PATTERN = re.compile(r"^req_[0-9a-f]{12}$")


@pytest.fixture
def app() -> FastAPI:
    """Create test app with RequestIDMiddleware and error handlers."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/not-found")
    async def not_found():
        raise HTTPException(status_code=404, detail="Not found")

    @app.get("/server-error")
    async def server_error():
        raise RuntimeError("Unexpected failure")

    @app.get("/echo-request-id")
    async def echo_request_id(request):
        """Echo the request ID from request.state for testing."""
        return {"request_id": request.state.request_id}

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


class TestRequestIDOnSuccessfulRequests:
    """Request ID middleware adds X-Request-ID to successful responses."""

    def test_success_response_has_request_id_header(self, client: TestClient) -> None:
        """Successful requests include X-Request-ID response header."""
        response = client.get("/ok")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    def test_generated_id_matches_format(self, client: TestClient) -> None:
        """Generated request IDs match req_{12_hex_chars} format."""
        response = client.get("/ok")
        request_id = response.headers["X-Request-ID"]
        assert REQUEST_ID_PATTERN.match(request_id), (
            f"Request ID '{request_id}' does not match expected format req_{{12_hex_chars}}"
        )

    def test_unique_ids_per_request(self, client: TestClient) -> None:
        """Each request gets a unique request ID."""
        ids = {client.get("/ok").headers["X-Request-ID"] for _ in range(10)}
        assert len(ids) == 10, "Expected 10 unique request IDs"


class TestRequestIDOnErrorResponses:
    """Request ID middleware works with error handler responses."""

    def test_error_response_has_request_id_header(self, client: TestClient) -> None:
        """Error responses include X-Request-ID header from middleware."""
        response = client.get("/not-found")
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers

    def test_error_body_matches_header(self, client: TestClient) -> None:
        """Error response body request_id matches X-Request-ID header."""
        response = client.get("/not-found")
        header_id = response.headers["X-Request-ID"]
        body_id = response.json()["request_id"]
        assert header_id == body_id, f"Header ID '{header_id}' should match body ID '{body_id}'"

    def test_server_error_has_request_id(self, client: TestClient) -> None:
        """Unhandled exception responses include X-Request-ID."""
        response = client.get("/server-error")
        assert response.status_code == 500
        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert REQUEST_ID_PATTERN.match(request_id)


class TestClientProvidedRequestID:
    """Client-provided X-Request-ID is preserved and echoed."""

    def test_client_id_echoed_on_success(self, client: TestClient) -> None:
        """Client-provided X-Request-ID is returned in response."""
        custom_id = "my-custom-request-id-123"
        response = client.get("/ok", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id

    def test_client_id_echoed_on_error(self, client: TestClient) -> None:
        """Client-provided X-Request-ID is returned on error responses too."""
        custom_id = "client-error-trace-456"
        response = client.get("/not-found", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id
        # Body should also contain the client-provided ID
        assert response.json()["request_id"] == custom_id

    def test_client_id_available_in_request_state(self, client: TestClient) -> None:
        """Client-provided ID is stored in request.state.request_id."""
        custom_id = "state-check-789"
        response = client.get("/echo-request-id", headers={"X-Request-ID": custom_id})
        assert response.json()["request_id"] == custom_id


class TestRequestStateAvailability:
    """Request ID is available in request.state for downstream handlers."""

    def test_request_state_has_generated_id(self, client: TestClient) -> None:
        """Generated request ID is available via request.state."""
        response = client.get("/echo-request-id")
        body_id = response.json()["request_id"]
        header_id = response.headers["X-Request-ID"]
        assert body_id == header_id
        assert REQUEST_ID_PATTERN.match(body_id)


class TestHelperFunction:
    """Tests for the _get_request_id fallback function."""

    def test_get_request_id_with_state(self) -> None:
        """_get_request_id returns request.state.request_id when set."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.errors.handlers import _get_request_id

        request = MagicMock()
        request.state.request_id = "req_aabbccddee00"
        assert _get_request_id(request) == "req_aabbccddee00"

    def test_get_request_id_without_state(self) -> None:
        """_get_request_id generates fallback when state not set."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.errors.handlers import _get_request_id

        request = MagicMock(spec=[])
        # No request.state attribute at all
        result = _get_request_id(request)
        assert REQUEST_ID_PATTERN.match(result)

    def test_generate_request_id_still_works(self) -> None:
        """generate_request_id function still exists as fallback."""
        from mlx_manager.mlx_server.errors.handlers import generate_request_id

        rid = generate_request_id()
        assert REQUEST_ID_PATTERN.match(rid)

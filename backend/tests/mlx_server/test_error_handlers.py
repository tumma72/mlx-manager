"""Tests for RFC 7807 error handlers."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlx_manager.mlx_server.errors import (
    ProblemDetail,
    TimeoutHTTPException,
    register_error_handlers,
)


@pytest.fixture
def app() -> FastAPI:
    """Create test app with error handlers."""
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/not-found")
    async def not_found():
        raise HTTPException(status_code=404, detail="Model not found")

    @app.get("/timeout")
    async def timeout():
        raise TimeoutHTTPException(timeout_seconds=60.0)

    @app.get("/server-error")
    async def server_error():
        raise RuntimeError("Unexpected failure")

    @app.post("/validation")
    async def validation(data: dict):
        if "required_field" not in data:
            raise HTTPException(status_code=422, detail="Missing required_field")
        return data

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_successful_request(client: TestClient) -> None:
    """Successful requests return normal JSON."""
    response = client.get("/ok")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_http_exception_returns_problem_details(client: TestClient) -> None:
    """HTTPException returns RFC 7807 Problem Details."""
    response = client.get("/not-found")
    assert response.status_code == 404
    data = response.json()

    # Verify Problem Details structure
    assert data["type"] == "https://mlx-manager.dev/errors/not-found"
    assert data["title"] == "Not Found"
    assert data["status"] == 404
    assert data["detail"] == "Model not found"
    assert "request_id" in data
    assert data["request_id"].startswith("req_")

    # Verify X-Request-ID header
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == data["request_id"]


def test_timeout_exception_returns_timeout_problem(client: TestClient) -> None:
    """TimeoutHTTPException returns specialized timeout Problem Details."""
    response = client.get("/timeout")
    assert response.status_code == 408
    data = response.json()

    assert data["type"] == "https://mlx-manager.dev/errors/timeout"
    assert data["title"] == "Request Timeout"
    assert data["status"] == 408
    assert data["timeout_seconds"] == 60.0
    assert "request_id" in data


def test_generic_exception_returns_500_without_internals(client: TestClient) -> None:
    """Unhandled exceptions return 500 without exposing internals."""
    response = client.get("/server-error")
    assert response.status_code == 500
    data = response.json()

    assert data["type"] == "https://mlx-manager.dev/errors/internal-error"
    assert data["title"] == "Internal Server Error"
    assert data["status"] == 500
    # Must NOT contain stack trace or internal error message
    assert "RuntimeError" not in str(data)
    assert "Unexpected failure" not in str(data)
    assert "request_id" in data


def test_request_id_unique_per_request(client: TestClient) -> None:
    """Each error request gets unique request_id."""
    response1 = client.get("/not-found")
    response2 = client.get("/not-found")

    id1 = response1.json()["request_id"]
    id2 = response2.json()["request_id"]

    assert id1 != id2


def test_problem_detail_model_structure() -> None:
    """ProblemDetail model has correct fields."""
    problem = ProblemDetail(
        type="https://mlx-manager.dev/errors/test",
        title="Test Error",
        status=400,
        detail="Test detail",
        instance="/test/path",
        request_id="req_123456789012",
    )

    data = problem.model_dump(exclude_none=True)
    assert data["type"] == "https://mlx-manager.dev/errors/test"
    assert data["title"] == "Test Error"
    assert data["status"] == 400
    assert data["detail"] == "Test detail"
    assert data["instance"] == "/test/path"
    assert data["request_id"] == "req_123456789012"


def test_timeout_http_exception() -> None:
    """TimeoutHTTPException carries timeout information."""
    exc = TimeoutHTTPException(timeout_seconds=30.0)
    assert exc.status_code == 408
    assert exc.timeout_seconds == 30.0
    assert "30" in exc.detail

    exc_custom = TimeoutHTTPException(timeout_seconds=60.0, detail="Custom timeout message")
    assert exc_custom.detail == "Custom timeout message"

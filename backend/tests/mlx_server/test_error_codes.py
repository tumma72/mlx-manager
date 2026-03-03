"""Tests for ErrorCode enum and error_code field in Problem Details."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlx_manager.mlx_server.errors import (
    ErrorCode,
    ProblemDetail,
    TimeoutHTTPException,
    TimeoutProblem,
    register_error_handlers,
)

# --- ErrorCode enum tests ---


class TestErrorCodeEnum:
    """Tests for ErrorCode string enum values."""

    def test_validation_error_value(self) -> None:
        assert ErrorCode.VALIDATION_ERROR == "validation_error"

    def test_field_max_length_exceeded_value(self) -> None:
        assert ErrorCode.FIELD_MAX_LENGTH_EXCEEDED == "field_max_length_exceeded"

    def test_invalid_request_value(self) -> None:
        assert ErrorCode.INVALID_REQUEST == "invalid_request"

    def test_model_not_found_value(self) -> None:
        assert ErrorCode.MODEL_NOT_FOUND == "model_not_found"

    def test_resource_not_found_value(self) -> None:
        assert ErrorCode.RESOURCE_NOT_FOUND == "resource_not_found"

    def test_unauthorized_value(self) -> None:
        assert ErrorCode.UNAUTHORIZED == "unauthorized"

    def test_forbidden_value(self) -> None:
        assert ErrorCode.FORBIDDEN == "forbidden"

    def test_rate_limited_value(self) -> None:
        assert ErrorCode.RATE_LIMITED == "rate_limited"

    def test_request_timeout_value(self) -> None:
        assert ErrorCode.REQUEST_TIMEOUT == "request_timeout"

    def test_internal_error_value(self) -> None:
        assert ErrorCode.INTERNAL_ERROR == "internal_error"

    def test_service_unavailable_value(self) -> None:
        assert ErrorCode.SERVICE_UNAVAILABLE == "service_unavailable"

    def test_inference_error_value(self) -> None:
        assert ErrorCode.INFERENCE_ERROR == "inference_error"

    def test_generation_error_value(self) -> None:
        assert ErrorCode.GENERATION_ERROR == "generation_error"

    def test_is_str_enum(self) -> None:
        """ErrorCode values are strings usable in JSON."""
        assert isinstance(ErrorCode.VALIDATION_ERROR, str)
        assert ErrorCode.INTERNAL_ERROR == "internal_error"


# --- ProblemDetail error_code field tests ---


class TestProblemDetailErrorCode:
    """Tests for error_code field on ProblemDetail model."""

    def test_error_code_included_when_provided(self) -> None:
        """error_code is present in JSON when explicitly set."""
        problem = ProblemDetail(
            title="Bad Request",
            status=400,
            request_id="req_test123",
            error_code=ErrorCode.INVALID_REQUEST,
        )
        data = problem.model_dump(exclude_none=True)
        assert data["error_code"] == "invalid_request"

    def test_error_code_omitted_when_none(self) -> None:
        """error_code is excluded from JSON when None (exclude_none)."""
        problem = ProblemDetail(
            title="Error",
            status=500,
            request_id="req_test456",
        )
        data = problem.model_dump(exclude_none=True)
        assert "error_code" not in data

    def test_error_code_default_is_none(self) -> None:
        """error_code defaults to None."""
        problem = ProblemDetail(
            title="Error",
            status=500,
            request_id="req_test789",
        )
        assert problem.error_code is None

    def test_error_code_accepts_string(self) -> None:
        """error_code accepts plain strings (not just ErrorCode enum)."""
        problem = ProblemDetail(
            title="Custom Error",
            status=400,
            request_id="req_custom",
            error_code="custom_error_code",
        )
        assert problem.error_code == "custom_error_code"

    def test_timeout_problem_inherits_error_code(self) -> None:
        """TimeoutProblem also supports error_code field."""
        problem = TimeoutProblem(
            request_id="req_timeout",
            timeout_seconds=30.0,
            error_code=ErrorCode.REQUEST_TIMEOUT,
        )
        data = problem.model_dump(exclude_none=True)
        assert data["error_code"] == "request_timeout"
        assert data["timeout_seconds"] == 30.0


# --- Integration tests: error handlers include error_code ---


@pytest.fixture
def error_app() -> FastAPI:
    """Create test app with error handlers registered."""
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/bad-request")
    async def bad_request():
        raise HTTPException(status_code=400, detail="Invalid input")

    @app.get("/not-found")
    async def not_found():
        raise HTTPException(status_code=404, detail="Resource missing")

    @app.get("/timeout")
    async def timeout():
        raise TimeoutHTTPException(timeout_seconds=30.0)

    @app.get("/server-error")
    async def server_error():
        raise RuntimeError("Boom")

    @app.get("/unauthorized")
    async def unauthorized():
        raise HTTPException(status_code=401, detail="Not authenticated")

    @app.get("/forbidden")
    async def forbidden():
        raise HTTPException(status_code=403, detail="Access denied")

    @app.get("/rate-limited")
    async def rate_limited():
        raise HTTPException(status_code=429, detail="Too many requests")

    @app.get("/service-unavailable")
    async def service_unavailable():
        raise HTTPException(status_code=503, detail="Service down")

    @app.get("/teapot")
    async def teapot():
        raise HTTPException(status_code=418, detail="I'm a teapot")

    return app


@pytest.fixture
def error_client(error_app: FastAPI) -> TestClient:
    return TestClient(error_app, raise_server_exceptions=False)


class TestHandlerErrorCodes:
    """Tests that error handlers include correct error_code values."""

    def test_400_returns_invalid_request(self, error_client: TestClient) -> None:
        response = error_client.get("/bad-request")
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "invalid_request"

    def test_401_returns_unauthorized(self, error_client: TestClient) -> None:
        response = error_client.get("/unauthorized")
        assert response.status_code == 401
        data = response.json()
        assert data["error_code"] == "unauthorized"

    def test_403_returns_forbidden(self, error_client: TestClient) -> None:
        response = error_client.get("/forbidden")
        assert response.status_code == 403
        data = response.json()
        assert data["error_code"] == "forbidden"

    def test_404_returns_resource_not_found(self, error_client: TestClient) -> None:
        response = error_client.get("/not-found")
        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "resource_not_found"

    def test_422_http_exception_returns_validation_error(
        self, error_app: FastAPI, error_client: TestClient
    ) -> None:
        """HTTP 422 via HTTPException returns validation_error error_code."""

        @error_app.get("/validation-http")
        async def validation_http():
            raise HTTPException(status_code=422, detail="Invalid data")

        response = error_client.get("/validation-http")
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == "validation_error"

    def test_422_pydantic_returns_validation_error(
        self, error_app: FastAPI, error_client: TestClient
    ) -> None:
        """Pydantic validation errors return validation_error error_code."""
        from pydantic import BaseModel, Field

        class StrictModel(BaseModel):
            name: str = Field(..., min_length=1)

        @error_app.post("/strict-model")
        async def strict_endpoint(data: StrictModel):
            return data.model_dump()

        response = error_client.post("/strict-model", json={"name": ""})
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == "validation_error"

    def test_429_returns_rate_limited(self, error_client: TestClient) -> None:
        response = error_client.get("/rate-limited")
        assert response.status_code == 429
        data = response.json()
        assert data["error_code"] == "rate_limited"

    def test_timeout_returns_request_timeout(self, error_client: TestClient) -> None:
        response = error_client.get("/timeout")
        assert response.status_code == 408
        data = response.json()
        assert data["error_code"] == "request_timeout"

    def test_500_generic_returns_internal_error(self, error_client: TestClient) -> None:
        response = error_client.get("/server-error")
        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "internal_error"

    def test_503_returns_service_unavailable(self, error_client: TestClient) -> None:
        response = error_client.get("/service-unavailable")
        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "service_unavailable"

    def test_unmapped_status_code_has_no_error_code(self, error_client: TestClient) -> None:
        """Status codes without a mapping have no error_code in response."""
        response = error_client.get("/teapot")
        assert response.status_code == 418
        data = response.json()
        # error_code should be omitted (exclude_none=True in handler)
        assert "error_code" not in data

    def test_error_code_coexists_with_existing_fields(self, error_client: TestClient) -> None:
        """error_code is an addition that doesn't replace existing fields."""
        response = error_client.get("/not-found")
        data = response.json()
        # All existing fields still present
        assert data["type"] == "https://mlx-manager.dev/errors/not-found"
        assert data["title"] == "Not Found"
        assert data["status"] == 404
        assert data["detail"] == "Resource missing"
        assert "request_id" in data
        # Plus the new error_code
        assert data["error_code"] == "resource_not_found"

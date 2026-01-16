# Testing Patterns

**Analysis Date:** 2026-01-16

## Test Framework

**Backend:**
- Framework: pytest 8.0+ with pytest-asyncio 0.24+
- Coverage: pytest-cov 4.0+
- Config: `backend/pyproject.toml`

**Frontend:**
- Unit tests: Vitest 4.0+ with @testing-library/svelte
- E2E tests: Playwright 1.57+
- Coverage: @vitest/coverage-v8
- Config: `frontend/vitest.config.ts`, `frontend/playwright.config.ts`

**Run Commands:**
```bash
# Backend
cd backend && source .venv/bin/activate
pytest -v                           # Run all tests
pytest --cov=mlx_manager            # Run with coverage
pytest tests/test_profiles.py -v    # Run specific file
pytest -x -q                        # Quick run, stop on first failure

# Frontend Unit
cd frontend
npm run test                        # Run all tests
npm run test:watch                  # Watch mode
npm run test:coverage               # With coverage

# Frontend E2E
npm run test:e2e                    # Run Playwright tests
npm run test:e2e:ui                 # With Playwright UI
```

## Test File Organization

**Backend:**
- Location: `backend/tests/`
- Naming: `test_{module_name}.py` (e.g., `test_profiles.py`, `test_services_hf_client.py`)
- Structure: Separate test file per module or feature

```
backend/tests/
├── __init__.py
├── conftest.py                      # Shared fixtures
├── test_profiles.py                 # Router tests
├── test_servers.py
├── test_system.py
├── test_models.py
├── test_services_server_manager.py  # Service tests
├── test_services_hf_client.py
├── test_services_health_checker.py
├── test_services_launchd.py
├── test_utils_command_builder.py    # Utility tests
├── test_utils_security.py
├── test_fuzzy_matcher.py
└── test_model_detection.py
```

**Frontend:**
- Location: Co-located with source files
- Naming: `{source_name}.test.ts`

```
frontend/src/
├── lib/
│   ├── api/
│   │   ├── client.ts
│   │   └── client.test.ts           # Co-located
│   ├── utils/
│   │   ├── format.ts
│   │   ├── format.test.ts
│   │   ├── reconcile.ts
│   │   └── reconcile.test.ts
│   └── services/
│       ├── polling-coordinator.svelte.ts
│       └── polling-coordinator.test.ts
├── tests/
│   └── setup.ts                     # Global test setup
└── e2e/
    └── (Playwright tests)
```

## Test Structure

**Backend Suite Organization (pytest):**
```python
"""Tests for the server manager service."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from mlx_manager.services.server_manager import ServerManager


@pytest.fixture
def server_manager_instance():
    """Create a fresh ServerManager instance."""
    return ServerManager()


@pytest.fixture
def sample_profile():
    """Create a sample ServerProfile for testing."""
    return ServerProfile(
        id=1,
        name="Test Profile",
        model_path="mlx-community/test-model",
        port=10240,
    )


class TestServerManagerStartServer:
    """Tests for the start_server method."""

    @pytest.mark.asyncio
    async def test_start_already_running_server(self, server_manager_instance, sample_profile):
        """Test starting a server that's already running raises error."""
        # Arrange
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        server_manager_instance.processes[sample_profile.id] = mock_proc

        # Act & Assert
        with pytest.raises(RuntimeError, match="already running"):
            await server_manager_instance.start_server(sample_profile)

    @pytest.mark.asyncio
    async def test_start_server_success(self, server_manager_instance, sample_profile):
        """Test successfully starting a server."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("asyncio.sleep"):
                pid = await server_manager_instance.start_server(sample_profile)

        assert pid == 12345
        assert sample_profile.id in server_manager_instance.processes
```

**Frontend Suite Organization (Vitest):**
```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { profiles, ApiError } from "./client";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

function mockResponse(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  };
}

beforeEach(() => {
  mockFetch.mockReset();
});

describe("profiles API", () => {
  describe("list", () => {
    it("fetches all profiles", async () => {
      const mockProfiles = [{ id: 1, name: "Test" }];
      mockFetch.mockResolvedValueOnce(mockResponse(mockProfiles));

      const result = await profiles.list();

      expect(mockFetch).toHaveBeenCalledWith("/api/profiles");
      expect(result).toEqual(mockProfiles);
    });
  });

  describe("get", () => {
    it("throws ApiError on 404", async () => {
      mockFetch.mockResolvedValueOnce(
        mockErrorResponse("Profile not found", 404)
      );

      await expect(profiles.get(999)).rejects.toThrow(ApiError);
    });
  });
});
```

## Mocking

**Backend (unittest.mock):**
```python
from unittest.mock import MagicMock, AsyncMock, patch

# Patching module-level imports
with patch("mlx_manager.services.hf_client.settings") as mock_settings:
    mock_settings.hf_cache_path = tmp_path
    mock_settings.offline_mode = False

# Patching external services
with patch("httpx.AsyncClient") as mock_client:
    mock_client.return_value.__aenter__.return_value.get = AsyncMock(
        return_value=mock_response
    )

# Creating mock processes
mock_proc = MagicMock()
mock_proc.pid = 12345
mock_proc.poll.return_value = None  # Running
mock_proc.wait.return_value = 0

# Patching psutil
mock_psutil = MagicMock()
mock_psutil.memory_info.return_value.rss = 1024 * 1024 * 512
mock_psutil.cpu_percent.return_value = 15.5
with patch("psutil.Process", return_value=mock_psutil):
```

**Frontend (Vitest vi):**
```typescript
import { vi, beforeEach, afterEach } from "vitest";

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock EventSource for SSE
class MockEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close = vi.fn();
}
global.EventSource = MockEventSource as unknown as typeof EventSource;

// Mock timers for polling tests
beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

// Advance timers
await vi.advanceTimersByTimeAsync(5000);

// Mock document for visibility
const mockDocument = {
  visibilityState: 'visible' as 'visible' | 'hidden',
  addEventListener: vi.fn(),
};
// @ts-expect-error - mocking document
globalThis.document = mockDocument;
```

**What to Mock:**
- External HTTP calls (HuggingFace API, health checks)
- File system operations (model cache paths)
- System resources (psutil, subprocess)
- Timers and intervals
- Browser APIs (fetch, EventSource, document)

**What NOT to Mock:**
- Pure functions (formatters, validators)
- SQLModel/database in integration tests (use in-memory SQLite)
- Core business logic under test

## Fixtures and Factories

**Backend Fixtures (`conftest.py`):**
```python
@pytest.fixture(autouse=True)
def mock_find_mlx_openai_server(request):
    """Mock find_mlx_openai_server globally since it's not available on Linux CI."""
    if "test_utils_command_builder" in request.fspath.basename:
        yield
        return
    with patch(
        "mlx_manager.utils.command_builder.find_mlx_openai_server",
        return_value="/usr/local/bin/mlx-openai-server",
    ):
        yield


@pytest.fixture(scope="function")
async def test_engine():
    """Create a test database engine with in-memory SQLite."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="function")
async def client(test_engine):
    """Create an async test client with test database."""
    # Override dependencies and create httpx AsyncClient
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    return {
        "name": "Test Profile",
        "model_path": "mlx-community/test-model-4bit",
        "port": 10240,
    }


@pytest.fixture
def mock_hf_client():
    """Mock HuggingFace client for testing."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.search_mlx_models = AsyncMock(return_value=[...])
        yield mock
```

**Frontend Test Setup (`src/tests/setup.ts`):**
```typescript
import "@testing-library/svelte/vitest";
import { vi, beforeEach } from "vitest";

// Mock fetch for API calls
global.fetch = vi.fn();

// Mock EventSource for SSE
class MockEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close = vi.fn();
}
global.EventSource = MockEventSource as unknown as typeof EventSource;

// Reset mocks between tests
beforeEach(() => {
  vi.clearAllMocks();
});
```

## Coverage

**Backend Requirements:** ~67% current coverage

| Module | Coverage |
|--------|----------|
| routers/profiles.py | 100% |
| routers/system.py | 92% |
| services/health_checker.py | 100% |
| services/launchd.py | 96% |
| services/hf_client.py | 89% |
| services/server_manager.py | 80% |

**Coverage Exclusions (`pyproject.toml`):**
```toml
[tool.coverage.run]
source = ["mlx_manager"]
omit = [
    "mlx_manager/cli.py",           # Requires interactive terminal
    "mlx_manager/menubar.py",       # macOS GUI-specific
    "mlx_manager/services/manager_launchd.py",  # System-level operations
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

**Frontend Coverage Exclusions (`vitest.config.ts`):**
```typescript
coverage: {
  exclude: [
    "node_modules/**",
    "src/tests/**",
    "**/*.d.ts",
    "**/*.config.*",
    "src/lib/components/ui/**",  // UI primitives
  ],
}
```

**View Coverage:**
```bash
# Backend
pytest --cov=mlx_manager --cov-report=term-missing

# Frontend
npm run test:coverage
```

## Test Types

**Unit Tests:**
- Scope: Individual functions, classes, methods
- Backend: Services, utilities, model validation
- Frontend: API client methods, utility functions, stores
- Isolation: Heavy mocking of external dependencies

**Integration Tests:**
- Scope: API endpoints with database
- Backend: Full request/response cycle with in-memory SQLite
- Use `httpx.AsyncClient` with `ASGITransport` for FastAPI testing

**E2E Tests (Playwright):**
- Scope: Full application flows
- Location: `frontend/e2e/`
- Config: Chrome in CI, Chrome + WebKit locally
- Requires dev server running

## Common Patterns

**Async Testing (Backend):**
```python
@pytest.mark.asyncio
async def test_async_operation(client):
    """Test async endpoint."""
    response = await client.get("/api/profiles")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_service_method(server_manager_instance, sample_profile):
    """Test async service method."""
    result = await server_manager_instance.check_health(sample_profile)
    assert result["status"] == "healthy"
```

**Async Testing (Frontend):**
```typescript
it("fetches all profiles", async () => {
  mockFetch.mockResolvedValueOnce(mockResponse(mockProfiles));
  const result = await profiles.list();
  expect(result).toEqual(mockProfiles);
});
```

**Error Testing (Backend):**
```python
@pytest.mark.asyncio
async def test_get_profile_not_found(client):
    """Test getting a non-existent profile."""
    response = await client.get("/api/profiles/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Profile not found"

@pytest.mark.asyncio
async def test_start_server_failure(server_manager_instance, sample_profile):
    """Test server start failure."""
    with pytest.raises(RuntimeError, match="failed to start"):
        await server_manager_instance.start_server(sample_profile)
```

**Error Testing (Frontend):**
```typescript
it("throws ApiError on 404", async () => {
  mockFetch.mockResolvedValueOnce(mockErrorResponse("Not found", 404));
  await expect(profiles.get(999)).rejects.toThrow(ApiError);
});

it("includes status code and message", async () => {
  mockFetch.mockResolvedValueOnce(mockErrorResponse("Error", 500));
  try {
    await profiles.list();
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(500);
  }
});
```

**Testing with tmp_path (Backend):**
```python
def test_downloaded_with_snapshot(self, hf_client_instance, tmp_path):
    """Test model is downloaded when snapshot exists."""
    model_dir = tmp_path / "models--mlx-community--test-model" / "snapshots"
    model_dir.mkdir(parents=True)
    (model_dir / "abc123").mkdir()

    result = hf_client_instance._is_downloaded("mlx-community/test-model")
    assert result is True
```

**Testing Time-Dependent Code (Frontend):**
```typescript
beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

it("calls refresh at configured interval", async () => {
  coordinator.start("servers");
  expect(refreshFn).toHaveBeenCalledTimes(1);

  await vi.advanceTimersByTimeAsync(5000);
  expect(refreshFn).toHaveBeenCalledTimes(2);
});
```

## Pre-commit Test Hooks

Tests run automatically via pre-commit on push:
```yaml
- repo: local
  hooks:
    - id: backend-tests
      name: backend-tests
      entry: bash -c 'cd backend && source .venv/bin/activate && pytest -x -q'
      language: system
      files: ^backend/
      stages: [pre-push]

    - id: frontend-tests
      name: frontend-tests
      entry: bash -c 'cd frontend && npm run test'
      language: system
      files: ^frontend/
      stages: [pre-push]
```

---

*Testing analysis: 2026-01-16*

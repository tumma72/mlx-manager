# Testing Patterns

**Analysis Date:** 2026-01-27

## Test Framework

**Backend:**
- **Runner:** pytest 8.0.0+
- **Async Support:** pytest-asyncio 0.24.0+ with `asyncio_mode = "auto"`
- **Coverage:** pytest-cov 4.0.0+ with coverage target 95% (fail_under=95)
- **Config:** `backend/pyproject.toml` with `[tool.pytest.ini_options]`
- **Run Commands:**
  ```bash
  pytest -v                                    # Run all tests
  pytest -k test_name                          # Run specific test
  pytest tests/test_profiles.py -v             # Run test file
  pytest --cov=mlx_manager --cov-report=term  # Coverage report
  ```

**Frontend:**
- **Runner:** Vitest 4.0.16
- **Environment:** jsdom (browser simulation)
- **Coverage:** v8 provider with 95% statements/functions/lines, 90% branches
- **Testing Library:** @testing-library/svelte 5.3.1
- **Config:** `frontend/vitest.config.ts`
- **Run Commands:**
  ```bash
  npm run test              # Run all tests
  npm run test:watch       # Watch mode
  npm run test:coverage    # Coverage report
  ```

**E2E Tests:**
- **Framework:** Playwright 1.57.0
- **Config:** `frontend/playwright.config.ts`
- **Run Commands:**
  ```bash
  npm run test:e2e         # Run E2E tests
  npm run test:e2e:ui      # Interactive mode with UI
  ```

## Test File Organization

**Backend:**
- **Location:** `backend/tests/` parallel to source structure
- **Naming:** `test_*.py` for test files, matches module being tested
- **Examples:**
  - `tests/test_profiles.py` → tests `mlx_manager/routers/profiles.py`
  - `tests/test_servers.py` → tests `mlx_manager/routers/servers.py`
  - `tests/conftest.py` → shared fixtures for all tests

**Frontend:**
- **Location:** Co-located with source files using `.test.ts` or `.spec.ts` suffix
- **Pattern:** `src/lib/stores/models.svelte.ts` has `src/lib/stores/models.svelte.test.ts`
- **Utilities:** `src/tests/setup.ts` for global test configuration

## Test Structure

**Backend Test Suite Organization:**

```python
@pytest.mark.asyncio
async def test_list_running_servers_empty(auth_client, mock_server_manager):
    """Test listing running servers when none exist."""
    response = await auth_client.get("/api/servers")
    assert response.status_code == 200
    assert response.json() == []
```

**Patterns:**
- Async tests decorated with `@pytest.mark.asyncio`
- Fixtures injected as function parameters (e.g., `auth_client`, `test_session`, `mock_server_manager`)
- Docstrings describe what is being tested
- One logical assertion per test (avoid combining multiple behaviors)
- Arrange-Act-Assert pattern: setup data → call endpoint → verify response

**Frontend Test Suite Organization:**

```typescript
describe("ModelConfigStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    modelConfigStore.clearAll();
  });

  describe("getConfig", () => {
    it("returns undefined for uncached model", () => {
      const result = modelConfigStore.getConfig("test-model");
      expect(result).toBeUndefined();
    });
  });
});
```

**Patterns:**
- Nested `describe` blocks for logical grouping by functionality
- `beforeEach`/`afterEach` for setup/teardown per test
- Test names start with lowercase and describe behavior (e.g., "returns", "sets", "raises")
- Each test isolated; no test dependencies
- Use `flushSync()` from Svelte to flush reactive state before assertions

## Mocking

**Backend Mocking Framework:** `unittest.mock` (Python stdlib)

**Patterns:**
```python
from unittest.mock import AsyncMock, MagicMock, patch

# Mock async function in router
with patch("mlx_manager.routers.servers.server_manager") as mock:
    mock.start_server = AsyncMock(side_effect=RuntimeError("Server already running"))
    response = await auth_client.post(f"/api/servers/{profile_id}/start")
    assert response.status_code == 409
```

**What to Mock:**
- External services: server subprocess management (`server_manager`)
- System calls: filesystem, launchd operations (`subprocess.Popen`)
- Network calls: HuggingFace Hub API (`huggingface_hub` calls)
- Database: Test database always in-memory (`:memory:` SQLite)
- Health checker: Prevents background tasks during tests

**Global Mocks in conftest.py:**
- `mock_find_mlx_openai_server`: Mocks platform-specific server binary discovery (not available on Linux CI)
  - Skipped for `test_utils_command_builder.py` which tests the function directly

**Frontend Mocking:**

```typescript
vi.mock("$api", () => ({
  models: {
    getConfig: vi.fn(),
  },
}));

import { models as modelsApi } from "$api";

describe("ModelConfigStore", () => {
  it("returns cached state after fetch", async () => {
    vi.mocked(modelsApi.getConfig).mockResolvedValue(mockCharacteristics);
    await modelConfigStore.fetchConfig("test-model");
    const result = modelConfigStore.getConfig("test-model");
    expect(result?.characteristics).toEqual(mockCharacteristics);
  });
});
```

**Global Mocks in setup.ts:**
- `global.fetch`: Mocked for API calls (required for AsyncClient)
- `Element.prototype.scrollIntoView`: Mocked for bits-ui components
- `EventSource`: Mocked for Server-Sent Events (chat streaming)
- `beforeEach`: `vi.clearAllMocks()` resets all mocks between tests

**What to Mock (Frontend):**
- API client calls (mocked in test setup)
- Browser APIs that don't exist in jsdom (scrollIntoView, EventSource)
- Large third-party libraries if unnecessary for behavior testing

## Fixtures and Factories

**Backend Fixtures (conftest.py):**

```python
@pytest.fixture(scope="function")
async def test_session(test_engine):
    """Create a test database session."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session
```

**Fixture Types:**
- `test_engine`: In-memory SQLite database with schema
- `test_session`: Async session for direct database access
- `client`: FastAPI test client with mocked dependencies
- `auth_client`: Authenticated test client (token in headers)
- `sample_profile_data`: Reusable test data dict for ServerProfile
- `mock_server_manager`: Pre-configured mock of ServerManager

**Test Data Location:**
- Fixtures defined in `backend/tests/conftest.py` (shared across all tests)
- Inline test data: Simple JSON dicts in test functions (e.g., `sample_profile_data`)
- Seed data: Database populated in fixtures via SQLModel ORM

## Coverage

**Backend Requirements:**
- Target: 95% (fail_under=95)
- Excluded files (marked in `pyproject.toml`):
  - `mlx_manager/cli.py`: Requires terminal/typer/uvicorn mocking; low value
  - `mlx_manager/menubar.py`: macOS GUI library (rumps) unavailable in CI
  - `mlx_manager/services/manager_launchd.py`: Requires filesystem/launchctl mocking

**Current Coverage by Module:**
- `routers/profiles.py`: 100%
- `routers/system.py`: 92%
- `services/health_checker.py`: 100%
- `services/launchd.py`: 96%
- `services/hf_client.py`: 89%
- `services/server_manager.py`: 80%

**Frontend Requirements:**
- Statements: 95%
- Branches: 90% (lower due to Svelte 5 compiled template artifacts)
- Functions: 95%
- Lines: 95%

**Excluded from Coverage:**
- `src/tests/**`: Test utilities themselves
- `**/*.d.ts`: Type definitions
- `**/*.config.*`: Configuration files
- `src/lib/components/ui/**`: UI primitives (bits-ui wrappers)
- `**/index.ts`: Barrel re-exports

**View Coverage:**
```bash
# Backend
pytest --cov=mlx_manager --cov-report=html
# Open backend/htmlcov/index.html

# Frontend
npm run test:coverage
# Open frontend/coverage/index.html
```

## Test Types

**Unit Tests:**
- **Scope:** Single function/method in isolation
- **Approach (Backend):** Mock all dependencies, test business logic
  - Example: `test_list_profiles()` in `routers/profiles.py` tests query execution with mocked database
- **Approach (Frontend):** Mock API client, test store logic and formatting functions
  - Example: `test_getConfig()` in `models.svelte.test.ts` verifies caching behavior without network calls

**Integration Tests:**
- **Scope:** Multiple components together (e.g., router + database + model validation)
- **Approach:** Use real in-memory database, mock external services only
  - Example: `test_create_profile()` verifies profile creation with database validation, auth checks, and unique constraint enforcement
- **Test Client:** FastAPI AsyncClient with mocked ServiceManager and HealthChecker

**E2E Tests:**
- **Scope:** Full user workflow through browser
- **Approach:** Mock API responses via Playwright route handlers
- **Example:** `e2e/app.spec.ts` mocks API endpoints, logs in user, loads profile list page, verifies rendering
- **Setup:** Uses `page.route()` to intercept API calls and return mock data before navigation

## Common Patterns

**Async Testing (Backend):**

```python
@pytest.mark.asyncio
async def test_start_server(auth_client, sample_profile_data, mock_server_manager):
    """Test starting a server."""
    # Create profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Start server
    response = await auth_client.post(f"/api/servers/{profile_id}/start")
    assert response.status_code == 200
    assert response.json()["pid"] == 12345
```

**Async Testing (Frontend):**

```typescript
it("returns cached state after fetch", async () => {
  const mockCharacteristics = {
    architecture_family: "Llama",
    is_multimodal: false,
    quantization_bits: 4,
  };
  vi.mocked(modelsApi.getConfig).mockResolvedValue(mockCharacteristics);

  await modelConfigStore.fetchConfig("test-model");
  flushSync();  // Force Svelte reactivity update

  const result = modelConfigStore.getConfig("test-model");
  expect(result?.characteristics).toEqual(mockCharacteristics);
});
```

**Error Testing (Backend):**

```python
@pytest.mark.asyncio
async def test_create_profile_duplicate_name(auth_client, sample_profile_data):
    """Test creating profile with duplicate name raises 409."""
    # Create first profile
    await auth_client.post("/api/profiles", json=sample_profile_data)

    # Try to create duplicate
    response = await auth_client.post("/api/profiles", json=sample_profile_data)
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
```

**Error Testing (Frontend):**

```typescript
it("sets error state on fetch failure", async () => {
  vi.mocked(modelsApi.getConfig).mockRejectedValue(
    new Error("Network error")
  );

  await modelConfigStore.fetchConfig("failing-model");
  flushSync();

  const result = modelConfigStore.getConfig("failing-model");
  expect(result?.error).toContain("Network error");
  expect(result?.loading).toBe(false);
});
```

**E2E Test Pattern:**

```typescript
test("profile list loads with mock API", async ({ page }) => {
  // Setup auth before navigation
  await setupAuth(page);

  // Mock API responses
  await page.route("/api/profiles", (route) => {
    route.abort("blockedbyclient");  // Or route.continue with mocked response
  });

  // Navigate and verify
  await page.goto("/profiles");
  await expect(page.locator("text=Test Profile")).toBeVisible();
});
```

## Test Configuration

**Backend (pyproject.toml):**
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.coverage.run]
source = ["mlx_manager"]
omit = ["mlx_manager/cli.py", "mlx_manager/menubar.py", ...]

[tool.coverage.report]
fail_under = 95
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

**Frontend (vitest.config.ts):**
```typescript
test: {
  include: ["src/**/*.{test,spec}.{js,ts}"],
  environment: "jsdom",
  globals: true,
  setupFiles: ["./src/tests/setup.ts"],
  coverage: {
    provider: "v8",
    thresholds: {
      statements: 95,
      branches: 90,
      functions: 95,
      lines: 95,
    },
  },
}
```

---

*Testing analysis: 2026-01-27*

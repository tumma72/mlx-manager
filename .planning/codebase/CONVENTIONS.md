# Coding Conventions

**Analysis Date:** 2026-01-16

## Naming Patterns

**Files:**
- Backend Python: `snake_case.py` (e.g., `server_manager.py`, `hf_client.py`)
- Frontend TypeScript: `kebab-case.ts` or `camelCase.ts` (e.g., `client.ts`, `polling-coordinator.svelte.ts`)
- Svelte components: `PascalCase.svelte` (e.g., `ModelCard.svelte`, `ProfileForm.svelte`)
- Test files: `{source_name}.test.ts` (frontend), `test_{source_name}.py` (backend)

**Functions:**
- Python: `snake_case` (e.g., `start_server`, `get_local_path`, `check_health`)
- TypeScript/JavaScript: `camelCase` (e.g., `handleDownload`, `formatBytes`, `getNextPort`)
- Async functions: Named by action, no special prefix (e.g., `async def start_server`, `async function handleDownload`)

**Variables:**
- Python: `snake_case` (e.g., `profile_id`, `model_path`, `download_tasks`)
- TypeScript: `camelCase` (e.g., `downloadState`, `isDownloaded`, `showDeleteConfirm`)
- Constants: `UPPER_SNAKE_CASE` in Python (e.g., `STATIC_DIR`), `UPPER_SNAKE_CASE` or `camelCase` in TypeScript (e.g., `API_BASE`)

**Types/Classes:**
- Python classes: `PascalCase` (e.g., `ServerManager`, `HuggingFaceClient`, `ServerProfile`)
- Python SQLModel: `PascalCase` with descriptive suffixes (e.g., `ServerProfileCreate`, `ServerProfileUpdate`, `ServerProfileResponse`)
- TypeScript interfaces: `PascalCase` (e.g., `ServerProfile`, `ModelSearchResult`, `HealthStatus`)
- Svelte stores: `camelCase` class instance (e.g., `profileStore`, `serverStore`, `downloadsStore`)

## Code Style

**Formatting:**
- Backend: Ruff formatter with line length 100
- Frontend: Prettier with svelte-plugin
- Both enforced via pre-commit hooks

**Linting:**
- Backend: Ruff with rules `["E", "F", "I", "N", "W", "UP"]`
  - E: pycodestyle errors
  - F: pyflakes
  - I: isort (import sorting)
  - N: pep8-naming
  - W: pycodestyle warnings
  - UP: pyupgrade
- Frontend: ESLint with TypeScript and Svelte plugins
- Type checking: mypy (backend), svelte-check with strict TypeScript (frontend)

**Key Settings:**
- Python target version: 3.11
- TypeScript: strict mode enabled
- Line length: 100 (backend), default Prettier (frontend)

## Import Organization

**Python (enforced by Ruff isort):**
1. Standard library imports
2. Third-party imports
3. Local application imports

Example from `backend/mlx_manager/services/server_manager.py`:
```python
import asyncio
import logging
import signal
import subprocess

import httpx
import psutil

from mlx_manager.models import ServerProfile
from mlx_manager.types import HealthCheckResult, RunningServerInfo, ServerStats
from mlx_manager.utils.command_builder import build_mlx_server_command, get_server_log_path
```

**TypeScript/Svelte:**
1. Framework imports (svelte, sveltekit)
2. External library imports
3. Local imports with path aliases

Path aliases configured:
- `$lib` -> `src/lib`
- `$api` -> `src/lib/api` (inferred from usage)
- `$components` -> `src/lib/components`

Example from `frontend/src/lib/components/models/ModelCard.svelte`:
```typescript
import type { ModelSearchResult } from '$api';
import { models } from '$api';
import { formatNumber, formatBytes } from '$lib/utils/format';
import { Card, Button, Badge, ConfirmDialog } from '$components/ui';
import { Download, Trash2, Check, HardDrive, Heart, ArrowDownToLine } from 'lucide-svelte';
```

## Error Handling

**Backend Patterns:**
- Use `HTTPException` for API errors with appropriate status codes:
  ```python
  raise HTTPException(status_code=404, detail="Profile not found")
  raise HTTPException(status_code=409, detail="Profile name already exists")
  ```
- Use `RuntimeError` for internal service errors that propagate to API:
  ```python
  raise RuntimeError(f"Server for profile {profile.name} is already running")
  ```
- Return explicit error status in typed dicts for health checks:
  ```python
  return HealthCheckResult(status="unhealthy", error=str(e))
  ```
- Log errors with context before raising:
  ```python
  logger.error(f"Server failed to start for profile '{profile.name}' (exit_code={proc.poll()})")
  ```

**Frontend Patterns:**
- Custom `ApiError` class for API failures:
  ```typescript
  class ApiError extends Error {
    constructor(public status: number, message: string) {
      super(message);
      this.name = "ApiError";
    }
  }
  ```
- Centralized response handling in API client:
  ```typescript
  async function handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      // Parse and format error details
      throw new ApiError(response.status, message);
    }
  }
  ```
- Component-level error state:
  ```typescript
  let error = $state<string | null>(null);
  // In catch block:
  error = e instanceof Error ? e.message : 'Operation failed';
  ```

## Logging

**Framework:** Python `logging` module (backend), `console` (frontend)

**Backend Patterns:**
- Module-level logger: `logger = logging.getLogger(__name__)`
- Log levels used appropriately:
  - `INFO`: Operation started/completed (e.g., "Starting server for profile...")
  - `DEBUG`: Detailed internal state (e.g., "Log file: {log_path}")
  - `WARNING`: Non-fatal issues (e.g., "Server already running")
  - `ERROR`: Failures (e.g., "Server failed to start")
- Include context in log messages:
  ```python
  logger.info(f"Starting server for profile '{profile.name}' (id={profile.id})")
  logger.error(f"Server log: {error_msg[:500]}")
  ```
- Third-party loggers suppressed to WARNING:
  ```python
  logging.getLogger("httpx").setLevel(logging.WARNING)
  logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
  ```

**Frontend Patterns:**
- Use `console.warn` and `console.error` for debugging
- Polling coordinator logs errors with context prefix:
  ```typescript
  console.error('[PollingCoordinator] Refresh failed for servers:', error);
  ```

## Comments

**When to Comment:**
- Module-level docstrings explaining purpose
- Class docstrings explaining responsibility
- Function docstrings for public APIs with parameter descriptions
- Inline comments for non-obvious logic or workarounds

**Python Docstrings (Google style):**
```python
"""Server process manager service."""

class ServerManager:
    """Manages mlx-openai-server processes."""

    async def start_server(self, profile: ServerProfile) -> int:
        """Start an mlx-openai-server instance for the given profile."""
```

**TypeScript JSDoc:**
```typescript
/**
 * Profile state management using Svelte 5 runes.
 *
 * Refactored to use:
 * - In-place array reconciliation (prevents unnecessary re-renders)
 * - Polling coordinator for centralized refresh management
 */

/**
 * Custom equality function for ServerProfile.
 * Compares all fields that affect the UI.
 */
function profilesEqual(a: ServerProfile, b: ServerProfile): boolean {
```

## Function Design

**Size:** Keep functions focused on single responsibility. Most are under 30 lines.

**Parameters:**
- Python: Use type hints for all parameters and return types
- TypeScript: Use explicit types, prefer interfaces for complex objects
- Use Optional/nullable types where appropriate: `profile_id: int | None = None`

**Return Values:**
- Backend API endpoints return typed response models or dicts
- Services return typed results (`HealthCheckResult`, `ServerStats`)
- Frontend API client returns typed Promises: `Promise<ServerProfile[]>`

**Async Pattern:**
- Use `async/await` consistently
- Backend: All database operations and HTTP calls are async
- Frontend: All API calls are async

## Module Design

**Backend Exports:**
- Routers exported from `routers/__init__.py`:
  ```python
  from mlx_manager.routers.models import router as models_router
  from mlx_manager.routers.profiles import router as profiles_router
  ```
- Services use singleton pattern with module-level instance:
  ```python
  # At bottom of server_manager.py
  server_manager = ServerManager()
  ```

**Frontend Exports:**
- Barrel files for components: `src/lib/components/ui/index.ts`
- Store instances exported from `src/lib/stores/index.ts`
- API functions grouped by domain in `src/lib/api/client.ts`

**Barrel Files:**
- Use `index.ts` files for convenient imports
- Group related exports:
  ```typescript
  export { profileStore } from './profiles.svelte';
  export { serverStore } from './servers.svelte';
  export { systemStore } from './system.svelte';
  ```

## SQLModel Patterns

**Model Organization:**
- Base model with shared fields: `ServerProfileBase(SQLModel)`
- Table model extends base: `ServerProfile(ServerProfileBase, table=True)`
- Create schema for POST: `ServerProfileCreate(ServerProfileBase)`
- Update schema with all optional: `ServerProfileUpdate(SQLModel)` with `field: type | None = None`
- Response model adds computed fields: `ServerProfileResponse(ServerProfileBase)`

**Field Definitions:**
```python
name: str = Field(index=True)
port: int
description: str | None = None
max_concurrency: int = Field(default=1)
created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
```

## Svelte 5 Patterns

**Runes:**
- State: `let value = $state<Type>(initial)`
- Derived: `let computed = $derived(expression)`
- Effects: `$effect(() => { ... })`

**Component Props:**
```typescript
interface Props {
  model: ModelSearchResult;
  onUse?: (modelId: string) => void;
  onDeleted?: () => void;
}

let { model, onUse, onDeleted }: Props = $props();
```

**Store Classes:**
```typescript
class ProfileStore {
  profiles = $state<ServerProfile[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() { ... }
}

export const profileStore = new ProfileStore();
```

## FastAPI Patterns

**Router Definition:**
```python
router = APIRouter(prefix="/api/profiles", tags=["profiles"])

@router.get("", response_model=list[ServerProfileResponse])
async def list_profiles(session: AsyncSession = Depends(get_db)):
```

**Dependency Injection:**
- Database session: `session: AsyncSession = Depends(get_db)`
- Profile lookup: `profile: ServerProfile = Depends(get_profile_or_404)`

**Response Status Codes:**
- 200: Success (default)
- 201: Created (`status_code=201`)
- 204: No content (delete operations)
- 404: Not found
- 409: Conflict (duplicate name/port)
- 422: Validation error (automatic from Pydantic)

---

*Convention analysis: 2026-01-16*

# Coding Conventions

**Analysis Date:** 2026-01-27

## Naming Patterns

**Files:**
- Python modules use `snake_case` (e.g., `server_manager.py`, `model_detection.py`)
- TypeScript/Svelte files use `kebab-case` for most files, `camelCase` for store files (e.g., `models.svelte.ts`, `auth.svelte.ts`)
- Test files append `.test.ts`, `.spec.ts`, or `.svelte.test.ts` suffix
- Component files in SvelteKit use directory-based routing (e.g., `src/routes/(protected)/profiles/[id]/+page.svelte`)

**Functions:**
- Python: `snake_case` for all functions and methods (e.g., `list_profiles()`, `create_profile()`, `start_server()`)
- TypeScript/JavaScript: `camelCase` for functions (e.g., `formatBytes()`, `getAuthHeaders()`, `handleResponse()`)
- Async functions prefix with `async` keyword in declarations

**Variables:**
- Python: `snake_case` for all variables and constants (e.g., `model_path`, `default_port_start`)
- TypeScript: `camelCase` for variables (e.g., `mockProfiles`, `authToken`)
- Constants in both: SCREAMING_SNAKE_CASE (e.g., `ARCHITECTURE_FAMILIES`, `TOOL_CAPABLE_FAMILIES`)
- Boolean flags use `is_` or `has_` prefixes (e.g., `is_admin`, `is_multimodal`, `has_error`)

**Types:**
- Python: SQLModel classes use `PascalCase` (e.g., `User`, `ServerProfile`, `UserStatus`)
- TypeScript: Interfaces and types use `PascalCase` (e.g., `ConfigState`, `ModelCharacteristics`)
- Pydantic request/response models append suffix: `Create`, `Update`, `Response` (e.g., `ServerProfileCreate`, `ServerProfileUpdate`, `ServerProfileResponse`)

## Code Style

**Formatting:**
- Python: Enforced via `ruff format` with 100-character line length
- TypeScript/JavaScript: Enforced via `prettier` (auto-formatting on save)
- Backend targets Python 3.11+ with strict type hints

**Linting:**
- Python: `ruff` with rules E, F, I, N, W, UP enabled
  - E402 exception per-file: `mlx_manager/main.py` allows module-level imports after logging setup
- TypeScript: ESLint with flat config (`eslint.config.js`)
  - Includes `@eslint/js`, `typescript-eslint`, `svelte/recommended`
  - Special handling for `.svelte.ts` files with `svelteParser` and TypeScript parser
  - Svelte navigation rule configured to allow dynamic path resolution in components

**Type Checking:**
- Python: `mypy` with `warn_return_any=true` and `warn_unused_configs=true`
- TypeScript: `svelte-check` enforces strict TypeScript, `skipLibCheck=false` disabled due to verbosity

## Import Organization

**Order (Python):**
1. Standard library (datetime, asyncio, logging, json, etc.)
2. Third-party packages (fastapi, sqlmodel, httpx, psutil, etc.)
3. Local application imports (mlx_manager modules)
4. Conditional/late imports within function bodies only when necessary (see `routers/profiles.py` for `sqlalchemy.desc` import)

**Order (TypeScript):**
1. Type imports with `type` keyword
2. Named imports from libraries
3. Destructured imports from relative paths (using aliases like `$api`, `$lib`)

**Path Aliases (TypeScript):**
- `$lib` → `src/lib/` (utilities, stores, components, API)
- `$api` → `src/lib/api/` (API client and types)
- Auto-configured by SvelteKit in `svelte.config.js`

**Barrel Exports:**
- `src/lib/stores/index.ts` re-exports all stores for convenience imports
- `src/lib/components/ui/index.ts` re-exports UI primitives
- Barrel files excluded from coverage calculations

## Error Handling

**Python Patterns:**
- HTTP errors: Raise `HTTPException` from FastAPI with explicit `status_code` and `detail` message
  - Example: `raise HTTPException(status_code=404, detail="Profile not found")`
  - Validation errors: `raise HTTPException(status_code=409, detail="Profile name already exists")`
  - Server errors: `raise HTTPException(status_code=500, detail=str(e))`
- Service-level exceptions: Custom exceptions (e.g., `RuntimeError`) raised in services, caught in routers
- Database errors: Logged with `logger.error()` and session rolled back, converted to HTTP responses

**TypeScript Patterns:**
- Custom `ApiError` class extends `Error` with `status` property (see `src/lib/api/client.ts`)
- Response handling: Parse error details from FastAPI validation arrays into human-readable messages
- Auth errors (401): Clear auth state and redirect to login page
- Non-200 responses: Extract detail from JSON or use text fallback

## Logging

**Framework:** Python uses `logging` module with `logger.getLogger(__name__)`

**Patterns:**
- INFO level: Application flow (server startup, operations) - `logger.info(f"Starting server for profile '{profile.name}'")`
- DEBUG level: Detailed diagnostics and internal state - `logger.debug(f"Command: {' '.join(cmd)}")`
- WARNING level: Recoverable issues - `logger.warning(f"Error reading model snapshots: {e}")`
- ERROR level: Failures that need attention - `logger.error(f"Database session error: {e}")`
- Formatted with f-strings, context includes relevant IDs and values

**No logging in TypeScript:** Frontend uses browser console only in development; production errors are silent to not expose internals

## Comments

**When to Comment:**
- Module docstrings: Required on every Python module (triple-quoted, describe purpose)
- Function docstrings: Required on public functions with Args, Returns sections in Google style
- Inline comments: Rare; only for non-obvious logic (e.g., "SQLModel types port as int, but it's a Column at runtime")
- Disabled linting: Document why (see `eslint-disable` comments in `models.svelte.ts`)
- TODOs: Tracked in GitHub Issues, not left in code

**JSDoc/TSDoc:**
- TypeScript: Comments above functions with description (e.g., `/** Get headers with auth token if available. */`)
- Python: Docstrings in Google format with sections for Args, Returns, Raises
- Example from `mlx_manager/utils/model_detection.py`: Clear module docstring explaining "OFFLINE-FIRST detection" approach

## Function Design

**Size:**
- Python: Generally 20-50 lines; larger functions (100+ lines) decomposed into smaller helpers
- TypeScript: Similarly concise; store functions in `*.svelte.ts` average 30-80 lines

**Parameters:**
- Python: Use `Annotated` for FastAPI dependency injection (see `routers/profiles.py`)
  - Pattern: `current_user: Annotated[User, Depends(get_current_user)]`
  - Async session: `session: AsyncSession = Depends(get_db)`
- TypeScript: Pass typed objects rather than many primitives; destructure when needed

**Return Values:**
- Python: Explicit types in function signature (e.g., `async def list_profiles(...) -> list[ServerProfileResponse]`)
- Pydantic models used for all HTTP responses, not raw dicts
- TypeScript: Declared return types on all functions, generics used for API responses

## Module Design

**Exports:**
- Python: No explicit `__all__` used; public functions at module level, private helpers prefixed with `_`
- TypeScript: Explicit exports from modules; stores exported as default and named exports

**Structure - Python Routers:**
- Located in `mlx_manager/routers/`
- Each router defined as `APIRouter(prefix="/api/...", tags=[...])`
- Endpoints marked with `@router.get`, `@router.post`, etc.
- Response models specified with `response_model=SchemaClass`
- Status codes explicit: `status_code=201` for POST, `status_code=204` for DELETE

**Structure - TypeScript Stores:**
- Located in `src/lib/stores/`
- Svelte 5 runes-based (`$state`, `$derived` for reactivity)
- Stores are singletons managing component state
- Fetch/mutation functions are synchronous, async operations handled internally

**Structure - TypeScript API Client:**
- Single `client.ts` file with namespace-organized API methods
- Type definitions in `types.ts` (separate, never executed)
- Auth headers attached to all requests
- Error handling centralized in `handleResponse<T>()`

## Database Models

**Naming:**
- Database table models inherit from `SQLModel` (e.g., `class User(UserBase, table=True)`)
- Separate Pydantic schemas for CRUD operations: `UserCreate`, `UserUpdate`, `UserPublic`
- Database model attributes typed explicitly with `Field()` annotations
- Timestamps use `datetime.now(tz=UTC)` for timezone awareness

**Pattern:**
```python
class UserBase(SQLModel):
    """Base model for users."""
    email: str = Field(unique=True, index=True)

class User(UserBase, table=True):
    """User database model."""
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
```

---

*Convention analysis: 2026-01-27*

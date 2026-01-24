---
phase: 06-bug-fixes-stability
plan: 07
subsystem: api
tags: [mcp, tools, fastapi, openai, function-calling, ast, security]

# Dependency graph
requires:
  - phase: 03-user-authentication
    provides: Authentication system with get_current_user dependency
provides:
  - MCP mock tools endpoint serving get_weather and calculate in OpenAI format
  - Safe tool execution endpoint with AST-based arithmetic evaluation
  - Deterministic mock weather data for testing tool-use models
affects: [chat, profiles, model-testing]

# Tech tracking
tech-stack:
  added: []
  patterns: [ast-based-safe-evaluation, openai-function-calling-format, deterministic-mocking]

key-files:
  created:
    - backend/mlx_manager/routers/mcp.py
    - backend/tests/test_mcp.py
  modified:
    - backend/mlx_manager/routers/__init__.py
    - backend/mlx_manager/main.py

key-decisions:
  - "Use AST parsing for safe calculator (no eval/exec code injection)"
  - "Deterministic mock weather based on location hash for reproducible tests"
  - "OpenAI function-calling format for tool definitions (compatible with mlx-openai-server)"
  - "Both endpoints require authentication via get_current_user dependency"

patterns-established:
  - "Safe expression evaluation: parse to AST, whitelist operators, reject anything else"
  - "Mock tools return deterministic results based on input hash for test reliability"
  - "Tool errors returned as {error: string} not HTTP errors (tool execution succeeded, tool logic failed)"

# Metrics
duration: 3min
completed: 2026-01-24
---

# Phase 6 Plan 7: MCP Mock Service Summary

**MCP mock service with OpenAI-compatible get_weather and calculate tools using safe AST evaluation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-24T11:01:31Z
- **Completed:** 2026-01-24T11:04:41Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- MCP router with two mock tools (get_weather, calculate) in OpenAI function-calling format
- Safe arithmetic calculator using AST parsing (no code injection possible)
- Deterministic mock weather data based on location hash
- Comprehensive test coverage (18 tests) including security validation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create MCP mock router with tool definitions and execution** - `1333d3a` (feat)
2. **Task 2: Add tests for MCP mock endpoints** - `22a4e76` (test)

## Files Created/Modified
- `backend/mlx_manager/routers/mcp.py` - MCP mock tools router with get_weather and calculate tools
- `backend/tests/test_mcp.py` - Comprehensive tests for MCP endpoints (18 test cases)
- `backend/mlx_manager/routers/__init__.py` - Export mcp_router
- `backend/mlx_manager/main.py` - Register MCP router in app

## Decisions Made

**Safe calculator implementation using AST:**
- Used Python's `ast` module to parse expressions into Abstract Syntax Tree
- Whitelisted only safe operators (Add, Sub, Mult, Div, Pow, USub)
- Rejected any unsupported AST nodes (imports, function calls, variables, etc.)
- Returns error dict on failure, not HTTP error (tool execution succeeded, calculation failed)

**Deterministic mock data:**
- Weather results based on `hash(location.lower())` for reproducibility
- Same location always returns same temperature, condition, humidity
- Enables reliable testing of tool-use models

**OpenAI function-calling format:**
- Tool definitions use OpenAI's format (type: function, function: {name, description, parameters})
- Compatible with mlx-openai-server's tool calling implementation
- Execution endpoint takes {name, arguments} and returns result dict

**Authentication requirement:**
- Both `/api/mcp/tools` and `/api/mcp/execute` require authentication
- Uses existing `get_current_user` dependency from auth system

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Type annotations for operator dict:**
- Initial attempt to type SAFE_OPERATORS with specific callable signatures failed
- mypy couldn't match BinOp operators (2 args) vs UnaryOp operators (1 arg) to union type
- Resolved by using `Callable[..., Any]` for operator functions
- Added explicit `float()` conversion and type: ignore for safe operator calls

All quality checks passed (ruff, mypy, pytest).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- MCP mock tools ready for testing tool-use capable models
- Users can configure profiles with `tool_call_parser` and test tool invocation via chat
- Tools provide deterministic, safe execution for validation
- Foundation for potential future expansion to real MCP protocol integration

**Test coverage:**
- All endpoints tested (tools listing, execution)
- Security validated (code injection rejected, auth required)
- Edge cases covered (invalid syntax, empty args, unknown tools)
- Full test suite passes (550 tests)

---
*Phase: 06-bug-fixes-stability*
*Completed: 2026-01-24*

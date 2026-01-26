---
status: resolved
trigger: "Investigate why the tool-use badge doesn't appear on any models, with backend logs showing '404: Config not found' errors."
created: 2026-01-24T10:30:00Z
updated: 2026-01-24T10:30:00Z
---

## Current Focus

hypothesis: CONFIRMED - Frontend fallback parsing missing tool-use detection
test: N/A - root cause confirmed
expecting: N/A - ready to implement fix
next_action: Add tool-use detection to frontend parseCharacteristicsFromName() function

## Symptoms

expected: Tool-use badges should appear on models with tool-calling capabilities
actual: No tool-use badges appear on any models
errors: "Database session error, rolling back: 404: Config not found" repeated for many models
reproduction: Browse models page, observe 404 errors in backend logs for /api/models/config/{model_id}
started: After plan 06-09 implementation

## Eliminated

- hypothesis: Model ID format mismatch (missing org prefix)
  evidence: Backend hf_client.search_mlx_models() returns full model_id from HF API (line 81) which includes org/model format
  timestamp: 2026-01-24T10:40:00Z

## Evidence

- timestamp: 2026-01-24T10:30:00Z
  checked: Backend endpoint /api/models/config/{model_id}
  found: Uses {model_id:path} parameter, accepts tags query param
  implication: Endpoint should accept org/model format with slashes

- timestamp: 2026-01-24T10:32:00Z
  checked: Frontend API client getConfig()
  found: Does NOT use encodeURIComponent, comment says "backend uses {model_id:path}"
  implication: Frontend expects slashes to pass through as literal path segments

- timestamp: 2026-01-24T10:33:00Z
  checked: ModelConfigStore fetchConfig()
  found: Receives modelId and tags from ModelCard component
  implication: Model ID format depends on what ModelCard passes

- timestamp: 2026-01-24T10:35:00Z
  checked: Backend extract_characteristics_from_model()
  found: Calls read_model_config() → get_local_model_path() which builds cache path as "models--{org}--{name}"
  implication: Requires full org/model format to find cached files

- timestamp: 2026-01-24T10:37:00Z
  checked: Backend fetch_remote_config()
  found: Builds URL as "https://huggingface.co/{model_id}/resolve/main/config.json"
  implication: Also requires full org/model format for remote fetch

- timestamp: 2026-01-24T10:42:00Z
  checked: database.py get_db() dependency
  found: Catches ALL exceptions (including HTTPException) and logs "Database session error, rolling back"
  implication: The "Database session error" log is misleading - it's catching HTTPException(404) from get_model_config endpoint

- timestamp: 2026-01-24T10:44:00Z
  checked: Backend search_mlx_models()
  found: Returns ModelSearchResult with full model_id from HF API
  implication: Model IDs ARE in correct format (org/model)

- timestamp: 2026-01-24T10:47:00Z
  checked: Frontend parseCharacteristicsFromName() fallback
  found: Lines 93-128 only detect architecture, quantization, and multimodal - NO tool-use detection
  implication: When API returns 404 (model not cached, remote fetch fails), frontend falls back but loses tool-use detection

- timestamp: 2026-01-24T10:49:00Z
  checked: Backend detect_tool_use() function
  found: Lines 343-388 check tags for "tool-use", "function-calling", "tools", etc. and config for tool_call_parser
  implication: Backend has the logic, frontend fallback is missing it

## Resolution

root_cause: Two issues:
1. Frontend parseCharacteristicsFromName() fallback function missing tool-use detection logic. When API returns 404 (models not locally cached), frontend falls back to name/tag parsing but only detects architecture, quantization, and multimodal - NOT tool-use.
2. Backend database.py get_db() and get_session() log HTTPException as "Database session error" which is misleading.

fix:
1. Added TOOL_USE_PATTERNS to frontend and tool-use detection in parseCharacteristicsFromName()
2. Modified get_db() and get_session() to skip logging HTTPException (not database errors)

verification:
- Type checking passed for both frontend and backend
- Pattern matching tested with 6 test cases - all pass
- Tool-use detection now works in fallback mode when API returns 404
- HTTPException no longer logged as database errors

files_changed:
  - frontend/src/lib/stores/models.svelte.ts (added TOOL_USE_PATTERNS and tool-use detection)
  - backend/mlx_manager/database.py (skip logging HTTPException in get_db() and get_session())

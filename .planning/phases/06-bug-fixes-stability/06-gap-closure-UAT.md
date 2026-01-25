---
status: diagnosed
phase: 06-bug-fixes-stability
source: 06-14-SUMMARY.md, 06-15-SUMMARY.md, 06-16-SUMMARY.md
started: 2026-01-24T21:00:00Z
updated: 2026-01-25T10:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. No console errors during server startup
expected: Start a server profile from the UI. Open browser DevTools Console. During model loading, you should NOT see any "Failed to load resource" or "Could not connect to the server" errors for /v1/models. The tile should show "Starting" and transition to "Running" cleanly.
result: pass

### 2. Text file attachments accepted (.log, .md, .yaml, .toml)
expected: In chat, attach a text file with one of these extensions: .log, .md, .yaml, .yml, or .toml. The file should be accepted without error and show as an attachment. When sent, the model receives the file content as text (not base64 image).
result: issue
reported: "Files without extensions (e.g., README, Makefile, Dockerfile, LICENSE) are rejected even though they are valid text files. Extension-based detection fails for extensionless files."
severity: major

### 3. Tool-use badge on capable models
expected: Browse models panel. Models with tool-use/function-calling tags (e.g., Qwen function-calling models) should show an amber "Tool Use" badge with wrench icon on their tile.
result: issue
reported: "Two problems: (1) 404 console errors for every model when browsing — config.json fetches failing and showing in browser console. (2) Badge detection unreliable — local models known to support tool use (Qwen3, GLM-4.7, MiniMax) show no badge because detection only relies on HuggingFace config.json tags which many model creators don't set. Need a better detection approach."
severity: major

### 4. Tool calls displayed in collapsible panel
expected: In chat with MCP tools enabled, trigger a tool call (e.g., ask about weather or a calculation). Tool calls should appear in a collapsible panel with wrench icon and amber border — NOT as bold inline markdown. Arguments should be code-formatted. Results should have green background.
result: pass

### 5. Memory display uses GB units
expected: Start a server with a loaded model. The server tile's memory gauge should display values in GB when the value is large (e.g., "16.9 GB" not "17307 MB").
result: pass

## Summary

total: 5
passed: 3
issues: 2
pending: 0
skipped: 0

## Gaps

- truth: "Text files without extensions (README, Makefile, Dockerfile, LICENSE) accepted as attachments"
  status: diagnosed
  reason: "User reported: Files without extensions (e.g., README, Makefile, Dockerfile, LICENSE) are rejected even though they are valid text files. Extension-based detection fails for extensionless files."
  severity: major
  test: 2
  root_cause: "Code uses file.name.split('.').pop() which returns the filename itself when there's no extension (e.g., 'README' → ext='readme'). Since 'readme' is not in TEXT_EXTENSIONS set, file is rejected. Need allowlist of known extensionless text filenames."
  artifacts:
    - frontend/src/routes/(protected)/chat/+page.svelte:143-144
  missing:
    - Allowlist of known extensionless text filenames (README, Makefile, Dockerfile, LICENSE, Procfile, Gemfile, Brewfile, etc.)
    - Detection logic for files where split('.').length === 1
  debug_session: ""

- truth: "Tool-use badge reliably shown on capable models without console errors"
  status: diagnosed
  reason: "User reported: (1) 404 console errors for every model when browsing — config.json fetches failing in browser. (2) Badge detection unreliable — Qwen3, GLM-4.7, MiniMax show no badge because detection relies solely on HuggingFace config.json tags which many model creators don't set."
  severity: major
  test: 3
  root_cause: |
    Two sub-issues:
    A) 404 console errors: Browser logs network failures BEFORE JS catch block handles them. Frontend fetchConfig() catches the error gracefully but browser console still shows the 404. Need backend to return 204/null instead of 404 for missing configs, or pre-filter which models to fetch configs for.
    B) Unreliable detection: TOOL_USE_PATTERNS only checks explicit keywords (tool-use, function-calling, tools). Doesn't recognize known tool-capable model families. Need model family allowlist approach: Qwen (all), GLM-4, MiniMax, DeepSeek, Hermes support tool use regardless of HF tags.
  artifacts:
    - frontend/src/lib/stores/models.svelte.ts:92-97 (TOOL_USE_PATTERNS too narrow)
    - frontend/src/lib/stores/models.svelte.ts:178-185 (fetch causes browser 404 log)
    - backend/mlx_manager/routers/models.py (returns 404 for missing configs)
  missing:
    - Model family allowlist for tool-use detection (Qwen, GLM-4, MiniMax, DeepSeek, Hermes, etc.)
    - Backend change: return null/204 instead of 404 for missing model configs
  debug_session: ""

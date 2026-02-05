# Phase 13: MLX Server Integration - UAT Report

**Date:** 2026-02-02
**Tester:** User
**Status:** Issues Fixed

## Summary

6 issues identified during user acceptance testing of the MLX Server integration.
All issues have been fixed.

## Issues

### BUG-13-01: Thinking/Reasoning Content Not Displayed Correctly ✅ FIXED
**Severity:** High
**Component:** Frontend Chat + Backend Inference

**Description:**
When using Qwen3 (a reasoning model), the `<think>` tags content is displayed inline in the chat bubble instead of being extracted and shown in a separate thinking block.

**Expected:** Thinking content should be extracted by the server into `reasoning_content` field and displayed by frontend in a collapsible/distinct thinking block.

**Actual:** Thinking content appears inline with the response text.

**Root Cause:** The tag parsing in `chat.py` checked for complete `<think>` tags within a single SSE chunk, but when streaming token-by-token, tags are split across chunks (e.g., `<th` in one chunk, `ink>` in another).

**Fix:** Implemented buffered tag detection in `backend/mlx_manager/routers/chat.py` that accumulates content and correctly detects tags split across chunks.

---

### BUG-13-02: Tool Calling Not Working ✅ FIXED
**Severity:** High
**Component:** Backend Chat Endpoint + Inference Service

**Description:**
With tools enabled in chat, the model doesn't see the tools and can't respond to tool-requiring requests (e.g., weather query).

**Expected:** Tools should be injected into the prompt, model should recognize them and generate tool calls.

**Actual:** Model responds as if no tools are available. Server log shows `tools=no`.

**Root Cause:** `chat.py` was not passing `request.tools` to `generate_chat_completion()`.

**Fix:** Added `tools=request.tools` parameter to the `generate_chat_completion()` call in `backend/mlx_manager/routers/chat.py:118`.

---

### BUG-13-03: API Key Decryption Failing (InvalidToken) ✅ FIXED
**Severity:** High
**Component:** Encryption Service

**Description:**
When clicking "Test Connection" for OpenAI/Anthropic providers, server throws `cryptography.fernet.InvalidToken` error.

**Error Stack:**
```
File "mlx_manager/services/encryption_service.py", line 93, in decrypt_api_key
    return f.decrypt(encrypted_key.encode()).decode()
cryptography.fernet.InvalidToken
```

**Expected:** Previously saved API keys should decrypt successfully.

**Actual:** Decryption fails with signature mismatch.

**Root Cause:** API keys were encrypted with a different salt/jwt_secret combination. Old keys cannot be decrypted with the new configuration.

**Fix:** Added graceful error handling in `backend/mlx_manager/routers/settings.py` that catches `InvalidToken` exception and returns a user-friendly error message asking to re-enter the API key.

---

### BUG-13-04: Audit Logs Fetch Failing ✅ FIXED
**Severity:** Medium
**Component:** System Router

**Description:**
Console shows repeated warnings: `Failed to fetch audit logs: All connection attempts failed`

**Expected:** Audit logs should be fetched from database successfully.

**Actual:** Connection attempts fail.

**Root Cause:** `system.py` was proxying audit logs to `localhost:10242` but the embedded MLX Server runs on port 10242 at `/v1` prefix.

**Fix:** Updated `MLX_SERVER_URL` and WebSocket URL in `backend/mlx_manager/routers/system.py` to use `http://localhost:10242/v1` and `ws://localhost:10242/v1` respectively.

---

### BUG-13-05: Settings Menu Doesn't Close ✅ FIXED
**Severity:** Low
**Component:** Frontend Navigation

**Description:**
When clicking Settings in the user dropdown menu, the Settings page loads but the dropdown menu remains open.

**Expected:** Menu should close when an item is clicked.

**Actual:** Menu stays open.

**Root Cause:** The Settings and Users links used `<a>` tags inside the dropdown, which don't trigger the menu close behavior. The Logout button correctly used `DropdownMenu.Item` with `onSelect`.

**Fix:** Converted Settings and Users links from `<a>` tags to `DropdownMenu.Item` components with `onSelect={() => goto(resolve('/path'))}` in `frontend/src/lib/components/layout/Navbar.svelte`.

---

### BUG-13-06: Profile Status Not Updating Immediately ✅ FIXED
**Severity:** Low
**Component:** Frontend Profiles Page

**Description:**
After loading a model via chat, the Profiles panel doesn't show the "Running" status until navigating away and back.

**Expected:** Profile status should update reactively when model loads.

**Actual:** Status only updates after navigation refresh.

**Root Cause:** The serverStore only polls every 5 seconds, but when a model loads via chat (not via Profile Start button), the status update wasn't triggered immediately.

**Fix:** Added `serverStore.refresh()` call in `frontend/src/routes/(protected)/chat/+page.svelte` when the first content chunk is received (which indicates the model is loaded).

---

## Priority Order

1. BUG-13-03 (API Key Decryption) - Blocking provider testing
2. BUG-13-01 (Thinking Display) - Core feature broken
3. BUG-13-02 (Tool Calling) - Core feature broken
4. BUG-13-04 (Audit Logs) - Feature broken
5. BUG-13-05 (Settings Menu) - UX polish
6. BUG-13-06 (Profile Status) - UX polish

## Resolution

All 6 issues have been fixed:

| Bug | Status | Fix Location |
|-----|--------|--------------|
| BUG-13-01 | ✅ Fixed | `backend/mlx_manager/routers/chat.py` - buffered tag detection |
| BUG-13-02 | ✅ Fixed | `backend/mlx_manager/routers/chat.py:118` - pass tools to inference |
| BUG-13-03 | ✅ Fixed | `backend/mlx_manager/routers/settings.py` - graceful InvalidToken handling |
| BUG-13-04 | ✅ Fixed | `backend/mlx_manager/routers/system.py` - correct embedded server URLs |
| BUG-13-05 | ✅ Fixed | `frontend/src/lib/components/layout/Navbar.svelte` - DropdownMenu.Item |
| BUG-13-06 | ✅ Fixed | `frontend/src/routes/(protected)/chat/+page.svelte` - serverStore refresh |

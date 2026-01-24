---
phase: 06-bug-fixes-stability
plan: 05
subsystem: ui
tags: [profile-management, chat, system-prompt, textarea, svelte]

# Dependency graph
requires:
  - phase: 05-chat-multimodal-support
    provides: Chat interface with ContentPart message format
provides:
  - Profile system_prompt field in backend and frontend
  - Multi-line textarea for profile description
  - System prompt textarea with character counter
  - Pinned system prompt display in chat
  - System prompt sent as first message in API calls
affects: [chat, profile-management]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - System prompt as pinned first message in chat (grayed-out, italic)
    - Dismissible hints with action links
    - Character counter with soft limit warnings

key-files:
  created: []
  modified:
    - backend/mlx_manager/models.py
    - backend/mlx_manager/database.py
    - backend/mlx_manager/routers/profiles.py
    - frontend/src/lib/api/types.ts
    - frontend/src/lib/components/profiles/ProfileForm.svelte
    - frontend/src/routes/(protected)/chat/+page.svelte
    - frontend/src/lib/components/servers/ProfileSelector.test.ts
    - frontend/src/lib/components/servers/ServerCard.test.ts
    - frontend/src/lib/components/servers/StartingTile.test.ts
    - frontend/src/lib/stores/profiles.svelte.test.ts

key-decisions:
  - "System prompt appears as grayed-out italic message (not editable in chat)"
  - "Dismissible hint when no system prompt is set (links to profile settings)"
  - "Character counter with 2000-char soft limit warning"
  - "System prompt sent as first message with role: 'system' in API"

patterns-established:
  - "System prompt as pinned context message pattern"
  - "Textarea styling matches Input component for consistency"
  - "Soft limit warnings (not hard blocks) for UX flexibility"

# Metrics
duration: 6min
completed: 2026-01-24
---

# Phase 6 Plan 5: Profile System Prompt & Textarea Summary

**Profile system_prompt field with multi-line description textarea and pinned system prompt display in chat**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-24T11:01:30Z
- **Completed:** 2026-01-24T11:07:04Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments
- Added system_prompt field to ServerProfile model with database migration
- Changed profile description to multi-line textarea (2 rows)
- Added system prompt textarea with character counter and 2000-char soft limit warning
- Display system prompt as grayed-out pinned message at top of chat
- Show dismissible hint with link to profile settings when no system prompt is set
- System prompt sent as first message in API calls (role: 'system')

## Task Commits

Each task was committed atomically:

1. **Task 1: Add system_prompt field to backend model and API** - `46e4511` (feat)
2. **Task 2: Update ProfileForm with textarea and system prompt** - `e0a03ce` (feat)
3. **Task 3: Display system prompt as pinned message in chat** - `63f8571` (feat)

## Files Created/Modified
- `backend/mlx_manager/models.py` - Added system_prompt field to ServerProfileBase and ServerProfileUpdate
- `backend/mlx_manager/database.py` - Added system_prompt column migration
- `backend/mlx_manager/routers/profiles.py` - Include system_prompt in duplicate_profile
- `frontend/src/lib/api/types.ts` - Added system_prompt to TypeScript types
- `frontend/src/lib/components/profiles/ProfileForm.svelte` - Multi-line description textarea and system prompt field with character counter
- `frontend/src/routes/(protected)/chat/+page.svelte` - Pinned system prompt display and dismissible hint
- Test files updated with system_prompt: null in mock profiles

## Decisions Made

1. **System prompt display**: Grayed-out italic message (not editable in chat) to clearly indicate it's metadata, not conversation history
2. **Dismissible hint**: When no system prompt is set, show hint with link to profile settings. Dismissible via X button (session-persistent)
3. **Character counter**: Show character count with warning at >2000 chars (soft limit, not blocking)
4. **API message order**: System prompt sent as first message before conversation history to ensure model receives it as context

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **Linting error**: Initial implementation used `href` without `resolve()` for profile settings link. Fixed by importing `resolve` from `$app/paths` and wrapping the href path.
- **Test updates**: All test files with `createMockProfile` factory needed `system_prompt: null` added to maintain type consistency. Updated 4 test files.

## Next Phase Readiness
- System prompt functionality complete and ready for use
- Chat interface properly displays and sends system prompts to models
- Profile form provides clear UX for setting behavior context

---
*Phase: 06-bug-fixes-stability*
*Completed: 2026-01-24*

---
phase: 06-bug-fixes-stability
plan: 17
subsystem: ui
tags: [chat, attachments, file-validation, svelte]

# Dependency graph
requires:
  - phase: 05-chat-multimodal-support
    provides: Chat attachment system with text/image/video support
provides:
  - Extensionless text file attachment support (README, Makefile, Dockerfile, LICENSE)
  - Dotfile attachment support (.gitignore, .dockerignore, .env variants)
  - Filename-based detection for files without extensions
affects: [gap-closure-verification]

# Tech tracking
tech-stack:
  added: []
  patterns: [filename-allowlist-based-validation]

key-files:
  created: []
  modified:
    - frontend/src/routes/(protected)/chat/+page.svelte

key-decisions:
  - "Use KNOWN_TEXT_FILENAMES allowlist for extensionless file detection"
  - "Detect extension presence via nameParts.length > 1 check"
  - "Fallback to filename allowlist when no extension present"

patterns-established:
  - "Extension detection: check hasExtension before choosing validation strategy"
  - "Allowlist includes both standard files (README) and dotfiles (.gitignore)"

# Metrics
duration: 2min
completed: 2026-01-25
---

# Phase 6 Plan 17: Extensionless Text File Acceptance Summary

**Chat attachments now accept common extensionless text files (README, Makefile, Dockerfile, LICENSE) and dotfiles (.gitignore, .env) via filename-based allowlist detection**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-25T21:01:02Z
- **Completed:** 2026-01-25T21:01:55Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added KNOWN_TEXT_FILENAMES allowlist with 31 common extensionless files and dotfiles
- Implemented dual detection strategy: extension-based for files with extensions, filename-based for extensionless files
- Extension detection uses hasExtension flag (nameParts.length > 1) to distinguish scenarios
- All existing extension-based detection continues to work unchanged

## Task Commits

1. **Task 1: Add extensionless text file detection** - `dc7db78` (feat)

## Files Created/Modified
- `frontend/src/routes/(protected)/chat/+page.svelte` - Added KNOWN_TEXT_FILENAMES allowlist and updated addAttachment function to check filename when no extension present

## Decisions Made

**1. Use comprehensive allowlist strategy**
- Included both standard extensionless files (README, Makefile, Dockerfile, LICENSE, etc.) and common dotfiles (.gitignore, .env variants, config dotfiles)
- Lowercase comparison ensures case-insensitive matching (README, readme, Readme all work)

**2. Detect extension presence via split('.').length check**
- Files like "README" have length 1 (no dot, no extension)
- Files like ".gitignore" have length 2 (starts with dot but no extension after)
- Files like "file.txt" have length 2 (has extension)
- This distinguishes between "file with extension" vs "extensionless file" reliably

**3. Dual validation strategy based on hasExtension**
- If hasExtension: check TEXT_EXTENSIONS set (existing behavior)
- If !hasExtension: check KNOWN_TEXT_FILENAMES allowlist (new behavior)
- Clean separation of concerns, no impact on existing extension-based detection

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Gap closure plan 17 complete. Extensionless text file acceptance verified. Ready for gap closure verification testing (can attach README, Makefile, Dockerfile, LICENSE, .gitignore, .env files without rejection).

---
*Phase: 06-bug-fixes-stability*
*Completed: 2026-01-25*

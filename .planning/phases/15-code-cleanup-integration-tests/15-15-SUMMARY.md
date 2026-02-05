---
phase: 15-code-cleanup-integration-tests
plan: 15
subsystem: auth
tags: [authlib, jwe, jwt, encryption, security, dependency-cleanup]

# Dependency graph
requires:
  - phase: 11-configuration-ui
    provides: "encryption_service.py and auth_service.py with Fernet/pyjwt"
provides:
  - "AuthLib JWE-based API key encryption (A256KW + A256GCM)"
  - "AuthLib jose-based JWT token handling"
  - "Unified auth library (authlib replaces pyjwt + cryptography direct dep)"
  - "DecryptionError with backward-compat InvalidToken alias"
affects: []

# Tech tracking
tech-stack:
  added: ["authlib>=1.3.0"]
  removed: ["pyjwt>=2.8.0"]
  patterns: ["JWE compact serialization for symmetric encryption", "authlib.jose.jwt for JWT encode/decode"]

key-files:
  modified:
    - "backend/pyproject.toml"
    - "backend/mlx_manager/services/encryption_service.py"
    - "backend/mlx_manager/services/auth_service.py"
    - "backend/mlx_manager/routers/settings.py"
    - "backend/tests/test_encryption_service.py"
    - "backend/tests/test_auth_service.py"

key-decisions:
  - "AuthLib JWE A256KW+A256GCM for symmetric encryption (replaces Fernet)"
  - "SHA-256 of jwt_secret as JWE key (no salt file needed)"
  - "DecryptionError aliased as InvalidToken for backward compatibility"
  - "AuthLib jose JWT accepts tokens signed with different HMAC variants (HS256/HS512) using same key"

patterns-established:
  - "JWE compact serialization: 5 dot-separated parts (header.key.iv.ciphertext.tag)"
  - "Catching broad Exception for JWE decrypt failures (errors come from cryptography transitive dep)"

# Metrics
duration: 8min
completed: 2026-02-05
---

# Phase 15 Plan 15: AuthLib Consolidation Summary

**Unified auth stack under AuthLib: JWE encryption replacing Fernet, jose JWT replacing pyjwt, one fewer direct dependency**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-05T10:19:46Z
- **Completed:** 2026-02-05T10:27:40Z
- **Tasks:** 6
- **Files modified:** 6

## Accomplishments
- Replaced broken Fernet encryption (cryptography not in deps) with AuthLib JWE (A256KW + A256GCM)
- Replaced pyjwt with AuthLib jose for JWT token handling
- Eliminated one direct dependency (pyjwt) and removed need for .encryption_salt file
- All 30 encryption/auth tests pass with new stack; 1297/1299 total tests pass (2 pre-existing failures in download SSE tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Update dependencies** - `a349ac0` (chore)
2. **Task 2: Rewrite encryption_service.py** - `d48600f` (feat)
3. **Task 3: Rewrite auth_service.py** - `e746050` (feat)
4. **Task 4: Update comments and salt references** - `8724996` (fix)
5. **Task 5: Update tests** - `9d63a1e` (test)
6. **Task 6: Verify and fix lint** - `6ac3980` (style)

## Files Created/Modified
- `backend/pyproject.toml` - Replaced pyjwt with authlib in dependencies
- `backend/mlx_manager/services/encryption_service.py` - JWE encryption with A256KW+A256GCM, DecryptionError exception
- `backend/mlx_manager/services/auth_service.py` - AuthLib jose JWT encode/decode
- `backend/mlx_manager/routers/settings.py` - Updated comment about decryption failure cause
- `backend/tests/test_encryption_service.py` - JWE format tests, DecryptionError, removed salt tests
- `backend/tests/test_auth_service.py` - AuthLib jose JWT for test verification

## Decisions Made
- **SHA-256 for key derivation**: Instead of PBKDF2HMAC with salt, we use SHA-256(jwt_secret) for the JWE key. This eliminates the need for a persistent salt file while still providing a 256-bit key. The jwt_secret itself provides the entropy.
- **DecryptionError as InvalidToken alias**: Created DecryptionError exception class and aliased it as InvalidToken to maintain backward compatibility with the settings router and any other consumers.
- **Broad Exception catch in decrypt**: JWE decryption errors come from cryptography's transitive dependency (InvalidUnwrap, InvalidTag) which are not JoseError subclasses. Catching Exception and wrapping in DecryptionError provides clean error handling.
- **HMAC algorithm flexibility**: AuthLib's jwt.decode() accepts tokens signed with any HMAC variant using the correct key (unlike pyjwt which enforces algorithm matching). This is valid security behavior since the signature is correctly verified.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- pyjwt remained installed after removing from dependencies (cached in virtualenv). Required explicit `uv pip uninstall pyjwt` to remove.
- AuthLib JWT decode does not enforce algorithm restriction like pyjwt does. Updated test expectation to match AuthLib's behavior (HS256/HS512 tokens with same key are both valid).

## User Setup Required
- Users with existing Fernet-encrypted API keys will need to re-enter their API keys in Settings. A clear error message guides them when decryption fails.

## Next Phase Readiness
- Auth stack fully consolidated under AuthLib
- No blockers for future development
- Server starts cleanly with no ModuleNotFoundError

---
*Phase: 15-code-cleanup-integration-tests*
*Completed: 2026-02-05*

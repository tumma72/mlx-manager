# Phase 3: User-Based Authentication - Context

**Gathered:** 2026-01-20
**Status:** Ready for planning

<domain>
## Phase Boundary

User registration and login with email+password, JWT-based session management. First user becomes admin and can approve/manage other users. All authenticated users have equal access to app features.

**Scope change:** Originally scoped as simple API key auth, redefined to user-based auth for better security and traceability when exposing to local network.

</domain>

<decisions>
## Implementation Decisions

### Registration flow
- Email + password only (no username required)
- No email verification (avoids SMTP complexity)
- First registered user automatically becomes admin
- Subsequent users must request an account and await admin approval

### Session handling
- JWT tokens for authentication
- 7-day session duration
- Token stored in frontend (localStorage)

### Login UX
- Dedicated /login page (not modal overlay)
- Unauthenticated users redirected to login
- After registration request, show "waiting for approval" page

### Admin capabilities
- User management only (create, approve, delete users, reset passwords)
- All users have equal access to app features (no feature-level permissions)
- Admin can delete own account only if another admin exists
- Can promote other users to admin

### Admin UI
- Dedicated /users page in navigation (admin-only visibility)
- Badge on nav item showing count of pending approval requests

### Claude's Discretion
- Password complexity requirements
- JWT token structure and refresh strategy
- Exact password hashing algorithm (bcrypt recommended)
- Database schema for users table
- Error message wording

</decisions>

<specifics>
## Specific Ideas

- Goal is to support running on 0.0.0.0 (local network access) with proper access control
- "First user is admin" pattern avoids bootstrap complexity
- Keep it simple for a local-first tool — no email infrastructure

</specifics>

<deferred>
## Deferred Ideas

- Named user accounts for traceability/audit logging — future enhancement
- Network binding configuration (0.0.0.0 vs localhost toggle) — separate phase
- OAuth2 flows (Google, GitHub login) — future phase
- Email-based password recovery — requires SMTP setup, defer

</deferred>

---

*Phase: 03-user-authentication*
*Context gathered: 2026-01-20*

# Phase 3: User-Based Authentication - Research

**Researched:** 2026-01-20
**Domain:** FastAPI JWT authentication, SQLModel user models, SvelteKit SPA auth
**Confidence:** HIGH

## Summary

This phase implements user registration and login with email+password, JWT-based sessions, and an admin approval flow. The research covers the standard Python authentication stack (PyJWT + pwdlib with Argon2), FastAPI security patterns using dependency injection, and SvelteKit client-side authentication for the existing SPA architecture.

Key findings:
- FastAPI's official documentation now recommends **PyJWT** (not python-jose) for JWT and **pwdlib with Argon2** for password hashing
- The frontend uses `adapter-static` (pure SPA), so authentication must be client-side with localStorage JWT storage and API-driven validation
- SQLModel schemas should separate concerns: UserCreate (with password), User (with hashed_password), UserPublic (no password)
- First-user-becomes-admin pattern is standard for bootstrap-free local applications

**Primary recommendation:** Use FastAPI dependency injection for route protection (not middleware), PyJWT for tokens, pwdlib[argon2] for password hashing, and client-side route guards in SvelteKit layout files.

## Standard Stack

The established libraries/tools for this domain.

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyJWT | >=2.8.0 | JWT encode/decode | FastAPI official recommendation (python-jose abandoned) |
| pwdlib[argon2] | >=0.3.0 | Password hashing | Modern passlib replacement, Argon2 is PHC winner |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic-settings | >=2.0.0 | JWT secret management | Already in project, use for auth secrets |
| email-validator | >=2.0.0 | Email format validation | Optional, Pydantic can validate with this |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyJWT | python-jose | python-jose abandoned (last release 3 years ago) |
| PyJWT | authlib/joserfc | More features (JWE) but heavier, overkill for this use case |
| pwdlib | passlib | passlib has Python 3.13+ compatibility issues |
| pwdlib | bcrypt directly | pwdlib provides cleaner API and Argon2 default |
| Argon2 | bcrypt | bcrypt still secure but Argon2 is modern standard |

**Installation:**

```bash
# Backend
pip install pyjwt "pwdlib[argon2]"
```

No frontend packages needed - uses native fetch with Authorization header.

## Architecture Patterns

### Recommended Project Structure

```
backend/mlx_manager/
├── routers/
│   └── auth.py           # /api/auth/* endpoints (login, register, users)
├── services/
│   └── auth_service.py   # Password hashing, JWT creation/validation
├── models.py             # Add User, UserCreate, UserPublic, etc.
├── dependencies.py       # Add get_current_user, get_admin_user
└── database.py           # No changes needed (existing pattern works)

frontend/src/
├── lib/
│   ├── stores/
│   │   └── auth.svelte.ts    # Auth state, token management
│   └── api/
│       └── client.ts         # Add auth header injection
├── routes/
│   ├── (protected)/          # Route group for authenticated pages
│   │   ├── +layout.ts        # Auth guard, ssr=false
│   │   ├── servers/+page.svelte
│   │   ├── models/+page.svelte
│   │   └── ...
│   ├── (public)/             # Route group for auth pages
│   │   ├── +layout.ts        # ssr=false
│   │   └── login/+page.svelte
│   └── users/+page.svelte    # Admin-only user management
└── hooks.client.ts           # Optional: global 401 handling
```

### Pattern 1: FastAPI Dependency Injection for Auth

**What:** Use `Depends()` to inject authenticated user into route handlers
**When to use:** All protected API endpoints

```python
# Source: FastAPI official documentation
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        email: str | None = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception

    result = await session.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    if user is None or not user.is_approved:
        raise credentials_exception
    return user

async def get_admin_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Usage in routes
@router.get("/api/profiles")
async def list_profiles(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db)
):
    # Only authenticated users reach here
    ...

@router.get("/api/users")
async def list_users(
    admin: Annotated[User, Depends(get_admin_user)],
    session: AsyncSession = Depends(get_db)
):
    # Only admin users reach here
    ...
```

### Pattern 2: SQLModel User Schema Separation

**What:** Separate models for different operations to control field exposure
**When to use:** Always for user models with sensitive data

```python
# Source: SQLModel official documentation
from datetime import UTC, datetime
from enum import Enum
from sqlmodel import Field, SQLModel

class UserStatus(str, Enum):
    PENDING = "pending"      # Awaiting admin approval
    APPROVED = "approved"    # Can access the app
    DISABLED = "disabled"    # Account disabled by admin

class UserBase(SQLModel):
    """Shared fields for all user schemas."""
    email: str = Field(unique=True, index=True)

class User(UserBase, table=True):
    """Database model - includes sensitive fields."""
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    hashed_password: str
    is_admin: bool = Field(default=False)
    status: UserStatus = Field(default=UserStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    approved_at: datetime | None = None
    approved_by: int | None = Field(default=None, foreign_key="users.id")

class UserCreate(SQLModel):
    """Schema for registration - accepts plain password."""
    email: str
    password: str

class UserPublic(UserBase):
    """Response model - no password."""
    id: int
    is_admin: bool
    status: UserStatus
    created_at: datetime

class UserLogin(SQLModel):
    """Schema for login request."""
    email: str
    password: str

class Token(SQLModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
```

### Pattern 3: SvelteKit SPA Client-Side Auth Guard

**What:** Route protection using layout files with localStorage JWT
**When to use:** All protected routes in SPA mode

```typescript
// Source: SvelteKit SPA patterns
// src/lib/stores/auth.svelte.ts
function createAuthStore() {
    const TOKEN_KEY = 'mlx_auth_token';
    const USER_KEY = 'mlx_auth_user';

    let token = $state<string | null>(null);
    let user = $state<User | null>(null);
    let loading = $state(true);

    // Initialize from localStorage (client-side only)
    if (typeof window !== 'undefined') {
        token = localStorage.getItem(TOKEN_KEY);
        const userJson = localStorage.getItem(USER_KEY);
        if (userJson) {
            try {
                user = JSON.parse(userJson);
            } catch {
                localStorage.removeItem(USER_KEY);
            }
        }
        loading = false;
    }

    return {
        get token() { return token; },
        get user() { return user; },
        get loading() { return loading; },
        get isAuthenticated() { return !!token && !!user; },
        get isAdmin() { return user?.is_admin ?? false; },

        setAuth(newToken: string, newUser: User) {
            token = newToken;
            user = newUser;
            localStorage.setItem(TOKEN_KEY, newToken);
            localStorage.setItem(USER_KEY, JSON.stringify(newUser));
        },

        clearAuth() {
            token = null;
            user = null;
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USER_KEY);
        }
    };
}

export const authStore = createAuthStore();

// src/routes/(protected)/+layout.ts
import { redirect } from '@sveltejs/kit';
import { authStore } from '$stores';

export const ssr = false;  // Required for SPA mode

export async function load() {
    // Wait for auth store to initialize
    if (authStore.loading) {
        await new Promise(resolve => setTimeout(resolve, 50));
    }

    if (!authStore.isAuthenticated) {
        throw redirect(302, '/login');
    }

    // Validate token with backend
    try {
        const res = await fetch('/api/auth/me', {
            headers: { Authorization: `Bearer ${authStore.token}` }
        });
        if (!res.ok) {
            authStore.clearAuth();
            throw redirect(302, '/login');
        }
    } catch {
        authStore.clearAuth();
        throw redirect(302, '/login');
    }

    return { user: authStore.user };
}
```

### Pattern 4: API Client Auth Header Injection

**What:** Automatically add Authorization header to all API requests
**When to use:** Modify existing API client

```typescript
// src/lib/api/client.ts modifications
import { authStore } from '$stores';

async function handleResponse<T>(response: Response): Promise<T> {
    if (response.status === 401) {
        // Token expired or invalid
        authStore.clearAuth();
        window.location.href = '/login';
        throw new ApiError(401, 'Session expired');
    }
    // ... existing error handling
}

function getAuthHeaders(): HeadersInit {
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (authStore.token) {
        headers['Authorization'] = `Bearer ${authStore.token}`;
    }
    return headers;
}

// Update all fetch calls to use getAuthHeaders()
export const profiles = {
    list: async (): Promise<ServerProfile[]> => {
        const res = await fetch(`${API_BASE}/profiles`, {
            headers: getAuthHeaders()
        });
        return handleResponse(res);
    },
    // ... etc
};
```

### Anti-Patterns to Avoid

- **Using middleware for auth:** FastAPI dependency injection is more flexible and testable than middleware
- **Storing JWT in cookies for SPA:** The frontend uses adapter-static (pure client-side), localStorage is appropriate here
- **python-jose:** Abandoned library, use PyJWT instead
- **passlib:** Python 3.13+ compatibility issues, use pwdlib instead
- **Checking auth in +layout.server.ts:** Won't work with adapter-static, use +layout.ts with ssr=false
- **Global auth state without persistence:** Always sync auth state with localStorage

## Don't Hand-Roll

Problems that look simple but have existing solutions.

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Password hashing | Custom hash function | pwdlib[argon2] | Argon2 is PHC winner, proper salt handling |
| JWT creation/validation | Manual string manipulation | PyJWT | Handles expiry, signatures, algorithm security |
| Email validation | Regex patterns | Pydantic EmailStr or email-validator | Handles edge cases, internationalized emails |
| Token refresh | Custom expiry tracking | JWT exp claim + PyJWT decode | Built-in expiration handling |
| First-user detection | Count query on every register | Check during startup + cache | Avoid race conditions |

**Key insight:** Authentication is a security-critical domain where custom solutions inevitably have vulnerabilities. Use battle-tested libraries for all cryptographic operations.

## Common Pitfalls

### Pitfall 1: Token Stored But Not Validated

**What goes wrong:** Frontend stores token, assumes valid, backend rejects
**Why it happens:** Token expires, user status changes, or token corrupted
**How to avoid:** Always validate token on protected route load, not just presence check
**Warning signs:** Users report random logouts, 401 errors in console

### Pitfall 2: Race Condition in First-User Admin Assignment

**What goes wrong:** Two simultaneous registrations both become admin
**Why it happens:** Check-then-insert without transaction isolation
**How to avoid:** Use database constraint or atomic operation

```python
# BAD: Race condition possible
user_count = await session.execute(select(func.count(User.id)))
if user_count.scalar() == 0:
    new_user.is_admin = True

# GOOD: Atomic with database constraint
# Add unique constraint on is_admin=True with status=approved
# Or use SELECT ... FOR UPDATE in transaction
async with session.begin():
    result = await session.execute(
        select(User).where(User.is_admin == True).with_for_update()
    )
    if result.scalar_one_or_none() is None:
        new_user.is_admin = True
        new_user.status = UserStatus.APPROVED
```

### Pitfall 3: Exposing Pending Users Count to Non-Admins

**What goes wrong:** Badge count endpoint accessible to all authenticated users
**Why it happens:** Forgot admin-only dependency on the endpoint
**How to avoid:** Use `get_admin_user` dependency on pending count endpoint
**Warning signs:** Non-admin users see badge count (information disclosure)

### Pitfall 4: JWT Secret in Code

**What goes wrong:** Secret committed to git, compromised
**Why it happens:** Hardcoded for development, forgotten in production
**How to avoid:** Use environment variable via pydantic-settings (existing pattern)

```python
# In config.py (extend existing Settings class)
class Settings(BaseSettings):
    # ... existing settings
    jwt_secret: str = Field(default="CHANGE_ME_IN_PRODUCTION")
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 7
```

### Pitfall 5: SPA Route Guard Without Backend Validation

**What goes wrong:** User modifies localStorage, bypasses frontend guard
**Why it happens:** Only checking token existence, not validity
**How to avoid:** Call `/api/auth/me` in route guard, clear on 401
**Warning signs:** Users can access protected pages after token expiry

## Code Examples

Verified patterns from official sources.

### Password Hashing with pwdlib

```python
# Source: pwdlib GitHub, FastAPI official docs
from pwdlib import PasswordHash

# Create once, reuse (module level singleton)
password_hash = PasswordHash.recommended()

def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return password_hash.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return password_hash.verify(plain_password, hashed_password)
```

### JWT Token Creation

```python
# Source: FastAPI official docs, PyJWT
from datetime import datetime, timedelta, timezone
import jwt
from mlx_manager.config import settings

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_expire_days)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.InvalidTokenError:
        return None
```

### Login Endpoint

```python
# Source: FastAPI official docs pattern
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: AsyncSession = Depends(get_db)
):
    """Authenticate user and return JWT token."""
    result = await session.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.status != UserStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending approval" if user.status == UserStatus.PENDING else "Account disabled"
        )

    access_token = create_access_token(data={"sub": user.email})
    return Token(access_token=access_token)
```

### Registration Endpoint

```python
@router.post("/register", response_model=UserPublic, status_code=201)
async def register(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_db)
):
    """Register a new user account."""
    # Check if email exists
    result = await session.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Check if this is the first user (becomes admin)
    count_result = await session.execute(select(func.count(User.id)))
    is_first_user = count_result.scalar() == 0

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        is_admin=is_first_user,
        status=UserStatus.APPROVED if is_first_user else UserStatus.PENDING
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    return user
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| passlib for hashing | pwdlib with Argon2 | 2024-2025 | passlib has Python 3.13+ issues |
| python-jose for JWT | PyJWT | 2024 | python-jose abandoned, PyJWT actively maintained |
| bcrypt default | Argon2 default | 2024-2025 | Argon2 is PHC winner, more secure |
| Middleware auth | Dependency injection | Always for FastAPI | More flexible, better testing |

**Deprecated/outdated:**
- **python-jose:** Abandoned, last release ~3 years ago. FastAPI docs updated to use PyJWT
- **passlib:** Not actively maintained, Python 3.13+ compatibility issues
- **bcrypt as primary:** Still secure but Argon2 is the modern recommendation

## Open Questions

Things that couldn't be fully resolved.

1. **Token Refresh Strategy**
   - What we know: 7-day session duration is specified in requirements
   - What's unclear: Whether to implement refresh tokens or just long-lived access tokens
   - Recommendation: Use single 7-day access token (simpler for local app, refresh adds complexity without proportional benefit for local-first tool)

2. **Password Complexity Requirements**
   - What we know: Claude's discretion per CONTEXT.md
   - What's unclear: What minimum requirements are appropriate
   - Recommendation: Minimum 8 characters, no other restrictions (local tool, not enterprise)

3. **Admin Cannot Delete Self (Edge Case)**
   - What we know: Admin can delete own account only if another admin exists
   - What's unclear: UI flow when deletion is blocked
   - Recommendation: Disable delete button with tooltip "Promote another user to admin first"

## Sources

### Primary (HIGH confidence)
- [FastAPI Official JWT Tutorial](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/) - Password hashing, JWT patterns, dependency injection
- [pwdlib GitHub](https://github.com/frankie567/pwdlib) - Modern password hashing library
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/tutorial/fastapi/update-extra-data/) - Schema separation pattern

### Secondary (MEDIUM confidence)
- [FastAPI GitHub Discussion #9587](https://github.com/fastapi/fastapi/discussions/9587) - python-jose abandonment, PyJWT recommendation
- [SvelteKit Auth Docs](https://svelte.dev/docs/kit/auth) - Official auth guidance
- [SvelteKit SPA Protected Routes](https://sveltestarterkit.com/blog/sveltekit-spa-protected-routes) - Adapter-static auth patterns

### Tertiary (LOW confidence)
- WebSearch results for bcrypt vs Argon2 comparison - Multiple sources agree on Argon2 recommendation
- WebSearch results for JWT library comparison - Confirmed python-jose abandoned status

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official FastAPI docs updated to PyJWT + pwdlib
- Architecture: HIGH - Follows existing codebase patterns (routers, services, dependencies)
- Pitfalls: MEDIUM - Based on common patterns and security best practices
- Frontend patterns: MEDIUM - Adapter-static limits options, patterns verified but less documented

**Research date:** 2026-01-20
**Valid until:** 2026-02-20 (30 days - stable domain)

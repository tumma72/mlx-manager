---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/auth_service.py
type: service
updated: 2026-01-21
status: active
---

# auth_service.py

## Purpose

Authentication service for password hashing and JWT token management. Uses Argon2 for secure password hashing via pwdlib and PyJWT for token creation/verification. Provides the cryptographic primitives used by the auth router for user authentication.

## Exports

- `password_hash` - Module-level PasswordHash instance (Argon2)
- `hash_password(password: str) -> str` - Hash a password for storage
- `verify_password(plain_password: str, hashed_password: str) -> bool` - Verify password
- `create_access_token(data: dict, expires_delta?: timedelta) -> str` - Create JWT token
- `decode_token(token: str) -> dict | None` - Decode and validate JWT token

## Dependencies

- [[backend-mlx_manager-config]] - JWT settings (secret, algorithm, expiry)
- pwdlib - Password hashing library (Argon2)
- jwt (PyJWT) - JWT token handling

## Used By

TBD

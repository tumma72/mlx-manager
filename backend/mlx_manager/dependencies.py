"""FastAPI dependencies for common operations."""

from typing import Annotated

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.models import ExecutionProfile, User, UserStatus
from mlx_manager.services.auth_service import decode_token

# OAuth2 password bearer for JWT token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_profile_or_404(
    profile_id: int,
    session: AsyncSession = Depends(get_db),
) -> ExecutionProfile:
    """Get a profile by ID or raise 404."""
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(ExecutionProfile)
        .where(ExecutionProfile.id == profile_id)
        .options(selectinload(ExecutionProfile.model))  # type: ignore[arg-type]
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: AsyncSession = Depends(get_db),
) -> User:
    """Get the current authenticated user from JWT token.

    Raises:
        HTTPException 401: If token is invalid or user not found/approved.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Decode and validate token
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    # Query user by email
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None or user.status != UserStatus.APPROVED:
        raise credentials_exception

    return user


async def get_current_user_from_token(
    token: str = Query(..., description="JWT token for SSE/WS auth"),
    session: AsyncSession = Depends(get_db),
) -> User:
    """Get the current authenticated user from a query-parameter JWT token.

    This is used for SSE and WebSocket endpoints where browser EventSource
    cannot send custom Authorization headers.

    Raises:
        HTTPException 401: If token is invalid or user not found/approved.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    # Decode and validate token
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    # Query user by email
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None or user.status != UserStatus.APPROVED:
        raise credentials_exception

    return user


async def get_admin_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get the current user if they are an admin.

    Raises:
        HTTPException 403: If user is not an admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user

"""Authentication API router."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_admin_user, get_current_user
from mlx_manager.models import (
    PasswordReset,
    Token,
    User,
    UserCreate,
    UserPublic,
    UserStatus,
    UserUpdate,
)
from mlx_manager.services.auth_service import (
    create_access_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserPublic, status_code=201)
async def register(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_db),
) -> User:
    """Register a new user account.

    The first user to register becomes admin and is auto-approved.
    Subsequent users are created with pending status.
    """
    # Check if email already exists
    result = await session.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Check if this is the first user (becomes admin)
    count_result = await session.execute(
        select(func.count(User.id))  # type: ignore[arg-type]
    )
    is_first_user = count_result.scalar() == 0

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        is_admin=is_first_user,
        status=UserStatus.APPROVED if is_first_user else UserStatus.PENDING,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: AsyncSession = Depends(get_db),
) -> Token:
    """Authenticate user and return JWT token.

    Uses OAuth2PasswordRequestForm where username field contains email.
    """
    # Query user by email
    result = await session.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    # Verify user exists and password is correct
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check user status
    if user.status == UserStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending approval",
        )
    if user.status == UserStatus.DISABLED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account disabled",
        )

    # Create access token
    access_token = create_access_token(data={"sub": user.email})
    return Token(access_token=access_token)


@router.get("/me", response_model=UserPublic)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current authenticated user info."""
    return current_user


# Admin endpoints


@router.get("/users", response_model=list[UserPublic])
async def list_users(
    _admin: Annotated[User, Depends(get_admin_user)],
    session: AsyncSession = Depends(get_db),
) -> list[User]:
    """List all users (admin only)."""
    result = await session.execute(
        select(User).order_by(desc(User.created_at))  # type: ignore[arg-type]
    )
    return list(result.scalars().all())


@router.get("/users/pending/count")
async def get_pending_count(
    _admin: Annotated[User, Depends(get_admin_user)],
    session: AsyncSession = Depends(get_db),
) -> dict[str, int]:
    """Get count of pending users (admin only)."""
    result = await session.execute(
        select(func.count(User.id)).where(User.status == UserStatus.PENDING)  # type: ignore[arg-type]
    )
    count = result.scalar() or 0
    return {"count": count}


@router.put("/users/{user_id}", response_model=UserPublic)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    admin: Annotated[User, Depends(get_admin_user)],
    session: AsyncSession = Depends(get_db),
) -> User:
    """Update a user (admin only).

    Can update email, is_admin, and status.
    Prevents demoting the last admin.
    """
    # Get user to update
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent admin from demoting self if they're the only admin
    if user_data.is_admin is False and user_id == admin.id:
        admin_count_result = await session.execute(
            select(func.count(User.id)).where(  # type: ignore[arg-type]
                User.is_admin == True  # noqa: E712
            )
        )
        admin_count = admin_count_result.scalar() or 0
        if admin_count == 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove last admin",
            )

    # Prevent admin from disabling self if they're the only active admin
    if user_data.status == UserStatus.DISABLED and user_id == admin.id:
        active_admin_count_result = await session.execute(
            select(func.count(User.id)).where(  # type: ignore[arg-type]
                User.is_admin == True,  # noqa: E712
                User.status == UserStatus.APPROVED,
            )
        )
        active_admin_count = active_admin_count_result.scalar() or 0
        if active_admin_count == 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot disable last active admin",
            )

    # Update provided fields
    update_data = user_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    # If status changed to APPROVED, set approval metadata
    if user_data.status == UserStatus.APPROVED and user.approved_at is None:
        user.approved_at = datetime.now(tz=UTC)
        user.approved_by = admin.id

    session.add(user)
    await session.commit()
    await session.refresh(user)

    return user


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    admin: Annotated[User, Depends(get_admin_user)],
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete a user (admin only).

    Prevents deleting self if you're the only admin.
    """
    # Get user to delete
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent admin from deleting self if only admin
    if user_id == admin.id:
        admin_count_result = await session.execute(
            select(func.count(User.id)).where(  # type: ignore[arg-type]
                User.is_admin == True  # noqa: E712
            )
        )
        admin_count = admin_count_result.scalar() or 0
        if admin_count == 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete last admin",
            )

    await session.delete(user)
    await session.commit()


@router.post("/users/{user_id}/reset-password")
async def reset_password(
    user_id: int,
    password_data: PasswordReset,
    _admin: Annotated[User, Depends(get_admin_user)],
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Reset a user's password (admin only)."""
    # Get user
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update password
    user.hashed_password = hash_password(password_data.password)
    session.add(user)
    await session.commit()

    return {"message": "Password reset successfully"}

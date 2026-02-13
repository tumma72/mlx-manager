"""Auth DTOs - user registration, login, and management."""

from datetime import datetime

from pydantic import BaseModel

from mlx_manager.models.entities import UserBase
from mlx_manager.models.enums import UserStatus

__all__ = [
    "UserCreate",
    "UserPublic",
    "UserLogin",
    "UserUpdate",
    "Token",
    "PasswordReset",
]


class UserCreate(BaseModel):
    """Schema for creating a user (registration)."""

    email: str
    password: str


class UserPublic(UserBase):
    """Public response model for user (no password)."""

    id: int
    is_admin: bool
    status: UserStatus
    created_at: datetime


class UserLogin(BaseModel):
    """Schema for login request."""

    email: str
    password: str


class UserUpdate(BaseModel):
    """Schema for admin user updates."""

    email: str | None = None
    is_admin: bool | None = None
    status: UserStatus | None = None


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class PasswordReset(BaseModel):
    """Schema for admin password reset."""

    password: str

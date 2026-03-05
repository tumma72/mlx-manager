"""Auth DTOs - user registration, login, and management."""

import re
from datetime import datetime

from pydantic import BaseModel, field_validator
from pydantic import Field as PydanticField

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


_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _validate_email(value: str) -> str:
    """Validate email format using a simple regex."""
    if not _EMAIL_RE.match(value):
        msg = "Invalid email address format"
        raise ValueError(msg)
    return value.lower().strip()


class UserCreate(BaseModel):
    """Schema for creating a user (registration)."""

    email: str
    password: str = PydanticField(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _validate_email(v)


class UserPublic(UserBase):
    """Public response model for user (no password)."""

    id: int
    is_admin: bool
    status: UserStatus
    created_at: datetime


class UserLogin(BaseModel):
    """Schema for login request."""

    email: str
    password: str = PydanticField(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _validate_email(v)


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

    password: str = PydanticField(min_length=8, max_length=128)

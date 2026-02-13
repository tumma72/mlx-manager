"""MCP DTOs - tool execution requests."""

from typing import Any

from pydantic import BaseModel

__all__ = ["ToolExecuteRequest"]


class ToolExecuteRequest(BaseModel):
    """Request model for tool execution."""

    name: str
    arguments: dict[str, Any]

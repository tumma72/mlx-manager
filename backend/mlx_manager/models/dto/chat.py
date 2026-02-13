"""Chat DTOs - chat completion requests."""

from pydantic import BaseModel

__all__ = ["ChatRequest"]


class ChatRequest(BaseModel):
    """Chat completion request."""

    profile_id: int
    messages: list[dict]  # OpenAI-compatible message format
    tools: list[dict] | None = None  # OpenAI function definitions array
    tool_choice: str | None = None  # "auto", "none", or specific function
    # Generation parameters - override profile defaults if provided
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

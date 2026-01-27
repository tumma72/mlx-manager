"""OpenAI-compatible request/response schemas.

Reference: https://platform.openai.com/docs/api-reference/chat
"""

import time
from typing import Literal

from pydantic import BaseModel, Field


# --- Request Models ---


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion request."""

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | str | None = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    # n: int = 1  # Not supported initially (always 1)


class CompletionRequest(BaseModel):
    """OpenAI Completion request (legacy)."""

    model: str
    prompt: str | list[str]
    max_tokens: int | None = Field(default=16, ge=1, le=128000)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | str | None = None
    echo: bool = False


# --- Response Models ---


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class CompletionChoice(BaseModel):
    """A single choice in completion response."""

    index: int
    text: str
    finish_reason: Literal["stop", "length"] | None = None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class CompletionResponse(BaseModel):
    """OpenAI Completion response."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# --- Streaming Response Models ---


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming response."""

    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in streaming chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI Chat Completion streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


# --- Models Endpoint ---


class ModelInfo(BaseModel):
    """Information about a single model."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlx-community"


class ModelListResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]

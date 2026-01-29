"""Anthropic Messages API request/response schemas.

Reference: https://platform.claude.com/docs/en/api/messages
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Content Block Types ---


class ImageSource(BaseModel):
    """Image source for image content blocks."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str  # Base64-encoded image data


class TextBlockParam(BaseModel):
    """Text content block in a message."""

    type: Literal["text"] = "text"
    text: str


class ImageBlockParam(BaseModel):
    """Image content block in a message."""

    type: Literal["image"] = "image"
    source: ImageSource


# Union type for content blocks
ContentBlock = TextBlockParam | ImageBlockParam


# --- Message Types ---


class MessageParam(BaseModel):
    """A message in the conversation.

    Note: System messages are NOT allowed here - use the system parameter instead.
    """

    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]


# --- Request Model ---


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request.

    Reference: https://platform.claude.com/docs/en/api/messages

    Key differences from OpenAI:
    - max_tokens is REQUIRED (not optional)
    - system is a separate field (not in messages array)
    - content can be string or array of content blocks
    - temperature range is 0.0 to 1.0 (not 0.0 to 2.0)
    """

    model: str
    max_tokens: int = Field(ge=1)  # Required - no default
    messages: list[MessageParam]
    system: str | list[TextBlockParam] | None = None
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None


# --- Response Models ---


class TextBlock(BaseModel):
    """Text content block in response."""

    type: Literal["text"] = "text"
    text: str


class Usage(BaseModel):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response.

    Reference: https://platform.claude.com/docs/en/api/messages
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[TextBlock]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None = None
    usage: Usage


# --- Streaming Event Models ---


class MessageStartMessage(BaseModel):
    """Message object in message_start event."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[Any] = Field(default_factory=list)
    model: str
    stop_reason: None = None
    stop_sequence: None = None
    usage: Usage


class MessageStartEvent(BaseModel):
    """SSE event: message_start."""

    type: Literal["message_start"] = "message_start"
    message: MessageStartMessage


class ContentBlockStartBlock(BaseModel):
    """Content block in content_block_start event."""

    type: Literal["text"] = "text"
    text: str = ""


class ContentBlockStartEvent(BaseModel):
    """SSE event: content_block_start."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlockStartBlock


class TextDelta(BaseModel):
    """Text delta in content_block_delta event."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ContentBlockDeltaEvent(BaseModel):
    """SSE event: content_block_delta."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta


class ContentBlockStopEvent(BaseModel):
    """SSE event: content_block_stop."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaDelta(BaseModel):
    """Delta in message_delta event."""

    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None = None


class MessageDeltaUsage(BaseModel):
    """Usage in message_delta event."""

    output_tokens: int


class MessageDeltaEvent(BaseModel):
    """SSE event: message_delta."""

    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaDelta
    usage: MessageDeltaUsage


class MessageStopEvent(BaseModel):
    """SSE event: message_stop."""

    type: Literal["message_stop"] = "message_stop"


# --- Helper Functions ---


def extract_anthropic_content(
    content: str | list[ContentBlock],
) -> str:
    """Extract text from Anthropic content (string or content blocks).

    Args:
        content: Either a string or list of content blocks

    Returns:
        Concatenated text content
    """
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            # Handle dict form (from JSON parsing)
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        elif hasattr(block, "type") and block.type == "text":
            # Handle Pydantic model form
            text_parts.append(block.text)

    return " ".join(text_parts)

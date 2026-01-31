"""Chat completions router with streaming support.

This router uses the embedded MLX Server for inference, calling the
generate_chat_completion() and generate_vision_completion() functions
directly rather than proxying to an external server process.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import cast

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from ..database import get_db
from ..dependencies import get_current_user
from ..models import ServerProfile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat completion request."""

    profile_id: int
    messages: list[dict]  # OpenAI-compatible message format
    tools: list[dict] | None = None  # OpenAI function definitions array
    tool_choice: str | None = None  # "auto", "none", or specific function


@router.post("/completions")
async def chat_completions(
    request: ChatRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),  # Auth required, user object unused
):
    """
    Stream chat completions from embedded MLX Server.

    Uses the embedded inference service directly without external server proxy.
    Parses thinking tags and emits typed chunks for frontend rendering.

    SSE Event Types:
    - thinking: Content inside <think>...</think> tags
    - thinking_done: Emitted when </think> is encountered, includes duration
    - response: Regular response content (outside thinking tags)
    - error: Error message (model loading failure, inference error)
    - done: Stream complete

    Thinking Detection:
    This endpoint supports two thinking extraction mechanisms:
    1. Server-side reasoning_parser (e.g., reasoning_parser=glm4_moe in profile config)
       - Server extracts thinking into delta.reasoning_content field
       - We detect transition when reasoning_content stops and content begins
    2. Raw <think> tags in delta.content field (fallback for models without parser)
       - We parse tags character-by-character and emit thinking/thinking_done events

    For GLM-4 models:
    - If the model's chat template outputs <think> tags, they'll be parsed (method 2)
    - If reasoning_parser=glm4_moe is configured, thinking goes to reasoning_content
      (method 1)
    - If neither mechanism activates, thinking appears as regular response text
      (acceptable fallback)
    """
    # Get profile
    profile = await db.get(ServerProfile, request.profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    model_id = profile.model_path

    # Detect if request has images (multimodal request)
    has_images = _has_images_in_messages(request.messages)

    async def generate() -> AsyncGenerator[str, None]:
        thinking_start: float | None = None
        in_thinking = False
        first_chunk_logged = False

        try:
            # Import inference services
            from mlx_manager.mlx_server.services.inference import generate_chat_completion
            from mlx_manager.mlx_server.services.vision import generate_vision_completion
            from mlx_manager.mlx_server.services.image_processor import preprocess_images
            from mlx_manager.mlx_server.models.detection import detect_model_type
            from mlx_manager.mlx_server.models.types import ModelType

            # Detect model type
            model_type = detect_model_type(model_id)

            # Determine which inference service to use
            if has_images or model_type == ModelType.VISION:
                # Vision model inference
                text_prompt, image_urls = _extract_text_and_images(request.messages)
                images = await preprocess_images(image_urls) if image_urls else []

                gen = generate_vision_completion(
                    model_id=model_id,
                    text_prompt=text_prompt,
                    images=images,
                    max_tokens=4096,
                    temperature=0.7,
                    stream=True,
                )
            else:
                # Text model inference
                gen = generate_chat_completion(
                    model_id=model_id,
                    messages=request.messages,
                    max_tokens=4096,
                    temperature=0.7,
                    stream=True,
                )

            # Cast to async generator (generate_chat_completion returns Union type)
            async_gen = cast(AsyncGenerator[dict, None], gen)

            # Consume the async generator directly
            async for chunk in async_gen:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning_content", "")

                # Diagnostic logging for first chunk
                if not first_chunk_logged:
                    finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                    logger.debug(
                        f"First chunk - delta keys: {list(delta.keys())}, "
                        f"finish_reason: {finish_reason}"
                    )
                    first_chunk_logged = True

                # Handle tool calls from model
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        tool_data = {
                            "type": "tool_call",
                            "tool_call": tc,  # {index, id, function: {name, arguments}}
                        }
                        yield f"data: {json.dumps(tool_data)}\n\n"

                # Handle server-extracted reasoning content
                # (when reasoning_parser is configured, server puts
                # thinking in reasoning_content field)
                if reasoning:
                    # Strip thinking tags — server may include them
                    clean = reasoning.replace("<think>", "").replace("</think>", "")
                    if not in_thinking:
                        in_thinking = True
                        thinking_start = time.time()
                    if clean:
                        thinking_data = {
                            "type": "thinking",
                            "content": clean,
                        }
                        yield f"data: {json.dumps(thinking_data)}\n\n"
                elif in_thinking and not reasoning and content:
                    # Transitioned from reasoning to content — thinking done
                    in_thinking = False
                    duration = time.time() - thinking_start if thinking_start else 0
                    done_data = {
                        "type": "thinking_done",
                        "duration": round(duration, 1),
                    }
                    yield f"data: {json.dumps(done_data)}\n\n"

                # Handle content field
                if content:
                    # Parse <think> tags from content (fallback for
                    # models that output raw tags without server parser)
                    if "<think>" in content or "</think>" in content:
                        i = 0
                        while i < len(content):
                            if content[i : i + 7] == "<think>":
                                in_thinking = True
                                thinking_start = time.time()
                                i += 7
                                continue

                            if content[i : i + 8] == "</think>":
                                in_thinking = False
                                duration = (
                                    time.time() - thinking_start
                                    if thinking_start
                                    else 0
                                )
                                done_data = {
                                    "type": "thinking_done",
                                    "duration": round(duration, 1),
                                }
                                yield f"data: {json.dumps(done_data)}\n\n"
                                i += 8
                                continue

                            char = content[i]
                            chunk_type = "thinking" if in_thinking else "response"
                            chunk_data = {
                                "type": chunk_type,
                                "content": char,
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            i += 1
                    elif in_thinking:
                        # Still in thinking mode (from tag parsing)
                        thinking_data = {
                            "type": "thinking",
                            "content": content,
                        }
                        yield f"data: {json.dumps(thinking_data)}\n\n"
                    else:
                        # Regular response content
                        response_data = {
                            "type": "response",
                            "content": content,
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

                # Check finish_reason for tool_calls completion
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                if finish_reason == "tool_calls":
                    yield f"data: {json.dumps({'type': 'tool_calls_done'})}\n\n"

            # If still in thinking when stream ends, emit thinking_done
            if in_thinking:
                duration = time.time() - thinking_start if thinking_start else 0
                done_data = {
                    "type": "thinking_done",
                    "duration": round(duration, 1),
                }
                yield f"data: {json.dumps(done_data)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except FileNotFoundError as e:
            # Model not found locally
            error_msg = f"Model not available. Download the model first: {e}"
            logger.error(f"Chat inference error: {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        except Exception as e:
            # Generic error handling
            error_msg = str(e)
            logger.error(f"Chat inference error: {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _has_images_in_messages(messages: list[dict]) -> bool:
    """Check if any message contains image content parts."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _extract_text_and_images(messages: list[dict]) -> tuple[str, list[str]]:
    """Extract text prompt and image URLs from messages.

    For vision models, we need to:
    1. Combine all text content into a single prompt
    2. Extract all image URLs/data URIs

    Args:
        messages: OpenAI-format messages list

    Returns:
        Tuple of (combined text prompt, list of image URLs)
    """
    text_parts: list[str] = []
    image_urls: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            # Simple string content
            text_parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            # Multimodal content array
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(f"{role}: {part.get('text', '')}")
                    elif part.get("type") == "image_url":
                        image_url_obj = part.get("image_url", {})
                        if isinstance(image_url_obj, dict):
                            url = image_url_obj.get("url", "")
                        else:
                            url = str(image_url_obj)
                        if url:
                            image_urls.append(url)

    # Combine text parts into a single prompt
    combined_prompt = "\n".join(text_parts)

    return combined_prompt, image_urls

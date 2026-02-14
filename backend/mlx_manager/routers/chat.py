"""Chat completions router with streaming support.

This router uses the embedded MLX Server for inference, calling the
generate_chat_completion() function
directly rather than proxying to an external server process.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import cast

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.orm import selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ..database import get_db
from ..dependencies import get_current_user
from ..models import ExecutionProfile
from ..models.dto.chat import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/completions")
async def chat_completions(
    request: ChatRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),  # Auth required, user object unused
):
    """
    Stream chat completions from embedded MLX Server.

    Uses the embedded inference service directly without external server proxy.
    Server handles thinking tag extraction and sends OpenAI-compatible deltas.

    SSE Event Types:
    - thinking: Reasoning content from thinking models (from delta.reasoning_content)
    - thinking_done: Emitted when thinking phase ends, includes duration
    - response: Regular response content (from delta.content)
    - tool_call: Tool call from model
    - tool_calls_done: All tool calls complete
    - error: Error message (model loading failure, inference error)
    - done: Stream complete

    Thinking Detection:
    The inference server extracts <think> tags and streams content via
    delta.reasoning_content field (following OpenAI o1/o3 API spec).
    This endpoint simply forwards the structured data without tag parsing.
    """
    # Get profile with model relationship
    result = await db.execute(
        select(ExecutionProfile)
        .where(ExecutionProfile.id == request.profile_id)
        .options(selectinload(ExecutionProfile.model))  # type: ignore[arg-type]
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    model_id = profile.model.repo_id

    # Detect if request has images (multimodal request)
    has_images = _has_images_in_messages(request.messages)

    async def generate() -> AsyncGenerator[str, None]:
        thinking_start: float | None = None
        in_thinking = False
        first_chunk_logged = False

        try:
            # Import inference services
            from mlx_manager.mlx_server.models.detection import detect_model_type
            from mlx_manager.mlx_server.models.types import ModelType
            from mlx_manager.mlx_server.services.image_processor import preprocess_images
            from mlx_manager.mlx_server.services.inference import generate_chat_completion

            # Get generation parameters: request overrides profile defaults
            temp = (
                request.temperature
                if request.temperature is not None
                else profile.default_temperature
            )
            max_tok = (
                request.max_tokens if request.max_tokens is not None else profile.default_max_tokens
            )
            top_p_val = request.top_p if request.top_p is not None else profile.default_top_p

            # Lower temperature when tools are enabled for more reliable tool calling
            # Some models (like GLM-4.7) are verbose at higher temps and may not call tools
            if request.tools and request.temperature is None:
                temp = min(temp, 0.3)

            # Validate model type if images are present
            if has_images:
                model_type = detect_model_type(model_id)
                if model_type != ModelType.VISION:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{model_id}' is type '{model_type.value}', "
                        f"but request contains images. Use a vision model (e.g., Qwen2-VL).",
                    )

            # Preprocess images if present
            images = None
            if has_images:
                _, image_urls = _extract_text_and_images(request.messages)
                images = await preprocess_images(image_urls) if image_urls else None

            # Unified text and vision inference
            # Images (if present) will be handled by the adapter's prepare_input()
            gen = await generate_chat_completion(
                model_id=model_id,
                messages=request.messages,
                max_tokens=max_tok,
                temperature=temp,
                top_p=top_p_val,
                stream=True,
                tools=request.tools,
                enable_prompt_injection=profile.default_enable_tool_injection,
                images=images,
            )

            # Cast to async generator (both functions return AsyncGenerator when stream=True)
            async_gen = cast(AsyncGenerator[dict, None], gen)

            # Consume the async generator directly
            # Server now sends OpenAI-compatible deltas with reasoning_content
            # No tag parsing needed here - server handles it
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

                # Handle reasoning content (thinking models)
                # Server sends thinking content in delta.reasoning_content field
                if reasoning:
                    if not in_thinking:
                        in_thinking = True
                        thinking_start = time.time()
                    yield f"data: {json.dumps({'type': 'thinking', 'content': reasoning})}\n\n"

                # Handle transition from thinking to content
                if in_thinking and content and not reasoning:
                    # Transitioned from reasoning to content - thinking done
                    in_thinking = False
                    duration = time.time() - thinking_start if thinking_start else 0
                    done_data = {"type": "thinking_done", "duration": round(duration, 1)}
                    yield f"data: {json.dumps(done_data)}\n\n"

                # Handle regular content
                if content:
                    yield f"data: {json.dumps({'type': 'response', 'content': content})}\n\n"

                # Check finish_reason for tool_calls completion
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                if finish_reason == "tool_calls":
                    yield f"data: {json.dumps({'type': 'tool_calls_done'})}\n\n"

            # If still in thinking when stream ends, emit thinking_done
            if in_thinking:
                duration = time.time() - thinking_start if thinking_start else 0
                done_data = {"type": "thinking_done", "duration": round(duration, 1)}
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

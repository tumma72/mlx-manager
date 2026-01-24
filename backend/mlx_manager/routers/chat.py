"""Chat completions router with streaming support."""

import json
import logging
import time
from collections.abc import AsyncGenerator

import httpx
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
    Stream chat completions from mlx-openai-server.

    Proxies to the MLX server running for the given profile.
    Parses thinking tags and emits typed chunks for frontend rendering.

    SSE Event Types:
    - thinking: Content inside <think>...</think> tags
    - thinking_done: Emitted when </think> is encountered, includes duration
    - response: Regular response content (outside thinking tags)
    - error: Error message (connection failure, server error)
    - done: Stream complete

    Note: Connection errors (httpx.ConnectError) indicate the server is not running.
    This is the appropriate check - we don't need to pre-verify server state since
    the connection attempt itself is the verification.

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

    # Build MLX server URL
    server_url = f"http://{profile.host}:{profile.port}/v1/chat/completions"

    async def generate() -> AsyncGenerator[str, None]:
        thinking_start: float | None = None
        in_thinking = False
        first_chunk_logged = False

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                body: dict = {
                    "model": profile.model_path,
                    "messages": request.messages,
                    "stream": True,
                }
                if request.tools:
                    body["tools"] = request.tools
                    body["tool_choice"] = request.tool_choice or "auto"

                async with client.stream(
                    "POST",
                    server_url,
                    json=body,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_data = {"type": "error", "content": error_text.decode()}
                        yield f"data: {json.dumps(error_data)}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning_content", "")

                            # Diagnostic logging for first chunk (helps debug thinking extraction)
                            if not first_chunk_logged:
                                finish_reason = data.get("choices", [{}])[0].get("finish_reason")
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
                            finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                            if finish_reason == "tool_calls":
                                yield f"data: {json.dumps({'type': 'tool_calls_done'})}\n\n"

                        except json.JSONDecodeError:
                            continue

                    # If still in thinking when stream ends, emit thinking_done
                    if in_thinking:
                        duration = time.time() - thinking_start if thinking_start else 0
                        done_data = {
                            "type": "thinking_done",
                            "duration": round(duration, 1),
                        }
                        yield f"data: {json.dumps(done_data)}\n\n"

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except httpx.ConnectError:
                error_msg = "Failed to connect to MLX server. Is it running?"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            except httpx.TimeoutException:
                timeout_msg = "Request timed out. The model may be processing a complex request."
                yield f"data: {json.dumps({'type': 'error', 'content': timeout_msg})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

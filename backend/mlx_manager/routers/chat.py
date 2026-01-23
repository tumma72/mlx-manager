"""Chat completions router with streaming support."""

import json
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

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat completion request."""

    profile_id: int
    messages: list[dict]  # OpenAI-compatible message format


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
    """
    # Get profile
    profile = await db.get(ServerProfile, request.profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Build MLX server URL
    server_url = f"http://{profile.host}:{profile.port}/v1/chat/completions"

    # Check if profile has reasoning parser enabled
    has_reasoning = bool(profile.reasoning_parser)

    async def generate() -> AsyncGenerator[str, None]:
        thinking_start: float | None = None
        in_thinking = False

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream(
                    "POST",
                    server_url,
                    json={
                        "model": profile.model_path,
                        "messages": request.messages,
                        "stream": True,
                    },
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

                            if not content:
                                continue

                            # Parse thinking tags if reasoning enabled
                            if has_reasoning:
                                i = 0
                                while i < len(content):
                                    # Check for opening tag
                                    if content[i : i + 7] == "<think>":
                                        in_thinking = True
                                        thinking_start = time.time()
                                        i += 7
                                        continue

                                    # Check for closing tag
                                    if content[i : i + 8] == "</think>":
                                        in_thinking = False
                                        duration = (
                                            time.time() - thinking_start if thinking_start else 0
                                        )
                                        done_data = {
                                            "type": "thinking_done",
                                            "duration": round(duration, 1),
                                        }
                                        yield f"data: {json.dumps(done_data)}\n\n"
                                        i += 8
                                        continue

                                    # Emit character
                                    char = content[i]
                                    chunk_type = "thinking" if in_thinking else "response"
                                    chunk_data = {"type": chunk_type, "content": char}
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                                    i += 1
                            else:
                                # No reasoning parser, emit all as response
                                response_data = {"type": "response", "content": content}
                                yield f"data: {json.dumps(response_data)}\n\n"

                        except json.JSONDecodeError:
                            continue

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

"""TTS (text-to-speech) endpoint.

POST /v1/audio/speech - Generate audio from text.
OpenAI-compatible: https://platform.openai.com/docs/api-reference/audio/createSpeech
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from loguru import logger

from mlx_manager.mlx_server.schemas.openai import SpeechRequest
from mlx_manager.mlx_server.services.audio import generate_speech

router = APIRouter(tags=["audio"])

# Content type mapping for audio formats
AUDIO_CONTENT_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
}


@router.post("/audio/speech")
async def create_speech(request: SpeechRequest) -> Response:
    """Generate speech audio from text.

    Returns raw audio bytes in the requested format.

    Args:
        request: SpeechRequest with model, input text, voice, format, speed

    Returns:
        Response with audio bytes and appropriate content type
    """
    logger.info(
        f"TTS request: model={request.model}, text_len={len(request.input)}, voice={request.voice}"
    )

    try:
        audio_bytes, _sample_rate = await generate_speech(
            model_id=request.model,
            text=request.input,
            voice=request.voice,
            speed=request.speed,
            response_format=request.response_format,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.exception(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected TTS error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    content_type = AUDIO_CONTENT_TYPES.get(request.response_format, "application/octet-stream")

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"},
    )

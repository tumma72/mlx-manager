"""STT (speech-to-text) endpoint.

POST /v1/audio/transcriptions - Transcribe audio to text.
OpenAI-compatible: https://platform.openai.com/docs/api-reference/audio/createTranscription
"""

from fastapi import APIRouter, File, Form, UploadFile
from loguru import logger

from mlx_manager.mlx_server.schemas.openai import TranscriptionResponse
from mlx_manager.mlx_server.services.audio import transcribe_audio

router = APIRouter(tags=["audio"])


@router.post("/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(default=None),
) -> TranscriptionResponse:
    """Transcribe audio to text.

    Accepts multipart form data with an audio file.

    Args:
        file: Audio file upload (WAV, FLAC, MP3, etc.)
        model: STT model ID
        language: Optional language hint (e.g., "en", "es")

    Returns:
        TranscriptionResponse with transcribed text
    """
    logger.info(
        f"STT request: model={model}, "
        f"filename={file.filename}, language={language}"
    )

    # Read uploaded audio data
    audio_data = await file.read()

    if not audio_data:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Empty audio file")

    result = await transcribe_audio(
        model_id=model,
        audio_data=audio_data,
        language=language,
    )

    return TranscriptionResponse(text=result.get("text", ""))

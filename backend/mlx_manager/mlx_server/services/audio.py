"""Audio inference service for TTS and STT.

Delegates core generation logic to the model adapter.
This service handles pool access, logging, and observability.
"""

from loguru import logger

from mlx_manager.mlx_server.models.ir import AudioResult, TranscriptionResult

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


async def generate_speech(
    model_id: str,
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
) -> AudioResult:
    """Generate speech audio from text using a TTS model."""
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    assert loaded.adapter is not None, f"No adapter for model {model_id}"

    logger.info(
        f"Generating speech: model={model_id}, text_len={len(text)}, "
        f"voice={voice}, speed={speed}, format={response_format}"
    )

    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "tts_generation",
            model=model_id,
            text_length=len(text),
            voice=voice,
        )
        span_context.__enter__()

    try:
        result: AudioResult = await loaded.adapter.generate_speech(
            loaded.model,
            text,
            voice=voice,
            speed=speed,
            response_format=response_format,
        )

        logger.info(
            f"Speech generated: model={model_id}, "
            f"size={len(result.audio_bytes)} bytes, sample_rate={result.sample_rate}"
        )

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "tts_finished",
                model=model_id,
                audio_size_bytes=len(result.audio_bytes),
                sample_rate=result.sample_rate,
            )

        return result

    finally:
        if span_context:
            span_context.__exit__(None, None, None)


async def transcribe_audio(
    model_id: str,
    audio_data: bytes,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio to text using an STT model."""
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    assert loaded.adapter is not None, f"No adapter for model {model_id}"

    logger.info(
        f"Transcribing audio: model={model_id}, "
        f"audio_size={len(audio_data)} bytes, language={language}"
    )

    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "stt_transcription",
            model=model_id,
            audio_size_bytes=len(audio_data),
        )
        span_context.__enter__()

    try:
        result: TranscriptionResult = await loaded.adapter.transcribe(
            loaded.model,
            audio_data,
            language=language,
        )

        logger.info(f"Transcription complete: model={model_id}, text_len={len(result.text)}")

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "stt_finished",
                model=model_id,
                text_length=len(result.text),
            )

        return result

    finally:
        if span_context:
            span_context.__exit__(None, None, None)

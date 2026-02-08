"""Audio inference service for TTS and STT using mlx-audio.

CRITICAL: This module uses run_on_metal_thread utility to respect MLX Metal
thread affinity requirements.

TTS uses model.generate() which returns GenerationResult objects with audio data.
STT uses generate_transcription() which returns STTOutput with text segments.
"""

import io
from typing import Any

import numpy as np
from loguru import logger

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
) -> tuple[bytes, int]:
    """Generate speech audio from text using a TTS model.

    Args:
        model_id: HuggingFace model ID (must be a TTS audio model)
        text: Input text to convert to speech
        voice: Voice style identifier (model-specific)
        speed: Playback speed multiplier (1.0 = normal)
        response_format: Output audio format ("wav", "flac", "mp3")

    Returns:
        Tuple of (audio bytes in requested format, sample rate)

    Raises:
        RuntimeError: If model loading or generation fails
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool
    from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model

    logger.info(
        f"Generating speech: model={model_id}, text_len={len(text)}, "
        f"voice={voice}, speed={speed}, format={response_format}"
    )

    # LogFire span
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

        def run_tts() -> tuple[bytes, int]:
            """Run TTS generation in dedicated thread (owns Metal context)."""
            import mlx.core as mx

            # Generate audio using model.generate()
            # Returns an iterable of GenerationResult objects
            gen_kwargs: dict[str, Any] = {
                "text": text,
                "voice": voice,
                "speed": speed,
                "verbose": False,
            }

            results = model.generate(**gen_kwargs)

            # Collect all audio segments
            audio_segments = []
            sample_rate = getattr(model, "sample_rate", 24000)

            for result in results:
                audio_segments.append(result.audio)
                if hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate

            if not audio_segments:
                raise RuntimeError("TTS model produced no audio output")

            # Concatenate segments if multiple
            if len(audio_segments) > 1:
                audio = mx.concatenate(audio_segments, axis=0)
            else:
                audio = audio_segments[0]

            # Ensure computation is complete before converting
            # NOTE: mx.eval() is MLX framework's tensor evaluation, not Python eval()
            mx.eval(audio)

            # Convert to numpy
            audio_np = np.array(audio.tolist())

            # Write to bytes buffer using soundfile (used internally by mlx-audio)
            import soundfile as sf

            buffer = io.BytesIO()
            sf.write(buffer, audio_np, sample_rate, format=response_format.upper())
            audio_bytes = buffer.getvalue()

            return (audio_bytes, sample_rate)

        audio_bytes, sample_rate = await run_on_metal_thread(
            run_tts, error_context="TTS generation failed"
        )

        logger.info(
            f"Speech generated: model={model_id}, "
            f"size={len(audio_bytes)} bytes, sample_rate={sample_rate}"
        )

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "tts_finished",
                model=model_id,
                audio_size_bytes=len(audio_bytes),
                sample_rate=sample_rate,
            )

        return audio_bytes, sample_rate

    finally:
        if span_context:
            span_context.__exit__(None, None, None)


async def transcribe_audio(
    model_id: str,
    audio_data: bytes,
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe audio to text using an STT model.

    Args:
        model_id: HuggingFace model ID (must be an STT audio model)
        audio_data: Raw audio bytes (WAV, FLAC, MP3, etc.)
        language: Optional language hint for transcription

    Returns:
        Dict with "text" (str) and optionally "segments" (list of dicts),
        "language" (str), and timing metadata.

    Raises:
        RuntimeError: If model loading or transcription fails
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool
    from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model

    logger.info(
        f"Transcribing audio: model={model_id}, "
        f"audio_size={len(audio_data)} bytes, language={language}"
    )

    # LogFire span
    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "stt_transcription",
            model=model_id,
            audio_size_bytes=len(audio_data),
        )
        span_context.__enter__()

    try:

        def run_stt() -> dict[str, Any]:
            """Run STT transcription in dedicated thread (owns Metal context)."""
            import os
            import tempfile

            from mlx_audio.stt.generate import generate_transcription

            # Write audio data to a temporary file since generate_transcription
            # expects a file path for the audio parameter
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            try:
                # Build kwargs
                kwargs: dict[str, Any] = {}
                if language:
                    kwargs["language"] = language

                # Run transcription
                segments = generate_transcription(
                    model=model,
                    audio=tmp_path,
                    verbose=False,
                    **kwargs,
                )

                # Extract results from STTOutput
                result_dict: dict[str, Any] = {
                    "text": getattr(segments, "text", ""),
                }

                if hasattr(segments, "segments") and segments.segments:
                    result_dict["segments"] = segments.segments

                if hasattr(segments, "language") and segments.language:
                    result_dict["language"] = segments.language

                return result_dict

            finally:
                os.unlink(tmp_path)

        result = await run_on_metal_thread(run_stt, error_context="STT transcription failed")

        logger.info(
            f"Transcription complete: model={model_id}, text_len={len(result.get('text', ''))}"
        )

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "stt_finished",
                model=model_id,
                text_length=len(result.get("text", "")),
            )

        return result

    finally:
        if span_context:
            span_context.__exit__(None, None, None)

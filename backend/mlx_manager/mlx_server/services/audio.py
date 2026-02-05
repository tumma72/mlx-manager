"""Audio inference service for TTS and STT using mlx-audio.

CRITICAL: This module uses the same queue-based threading pattern as inference.py
and embeddings.py to respect MLX Metal thread affinity requirements.

TTS uses model.generate() which returns GenerationResult objects with audio data.
STT uses generate_transcription() which returns STTOutput with text segments.
"""

import asyncio
import io
import threading
from queue import Queue
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
        # Queue for passing result from generation thread
        result_queue: Queue[tuple[bytes, int] | Exception] = Queue()

        def run_tts() -> None:
            """Run TTS generation in dedicated thread (owns Metal context)."""
            try:
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

                # Write to bytes buffer using mlx-audio's audio_write
                from mlx_audio.tts.generate import audio_write

                buffer = io.BytesIO()
                audio_write(buffer, audio_np, sample_rate, format=response_format)
                audio_bytes = buffer.getvalue()

                result_queue.put((audio_bytes, sample_rate))

            except Exception as e:
                result_queue.put(e)

        # Start generation thread
        gen_thread = threading.Thread(target=run_tts, daemon=True)
        gen_thread.start()

        # Wait for result (with 5 minute timeout)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=300))

        gen_thread.join(timeout=1.0)

        # Check for exception
        if isinstance(result, Exception):
            raise RuntimeError(f"TTS generation failed: {result}") from result

        audio_bytes, sample_rate = result

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

        # Clear cache after generation
        from mlx_manager.mlx_server.utils.memory import clear_cache

        clear_cache()


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
        # Queue for passing result from transcription thread
        result_queue: Queue[dict[str, Any] | Exception] = Queue()

        def run_stt() -> None:
            """Run STT transcription in dedicated thread (owns Metal context)."""
            try:
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

                    result_queue.put(result_dict)

                finally:
                    os.unlink(tmp_path)

            except Exception as e:
                result_queue.put(e)

        # Start transcription thread
        stt_thread = threading.Thread(target=run_stt, daemon=True)
        stt_thread.start()

        # Wait for result (with 5 minute timeout)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=300))

        stt_thread.join(timeout=1.0)

        # Check for exception
        if isinstance(result, Exception):
            raise RuntimeError(f"STT transcription failed: {result}") from result

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

        # Clear cache after transcription
        from mlx_manager.mlx_server.utils.memory import clear_cache

        clear_cache()

"""Audio model probe strategy.

Detects TTS/STT capabilities based on model architecture,
then validates with lightweight generation tests.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


# Architecture patterns for TTS and STT models
_TTS_ARCHITECTURES = {"kokoro", "bark", "speecht5", "parler", "dia", "outetts", "spark", "soprano"}
_STT_ARCHITECTURES = {"whisper", "parakeet"}


class AudioProbe:
    """Probe strategy for audio models (TTS/STT)."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.AUDIO

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        result.model_type = ModelType.AUDIO

        # Step 1: Detect audio capabilities (TTS, STT, or both)
        yield ProbeStep(step="detect_audio_type", status="running")
        try:
            is_tts, is_stt = _detect_audio_capabilities(model_id)
            result.supports_tts = is_tts
            result.supports_stt = is_stt
            yield ProbeStep(
                step="detect_audio_type",
                status="completed",
                capability="supports_tts",
                value=is_tts,
            )
            yield ProbeStep(
                step="detect_audio_type",
                status="completed",
                capability="supports_stt",
                value=is_stt,
            )
        except Exception as e:
            logger.warning(f"Audio type detection failed for {model_id}: {e}")
            yield ProbeStep(step="detect_audio_type", status="failed", error=str(e))
            return  # Can't continue without knowing the type

        # Step 2: Test TTS if supported
        if is_tts:
            yield ProbeStep(step="test_tts", status="running")
            try:
                tts_ok = await _test_tts(loaded)
                yield ProbeStep(
                    step="test_tts",
                    status="completed" if tts_ok else "failed",
                    error=None if tts_ok else "TTS generation returned empty audio",
                )
            except Exception as e:
                logger.warning(f"TTS test failed for {model_id}: {e}")
                yield ProbeStep(step="test_tts", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_tts", status="skipped")

        # Step 3: Test STT if supported
        if is_stt:
            yield ProbeStep(step="test_stt", status="running")
            try:
                stt_ok = await _test_stt(loaded)
                yield ProbeStep(
                    step="test_stt",
                    status="completed" if stt_ok else "failed",
                    error=None if stt_ok else "STT transcription returned empty text",
                )
            except Exception as e:
                logger.warning(f"STT test failed for {model_id}: {e}")
                yield ProbeStep(step="test_stt", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_stt", status="skipped")


def _detect_audio_capabilities(model_id: str) -> tuple[bool, bool]:
    """Detect whether the model supports TTS, STT, or both.

    Uses config.json architecture and model_type fields to determine
    capabilities without loading the model.
    """
    from mlx_manager.utils.model_detection import read_model_config

    config = read_model_config(model_id)
    name_lower = model_id.lower()

    is_tts = False
    is_stt = False

    if config:
        # Check architecture
        arch_list = config.get("architectures", [])
        arch_str = arch_list[0].lower() if arch_list else ""
        model_type = config.get("model_type", "").lower()

        for pattern in _TTS_ARCHITECTURES:
            if pattern in arch_str or pattern in model_type:
                is_tts = True

        for pattern in _STT_ARCHITECTURES:
            if pattern in arch_str or pattern in model_type:
                is_stt = True

        # Config-based hints
        if "tts_config" in config or "vocoder_config" in config:
            is_tts = True
        if "stt_config" in config:
            is_stt = True

    # Name-based fallback
    if not is_tts and not is_stt:
        for pattern in _TTS_ARCHITECTURES:
            if pattern in name_lower:
                is_tts = True
        for pattern in _STT_ARCHITECTURES:
            if pattern in name_lower:
                is_stt = True
        if "tts" in name_lower:
            is_tts = True
        if "stt" in name_lower or "asr" in name_lower:
            is_stt = True

    return is_tts, is_stt


async def _test_tts(loaded: LoadedModel) -> bool:
    """Test TTS by generating a short audio clip from minimal text."""
    try:
        from mlx_manager.mlx_server.services.audio import generate_speech

        audio_bytes, sample_rate = await generate_speech(
            model_id=loaded.model_id,
            text="Hello.",
            voice="af_heart",  # Kokoro default voice
            speed=1.0,
            response_format="wav",
        )
        return len(audio_bytes) > 0
    except Exception as e:
        logger.debug(f"TTS test error: {e}")
        raise


async def _test_stt(loaded: LoadedModel) -> bool:
    """Test STT by transcribing a short silent audio clip."""
    try:
        import struct

        from mlx_manager.mlx_server.services.audio import transcribe_audio

        # Generate a minimal WAV file with silence (0.5s at 16kHz)
        sample_rate = 16000
        duration = 0.5
        num_samples = int(sample_rate * duration)
        audio_data = struct.pack(f"<{num_samples}h", *([0] * num_samples))

        # WAV header
        wav_header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(audio_data),
            b"WAVE",
            b"fmt ",
            16,  # chunk size
            1,  # PCM format
            1,  # mono
            sample_rate,
            sample_rate * 2,  # byte rate
            2,  # block align
            16,  # bits per sample
            b"data",
            len(audio_data),
        )

        wav_bytes = wav_header + audio_data
        result = await transcribe_audio(
            model_id=loaded.model_id,
            audio_data=wav_bytes,
        )
        # STT should return something (even empty text for silence)
        return isinstance(result, dict) and "text" in result
    except Exception as e:
        logger.debug(f"STT test error: {e}")
        raise

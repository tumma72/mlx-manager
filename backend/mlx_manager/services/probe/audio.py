"""Audio model probe strategy.

Detects TTS/STT capabilities based on model architecture,
then validates with lightweight generation tests.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .base import BaseProbe
from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


# Architecture patterns for TTS and STT models
_TTS_ARCHITECTURES = {"kokoro", "bark", "speecht5", "parler", "dia", "outetts", "spark", "soprano"}
_STT_ARCHITECTURES = {"whisper", "parakeet"}

_TTS_TEST_TEXT = "Welcome to MLX Manager, where your locally downloaded models take flight!"

# Pre-generated WAV file (Kokoro TTS) used for STT testing
_STT_TEST_WAV = Path(__file__).parent / "stt_test.wav"


class AudioProbe(BaseProbe):
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
        tts_audio_bytes: bytes | None = None
        if is_tts:
            yield ProbeStep(step="test_tts", status="running")
            try:
                tts_ok, audio_bytes = await _test_tts(loaded)
                audio_b64 = None
                if tts_ok and audio_bytes:
                    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                    tts_audio_bytes = audio_bytes
                yield ProbeStep(
                    step="test_tts",
                    status="completed" if tts_ok else "failed",
                    value=audio_b64,
                    error=None if tts_ok else "TTS generation returned empty audio",
                )
            except Exception as e:
                logger.warning(f"TTS test failed for {model_id}: {e}")
                yield ProbeStep(step="test_tts", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_tts", status="skipped")

        # Step 3: Test STT if supported
        if is_stt:
            # Use TTS output from step 2, or the pre-generated fixture
            stt_input = tts_audio_bytes or _load_stt_test_wav()

            yield ProbeStep(step="test_stt", status="running")
            try:
                stt_ok, transcript, used_audio = await _test_stt(loaded, stt_input)

                stt_value: dict[str, Any] | None = None
                if stt_ok and transcript:
                    stt_value = {"transcript": transcript}
                    if used_audio is not None:
                        stt_value["audio_b64"] = base64.b64encode(used_audio).decode("ascii")

                yield ProbeStep(
                    step="test_stt",
                    status="completed" if stt_ok else "failed",
                    value=stt_value,
                    error=None if stt_ok else "STT transcription returned empty text",
                )
            except Exception as e:
                logger.warning(f"STT test failed for {model_id}: {e}")
                yield ProbeStep(step="test_stt", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_stt", status="skipped")


def _load_stt_test_wav() -> bytes | None:
    """Load the pre-generated STT test WAV file."""
    if _STT_TEST_WAV.exists():
        return _STT_TEST_WAV.read_bytes()
    logger.warning(f"STT test WAV not found at {_STT_TEST_WAV}")
    return None


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


async def _test_tts(loaded: LoadedModel) -> tuple[bool, bytes | None]:
    """Test TTS by generating a short audio clip from minimal text."""
    try:
        from mlx_manager.mlx_server.services.audio import generate_speech

        audio_bytes, sample_rate = await generate_speech(
            model_id=loaded.model_id,
            text=_TTS_TEST_TEXT,
            voice="af_heart",  # Kokoro default voice
            speed=1.0,
            response_format="wav",
        )
        if len(audio_bytes) > 0:
            return True, audio_bytes
        return False, None
    except Exception as e:
        logger.debug(f"TTS test error: {e}")
        raise


async def _test_stt(
    loaded: LoadedModel, audio_bytes: bytes | None = None
) -> tuple[bool, str | None, bytes | None]:
    """Test STT by transcribing audio.

    If audio_bytes is provided (from TTS test or fixture), uses that.
    Otherwise falls back to a short silent WAV clip.

    Returns (success, transcript, audio_bytes_used).
    """
    try:
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        if audio_bytes is None:
            import struct

            # Generate a minimal WAV file with silence (0.5s at 16kHz)
            sample_rate = 16000
            duration = 0.5
            num_samples = int(sample_rate * duration)
            audio_data = struct.pack(f"<{num_samples}h", *([0] * num_samples))

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

            audio_bytes = wav_header + audio_data

        result = await transcribe_audio(
            model_id=loaded.model_id,
            audio_data=audio_bytes,
        )
        if isinstance(result, dict) and "text" in result:
            return True, result["text"] or None, audio_bytes
        return False, None, audio_bytes
    except Exception as e:
        logger.debug(f"STT test error: {e}")
        raise

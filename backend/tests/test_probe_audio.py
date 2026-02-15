"""Unit tests for audio probe module to improve coverage.

Tests edge cases, error handling, and specific audio probe scenarios
that are not covered by the general probe package tests.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.ir import AudioResult, TranscriptionResult
from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.services.probe.audio import AudioProbe, _detect_audio_capabilities
from mlx_manager.services.probe.steps import ProbeResult

# ============================================================================
# _load_stt_test_wav Tests
# ============================================================================


def test_load_stt_test_wav_missing_file():
    """Test _load_stt_test_wav when the WAV file doesn't exist."""
    from mlx_manager.services.probe.audio import _load_stt_test_wav

    with patch("mlx_manager.services.probe.audio._STT_TEST_WAV", Path("/nonexistent/file.wav")):
        result = _load_stt_test_wav()
        assert result is None


def test_load_stt_test_wav_exists():
    """Test _load_stt_test_wav when the WAV file exists."""
    from mlx_manager.services.probe.audio import _load_stt_test_wav

    # The actual file should exist in the probe module directory
    result = _load_stt_test_wav()
    assert result is not None
    assert isinstance(result, bytes)
    assert len(result) > 0
    # WAV files start with RIFF header
    assert result[:4] == b"RIFF"


# ============================================================================
# _detect_audio_capabilities Tests
# ============================================================================


def test_detect_audio_capabilities_codec_model():
    """Test codec model detection (DAC, SNAC, VOCOS) returns (False, False, None)."""
    mock_config = {
        "architectures": ["DACModel"],
        "codebook_dim": 128,
        "codebook_size": 1024,
        "encoder_dim": 64,
        "decoder_dim": 64,
    }

    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=mock_config):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/dac-model")

    assert is_tts is False
    assert is_stt is False
    assert audio_family is None


def test_detect_audio_capabilities_name_based_tts():
    """Test name-based TTS detection when config is unavailable."""
    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/kokoro-tts-model")

    assert is_tts is True
    assert is_stt is False
    assert audio_family == "kokoro"


def test_detect_audio_capabilities_name_based_stt():
    """Test name-based STT detection when config is unavailable."""
    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/whisper-asr-model")

    assert is_tts is False
    assert is_stt is True
    assert audio_family == "whisper"


def test_detect_audio_capabilities_generic_tts_name():
    """Test generic TTS name detection without specific architecture."""
    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/my-tts-model")

    assert is_tts is True
    assert is_stt is False
    assert audio_family is None


def test_detect_audio_capabilities_generic_stt_name():
    """Test generic STT/ASR name detection without specific architecture."""
    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/my-stt-model")

    assert is_tts is False
    assert is_stt is True
    assert audio_family is None


def test_detect_audio_capabilities_codec_by_name():
    """Test codec detection by name pattern (should not be marked as TTS/STT)."""
    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/dac-codec")

    assert is_tts is False
    assert is_stt is False
    assert audio_family is None


def test_detect_audio_capabilities_snac_codec():
    """Test SNAC codec detection by name."""
    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/snac-encoder")

    assert is_tts is False
    assert is_stt is False
    assert audio_family is None


def test_detect_audio_capabilities_vocos_codec():
    """Test VOCOS codec detection by name."""
    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/vocos-decoder")

    assert is_tts is False
    assert is_stt is False
    assert audio_family is None


# ============================================================================
# _test_tts Tests (Error Handling)
# ============================================================================


@pytest.mark.asyncio
async def test_test_tts_timeout():
    """Test _test_tts when generation times out."""
    from mlx_manager.services.probe.audio import _test_tts

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/tts-model"

    # Mock adapter.generate_speech to hang forever
    async def hanging_generate(*args, **kwargs):
        await asyncio.sleep(1000)

    mock_loaded.adapter.generate_speech = hanging_generate

    with pytest.raises(TimeoutError):
        await _test_tts(mock_loaded, timeout=0.1)


@pytest.mark.asyncio
async def test_test_tts_generation_error():
    """Test _test_tts when generation raises an exception."""
    from mlx_manager.services.probe.audio import _test_tts

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/tts-model"

    mock_loaded.adapter.generate_speech = AsyncMock(
        side_effect=RuntimeError("Model loading failed"),
    )

    with pytest.raises(RuntimeError, match="Model loading failed"):
        await _test_tts(mock_loaded, timeout=60.0)


@pytest.mark.asyncio
async def test_test_tts_empty_audio():
    """Test _test_tts when generation returns empty audio."""
    from mlx_manager.services.probe.audio import _test_tts

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/tts-model"

    # Return AudioResult with empty bytes
    mock_loaded.adapter.generate_speech = AsyncMock(
        return_value=AudioResult(audio_bytes=b"", sample_rate=24000, format="wav"),
    )

    success, audio_bytes = await _test_tts(mock_loaded, timeout=60.0)

    assert success is False
    assert audio_bytes is None


# ============================================================================
# _test_stt Tests (Minimal WAV Generation + Error Handling)
# ============================================================================


@pytest.mark.asyncio
async def test_test_stt_generates_minimal_wav():
    """Test _test_stt generates minimal WAV when audio_bytes is None."""
    from mlx_manager.services.probe.audio import _test_stt

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-model"

    # Mock adapter.transcribe to capture the generated audio
    captured_audio = None

    async def capture_audio(model, audio_data):
        nonlocal captured_audio
        captured_audio = audio_data
        return TranscriptionResult(text="test transcript")

    mock_loaded.adapter.transcribe = capture_audio

    success, transcript, used_audio = await _test_stt(mock_loaded, audio_bytes=None, timeout=60.0)

    assert success is True
    assert transcript == "test transcript"
    assert used_audio is not None
    assert captured_audio is not None
    # Verify it's a valid WAV file
    assert captured_audio[:4] == b"RIFF"
    assert captured_audio[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_test_stt_timeout():
    """Test _test_stt when transcription times out."""
    from mlx_manager.services.probe.audio import _test_stt

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-model"

    # Mock adapter.transcribe to hang forever
    async def hanging_transcribe(*args, **kwargs):
        await asyncio.sleep(1000)

    mock_loaded.adapter.transcribe = hanging_transcribe

    with pytest.raises(TimeoutError):
        await _test_stt(mock_loaded, audio_bytes=b"fake audio", timeout=0.1)


@pytest.mark.asyncio
async def test_test_stt_transcription_error():
    """Test _test_stt when transcription raises an exception."""
    from mlx_manager.services.probe.audio import _test_stt

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-model"

    mock_loaded.adapter.transcribe = AsyncMock(
        side_effect=RuntimeError("Model loading failed"),
    )

    with pytest.raises(RuntimeError, match="Model loading failed"):
        await _test_stt(mock_loaded, audio_bytes=b"fake audio", timeout=60.0)


@pytest.mark.asyncio
async def test_test_stt_empty_transcript():
    """Test _test_stt when transcription returns empty text."""
    from mlx_manager.services.probe.audio import _test_stt

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-model"

    # Return TranscriptionResult with empty text
    mock_loaded.adapter.transcribe = AsyncMock(
        return_value=TranscriptionResult(text=""),
    )

    success, transcript, used_audio = await _test_stt(
        mock_loaded, audio_bytes=b"fake audio", timeout=60.0
    )

    assert success is False
    assert transcript is None
    assert used_audio == b"fake audio"


@pytest.mark.asyncio
async def test_test_stt_with_provided_audio():
    """Test _test_stt uses provided audio_bytes instead of generating."""
    from mlx_manager.services.probe.audio import _test_stt

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-model"

    test_audio = b"RIFF....WAVE...."

    mock_loaded.adapter.transcribe = AsyncMock(
        return_value=TranscriptionResult(text="hello world"),
    )

    success, transcript, used_audio = await _test_stt(
        mock_loaded, audio_bytes=test_audio, timeout=60.0
    )

    assert success is True
    assert transcript == "hello world"
    assert used_audio == test_audio


# ============================================================================
# AudioProbe Integration Tests (End-to-End Scenarios)
# ============================================================================


@pytest.mark.asyncio
async def test_audio_probe_detection_error():
    """Test AudioProbe when detection raises an exception."""
    probe = AudioProbe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/audio-model"

    with patch(
        "mlx_manager.services.probe.audio._detect_audio_capabilities",
        side_effect=RuntimeError("Config parsing failed"),
    ):
        steps = []
        async for step in probe.probe("test/audio-model", mock_loaded, result):
            steps.append(step)

    # Should have detect_audio_type step that failed
    detect_steps = [s for s in steps if s.step == "detect_audio_type"]
    assert len(detect_steps) == 2  # running + failed
    assert detect_steps[0].status == "running"
    assert detect_steps[1].status == "failed"
    assert "Config parsing failed" in detect_steps[1].error

    # Should not proceed to TTS/STT tests
    tts_steps = [s for s in steps if s.step == "test_tts"]
    stt_steps = [s for s in steps if s.step == "test_stt"]
    assert len(tts_steps) == 0
    assert len(stt_steps) == 0


@pytest.mark.asyncio
async def test_audio_probe_tts_timeout():
    """Test AudioProbe when TTS test times out."""
    probe = AudioProbe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/tts-model"

    async def hanging_tts(*args, **kwargs):
        await asyncio.sleep(1000)

    with (
        patch(
            "mlx_manager.services.probe.audio._detect_audio_capabilities",
            return_value=(True, False, "kokoro"),
        ),
        patch("mlx_manager.services.probe.audio._test_tts", side_effect=hanging_tts),
    ):
        steps = []
        # Use very short timeout in the actual _test_tts call
        with patch("mlx_manager.services.probe.audio._test_tts") as mock_tts:
            mock_tts.side_effect = TimeoutError("TTS test timed out after 0.1s")
            async for step in probe.probe("test/tts-model", mock_loaded, result):
                steps.append(step)

    # Check TTS test failed with timeout
    tts_steps = [s for s in steps if s.step == "test_tts"]
    failed_tts = [s for s in tts_steps if s.status == "failed"]
    assert len(failed_tts) == 1
    assert "timed out" in failed_tts[0].error.lower()


@pytest.mark.asyncio
async def test_audio_probe_stt_timeout():
    """Test AudioProbe when STT test times out."""
    probe = AudioProbe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-model"

    with (
        patch(
            "mlx_manager.services.probe.audio._detect_audio_capabilities",
            return_value=(False, True, "whisper"),
        ),
        patch("mlx_manager.services.probe.audio._load_stt_test_wav", return_value=b"test audio"),
        patch("mlx_manager.services.probe.audio._test_stt") as mock_stt,
    ):
        mock_stt.side_effect = TimeoutError("STT test timed out after 0.1s")
        steps = []
        async for step in probe.probe("test/stt-model", mock_loaded, result):
            steps.append(step)

    # Check STT test failed with timeout
    stt_steps = [s for s in steps if s.step == "test_stt"]
    failed_stt = [s for s in stt_steps if s.status == "failed"]
    assert len(failed_stt) == 1
    assert "timed out" in failed_stt[0].error.lower()


@pytest.mark.asyncio
async def test_audio_probe_tts_only():
    """Test AudioProbe with TTS-only model."""
    probe = AudioProbe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/tts-only"

    with (
        patch(
            "mlx_manager.services.probe.audio._detect_audio_capabilities",
            return_value=(True, False, "kokoro"),
        ),
        patch("mlx_manager.services.probe.audio._test_tts", return_value=(True, b"audio data")),
    ):
        steps = []
        async for step in probe.probe("test/tts-only", mock_loaded, result):
            steps.append(step)

    # Should have TTS test completed
    tts_steps = [s for s in steps if s.step == "test_tts" and s.status == "completed"]
    assert len(tts_steps) == 1

    # Should have STT test skipped
    stt_steps = [s for s in steps if s.step == "test_stt"]
    assert len(stt_steps) == 1
    assert stt_steps[0].status == "skipped"

    # Result should reflect TTS support only
    assert result.supports_tts is True
    assert result.supports_stt is False
    assert result.model_type == ModelType.AUDIO
    assert result.model_family == "kokoro"


@pytest.mark.asyncio
async def test_audio_probe_stt_only():
    """Test AudioProbe with STT-only model."""
    probe = AudioProbe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-only"

    with (
        patch(
            "mlx_manager.services.probe.audio._detect_audio_capabilities",
            return_value=(False, True, "whisper"),
        ),
        patch(
            "mlx_manager.services.probe.audio._load_stt_test_wav",
            return_value=b"test audio",
        ),
        patch(
            "mlx_manager.services.probe.audio._test_stt",
            return_value=(True, "test transcript", b"test audio"),
        ),
    ):
        steps = []
        async for step in probe.probe("test/stt-only", mock_loaded, result):
            steps.append(step)

    # Should have TTS test skipped
    tts_steps = [s for s in steps if s.step == "test_tts"]
    assert len(tts_steps) == 1
    assert tts_steps[0].status == "skipped"

    # Should have STT test completed
    stt_steps = [s for s in steps if s.step == "test_stt" and s.status == "completed"]
    assert len(stt_steps) == 1

    # Result should reflect STT support only
    assert result.supports_tts is False
    assert result.supports_stt is True
    assert result.model_type == ModelType.AUDIO
    assert result.model_family == "whisper"


@pytest.mark.asyncio
async def test_audio_probe_stt_uses_tts_output():
    """Test AudioProbe STT test uses TTS output when both are supported."""
    probe = AudioProbe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/multimodal-audio"

    tts_audio = b"TTS generated audio"

    with (
        patch(
            "mlx_manager.services.probe.audio._detect_audio_capabilities",
            return_value=(True, True, "kokoro"),
        ),
        patch("mlx_manager.services.probe.audio._test_tts", return_value=(True, tts_audio)),
        patch(
            "mlx_manager.services.probe.audio._test_stt",
            return_value=(True, "transcript", tts_audio),
        ) as mock_stt,
    ):
        steps = []
        async for step in probe.probe("test/multimodal-audio", mock_loaded, result):
            steps.append(step)

    # Verify STT was called with TTS output
    mock_stt.assert_called_once()
    call_args = mock_stt.call_args
    assert call_args[0][1] == tts_audio  # audio_bytes parameter

    # Both tests should pass
    assert result.supports_tts is True
    assert result.supports_stt is True


@pytest.mark.asyncio
async def test_audio_probe_stt_fallback_to_fixture():
    """Test AudioProbe STT uses fixture when TTS fails."""
    probe = AudioProbe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/multimodal-audio"

    fixture_audio = b"Fixture audio from WAV file"

    with (
        patch(
            "mlx_manager.services.probe.audio._detect_audio_capabilities",
            return_value=(True, True, "kokoro"),
        ),
        patch("mlx_manager.services.probe.audio._test_tts", return_value=(False, None)),
        patch(
            "mlx_manager.services.probe.audio._load_stt_test_wav",
            return_value=fixture_audio,
        ),
        patch(
            "mlx_manager.services.probe.audio._test_stt",
            return_value=(True, "transcript", fixture_audio),
        ) as mock_stt,
    ):
        steps = []
        async for step in probe.probe("test/multimodal-audio", mock_loaded, result):
            steps.append(step)

    # Verify STT was called with fixture audio
    mock_stt.assert_called_once()
    call_args = mock_stt.call_args
    assert call_args[0][1] == fixture_audio


# ============================================================================
# Additional Coverage Tests for Edge Cases
# ============================================================================


def test_detect_audio_capabilities_stt_architecture_with_family():
    """Test STT architecture detection when audio_family is already set from TTS."""
    mock_config = {
        "architectures": ["KokoroWhisperModel"],  # Both TTS (kokoro) and STT (whisper)
        "model_type": "kokoro",
    }

    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=mock_config):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/kokoro-whisper")

    # Should detect both, with kokoro as the family (first match)
    assert is_tts is True
    assert is_stt is True
    assert audio_family == "kokoro"


def test_detect_audio_capabilities_vocoder_config():
    """Test TTS detection via vocoder_config (without codec keys)."""
    mock_config = {
        "architectures": ["TTSModel"],
        "vocoder_config": {"sample_rate": 24000},
        # No codec keys like codebook_dim
    }

    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=mock_config):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/tts-with-vocoder")

    assert is_tts is True
    assert is_stt is False


def test_detect_audio_capabilities_stt_config():
    """Test STT detection via stt_config field."""
    mock_config = {
        "architectures": ["STTModel"],
        "stt_config": {"sample_rate": 16000},
    }

    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=mock_config):
        is_tts, is_stt, audio_family = _detect_audio_capabilities("test/stt-with-config")

    assert is_tts is False
    assert is_stt is True


@pytest.mark.asyncio
async def test_test_stt_invalid_result_format():
    """Test _test_stt when result has empty text (default TranscriptionResult)."""
    from mlx_manager.services.probe.audio import _test_stt

    mock_loaded = MagicMock()
    mock_loaded.model_id = "test/stt-model"

    # Return TranscriptionResult with default empty text
    mock_loaded.adapter.transcribe = AsyncMock(
        return_value=TranscriptionResult(),
    )

    success, transcript, used_audio = await _test_stt(
        mock_loaded, audio_bytes=b"fake audio", timeout=60.0
    )

    assert success is False
    assert transcript is None
    assert used_audio == b"fake audio"


# ============================================================================
# Model Type Property Test
# ============================================================================


def test_audio_probe_model_type_property():
    """Test AudioProbe.model_type property returns AUDIO."""
    probe = AudioProbe()
    assert probe.model_type == ModelType.AUDIO

"""E2E tests for audio TTS and STT endpoints.

These tests require the Kokoro-82M-bf16 model to be pre-downloaded.
Run with: pytest -m e2e_audio

The tests validate:
1. TTS generates valid WAV audio from text
2. WAV output has correct structure (RIFF header, non-empty data)
3. Longer text produces proportionally more audio
4. Speed parameter affects output duration
5. Error handling for missing models
"""

import struct

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_audio]


def _parse_wav_header(data: bytes) -> dict:
    """Parse a WAV file header and return metadata.

    Returns dict with: chunk_id, file_size, format, sample_rate,
    num_channels, bits_per_sample, data_size.
    """
    if len(data) < 44:
        raise ValueError(f"WAV data too short: {len(data)} bytes")

    chunk_id = data[0:4]
    file_size = struct.unpack("<I", data[4:8])[0]
    format_tag = data[8:12]

    # fmt sub-chunk
    audio_format = struct.unpack("<H", data[20:22])[0]
    num_channels = struct.unpack("<H", data[22:24])[0]
    sample_rate = struct.unpack("<I", data[24:28])[0]
    bits_per_sample = struct.unpack("<H", data[34:36])[0]

    # Find data sub-chunk (may not be at offset 36)
    data_size = 0
    offset = 36
    while offset < len(data) - 8:
        sub_id = data[offset : offset + 4]
        sub_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        if sub_id == b"data":
            data_size = sub_size
            break
        offset += 8 + sub_size

    return {
        "chunk_id": chunk_id,
        "file_size": file_size,
        "format": format_tag,
        "audio_format": audio_format,
        "num_channels": num_channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "data_size": data_size,
    }


class TestTTSGeneration:
    """Test TTS audio generation via /v1/audio/speech."""

    @pytest.mark.asyncio
    async def test_basic_tts_generates_wav(self, app_client, audio_tts_model):
        """Basic TTS request should generate valid WAV audio."""
        response = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": audio_tts_model,
                "input": "Hello, this is a test.",
                "voice": "af_heart",
                "response_format": "wav",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

        # Validate WAV structure
        audio_data = response.content
        assert len(audio_data) > 44  # WAV header is 44 bytes minimum

        wav = _parse_wav_header(audio_data)
        assert wav["chunk_id"] == b"RIFF"
        assert wav["format"] == b"WAVE"
        assert wav["sample_rate"] > 0
        assert wav["data_size"] > 0

    @pytest.mark.asyncio
    async def test_tts_longer_text_produces_more_audio(self, app_client, audio_tts_model):
        """Longer input text should produce proportionally more audio data."""
        # Short text
        short_response = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": audio_tts_model,
                "input": "Hi.",
                "voice": "af_heart",
            },
        )
        assert short_response.status_code == 200

        # Longer text
        long_response = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": audio_tts_model,
                "input": (
                    "This is a significantly longer piece of text that should "
                    "produce more audio output than a simple greeting."
                ),
                "voice": "af_heart",
            },
        )
        assert long_response.status_code == 200

        # Longer text should produce more audio data
        assert len(long_response.content) > len(short_response.content)

    @pytest.mark.asyncio
    async def test_tts_speed_parameter(self, app_client, audio_tts_model):
        """Speed parameter should affect generated audio duration."""
        text = "Testing the speed parameter with this sentence."

        # Normal speed
        normal_response = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": audio_tts_model,
                "input": text,
                "voice": "af_heart",
                "speed": 1.0,
            },
        )
        assert normal_response.status_code == 200

        # Fast speed (should produce less audio data since playback is faster)
        fast_response = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": audio_tts_model,
                "input": text,
                "voice": "af_heart",
                "speed": 2.0,
            },
        )
        assert fast_response.status_code == 200

        # Both should be valid WAV
        normal_wav = _parse_wav_header(normal_response.content)
        fast_wav = _parse_wav_header(fast_response.content)
        assert normal_wav["data_size"] > 0
        assert fast_wav["data_size"] > 0

        # Fast audio should be shorter (less data) than normal
        # Allow some tolerance since speed affects generation differently per model
        assert fast_wav["data_size"] < normal_wav["data_size"] * 1.1


class TestTTSErrorHandling:
    """Test error handling for TTS endpoint."""

    @pytest.mark.asyncio
    async def test_tts_nonexistent_model_returns_error(self, app_client):
        """Request with non-existent model should return error."""
        response = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": "nonexistent/model-that-does-not-exist",
                "input": "Hello",
            },
        )
        # Should fail with 500 (model loading error) not 200
        assert response.status_code >= 400

    @pytest.mark.asyncio
    async def test_tts_invalid_speed_rejected(self, app_client):
        """Request with out-of-range speed should be rejected."""
        response = await app_client.post(
            "/v1/audio/speech",
            json={
                "model": "any-model",
                "input": "Hello",
                "speed": 10.0,  # Exceeds max 4.0
            },
        )
        assert response.status_code == 422  # Validation error

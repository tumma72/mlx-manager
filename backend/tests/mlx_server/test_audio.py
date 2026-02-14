"""Tests for audio (TTS/STT) schemas, endpoints, and service."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from mlx_manager.mlx_server.schemas.openai import (
    SpeechRequest,
    TranscriptionResponse,
)


class TestAudioSchemas:
    """Tests for audio request/response schemas."""

    def test_speech_request_defaults(self):
        """SpeechRequest should have sensible defaults."""
        req = SpeechRequest(model="test-tts", input="Hello world")
        assert req.voice == "af_heart"
        assert req.speed == 1.0
        assert req.response_format == "wav"

    def test_speech_request_custom_fields(self):
        """SpeechRequest should accept custom field values."""
        req = SpeechRequest(
            model="mlx-community/Kokoro-82M-bf16",
            input="Testing TTS generation",
            voice="af_bella",
            speed=1.5,
            response_format="flac",
        )
        assert req.model == "mlx-community/Kokoro-82M-bf16"
        assert req.input == "Testing TTS generation"
        assert req.voice == "af_bella"
        assert req.speed == 1.5
        assert req.response_format == "flac"

    def test_speech_request_speed_bounds(self):
        """Speed must be between 0.25 and 4.0."""
        # Valid boundaries
        SpeechRequest(model="m", input="hi", speed=0.25)
        SpeechRequest(model="m", input="hi", speed=4.0)

        # Below minimum
        with pytest.raises(ValidationError):
            SpeechRequest(model="m", input="hi", speed=0.1)

        # Above maximum
        with pytest.raises(ValidationError):
            SpeechRequest(model="m", input="hi", speed=5.0)

    def test_speech_request_empty_input_rejected(self):
        """Empty input text should be rejected."""
        with pytest.raises(ValidationError):
            SpeechRequest(model="m", input="")

    def test_speech_request_format_validation(self):
        """Only wav, flac, mp3 formats should be accepted."""
        # Valid formats
        for fmt in ("wav", "flac", "mp3"):
            req = SpeechRequest(model="m", input="hi", response_format=fmt)
            assert req.response_format == fmt

        # Invalid format
        with pytest.raises(ValidationError):
            SpeechRequest(model="m", input="hi", response_format="ogg")

    def test_transcription_response_structure(self):
        """TranscriptionResponse should contain text field."""
        resp = TranscriptionResponse(text="Hello, this is a test.")
        assert resp.text == "Hello, this is a test."


class TestSpeechEndpoint:
    """Tests for POST /v1/audio/speech."""

    @pytest.mark.asyncio
    async def test_speech_endpoint_returns_audio_bytes(self):
        """Speech endpoint should return audio bytes with correct content type."""
        from mlx_manager.mlx_server.api.v1.speech import create_speech
        from mlx_manager.mlx_server.models.ir import AudioResult

        request = SpeechRequest(
            model="mlx-community/Kokoro-82M-bf16",
            input="Hello world",
        )

        fake_audio = b"RIFF\x00\x00\x00\x00WAVEfmt "  # WAV header-ish bytes

        with patch(
            "mlx_manager.mlx_server.api.v1.speech.generate_speech",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = AudioResult(
                audio_bytes=fake_audio, sample_rate=24000, format="wav"
            )

            response = await create_speech(request)

            assert response.body == fake_audio
            assert response.media_type == "audio/wav"
            mock_gen.assert_called_once_with(
                model_id="mlx-community/Kokoro-82M-bf16",
                text="Hello world",
                voice="af_heart",
                speed=1.0,
                response_format="wav",
            )

    @pytest.mark.asyncio
    async def test_speech_endpoint_flac_content_type(self):
        """Speech endpoint should set flac content type for flac format."""
        from mlx_manager.mlx_server.api.v1.speech import create_speech
        from mlx_manager.mlx_server.models.ir import AudioResult

        request = SpeechRequest(
            model="test-tts",
            input="Hello",
            response_format="flac",
        )

        with patch(
            "mlx_manager.mlx_server.api.v1.speech.generate_speech",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = AudioResult(
                audio_bytes=b"fLaC", sample_rate=24000, format="flac"
            )

            response = await create_speech(request)
            assert response.media_type == "audio/flac"

    @pytest.mark.asyncio
    async def test_speech_endpoint_mp3_content_type(self):
        """Speech endpoint should set mp3 content type for mp3 format."""
        from mlx_manager.mlx_server.api.v1.speech import create_speech
        from mlx_manager.mlx_server.models.ir import AudioResult

        request = SpeechRequest(
            model="test-tts",
            input="Hello",
            response_format="mp3",
        )

        with patch(
            "mlx_manager.mlx_server.api.v1.speech.generate_speech",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = AudioResult(
                audio_bytes=b"\xff\xfb", sample_rate=24000, format="mp3"
            )

            response = await create_speech(request)
            assert response.media_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_speech_endpoint_runtime_error_returns_500(self):
        """RuntimeError from service should become HTTP 500."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.speech import create_speech

        request = SpeechRequest(model="bad-model", input="Hello")

        with patch(
            "mlx_manager.mlx_server.api.v1.speech.generate_speech",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.side_effect = RuntimeError("Failed to load model: not found")

            with pytest.raises(HTTPException) as exc_info:
                await create_speech(request)

            assert exc_info.value.status_code == 500
            assert "Failed to load model" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_speech_endpoint_unexpected_error_returns_500(self):
        """Unexpected exceptions should become HTTP 500 with generic message."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.speech import create_speech

        request = SpeechRequest(model="bad-model", input="Hello")

        with patch(
            "mlx_manager.mlx_server.api.v1.speech.generate_speech",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.side_effect = ValueError("something unexpected")

            with pytest.raises(HTTPException) as exc_info:
                await create_speech(request)

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Internal server error"


class TestTranscriptionsEndpoint:
    """Tests for POST /v1/audio/transcriptions."""

    @pytest.mark.asyncio
    async def test_transcription_endpoint_returns_text(self):
        """Transcription endpoint should return transcribed text."""
        from mlx_manager.mlx_server.api.v1.transcriptions import create_transcription
        from mlx_manager.mlx_server.models.ir import TranscriptionResult

        # Create a mock UploadFile
        mock_file = MagicMock()
        mock_file.filename = "test.wav"
        mock_file.read = AsyncMock(return_value=b"fake audio data")

        with patch(
            "mlx_manager.mlx_server.api.v1.transcriptions.transcribe_audio",
            new_callable=AsyncMock,
        ) as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(text="Hello, world!")

            response = await create_transcription(
                file=mock_file,
                model="mlx-community/whisper-large-v3-turbo",
                language=None,
            )

            assert isinstance(response, TranscriptionResponse)
            assert response.text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_transcription_with_language_hint(self):
        """Transcription should pass language hint to service."""
        from mlx_manager.mlx_server.api.v1.transcriptions import create_transcription
        from mlx_manager.mlx_server.models.ir import TranscriptionResult

        mock_file = MagicMock()
        mock_file.filename = "audio.wav"
        mock_file.read = AsyncMock(return_value=b"audio bytes")

        with patch(
            "mlx_manager.mlx_server.api.v1.transcriptions.transcribe_audio",
            new_callable=AsyncMock,
        ) as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(text="Hola, mundo!")

            await create_transcription(
                file=mock_file,
                model="whisper-model",
                language="es",
            )

            mock_transcribe.assert_called_once_with(
                model_id="whisper-model",
                audio_data=b"audio bytes",
                language="es",
            )

    @pytest.mark.asyncio
    async def test_transcription_empty_file_returns_400(self):
        """Empty audio file should return 400 error."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.transcriptions import create_transcription

        mock_file = MagicMock()
        mock_file.filename = "empty.wav"
        mock_file.read = AsyncMock(return_value=b"")

        with pytest.raises(HTTPException) as exc_info:
            await create_transcription(
                file=mock_file,
                model="whisper-model",
            )

        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_transcription_runtime_error_returns_500(self):
        """RuntimeError from service should become HTTP 500."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.transcriptions import create_transcription

        mock_file = MagicMock()
        mock_file.filename = "test.wav"
        mock_file.read = AsyncMock(return_value=b"fake audio data")

        with patch(
            "mlx_manager.mlx_server.api.v1.transcriptions.transcribe_audio",
            new_callable=AsyncMock,
        ) as mock_transcribe:
            mock_transcribe.side_effect = RuntimeError("Failed to load model")

            with pytest.raises(HTTPException) as exc_info:
                await create_transcription(file=mock_file, model="bad-model")

            assert exc_info.value.status_code == 500
            assert "Failed to load model" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_transcription_http_exception_reraise(self):
        """HTTPException from service is re-raised as-is."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.transcriptions import create_transcription

        mock_file = MagicMock()
        mock_file.filename = "test.wav"
        mock_file.read = AsyncMock(return_value=b"fake audio data")

        with patch(
            "mlx_manager.mlx_server.api.v1.transcriptions.transcribe_audio",
            new_callable=AsyncMock,
        ) as mock_transcribe:
            mock_transcribe.side_effect = HTTPException(
                status_code=400, detail="Invalid audio format"
            )

            with pytest.raises(HTTPException) as exc_info:
                await create_transcription(file=mock_file, model="whisper-model")

            assert exc_info.value.status_code == 400
            assert "Invalid audio format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_transcription_unexpected_error_returns_500_generic(self):
        """Unexpected non-Runtime exception returns generic 500."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.transcriptions import create_transcription

        mock_file = MagicMock()
        mock_file.filename = "test.wav"
        mock_file.read = AsyncMock(return_value=b"fake audio data")

        with patch(
            "mlx_manager.mlx_server.api.v1.transcriptions.transcribe_audio",
            new_callable=AsyncMock,
        ) as mock_transcribe:
            mock_transcribe.side_effect = ValueError("something unexpected")

            with pytest.raises(HTTPException) as exc_info:
                await create_transcription(file=mock_file, model="whisper-model")

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Internal server error"


class TestAudioService:
    """Tests for audio service function signatures."""

    def test_generate_speech_signature(self):
        """Verify generate_speech has correct parameters."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        sig = inspect.signature(generate_speech)
        params = list(sig.parameters.keys())
        assert "model_id" in params
        assert "text" in params
        assert "voice" in params
        assert "speed" in params
        assert "response_format" in params

    def test_generate_speech_is_async(self):
        """Verify generate_speech is an async function."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        assert inspect.iscoroutinefunction(generate_speech)

    def test_transcribe_audio_signature(self):
        """Verify transcribe_audio has correct parameters."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        sig = inspect.signature(transcribe_audio)
        params = list(sig.parameters.keys())
        assert "model_id" in params
        assert "audio_data" in params
        assert "language" in params

    def test_transcribe_audio_is_async(self):
        """Verify transcribe_audio is an async function."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        assert inspect.iscoroutinefunction(transcribe_audio)

    def test_speech_service_uses_pool(self):
        """Verify service uses model pool for model management."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        source = inspect.getsource(generate_speech)
        assert "get_model_pool" in source
        assert "model.generate" in source

    def test_transcribe_service_uses_pool(self):
        """Verify transcription service uses model pool."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        source = inspect.getsource(transcribe_audio)
        assert "get_model_pool" in source
        assert "generate_transcription" in source


class TestGenerateSpeechService:
    """Tests for generate_speech service function with mocked Metal thread."""

    @pytest.mark.asyncio
    async def test_generate_speech_single_segment(self):
        """Generate speech with a single audio segment from model."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        # Mock model that returns a single GenerationResult
        mock_result = MagicMock()
        mock_result.audio = MagicMock()
        mock_result.audio.tolist.return_value = [0.1, 0.2, 0.3]
        mock_result.sample_rate = 24000

        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_result]
        mock_model.sample_rate = 24000

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.concatenate") as mock_concat,
            patch("mlx.core.eval"),
            patch("soundfile.write") as mock_sf_write,
        ):
            # sf.write writes to the buffer
            def write_to_buffer(buf, data, sr, **kwargs):
                buf.write(b"fake-audio-bytes")

            mock_sf_write.side_effect = write_to_buffer

            result = await generate_speech(
                model_id="test-tts-model",
                text="Hello world",
                voice="af_heart",
                speed=1.0,
                response_format="wav",
            )

            assert result.audio_bytes == b"fake-audio-bytes"
            assert result.sample_rate == 24000
            assert result.format == "wav"
            mock_model.generate.assert_called_once()
            # Single segment means no concatenation
            mock_concat.assert_not_called()
            mock_pool.get_model.assert_called_once_with("test-tts-model")

    @pytest.mark.asyncio
    async def test_generate_speech_multiple_segments_concatenated(self):
        """Generate speech with multiple segments uses mx.concatenate."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        # Mock model returning multiple segments
        mock_result1 = MagicMock()
        mock_result1.audio = MagicMock()
        mock_result1.audio.tolist.return_value = [0.1, 0.2]
        mock_result1.sample_rate = 24000

        mock_result2 = MagicMock()
        mock_result2.audio = MagicMock()
        mock_result2.audio.tolist.return_value = [0.3, 0.4]
        mock_result2.sample_rate = 24000

        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_result1, mock_result2]
        mock_model.sample_rate = 24000

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        concatenated_audio = MagicMock()
        concatenated_audio.tolist.return_value = [0.1, 0.2, 0.3, 0.4]

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.concatenate", return_value=concatenated_audio) as mock_concat,
            patch("mlx.core.eval"),
            patch("soundfile.write") as mock_sf_write,
        ):

            def write_to_buffer(buf, data, sr, **kwargs):
                buf.write(b"concatenated-audio")

            mock_sf_write.side_effect = write_to_buffer

            result = await generate_speech(
                model_id="test-tts",
                text="Hello world, this is a longer text.",
                voice="af_bella",
                speed=1.5,
                response_format="flac",
            )

            assert result.audio_bytes == b"concatenated-audio"
            assert result.sample_rate == 24000
            assert result.format == "flac"
            mock_concat.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_speech_empty_output_raises(self):
        """Generate speech raises RuntimeError when model produces no audio."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_model = MagicMock()
        mock_model.generate.return_value = []  # No results
        mock_model.sample_rate = 24000

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.concatenate"),
            patch("mlx.core.eval"),
        ):
            with pytest.raises(RuntimeError, match="no audio output"):
                await generate_speech(
                    model_id="test-tts",
                    text="Hello",
                )

    @pytest.mark.asyncio
    async def test_generate_speech_uses_model_sample_rate_fallback(self):
        """If GenerationResult has no sample_rate attr, fall back to model.sample_rate."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_result = MagicMock(spec=["audio"])  # No sample_rate attribute
        mock_result.audio = MagicMock()
        mock_result.audio.tolist.return_value = [0.5]

        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_result]
        mock_model.sample_rate = 44100

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.eval"),
            patch("soundfile.write") as mock_sf_write,
        ):

            def write_to_buffer(buf, data, sr, **kwargs):
                buf.write(b"audio-data")

            mock_sf_write.side_effect = write_to_buffer

            result = await generate_speech(
                model_id="test-tts",
                text="Test",
            )

            assert result.sample_rate == 44100

    @pytest.mark.asyncio
    async def test_generate_speech_result_sample_rate_overrides(self):
        """GenerationResult.sample_rate overrides the model default."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_result = MagicMock()
        mock_result.audio = MagicMock()
        mock_result.audio.tolist.return_value = [0.5]
        mock_result.sample_rate = 48000

        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_result]
        mock_model.sample_rate = 24000  # Will be overridden

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.eval"),
            patch("soundfile.write") as mock_sf_write,
        ):

            def write_to_buffer(buf, data, sr, **kwargs):
                buf.write(b"audio-data")

            mock_sf_write.side_effect = write_to_buffer

            result = await generate_speech(
                model_id="test-tts",
                text="Test",
            )

            assert result.sample_rate == 48000

    @pytest.mark.asyncio
    async def test_generate_speech_passes_correct_gen_kwargs(self):
        """Verify generate_speech passes text, voice, speed, verbose to model.generate."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_result = MagicMock()
        mock_result.audio = MagicMock()
        mock_result.audio.tolist.return_value = [0.5]
        mock_result.sample_rate = 24000

        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_result]
        mock_model.sample_rate = 24000

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.eval"),
            patch("soundfile.write") as mock_sf_write,
        ):
            mock_sf_write.side_effect = lambda buf, *a, **kw: buf.write(b"data")

            await generate_speech(
                model_id="test-tts",
                text="Hello world",
                voice="af_bella",
                speed=2.0,
                response_format="mp3",
            )

            mock_model.generate.assert_called_once_with(
                text="Hello world",
                voice="af_bella",
                speed=2.0,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_generate_speech_model_no_sample_rate_attr(self):
        """Model without sample_rate attribute uses default 24000."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_result = MagicMock(spec=["audio"])
        mock_result.audio = MagicMock()
        mock_result.audio.tolist.return_value = [0.5]

        mock_model = MagicMock(spec=["generate"])
        mock_model.generate.return_value = [mock_result]

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.eval"),
            patch("soundfile.write") as mock_sf_write,
        ):
            mock_sf_write.side_effect = lambda buf, *a, **kw: buf.write(b"data")

            result = await generate_speech(
                model_id="test-tts",
                text="Test",
            )

            # Default sample_rate is 24000 per getattr fallback
            assert result.sample_rate == 24000

    @pytest.mark.asyncio
    async def test_generate_speech_run_on_metal_error_propagates(self):
        """Errors from run_on_metal_thread propagate as RuntimeError."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def raise_runtime_error(fn, **kwargs):
            raise RuntimeError("TTS generation failed: GPU error")

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=raise_runtime_error,
            ),
        ):
            with pytest.raises(RuntimeError, match="TTS generation failed"):
                await generate_speech(model_id="test-tts", text="Hello")


class TestTranscribeAudioService:
    """Tests for transcribe_audio service function with mocked Metal thread."""

    @pytest.mark.asyncio
    async def test_transcribe_audio_basic(self):
        """Transcribe audio returns text from STT model."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_stt_output = MagicMock()
        mock_stt_output.text = "Hello, world!"
        mock_stt_output.segments = None
        mock_stt_output.language = None

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch(
                "mlx_audio.stt.generate.generate_transcription",
                return_value=mock_stt_output,
            ),
        ):
            result = await transcribe_audio(
                model_id="test-stt-model",
                audio_data=b"fake-wav-data",
            )

            assert result.text == "Hello, world!"
            assert result.segments is None
            assert result.language is None
            mock_pool.get_model.assert_called_once_with("test-stt-model")

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_segments(self):
        """Transcribe audio includes segments when available."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_stt_output = MagicMock()
        mock_stt_output.text = "Hello world"
        mock_stt_output.segments = [
            {"start": 0.0, "end": 0.5, "text": "Hello"},
            {"start": 0.5, "end": 1.0, "text": "world"},
        ]
        mock_stt_output.language = "en"

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch(
                "mlx_audio.stt.generate.generate_transcription",
                return_value=mock_stt_output,
            ),
        ):
            result = await transcribe_audio(
                model_id="test-stt",
                audio_data=b"audio-bytes",
            )

            assert result.text == "Hello world"
            assert len(result.segments) == 2
            assert result.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_language_hint(self):
        """Transcribe audio passes language kwarg to generate_transcription."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_stt_output = MagicMock()
        mock_stt_output.text = "Hola mundo"
        mock_stt_output.segments = None
        mock_stt_output.language = "es"

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch(
                "mlx_audio.stt.generate.generate_transcription",
                return_value=mock_stt_output,
            ) as mock_gen_transcription,
        ):
            result = await transcribe_audio(
                model_id="test-stt",
                audio_data=b"audio-bytes",
                language="es",
            )

            assert result.text == "Hola mundo"
            # Verify language was passed
            call_kwargs = mock_gen_transcription.call_args[1]
            assert call_kwargs["language"] == "es"

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_language_hint(self):
        """Transcribe audio without language does not pass language kwarg."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_stt_output = MagicMock()
        mock_stt_output.text = "Test"
        mock_stt_output.segments = None
        mock_stt_output.language = None

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch(
                "mlx_audio.stt.generate.generate_transcription",
                return_value=mock_stt_output,
            ) as mock_gen_transcription,
        ):
            await transcribe_audio(
                model_id="test-stt",
                audio_data=b"audio-bytes",
                language=None,
            )

            # language should NOT be in kwargs
            call_kwargs = mock_gen_transcription.call_args[1]
            assert "language" not in call_kwargs

    @pytest.mark.asyncio
    async def test_transcribe_audio_writes_temp_file(self):
        """Transcribe audio writes audio_data to a temp file for STT."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_stt_output = MagicMock()
        mock_stt_output.text = "Transcribed text"
        mock_stt_output.segments = None
        mock_stt_output.language = None

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        captured_audio_path = []

        def capture_transcription_call(model, audio, verbose, **kwargs):
            captured_audio_path.append(audio)
            return mock_stt_output

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch(
                "mlx_audio.stt.generate.generate_transcription",
                side_effect=capture_transcription_call,
            ),
        ):
            await transcribe_audio(
                model_id="test-stt",
                audio_data=b"fake-wav-data",
            )

            # Verify a file path was passed (not raw bytes)
            assert len(captured_audio_path) == 1
            assert isinstance(captured_audio_path[0], str)
            assert captured_audio_path[0].endswith(".wav")

    @pytest.mark.asyncio
    async def test_transcribe_audio_cleans_up_temp_file(self):
        """Temp file is deleted after transcription even on success."""
        import os

        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_stt_output = MagicMock()
        mock_stt_output.text = "Test"
        mock_stt_output.segments = None
        mock_stt_output.language = None

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        captured_path = []

        def capture_path(model, audio, verbose, **kwargs):
            captured_path.append(audio)
            return mock_stt_output

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch(
                "mlx_audio.stt.generate.generate_transcription",
                side_effect=capture_path,
            ),
        ):
            await transcribe_audio(
                model_id="test-stt",
                audio_data=b"fake-wav-data",
            )

            # Temp file should have been deleted
            assert len(captured_path) == 1
            assert not os.path.exists(captured_path[0])

    @pytest.mark.asyncio
    async def test_transcribe_audio_run_on_metal_error_propagates(self):
        """Errors from run_on_metal_thread propagate as RuntimeError."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def raise_runtime_error(fn, **kwargs):
            raise RuntimeError("STT transcription failed: GPU error")

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=raise_runtime_error,
            ),
        ):
            with pytest.raises(RuntimeError, match="STT transcription failed"):
                await transcribe_audio(
                    model_id="test-stt",
                    audio_data=b"audio-bytes",
                )

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_text_attr_returns_empty(self):
        """If STT output has no text attribute, return empty string."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_stt_output = MagicMock(spec=[])  # No attributes at all

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch(
                "mlx_audio.stt.generate.generate_transcription",
                return_value=mock_stt_output,
            ),
        ):
            result = await transcribe_audio(
                model_id="test-stt",
                audio_data=b"audio-bytes",
            )

            # getattr(segments, "text", "") should give ""
            assert result.text == ""


class TestAudioRouteRegistration:
    """Tests for audio route registration in v1 router."""

    def test_speech_route_registered(self):
        """Verify /audio/speech route is registered."""
        from mlx_manager.mlx_server.api.v1 import v1_router

        paths = [r.path for r in v1_router.routes]
        assert "/audio/speech" in paths

    def test_transcriptions_route_registered(self):
        """Verify /audio/transcriptions route is registered."""
        from mlx_manager.mlx_server.api.v1 import v1_router

        paths = [r.path for r in v1_router.routes]
        assert "/audio/transcriptions" in paths

    def test_speech_route_is_post(self):
        """Verify /audio/speech is a POST endpoint."""
        from mlx_manager.mlx_server.api.v1 import v1_router

        for r in v1_router.routes:
            if getattr(r, "path", None) == "/audio/speech":
                assert "POST" in getattr(r, "methods", set())
                break

    def test_transcriptions_route_is_post(self):
        """Verify /audio/transcriptions is a POST endpoint."""
        from mlx_manager.mlx_server.api.v1 import v1_router

        for r in v1_router.routes:
            if getattr(r, "path", None) == "/audio/transcriptions":
                assert "POST" in getattr(r, "methods", set())
                break

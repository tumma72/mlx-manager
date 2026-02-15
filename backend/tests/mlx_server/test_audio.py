"""Tests for audio (TTS/STT) schemas, endpoints, and service."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from mlx_manager.mlx_server.models.ir import AudioResult, TranscriptionResult
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

    def test_speech_service_delegates_to_adapter(self):
        """Verify speech service delegates to adapter."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        source = inspect.getsource(generate_speech)
        assert "get_model_pool" in source
        assert "loaded.adapter.generate_speech" in source

    def test_transcribe_service_delegates_to_adapter(self):
        """Verify transcription service delegates to adapter."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        source = inspect.getsource(transcribe_audio)
        assert "get_model_pool" in source
        assert "loaded.adapter.transcribe" in source


class TestGenerateSpeechService:
    """Tests for generate_speech service function with mocked adapter."""

    @pytest.mark.asyncio
    async def test_generate_speech_returns_adapter_result(self):
        """Generate speech returns AudioResult from adapter."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        expected_result = AudioResult(
            audio_bytes=b"fake-audio-bytes",
            sample_rate=24000,
            format="wav",
        )

        mock_adapter = MagicMock()
        mock_adapter.generate_speech = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
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
            mock_pool.get_model.assert_called_once_with("test-tts-model")

    @pytest.mark.asyncio
    async def test_generate_speech_passes_all_params_to_adapter(self):
        """Verify all parameters are forwarded to adapter.generate_speech."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        expected_result = AudioResult(audio_bytes=b"data", sample_rate=24000, format="flac")

        mock_adapter = MagicMock()
        mock_adapter.generate_speech = AsyncMock(return_value=expected_result)

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            await generate_speech(
                model_id="test-tts",
                text="Hello world",
                voice="af_bella",
                speed=2.0,
                response_format="flac",
            )

            mock_adapter.generate_speech.assert_called_once_with(
                mock_model,
                "Hello world",
                voice="af_bella",
                speed=2.0,
                response_format="flac",
            )

    @pytest.mark.asyncio
    async def test_generate_speech_error_propagates(self):
        """Errors from adapter propagate as RuntimeError."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_adapter = MagicMock()
        mock_adapter.generate_speech = AsyncMock(
            side_effect=RuntimeError("TTS generation failed: GPU error")
        )

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with pytest.raises(RuntimeError, match="TTS generation failed"):
                await generate_speech(model_id="test-tts", text="Hello")

    @pytest.mark.asyncio
    async def test_generate_speech_no_adapter_raises(self):
        """AssertionError when loaded model has no adapter."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = None

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with pytest.raises(AssertionError, match="No adapter"):
                await generate_speech(model_id="test-tts", text="Hello")

    @pytest.mark.asyncio
    async def test_generate_speech_custom_voice_and_speed(self):
        """Custom voice and speed params are forwarded correctly."""
        from mlx_manager.mlx_server.services.audio import generate_speech

        expected_result = AudioResult(audio_bytes=b"audio", sample_rate=48000, format="mp3")

        mock_adapter = MagicMock()
        mock_adapter.generate_speech = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            result = await generate_speech(
                model_id="test-tts",
                text="Test",
                voice="af_bella",
                speed=1.5,
                response_format="mp3",
            )

            assert result.sample_rate == 48000
            assert result.format == "mp3"


class TestTranscribeAudioService:
    """Tests for transcribe_audio service function with mocked adapter."""

    @pytest.mark.asyncio
    async def test_transcribe_audio_basic(self):
        """Transcribe audio returns TranscriptionResult from adapter."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        expected_result = TranscriptionResult(
            text="Hello, world!",
            segments=None,
            language=None,
        )

        mock_adapter = MagicMock()
        mock_adapter.transcribe = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
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

        expected_result = TranscriptionResult(
            text="Hello world",
            segments=[
                {"start": 0.0, "end": 0.5, "text": "Hello"},
                {"start": 0.5, "end": 1.0, "text": "world"},
            ],
            language="en",
        )

        mock_adapter = MagicMock()
        mock_adapter.transcribe = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
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
        """Transcribe audio passes language kwarg to adapter."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        expected_result = TranscriptionResult(
            text="Hola mundo",
            segments=None,
            language="es",
        )

        mock_adapter = MagicMock()
        mock_adapter.transcribe = AsyncMock(return_value=expected_result)

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            result = await transcribe_audio(
                model_id="test-stt",
                audio_data=b"audio-bytes",
                language="es",
            )

            assert result.text == "Hola mundo"
            mock_adapter.transcribe.assert_called_once_with(
                mock_model, b"audio-bytes", language="es"
            )

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_language_hint(self):
        """Transcribe audio passes language=None to adapter when not specified."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        expected_result = TranscriptionResult(text="Test", segments=None, language=None)

        mock_adapter = MagicMock()
        mock_adapter.transcribe = AsyncMock(return_value=expected_result)

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            await transcribe_audio(
                model_id="test-stt",
                audio_data=b"audio-bytes",
                language=None,
            )

            mock_adapter.transcribe.assert_called_once_with(
                mock_model, b"audio-bytes", language=None
            )

    @pytest.mark.asyncio
    async def test_transcribe_audio_error_propagates(self):
        """Errors from adapter propagate as RuntimeError."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_adapter = MagicMock()
        mock_adapter.transcribe = AsyncMock(
            side_effect=RuntimeError("STT transcription failed: GPU error")
        )

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with pytest.raises(RuntimeError, match="STT transcription failed"):
                await transcribe_audio(
                    model_id="test-stt",
                    audio_data=b"audio-bytes",
                )

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_adapter_raises(self):
        """AssertionError when loaded model has no adapter."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = None

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with pytest.raises(AssertionError, match="No adapter"):
                await transcribe_audio(
                    model_id="test-stt",
                    audio_data=b"audio-bytes",
                )

    @pytest.mark.asyncio
    async def test_transcribe_audio_empty_text(self):
        """Adapter returning empty text is passed through."""
        from mlx_manager.mlx_server.services.audio import transcribe_audio

        expected_result = TranscriptionResult(text="", segments=None, language=None)

        mock_adapter = MagicMock()
        mock_adapter.transcribe = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            result = await transcribe_audio(
                model_id="test-stt",
                audio_data=b"audio-bytes",
            )

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

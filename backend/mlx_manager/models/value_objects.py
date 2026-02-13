"""Shared Pydantic value objects for MLX Manager."""

from pydantic import BaseModel, Field

__all__ = ["InferenceParams", "InferenceContext", "AudioDefaults"]


class InferenceParams(BaseModel):
    """Generation parameters settable at profile or request level."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class InferenceContext(BaseModel):
    """Execution context settings beyond raw generation params."""

    context_length: int | None = None
    system_prompt: str | None = None
    enable_tool_injection: bool = False


class AudioDefaults(BaseModel):
    """Default audio parameters for TTS/STT."""

    tts_voice: str | None = None
    tts_speed: float | None = Field(default=None, ge=0.25, le=4.0)
    tts_sample_rate: int | None = None
    stt_language: str | None = None

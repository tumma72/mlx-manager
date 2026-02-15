"""Composable adapter architecture with dependency-injected parsers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.ir import (
        AudioResult,
        EmbeddingResult,
        PreparedInput,
        TextResult,
        TranscriptionResult,
    )
    from mlx_manager.mlx_server.services.response_processor import StreamProcessor

from mlx_manager.mlx_server.models.adapters.configs import FAMILY_CONFIGS, FamilyConfig
from mlx_manager.mlx_server.parsers import (
    NullThinkingParser,
    NullToolParser,
    ThinkingParser,
    ToolCallParser,
)

# Common special tokens across model families
COMMON_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|end|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "assistant",  # Sometimes appears as raw text
]


class ModelAdapter:
    """Stateful adapter configured from FamilyConfig with injected parsers.

    Created once per loaded model. Holds tokenizer reference + parser
    instances. Behavior is driven by the FamilyConfig data object and
    optional strategy functions—no subclassing needed.
    """

    def __init__(
        self,
        config: FamilyConfig | None = None,
        tokenizer: Any | None = None,
        tool_parser: ToolCallParser | None = None,
        thinking_parser: ThinkingParser | None = None,
        model_id: str | None = None,
    ) -> None:
        self._config = config or FAMILY_CONFIGS["default"]
        self._tokenizer = tokenizer
        # Get actual tokenizer (Processor wraps tokenizer, regular is itself)
        # Audio adapters pass None; text/vision always have a tokenizer.
        self._actual_tokenizer: Any = (
            getattr(tokenizer, "tokenizer", tokenizer) if tokenizer is not None else None
        )
        # Parsers: explicit override > config factory > NullParser
        if tool_parser is not None:
            self._tool_parser = tool_parser
        elif self._config.tool_parser_factory:
            self._tool_parser = self._config.tool_parser_factory()
        else:
            self._tool_parser = NullToolParser()

        if thinking_parser is not None:
            self._thinking_parser = thinking_parser
        elif self._config.thinking_parser_factory:
            self._thinking_parser = self._config.thinking_parser_factory()
        else:
            self._thinking_parser = NullThinkingParser()

        # Pre-compute stop tokens at init
        self._stop_tokens = self._compute_stop_tokens()
        self._model_id = model_id

    @property
    def family(self) -> str:
        """Model family identifier."""
        return self._config.family

    @property
    def tool_parser(self) -> ToolCallParser:
        return self._tool_parser

    @property
    def thinking_parser(self) -> ThinkingParser:
        return self._thinking_parser

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def stop_tokens(self) -> list[int]:
        """Pre-computed stop token IDs."""
        return self._stop_tokens

    def _compute_stop_tokens(self) -> list[int]:
        """Compute stop tokens from tokenizer + config extra_stop_tokens."""
        if self._actual_tokenizer is None:
            return []
        eos = getattr(self._actual_tokenizer, "eos_token_id", None)
        stop_ids: list[int] = [eos] if eos is not None else []
        for token_str in self._config.extra_stop_tokens:
            try:
                token_id = self._actual_tokenizer.convert_tokens_to_ids(token_str)
                if (
                    token_id is not None
                    and token_id != getattr(self._actual_tokenizer, "unk_token_id", None)
                    and token_id not in stop_ids
                ):
                    stop_ids.append(token_id)
            except Exception:
                pass
        return stop_ids

    async def post_load_configure(self, model: Any, model_id: str) -> None:
        """Post-load configuration hook. Delegates to config's post_load_hook."""
        if self._config.post_load_hook:
            await self._config.post_load_hook(model, model_id)

    def supports_native_tools(self) -> bool:
        """Whether adapter passes tools= to apply_chat_template."""
        return self._config.native_tools

    def supports_tool_calling(self) -> bool:
        """Whether adapter supports tool calling (native or injected)."""
        return not isinstance(self._tool_parser, NullToolParser)

    def get_stream_markers(self) -> list[tuple[str, str]]:
        """Combined stream markers from tool + thinking parsers."""
        markers = list(self._tool_parser.stream_markers)
        markers.extend(self._thinking_parser.stream_markers)
        return markers

    # ── Tool delivery ────────────────────────────────────────────

    @staticmethod
    def _inject_tools(messages: list[dict[str, Any]], tool_prompt: str) -> list[dict[str, Any]]:
        """Inject tool definitions into the system message."""
        result = list(messages)
        system_idx = None
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                system_idx = i
                break
        if system_idx is not None:
            existing = result[system_idx].get("content", "")
            result[system_idx] = {
                **result[system_idx],
                "content": f"{existing}\n\n{tool_prompt}",
            }
        else:
            result.insert(0, {"role": "system", "content": tool_prompt})
        return result

    def _prepare_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Prepare messages for tool-aware template application.

        Returns:
            (effective_messages, native_tools) where native_tools is
            passed to tokenizer.apply_chat_template(tools=...) for
            models with native template support, or None when tools
            were injected into messages (or absent).
        """
        if not tools:
            return messages, None
        if self.supports_native_tools():
            return messages, tools
        # Adapter delivery: inject formatted tools into system message
        tool_prompt = self.format_tools_for_prompt(tools)
        if tool_prompt:
            return self._inject_tools(messages, tool_prompt), None
        return messages, None

    # ── Template application ─────────────────────────────────────

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Apply chat template with automatic tool delivery."""
        effective, native_tools = self._prepare_tools(messages, tools)
        if self._config.template_strategy:
            return self._config.template_strategy(
                self._actual_tokenizer,
                effective,
                add_generation_prompt,
                native_tools,
                enable_thinking,
            )
        # Default: call tokenizer.apply_chat_template() directly
        kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": False,
        }
        if native_tools:
            kwargs["tools"] = native_tools
        return cast(
            str,
            self._actual_tokenizer.apply_chat_template(effective, **kwargs),
        )

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tools for prompt injection. Delegates to config strategy."""
        if self._config.tool_format_strategy:
            return self._config.tool_format_strategy(tools)
        return ""

    def get_tool_call_stop_tokens(self) -> list[int]:
        """Additional stop tokens when tools are enabled."""
        if not self._config.tool_call_stop_tokens or self._actual_tokenizer is None:
            return []
        stop_tokens: list[int] = []
        for token_str in self._config.tool_call_stop_tokens:
            try:
                token_id = self._actual_tokenizer.convert_tokens_to_ids(token_str)
                if token_id is not None and token_id != getattr(
                    self._actual_tokenizer, "unk_token_id", None
                ):
                    stop_tokens.append(token_id)
            except Exception:
                pass
        return stop_tokens

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages. Delegates to config strategy or uses default fallback."""
        if self._config.message_convert_strategy:
            return self._config.message_convert_strategy(messages)
        # Default safe fallback: convert tool messages to user messages
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")
                converted.append(
                    {
                        "role": "user",
                        "content": (f"[Tool Result for {tool_call_id}]\n{content}"),
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                content = msg.get("content", "") or ""
                tool_calls = msg.get("tool_calls", [])
                tool_text_parts = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    tool_text_parts.append(f"[Tool Call: {name}({args})]")
                tool_text = "\n".join(tool_text_parts)
                if tool_text_parts:
                    content = f"{content}\n{tool_text}" if content else tool_text
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append(msg)
        return converted

    def clean_response(self, text: str) -> str:
        """Clean response text by removing special tokens."""
        cleaned = text
        for token in COMMON_SPECIAL_TOKENS:
            cleaned = cleaned.replace(token, "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def create_stream_processor(self, prompt: str = "") -> StreamProcessor:
        """Create a StreamProcessor for this adapter.

        Factory method that creates a properly configured processor using
        the adapter's parsers. Detects whether the prompt ends with a
        thinking start tag (from this adapter's thinking parser markers)
        to initialize the processor in thinking mode.

        Args:
            prompt: The prompt string being sent to the model. Used to
                    detect if the model should start in thinking mode.

        Returns:
            StreamProcessor configured with this adapter's parsers
        """
        # Detect if prompt ends with a thinking start tag
        starts_in_thinking = False
        if prompt:
            stripped = prompt.rstrip()
            thinking_starts = [start for start, _ in self._thinking_parser.stream_markers]
            starts_in_thinking = any(stripped.endswith(tag) for tag in thinking_starts)

        from mlx_manager.mlx_server.services.response_processor import StreamProcessor

        return StreamProcessor(adapter=self, starts_in_thinking=starts_in_thinking)

    def prepare_input(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_prompt_injection: bool = False,
        images: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> PreparedInput:
        """Prepare model-ready input from messages and optional tools.

        Encapsulates the full input pipeline:
        1. convert_messages() — format conversion
        2. Tool capability check
        3. apply_chat_template() — prompt generation
        4. Stop token aggregation

        When images are provided (vision models), uses mlx-vlm chat template
        instead of the regular tokenizer template.
        """
        from mlx_manager.mlx_server.models.ir import PreparedInput

        # Vision path: use mlx-vlm chat template
        if images:
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
            from mlx_vlm.utils import load_config

            # Build flat text prompt from messages
            text_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multipart content - extract text parts only
                    text = " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    text = str(content) if content else ""
                if text:
                    if role == "system":
                        text_parts.append(f"System: {text}")
                    elif role == "user":
                        text_parts.append(text)
                    elif role == "assistant":
                        text_parts.append(f"Assistant: {text}")
            text_prompt = "\n".join(text_parts)

            # Load model config and apply vision chat template
            config = load_config(self._model_id)
            prompt = vlm_apply_chat_template(
                self._tokenizer, config, text_prompt, num_images=len(images)
            )

            stop_ids: set[int] = set(self.stop_tokens)

            return PreparedInput(
                prompt=prompt,
                stop_token_ids=list(stop_ids),
                pixel_values=images,
            )

        # Text path: regular chat template
        converted = self.convert_messages(messages)

        use_tools = tools and (self.supports_tool_calling() or enable_prompt_injection)
        effective_tools = tools if use_tools else None

        prompt = self.apply_chat_template(
            messages=converted,
            add_generation_prompt=True,
            tools=effective_tools,
            enable_thinking=enable_thinking,
        )

        stop_ids_set = set(self.stop_tokens)
        if use_tools:
            stop_ids_set.update(self.get_tool_call_stop_tokens())

        return PreparedInput(
            prompt=prompt,
            stop_token_ids=list(stop_ids_set),
        )

    def process_complete(self, raw_text: str, finish_reason: str = "stop") -> TextResult:
        """Post-process raw model output into a TextResult.

        Encapsulates the full output pipeline:
        1. Tool call extraction
        2. Thinking/reasoning extraction
        3. Response cleaning
        """
        from mlx_manager.mlx_server.models.ir import TextResult

        tool_calls_list = self.tool_parser.extract(raw_text)
        reasoning_content = self.thinking_parser.extract(raw_text)

        final_content = raw_text
        if reasoning_content:
            final_content = self.thinking_parser.remove(final_content)
        final_content = self.clean_response(final_content)

        tool_calls = None
        if tool_calls_list:
            tool_calls = [tc.model_dump() for tc in tool_calls_list]
            finish_reason = "tool_calls"

        return TextResult(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    # ── Full generation pipeline ─────────────────────────────────

    async def generate(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
        enable_prompt_injection: bool = False,
        images: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> TextResult:
        """Full generation pipeline: prepare → generate → process_complete.

        Owns the complete text/vision inference path. Services call this
        instead of orchestrating prepare_input/generate/process_complete
        themselves.
        """
        from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

        prepared = self.prepare_input(
            messages,
            tools=tools,
            enable_prompt_injection=enable_prompt_injection,
            images=images,
            enable_thinking=enable_thinking,
        )

        if images:
            # Vision generation via mlx-vlm
            def run_vision_gen() -> tuple[str, str]:
                from mlx_vlm import generate as vlm_generate

                response = vlm_generate(
                    model,
                    self._tokenizer,
                    prepared.prompt,
                    prepared.pixel_values,
                    max_tokens=max_tokens,
                    temp=temperature,
                    verbose=False,
                )
                return (str(response.text), "stop")

            raw_text, finish_reason = await run_on_metal_thread(run_vision_gen, timeout=600.0)
        else:
            # Text generation via mlx-lm
            stop_ids = set(prepared.stop_token_ids or [])

            def run_text_gen() -> tuple[str, str]:
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_sampler

                text = ""
                finish = "length"
                sampler = make_sampler(temp=temperature, top_p=top_p)

                for resp in stream_generate(
                    model,
                    self._tokenizer,  # type: ignore[arg-type]
                    prepared.prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    token_id = getattr(resp, "token", None)
                    if token_id is not None and token_id in stop_ids:
                        finish = "stop"
                        break
                    text += getattr(resp, "text", str(resp))
                return (text, finish)

            raw_text, finish_reason = await run_on_metal_thread(run_text_gen)

        return self.process_complete(raw_text, finish_reason)

    async def generate_step(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
        enable_prompt_injection: bool = False,
        images: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> Any:
        """Streaming generation yielding IR events.

        Yields StreamEvent for each token, then a final TextResult.
        Returns an async generator (use `async for event in adapter.generate_step(...)`).
        """
        from collections.abc import Iterator

        from mlx_manager.mlx_server.models.ir import StreamEvent

        prepared = self.prepare_input(
            messages,
            tools=tools,
            enable_prompt_injection=enable_prompt_injection,
            images=images,
            enable_thinking=enable_thinking,
        )

        if images:
            # Vision: non-streaming, simulate single event
            from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

            def run_vision_gen() -> str:
                from mlx_vlm import generate as vlm_generate

                response = vlm_generate(
                    model,
                    self._tokenizer,
                    prepared.prompt,
                    prepared.pixel_values,
                    max_tokens=max_tokens,
                    temp=temperature,
                    verbose=False,
                )
                return str(response.text)

            response_text = await run_on_metal_thread(run_vision_gen, timeout=600.0)
            yield StreamEvent(type="content", content=response_text)
            yield self.process_complete(response_text, "stop")
            return

        # Text: streaming generation
        from mlx_manager.mlx_server.utils.metal import stream_from_metal_thread

        stop_ids = set(prepared.stop_token_ids or [])
        stream_processor = self.create_stream_processor(prompt=prepared.prompt)

        def produce_tokens() -> Iterator[tuple[str, int | None, bool]]:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=temperature, top_p=top_p)
            for resp in stream_generate(
                model,
                self._tokenizer,  # type: ignore[arg-type]
                prepared.prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                token_id = getattr(resp, "token", None)
                token_text = getattr(resp, "text", str(resp))
                is_stop = token_id is not None and token_id in stop_ids
                yield (token_text, token_id, is_stop)
                if is_stop:
                    return

        finish_reason = "length"
        async for token_text, token_id, is_stop in stream_from_metal_thread(produce_tokens):
            if is_stop:
                finish_reason = "stop"
                stream_processor.feed(token_text)
                break
            event = stream_processor.feed(token_text)
            if event.reasoning_content or event.content:
                yield event

        raw_text = stream_processor.get_accumulated_text()
        yield self.process_complete(raw_text, finish_reason)

    # ── Embeddings generation ────────────────────────────────────

    async def generate_embeddings(
        self,
        model: Any,
        texts: list[str],
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Owns the complete embeddings inference path. Services call this
        instead of orchestrating tokenization/forward pass themselves.
        """
        from mlx_manager.mlx_server.models.ir import EmbeddingResult
        from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

        tokenizer = self._tokenizer
        assert tokenizer is not None, "Tokenizer required for embeddings generation"

        def run_embeddings() -> tuple[list[list[float]], int]:
            """Run embedding generation in dedicated thread (owns Metal context)."""
            import mlx.core as mx

            # Use inner tokenizer's __call__ for batch encoding.
            # TokenizerWrapper from mlx-embeddings is not callable and
            # batch_encode_plus was removed in transformers v5.
            inner_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
            encoded = inner_tokenizer(
                texts,
                return_tensors=None,
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: mx.array(v) for k, v in encoded.items()}

            # Count tokens per input (not padded batch)
            total_tokens = 0
            for text in texts:
                tokens = tokenizer.encode(text, truncation=True, max_length=512)
                total_tokens += len(tokens)

            # Forward pass
            outputs = model(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

            # text_embeds are ALREADY L2-normalized (mean pooled + normalized)
            embeddings = outputs.text_embeds

            # Convert to Python lists
            # NOTE: mx.eval() is the MLX framework tensor evaluation function
            # It ensures computation is complete before converting to Python lists
            mx.eval(embeddings)
            embeddings_list = embeddings.tolist()

            return (embeddings_list, total_tokens)

        embeddings_list, total_tokens = await run_on_metal_thread(
            run_embeddings, error_context="Embeddings generation failed"
        )

        return EmbeddingResult(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            total_tokens=total_tokens,
            finish_reason="stop",
        )

    # ── TTS generation ───────────────────────────────────────────

    async def generate_speech(
        self,
        model: Any,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        response_format: str = "wav",
    ) -> AudioResult:
        """Generate speech audio from text using a TTS model.

        Owns the complete TTS inference path. Services call this
        instead of orchestrating audio generation themselves.
        """
        from mlx_manager.mlx_server.models.ir import AudioResult
        from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

        def run_tts() -> tuple[bytes, int]:
            """Run TTS generation in dedicated thread (owns Metal context)."""
            import io

            import mlx.core as mx
            import numpy as np

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
            # NOTE: mx.eval() is the MLX framework tensor evaluation function
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

        return AudioResult(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            format=response_format,
        )

    # ── STT transcription ────────────────────────────────────────

    async def transcribe(
        self,
        model: Any,
        audio_data: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text using an STT model.

        Owns the complete STT inference path. Services call this
        instead of orchestrating transcription themselves.
        """
        from mlx_manager.mlx_server.models.ir import TranscriptionResult
        from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

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

        return TranscriptionResult(
            text=result.get("text", ""),
            segments=result.get("segments"),
            language=result.get("language"),
        )


# --- Backward-compatible aliases ---
# These allow existing code that checks isinstance() or references
# specific subclass names to continue working during migration.
DefaultAdapter = ModelAdapter


# --- Factory ---


def create_adapter(
    family: str,
    tokenizer: Any | None = None,
    tool_parser: ToolCallParser | None = None,
    thinking_parser: ThinkingParser | None = None,
    model_id: str | None = None,
) -> ModelAdapter:
    """Create a composable adapter for a model family.

    Args:
        family: Model family name (e.g., "qwen", "llama", "whisper")
        tokenizer: HuggingFace tokenizer or processor (None for audio)
        tool_parser: Override default tool parser
        thinking_parser: Override default thinking parser
        model_id: Model identifier (needed for vision models to load config)

    Returns:
        ModelAdapter instance
    """
    config = FAMILY_CONFIGS.get(family, FAMILY_CONFIGS["default"])
    return ModelAdapter(
        config=config,
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        thinking_parser=thinking_parser,
        model_id=model_id,
    )


# Backward-compatible registry: maps family name -> FAMILY_CONFIGS keys
# Code that used `FAMILY_REGISTRY` for checking registered families
# can use `FAMILY_CONFIGS` directly instead. This alias eases migration.
FAMILY_REGISTRY = FAMILY_CONFIGS

# MLX Server — Architecture Blueprint

This document defines the target architecture of the `mlx_server` component: an
embedded inference server providing OpenAI and Anthropic-compatible APIs for MLX
models on Apple Silicon. It is the authoritative reference for how the component
**should** work. Deviations from this document are tracked separately as
compliance issues.

---

## Overview: 3-Layer Adapter Pipeline

The mlx_server uses a **3-layer pipeline architecture** that cleanly separates
concerns and enables parallel request handling with zero per-request detection:

### Layer 1: ModelAdapter (Model-Scoped, Persistent)
- **Lifecycle**: Created once at model load time, destroyed at eviction
- **Scope**: One adapter per loaded model, shared across all requests to that model
- **Responsibility**: Full pipeline owner for BOTH input and output
  - **INPUT**: Message conversion, chat template, tool delivery, stop tokens
  - **OUTPUT**: Tool extraction, thinking extraction, response cleaning
- **Key Methods**:
  - `prepare_input(messages, tools, ...) → PreparedInput`
  - `create_stream_processor() → StreamProcessor` (factory)
  - `process_complete(raw_output) → AdapterResult`
- **Pre-computed**: Stop tokens, stream markers (computed at `__init__`, cached)
- **Composed**: Injected with ToolCallParser + ThinkingParser

### Layer 2: StreamProcessor (Request-Scoped)
- **Lifecycle**: Created per streaming request via `adapter.create_stream_processor()`
- **Scope**: One processor per stream, owned by that request
- **Responsibility**: Incremental pattern matching, per-request state
  - Yields protocol-neutral `StreamEvent` (IR) for each token
  - Buffers potential pattern matches across token boundaries
  - Tracks mode (content / thinking / tool)
  - Finalizes with complete extraction via adapter's parsers
- **Key Methods**:
  - `feed(token) → StreamEvent` (incremental)
  - `finalize() → AdapterResult` (complete extraction)

### Layer 3: ProtocolFormatter (Request-Scoped)
- **Lifecycle**: Created once per request by the router
- **Scope**: One formatter per request, determined by endpoint
- **Responsibility**: Convert protocol-neutral IR to protocol responses
  - `StreamEvent` → protocol-specific SSE chunks
  - `AdapterResult` → protocol-specific complete responses
  - OpenAIFormatter: IR → OpenAI format
  - AnthropicFormatter: IR → Anthropic format
- **Key Methods**:
  - `format_stream_event(StreamEvent) → ProtocolChunk | None`
  - `format_response(AdapterResult) → ProtocolResponse`

### Request Flow

```
[Router] → creates ProtocolFormatter
    ↓
[Service] → gets LoadedModel (with adapter)
    ↓
[Adapter.prepare_input()] → PreparedInput
    ↓
[Metal Thread] → raw token stream
    ↓
[StreamProcessor.feed()] → StreamEvent (IR)
    ↓
[ProtocolFormatter.format_stream_event()] → SSE chunk → client
    ↓ (on stream end)
[StreamProcessor.finalize()] → AdapterResult (IR)
    ↓
[ProtocolFormatter.format_response()] → final chunk → client
```

### Key Design Principles
1. **1 model + 1 adapter**: Configured at load time, reused across all requests
2. **Request-scoped sessions**: Each request gets own StreamProcessor + ProtocolFormatter
3. **Parallel request support**: Queued behind same model+adapter combo, each with own processor+formatter
4. **Zero-copy streaming**: Events flow through pipeline without buffering/duplication
5. **Model-type agnostic server**: Service orchestrates; adapters handle model-specific logic
6. **Vision = Text + multimodal input**: Vision models use text adapters with image preprocessing

---

## 1. Principles

1. **Single inference pipeline**: All protocols (OpenAI, Anthropic, frontend UI)
   converge into one service layer. There is no per-endpoint inference logic.
2. **Composable, stateful adapters**: Each loaded model has a dedicated adapter
   instance composed from reusable parsers. The adapter is created at load time
   and lives as long as the model is in the pool. No per-request detection.
3. **No data loss through layers**: Messages, including tool calls, tool results,
   and multimodal content, pass through every layer with full fidelity.
4. **One canonical type per concept**: Each domain concept (tool call, message,
   model type) has exactly one Pydantic model. No duplicates, no bridging.
5. **Shared infrastructure**: Cross-cutting concerns like Metal thread management
   and memory cleanup are factored into reusable utilities, not duplicated.
6. **Probe-first, detect-as-fallback**: Model capabilities are determined once
   during probing and stored in the database. Runtime detection is a fallback
   for unprobed models, not the primary path.
7. **Parser reuse across families**: Parsers (tool call, thinking) are standalone
   strategy objects decoupled from model families. The same parser can serve
   multiple families; different models within a family can use different parsers.

---

## 2. Component Map

```
mlx_server/
  main.py                 # FastAPI app factory, lifespan, health check
  config.py               # MLXServerSettings (env: MLX_SERVER_*)
  database.py             # SQLite engine for audit logs

  api/v1/                 # Protocol endpoints (thin — validate, dispatch, format)
    chat.py               # POST /v1/chat/completions (OpenAI)
    completions.py        # POST /v1/completions (OpenAI legacy)
    messages.py           # POST /v1/messages (Anthropic)
    embeddings.py         # POST /v1/embeddings
    speech.py             # POST /v1/audio/speech (TTS)
    transcriptions.py     # POST /v1/audio/transcriptions (STT)
    models.py             # GET /v1/models
    admin.py              # Pool status, preload, unload, audit

  schemas/                # One canonical Pydantic model per concept
    openai.py             # OpenAI request/response types
    anthropic.py          # Anthropic request/response types

  parsers/                # Standalone, reusable extraction strategies
    base.py               # ToolCallParser (ABC), ThinkingParser (ABC)
    tool_call.py          # Concrete tool call parsers
    thinking.py           # Concrete thinking/reasoning parsers
    registry.py           # Parser registry: string ID → parser class

  models/                 # Model lifecycle and family-specific behavior
    types.py              # ModelType enum (TEXT_GEN, VISION, EMBEDDINGS, AUDIO)
    detection.py          # detect_model_type(), detect_model_family() — fallback
    pool.py               # ModelPoolManager — LRU cache, adapter-aware loading
    adapters/
      base.py             # ModelAdapter (stateful base class)
      registry.py         # Family → adapter class mapping, adapter factory
      qwen.py             # QwenAdapter (defaults: HermesJsonParser, ThinkTagParser)
                          # QwenVisionAdapter extends QwenAdapter, adds image preprocessing
      glm4.py             # GLM4Adapter (defaults: Glm4NativeParser, ThinkTagParser)
      llama.py            # LlamaAdapter (defaults: LlamaXmlParser, ThinkTagParser)
      gemma.py            # GemmaAdapter (defaults: NullToolParser, NullThinkingParser)
                          # GemmaVisionAdapter extends GemmaAdapter, adds image preprocessing
      mistral.py          # MistralAdapter (defaults: NullToolParser, NullThinkingParser)
      liquid.py           # LiquidAdapter (defaults: LiquidPythonParser, ThinkTagParser)
      embedding.py        # EmbeddingAdapter (minimal: tokenize → normalize)
      audio.py            # WhisperAdapter (STT), KokoroAdapter (TTS)

  services/               # Inference orchestration (model-type agnostic)
    inference.py          # Universal chat/completion generation (text + vision)
    embeddings.py         # Embedding generation (mlx-embeddings)
    audio.py              # TTS / STT (mlx-audio)
    stream_processor.py   # Request-scoped StreamProcessor (adapter-created)
    formatters/           # Protocol-specific response formatters
      base.py             # ProtocolFormatter (ABC)
      openai.py           # OpenAIFormatter (IR → OpenAI responses)
      anthropic.py        # AnthropicFormatter (IR → Anthropic responses)
    structured_output.py  # JSON schema validation for responses
    image_processor.py    # Image URL fetching + resizing for vision
    audit.py              # Request tracking + logging
    cloud/                # Cloud backend routing (experimental)
    batching/             # Continuous batching (experimental)

  errors/                 # RFC 7807 error handling
  middleware/             # Per-endpoint timeouts
  observability/          # LogFire instrumentation
  utils/                  # Shared utilities (Metal thread runner, memory)
  benchmark/              # Token throughput benchmarks
```

---

## 3. Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer                           │
│  Thin protocol endpoints. Validates requests against    │
│  schemas, creates ProtocolFormatter, pipes through      │
│  inference service. NO protocol-specific logic.         │
├─────────────────────────────────────────────────────────┤
│              Protocol Formatter Layer                   │
│  Request-scoped formatters convert protocol-neutral IR  │
│  (StreamEvent, AdapterResult) to protocol responses.    │
│  OpenAIFormatter, AnthropicFormatter. One per request.  │
├─────────────────────────────────────────────────────────┤
│                   Service Layer                         │
│  Family-agnostic inference orchestration. Gets adapter  │
│  from LoadedModel, calls prepare_input → generate →     │
│  process_output pipeline. Model-type agnostic server.   │
├─────────────────────────────────────────────────────────┤
│                Stream Processor Layer                   │
│  Request-scoped processors created by adapters.         │
│  Incremental pattern matching, per-request state,       │
│  yields protocol-neutral StreamEvents. One per stream.  │
├─────────────────────────────────────────────────────────┤
│                    Adapter Layer                        │
│  Model-scoped, persistent. Full pipeline owner: input   │
│  preparation (message conversion, template, tools) AND  │
│  output processing (tool extraction, thinking, clean).  │
│  One adapter per loaded model, lives entire lifetime.   │
├─────────────────────────────────────────────────────────┤
│                    Model Layer                          │
│  Model pool with LRU eviction. Each LoadedModel holds   │
│  a stateful adapter instance composed from injected     │
│  parsers. Adapter created at load time from DB config.  │
├─────────────────────────────────────────────────────────┤
│                   Parser Layer                          │
│  Standalone, reusable strategy objects for extracting   │
│  tool calls and thinking blocks from model output.      │
│  Decoupled from families — same parser serves multiple  │
│  families. Provides both streaming markers and batch    │
│  extraction via a single interface.                     │
├─────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                    │
│  Configuration, persistence, error handling, Metal      │
│  thread utilities, timeouts, observability.             │
└─────────────────────────────────────────────────────────┘
```

**Layer rules:**
- Each layer communicates only with adjacent layers.
- The API layer never imports from `models/adapters/` or `parsers/`.
- The service layer accesses adapters only through `LoadedModel.adapter`.
- StreamProcessor is created by adapters, used by services.
- ProtocolFormatters consume IR only, never call adapters or services directly.
- Parsers are injected into adapters; they never call adapters or services.
- Adapters never call services or API endpoints.

---

## 4. Request Flow — 3-Layer Pipeline

### 4.1 Streaming Pipeline (All Text + Vision Models)

```
POST /v1/chat/completions (OpenAI)         POST /v1/messages (Anthropic)
    │                                           │
    ▼                                           ▼
api/v1/chat.py                             api/v1/messages.py
    │                                           │
    ├── Validate against schema                 ├── Validate against schema
    ├── Create OpenAIFormatter                  ├── Create AnthropicFormatter
    │                                           │
    └──────────────┬────────────────────────────┘
                   │
                   ▼  messages + tools + params + formatter
          services/inference.py
          ::generate_chat_completion()
                   │
                   ├── pool.get_model()         → LoadedModel (with adapter)
                   │
                   │   LAYER 1: ModelAdapter (model-scoped, persistent)
                   │  ┌─────────────────────────────────────────────┐
                   │  │  Created once at model load time            │
                   │  │  Pre-configured with family parsers         │
                   │  │  Owns BOTH input prep AND output processing │
                   │  └─────────────────────────────────────────────┘
                   │
                   ├── adapter.prepare_input(messages, tools, ...)
                   │   │
                   │   ├── convert_messages() — handle tool roles
                   │   ├── apply_chat_template() — messages → prompt
                   │   ├── format_tools() — native or injected
                   │   └── aggregate stop_tokens (pre-computed)
                   │   │
                   │   ▼  PreparedInput(prompt, stop_tokens, ...)
                   │
                   ▼  prompt string + stop tokens
          Metal Thread (stream_generate / generate)
                   │
                   ▼  raw token stream
                   │
                   │   LAYER 2: StreamProcessor (request-scoped)
                   │  ┌─────────────────────────────────────────────┐
                   │  │  Created per-request via adapter factory    │
                   │  │  Holds per-request state: buffers, mode     │
                   │  │  Incremental pattern matching               │
                   │  └─────────────────────────────────────────────┘
                   │
                   ├── stream_processor = adapter.create_stream_processor()
                   │
                   ├── For each token:
                   │   │
                   │   ├── stream_processor.feed(token) → StreamEvent (IR)
                   │   │   │
                   │   │   ├── Match thinking tags → reasoning_content
                   │   │   ├── Match tool markers → buffer for extraction
                   │   │   └── Regular text → content
                   │   │   │
                   │   │   ▼  StreamEvent (protocol-neutral)
                   │   │
                   │   │   LAYER 3: ProtocolFormatter (request-scoped)
                   │   │  ┌─────────────────────────────────────────┐
                   │   │  │  Determined by endpoint                 │
                   │   │  │  Converts IR → protocol-specific SSE    │
                   │   │  │  OpenAIFormatter / AnthropicFormatter   │
                   │   │  └─────────────────────────────────────────┘
                   │   │
                   │   └── formatter.format_stream_event(event) → SSE chunk
                   │       │
                   │       ▼  yield to client
                   │
                   ├── On stream end:
                   │   │
                   │   ├── stream_processor.finalize() → AdapterResult (IR)
                   │   │   │
                   │   │   ├── adapter.tool_parser.extract() → tool_calls
                   │   │   ├── adapter.thinking_parser.extract() → reasoning
                   │   │   └── Build TextResult with all extracted content
                   │   │
                   │   └── formatter.format_response(result) → final chunk
                   │       │
                   │       ▼  yield to client
```

**Key Design Points:**
- **Vision models use TEXT adapters** — Vision is TEXT_GEN + multimodal input
- **Same output pipeline** — Vision models produce TextResult, support tools/thinking
- **Vision adapters extend text family adapters** (e.g., GemmaVisionAdapter extends GemmaAdapter)
- **Additional input preprocessing** — Image download, PIL conversion, processor formatting
- **Zero protocol logic in service** — Service orchestrates, formatters handle protocol details

### 4.2 Non-Streaming Pipeline

```
[Router receives request]
    ↓
[Create ProtocolFormatter based on endpoint]
    ↓
[services/inference.py]
    ↓
[adapter.prepare_input(messages, tools)] → PreparedInput
    ↓
[model.generate(prepared_input)] → raw output (Metal thread)
    ↓
[adapter.process_complete(raw_output)] → AdapterResult (IR)
    │
    ├── tool_parser.extract() → tool_calls
    ├── thinking_parser.extract() → reasoning
    └── Build TextResult / EmbeddingResult / AudioResult
    ↓
[formatter.format_response(result)] → complete protocol response
```

### 4.3 Embeddings Pipeline

```
POST /v1/embeddings
    │
    ▼
services/embeddings.py::generate_embeddings()
    │
    ├── pool.get_model() (EMBEDDINGS type → mlx-embeddings)
    │   │
    │   └── LoadedModel with minimal adapter (NullToolParser, NullThinkingParser)
    │
    ├── adapter.prepare_input(texts) → tokenized batch
    ├── model.embed() (Metal thread)
    ├── adapter.process_complete(embeddings) → EmbeddingResult (IR)
    │   │
    │   └── Extract embeddings, normalize, compute dimensions
    │
    └── formatter.format_response(result) → EmbeddingResponse
```

### 4.4 Audio Pipeline

```
POST /v1/audio/speech        → services/audio.py::generate_speech()
POST /v1/audio/transcriptions → services/audio.py::transcribe_audio()
    │
    ├── pool.get_model() (AUDIO type → mlx-audio)
    │   │
    │   └── WhisperAdapter or KokoroAdapter
    │
    ├── TTS: adapter.prepare_input(text, voice) → audio_params
    │   │
    │   ├── model.generate_speech() (Metal thread)
    │   └── adapter.process_complete(audio) → AudioResult (IR)
    │
    ├── STT: adapter.prepare_input(audio_bytes) → audio_tensor
    │   │
    │   ├── model.transcribe() (Metal thread)
    │   └── adapter.process_complete(segments) → TranscriptionResult (IR)
    │
    └── formatter.format_response(result) → Audio bytes / JSON transcription
```

---

## 5. Parser Architecture

Parsers are standalone strategy objects that handle extraction of structured
content from model output. They are decoupled from model families and reusable:
the same parser can serve multiple families, and different models within a family
can use different parsers.

### 5.1 Parser Contracts

```python
class ToolCallParser(ABC):
    """Extracts tool calls from model output.

    Each implementation handles one specific output format.
    Used in both streaming (marker detection) and batch (full extraction).
    """

    @property
    @abstractmethod
    def parser_id(self) -> str:
        """Unique string identifier for DB storage (e.g., 'hermes_json')."""

    @property
    @abstractmethod
    def stream_markers(self) -> list[tuple[str, str]]:
        """(start_marker, end_marker) pairs for streaming detection."""

    @abstractmethod
    def extract(self, text: str) -> list[ToolCall]:
        """Extract all tool calls from complete text (batch mode)."""

    def validates(self, text: str, expected_fn: str) -> bool:
        """Check if output contains a valid call to expected_fn.
        Used by probe. Delegates to extract() — same code path as inference."""
        return any(tc.function.name == expected_fn for tc in self.extract(text))


class ThinkingParser(ABC):
    """Extracts thinking/reasoning blocks from model output.

    The enable_thinking parameter is part of the base contract because
    some models (GLM-4.7, Qwen3) support toggling thinking mode per-request
    via the chat template, while others always think or never think.
    """

    @property
    @abstractmethod
    def parser_id(self) -> str:
        """Unique string identifier for DB storage (e.g., 'think_tag')."""

    @property
    @abstractmethod
    def stream_markers(self) -> list[tuple[str, str]]:
        """(start_marker, end_marker) pairs for streaming detection."""

    @property
    def supports_toggle(self) -> bool:
        """Whether this parser supports enable_thinking parameter in templates.
        Override in subclasses for models that support toggling."""
        return False

    @abstractmethod
    def extract(self, text: str) -> str | None:
        """Extract thinking content from text. Returns None if no thinking."""

    @abstractmethod
    def remove(self, text: str) -> str:
        """Remove all thinking blocks from text, return cleaned content."""
```

### 5.2 Concrete Parsers

**Tool Call Parsers** (many-to-many with families):

| Parser ID          | Format                                                           | Used By (defaults)     |
|--------------------|------------------------------------------------------------------|------------------------|
| `hermes_json`      | `<tool_call>{"name": ..., "arguments": ...}</tool_call>`         | Qwen, Qwen3           |
| `glm4_native`      | `<tool_call>fn<arg_key>k</arg_key><arg_value>v</arg_value>`     | GLM-4.7                |
| `glm4_xml`         | `<tool_call><name>fn</name><arguments>{...}</arguments>`         | GLM-4                  |
| `llama_xml`        | `<function=name>{...}</function>`                                | Llama 3.x              |
| `llama_python`     | `<\|python_tag\|>module.method(args)<\|eom_id\|>`               | Llama (code)           |
| `null`             | No-op, never matches                                             | Gemma, Mistral, others |

**Thinking Parsers**:

| Parser ID      | Format                          | `supports_toggle` | Used By (defaults)    |
|----------------|---------------------------------|-------------------|-----------------------|
| `think_tag`    | `<think>...</think>`            | `True`            | Qwen3, GLM-4.7       |
| `null`         | No-op, no thinking blocks       | `False`           | Gemma, Mistral, Llama |

### 5.3 Parser Registry

A simple registry maps string IDs to parser classes:

```python
TOOL_PARSERS: dict[str, type[ToolCallParser]] = {
    "hermes_json": HermesJsonParser,
    "glm4_native": Glm4NativeParser,
    "glm4_xml":    Glm4XmlParser,
    "llama_xml":   LlamaXmlParser,
    "llama_python": LlamaPythonParser,
    "null":        NullToolParser,
}

THINKING_PARSERS: dict[str, type[ThinkingParser]] = {
    "think_tag": ThinkTagParser,
    "null":      NullThinkingParser,
}
```

When loading a model from DB, parser IDs are resolved to instances:
`"hermes_json"` → `HermesJsonParser()`. This decouples DB storage from Python
classes and makes the system extensible without schema migrations.

---

## 6. Adapter Architecture — 3-Layer Pipeline Owner

### 6.1 Layer 1: ModelAdapter (Model-Scoped, Persistent)

Each `LoadedModel` holds a dedicated `ModelAdapter` instance. The adapter is
**created once at model load time** and **destroyed when the model is evicted**.
It is the **full pipeline owner**: both INPUT preparation and OUTPUT processing.

```python
class ModelAdapter(ABC):
    """Stateful adapter, one instance per loaded model.

    Owns the complete inference pipeline:
    - INPUT: Message conversion, chat template, tool delivery
    - OUTPUT: Tool extraction, thinking extraction, response cleaning

    Composes parsers via dependency injection. Pre-computes stop tokens
    and stream markers at init. Lives as long as the model is in pool.
    """

    def __init__(
        self,
        tokenizer: Any,
        tool_parser: ToolCallParser,
        thinking_parser: ThinkingParser,
        capabilities: ModelCapabilities | None = None,
    ):
        self.tokenizer = tokenizer
        self.tool_parser = tool_parser
        self.thinking_parser = thinking_parser
        self.capabilities = capabilities
        self._stop_token_ids: list[int] = self._compute_stop_tokens()
        self._stream_markers: list[tuple[str, str]] = self._compute_stream_markers()

    # --- Identity ---

    @property
    @abstractmethod
    def family(self) -> str:
        """Model family identifier (qwen, glm4, llama, gemma, etc.)."""

    # --- INPUT PIPELINE ---

    def prepare_input(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> PreparedInput:
        """Full input preparation pipeline.

        Returns PreparedInput with:
        - prompt: str — formatted prompt ready for generation
        - stop_tokens: list[int] — aggregated stop tokens
        - metadata: dict — any additional info for generation
        """
        # 1. Message conversion (handle tool roles)
        converted = self.convert_messages(messages)

        # 2. Tool delivery (native, adapter, or injected)
        tool_format = self.capabilities.tool_format if self.capabilities else None
        if tools:
            if tool_format in ("template", "native") and self.supports_native_tools():
                prompt = self.apply_chat_template(converted, tools=tools, enable_thinking=enable_thinking)
            elif tool_format in ("adapter", "hermes"):
                tool_prompt = self.format_tools_for_prompt(tools)
                injected = self._inject_tool_prompt(converted, tool_prompt)
                prompt = self.apply_chat_template(injected, enable_thinking=enable_thinking)
            else:
                prompt = self.apply_chat_template(converted, enable_thinking=enable_thinking)
        else:
            prompt = self.apply_chat_template(converted, enable_thinking=enable_thinking)

        # 3. Stop tokens (pre-computed, cached)
        stop_tokens = self.get_stop_tokens()

        return PreparedInput(prompt=prompt, stop_tokens=stop_tokens, metadata={})

    @abstractmethod
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Apply tokenizer chat template. Messages → prompt string."""

    @abstractmethod
    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Transform tool/special roles to tokenizer-compatible format."""

    @abstractmethod
    def format_tools_for_prompt(self, tools: list[dict]) -> str:
        """Format tool definitions for system prompt injection fallback."""

    # --- OUTPUT PIPELINE ---

    def create_stream_processor(self) -> StreamProcessor:
        """Factory method for request-scoped stream processor.

        Pre-configures processor with this adapter's parsers and markers.
        Each request gets its own processor instance.
        """
        return StreamProcessor(
            tool_parser=self.tool_parser,
            thinking_parser=self.thinking_parser,
            stream_markers=self._stream_markers,
        )

    def process_complete(self, raw_output: str, **kwargs) -> AdapterResult:
        """Process complete (non-streaming) model output.

        Returns AdapterResult (TextResult for text/vision, EmbeddingResult, etc.)
        with extracted tool calls, thinking, and cleaned content.
        """
        # Extract tool calls via composed parser
        tool_calls = self.tool_parser.extract(raw_output)

        # Extract thinking via composed parser
        reasoning_content = self.thinking_parser.extract(raw_output)

        # Clean content (remove thinking tags)
        content = self.thinking_parser.remove(raw_output)

        return TextResult(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason="stop",
            prompt_tokens=kwargs.get("prompt_tokens", 0),
            completion_tokens=kwargs.get("completion_tokens", 0),
        )

    # --- Pre-Computed Properties ---

    def get_stop_tokens(self) -> list[int]:
        """Return pre-computed stop tokens. No runtime lookup."""
        return self._stop_token_ids

    @abstractmethod
    def _compute_stop_tokens(self) -> list[int]:
        """Compute stop tokens once at init. Called by __init__."""

    def _compute_stream_markers(self) -> list[tuple[str, str]]:
        """Combined markers from both parsers for single-pass streaming."""
        return (
            self.tool_parser.stream_markers
            + self.thinking_parser.stream_markers
        )

    # --- Tool Support ---

    def supports_native_tools(self) -> bool:
        """Whether tokenizer accepts tools= parameter natively."""
        if self.capabilities:
            return bool(self.capabilities.supports_native_tools)
        return False
```

**Vision Models as Text Adapters:**

Vision adapters **extend text family adapters** and add multimodal input handling:

```python
class GemmaVisionAdapter(GemmaAdapter):
    """Vision adapter extending GemmaAdapter.

    Adds image preprocessing to input pipeline.
    Uses SAME output pipeline: thinking, tools, response cleaning.
    Produces TextResult (not a separate result type).
    """

    def prepare_input(self, messages: list[dict], **kwargs) -> PreparedInput:
        # 1. Download and preprocess images
        processed_messages = self._preprocess_images(messages)

        # 2. Apply processor-based formatting (PIL → tensors)
        formatted = self._apply_vision_processor(processed_messages)

        # 3. Delegate to parent for chat template + tools
        return super().prepare_input(formatted, **kwargs)

    def _preprocess_images(self, messages: list[dict]) -> list[dict]:
        """Download URLs, convert to PIL Images."""
        ...

    def _apply_vision_processor(self, messages: list[dict]) -> list[dict]:
        """Use model's processor for image → tensor conversion."""
        ...
```

### 6.2 Layer 2: StreamProcessor (Request-Scoped)

Created per-request via `adapter.create_stream_processor()`. Holds per-request
state and performs incremental pattern matching to yield protocol-neutral IR.

```python
class StreamProcessor:
    """Request-scoped processor for streaming inference.

    Created by adapter's factory method. Holds per-request state:
    - accumulated_text: full output buffer
    - mode: current state (content / thinking / tool)
    - pattern_buffers: partial match tracking

    Yields StreamEvent (IR) for each token/chunk.
    Finalize() uses adapter's parsers for complete extraction.
    """

    def __init__(
        self,
        tool_parser: ToolCallParser,
        thinking_parser: ThinkingParser,
        stream_markers: list[tuple[str, str]],
    ):
        self.tool_parser = tool_parser
        self.thinking_parser = thinking_parser
        self.stream_markers = stream_markers

        self.accumulated_text = ""
        self.mode = "content"  # "content" | "thinking" | "tool"
        self._buffers: dict[str, str] = {}

    def feed(self, token: str) -> StreamEvent:
        """Process one token, return protocol-neutral IR event.

        Performs incremental pattern matching:
        - Thinking tags → yield reasoning_content
        - Tool markers → buffer silently, accumulate
        - Regular content → yield content

        Buffers potential pattern matches across token boundaries.
        """
        self.accumulated_text += token

        # Pattern matching logic (check all markers)
        for start_marker, end_marker in self.stream_markers:
            if start_marker in token:
                # Mode transition, update state
                ...
            elif end_marker in token:
                # End of special block
                ...

        # Yield appropriate event based on mode
        if self.mode == "thinking":
            return StreamEvent(reasoning_content=token)
        elif self.mode == "tool":
            return StreamEvent()  # Silent accumulation
        else:
            return StreamEvent(content=token)

    def finalize(self) -> AdapterResult:
        """Final extraction on complete accumulated text.

        Uses adapter's parsers for batch extraction.
        Returns complete AdapterResult (TextResult).
        """
        # Use injected parsers (same instances adapter uses)
        tool_calls = self.tool_parser.extract(self.accumulated_text)
        reasoning_content = self.thinking_parser.extract(self.accumulated_text)
        content = self.thinking_parser.remove(self.accumulated_text)

        return TextResult(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason="stop",
            prompt_tokens=0,  # Computed by caller
            completion_tokens=len(self.accumulated_text.split()),
        )
```

### 6.3 Family Subclasses

Subclasses provide **sensible defaults** for known families. The parsers can
be overridden at instantiation time (e.g., when probe discovers a model uses
a different parser than the family default).

```
ModelAdapter (abstract, stateful, full pipeline owner)
  ├── QwenAdapter         # Defaults: HermesJsonParser, ThinkTagParser
  │   └── QwenVisionAdapter  # Extends QwenAdapter, adds image preprocessing
  ├── GLM4Adapter         # Defaults: Glm4NativeParser, ThinkTagParser
  ├── LlamaAdapter        # Defaults: LlamaXmlParser, NullThinkingParser
  ├── GemmaAdapter        # Defaults: NullToolParser, NullThinkingParser
  │   └── GemmaVisionAdapter  # Extends GemmaAdapter, adds image preprocessing
  ├── MistralAdapter      # Defaults: NullToolParser, NullThinkingParser
  ├── LiquidAdapter       # Defaults: LiquidPythonParser, ThinkTagParser
  ├── WhisperAdapter      # Audio STT (minimal processing)
  ├── KokoroAdapter       # Audio TTS (minimal processing)
  └── DefaultAdapter      # Defaults: NullToolParser, NullThinkingParser
```

**Embeddings Adapters:**

```python
class EmbeddingAdapter(ModelAdapter):
    """Minimal adapter for embedding models.

    Input: Tokenize text batch
    Output: Extract embeddings, normalize → EmbeddingResult
    Parsers: NullToolParser, NullThinkingParser
    """

    def prepare_input(self, texts: list[str], **kwargs) -> PreparedInput:
        # Tokenization only
        tokens = [self.tokenizer.encode(t) for t in texts]
        return PreparedInput(prompt=tokens, stop_tokens=[], metadata={})

    def process_complete(self, embeddings: list[list[float]], **kwargs) -> EmbeddingResult:
        # Normalize, compute dimensions
        normalized = self._normalize(embeddings)
        return EmbeddingResult(
            embeddings=normalized,
            dimensions=len(normalized[0]),
            finish_reason="stop",
            prompt_tokens=kwargs.get("prompt_tokens", 0),
            completion_tokens=0,
        )
```

**Audio Adapters:**

```python
class KokoroAdapter(ModelAdapter):
    """TTS adapter for Kokoro models."""

    def prepare_input(self, text: str, voice: str, **kwargs) -> PreparedInput:
        # Audio-specific params
        return PreparedInput(
            prompt=text,
            stop_tokens=[],
            metadata={"voice": voice, "sample_rate": 24000},
        )

    def process_complete(self, audio_data: bytes, **kwargs) -> AudioResult:
        return AudioResult(
            audio_data=audio_data,
            sample_rate=kwargs.get("sample_rate", 24000),
            format="wav",
            finish_reason="stop",
            prompt_tokens=0,
            completion_tokens=0,
        )
```

### 6.4 Adapter Factory

The adapter factory creates adapter instances during model loading:

```python
def create_adapter(
    family: str,
    tokenizer: Any,
    model_type: ModelType,
    capabilities: ModelCapabilities | None = None,
) -> ModelAdapter:
    """Create an adapter instance for a loaded model.

    Uses model_type to select appropriate adapter class.
    If capabilities specify parser IDs (from probe), those parsers are
    injected. Otherwise, the family's defaults are used.
    """
    # Select adapter class based on model_type + family
    if model_type == ModelType.VISION:
        adapter_class = VISION_ADAPTER_REGISTRY.get(family, DefaultVisionAdapter)
    elif model_type == ModelType.EMBEDDINGS:
        adapter_class = EmbeddingAdapter
    elif model_type == ModelType.AUDIO:
        adapter_class = AUDIO_ADAPTER_REGISTRY.get(family, DefaultAudioAdapter)
    else:  # TEXT_GEN
        adapter_class = FAMILY_REGISTRY.get(family, DefaultAdapter)

    kwargs = {"tokenizer": tokenizer, "capabilities": capabilities}

    # Override parsers if DB specifies them (text/vision only)
    if model_type in (ModelType.TEXT_GEN, ModelType.VISION):
        if capabilities and capabilities.tool_parser_id:
            kwargs["tool_parser"] = resolve_tool_parser(capabilities.tool_parser_id)
        if capabilities and capabilities.thinking_parser_id:
            kwargs["thinking_parser"] = resolve_thinking_parser(
                capabilities.thinking_parser_id
            )

    return adapter_class(**kwargs)
```

### 6.5 Message Conversion Contract

`convert_messages()` must handle these OpenAI message types:

1. **`role: "tool"`** — Tool result messages. Convert to a format the tokenizer
   can process (typically `role: "user"` with structured text).
2. **`role: "assistant"` with `tool_calls`** — Assistant requesting tool use.
   Convert `tool_calls` list to inline text representation.
3. All other roles — Pass through unchanged.

This method is called **early in prepare_input()** before tool prompt injection
and chat template application. It ensures the tokenizer never sees message
structures it cannot handle.

---

## 7. Protocol-Neutral Intermediate Representation (IR)

### 7.1 Design Principle

The adapter layer produces **protocol-neutral IR types**. These types represent
the semantic content of model responses without any protocol-specific fields
or formatting. The `ProtocolFormatter` layer converts IR to protocol responses.

### 7.2 PreparedInput Type

The `PreparedInput` type is the output of `adapter.prepare_input()` and
the input to the generation layer:

```python
class PreparedInput:
    """Result of adapter input pipeline.

    Contains everything needed for generation on the Metal thread.
    Returned by ModelAdapter.prepare_input().
    """
    prompt: str | list[int] | Any  # Formatted prompt (string for text, tokens for embeddings, etc.)
    stop_tokens: list[int]          # Aggregated stop tokens (pre-computed + family-specific)
    metadata: dict[str, Any]        # Additional context (e.g., voice for TTS, processor for vision)
```

### 7.3 IR Type Hierarchy

```python
# Streaming IR
class StreamEvent:
    """Protocol-neutral streaming event.

    Yielded by StreamProcessor.feed() during token-by-token processing.
    Formatters convert to protocol-specific SSE chunks.
    """
    content: str | None = None              # Regular text content
    reasoning_content: str | None = None    # Thinking/reasoning content
    tool_call_delta: ToolCallDelta | None = None  # Incremental tool call
    is_complete: bool = False               # End-of-stream signal


# Complete Response IR
class AdapterResult(ABC):
    """Base class for complete adapter results.

    All results include token counts and finish reason.
    Subclasses add type-specific fields.
    """
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class TextResult(AdapterResult):
    """Result from TEXT_GEN and VISION models.

    Vision models produce TextResult because they use the same output
    pipeline as text models: tool extraction, thinking extraction, cleaning.
    """
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class EmbeddingResult(AdapterResult):
    """Result from EMBEDDINGS models."""
    embeddings: list[list[float]]
    dimensions: int


class AudioResult(AdapterResult):
    """Result from AUDIO TTS models."""
    audio_data: bytes
    sample_rate: int
    format: str  # "wav", "mp3", etc.


class TranscriptionResult(AdapterResult):
    """Result from AUDIO STT models."""
    text: str
    segments: list[Segment] | None = None
```

### 7.4 Canonical Tool Call Types

There is **one** set of Pydantic models for tool calls, used by parsers,
adapters, IR, and API schemas:

- `schemas/openai.py::ToolCall` — The canonical tool call type
- `schemas/openai.py::FunctionCall` — Function name + arguments string
- `schemas/openai.py::ToolCallDelta` — Incremental tool call for streaming

Parsers produce these types directly. Adapters pass them through to IR.
Formatters serialize them to protocol formats. No bridging or conversion.

### 7.5 Streaming Flow (Protocol-Neutral)

```
Token Stream (from Metal thread)
    │
    ▼
StreamProcessor.feed(token)  [created by adapter.create_stream_processor()]
    │
    ├── Incremental pattern matching against pre-configured markers
    │   (thinking tags, tool markers — from adapter's parsers)
    │
    ├── If inside <think>...</think> → yield StreamEvent(reasoning_content=token)
    ├── If inside tool markers → buffer silently (accumulate)
    ├── If regular content → yield StreamEvent(content=token)
    │
    ▼  StreamEvent (protocol-neutral IR)
    │
    ▼
ProtocolFormatter.format_stream_event(event)  [OpenAI or Anthropic]
    │
    ├── OpenAIFormatter → ChatCompletionChunk dict
    ├── AnthropicFormatter → content_block_delta / message_delta SSE events
    │
    ▼  Protocol-specific SSE chunk → client


On stream end:
    │
    ▼
StreamProcessor.finalize()
    │
    ├── adapter.tool_parser.extract(accumulated_text) → tool_calls
    ├── adapter.thinking_parser.extract(accumulated_text) → reasoning
    ├── adapter.thinking_parser.remove(accumulated_text) → clean content
    │
    ▼  TextResult (protocol-neutral IR)
    │
    ▼
ProtocolFormatter.format_response(result)
    │
    ├── OpenAIFormatter → ChatCompletionResponse
    ├── AnthropicFormatter → AnthropicMessagesResponse
    │
    ▼  Final protocol-specific chunk → client
```

---

## 9. Metal Thread Affinity

MLX Metal operations have GPU thread affinity — they must run on the same
thread that initialized the Metal context. All inference services use a
shared utility pattern:

```
    Async generator (main event loop)
         ▲
         │ Queue.get() via run_in_executor
         │
    Dedicated Thread (owns Metal context)
         │ Queue.put() per token/result
         ▼
    mlx_lm / mlx_vlm / mlx_embeddings / mlx_audio
```

The thread management boilerplate (Queue creation, thread spawn, error
propagation via Queue, cache clearing in `finally`) is factored into
`utils/metal.py::run_on_metal_thread()` to avoid duplication across the
four inference services.

**Key points:**
- One Metal thread per model pool (shared across all model types)
- Adapter pipeline (prepare_input, process_complete) runs on main thread
- Only MLX library calls (generate, embed, transcribe) run on Metal thread
- StreamProcessor runs on main thread (no GPU affinity needed)

---

## 8. Protocol Formatter Layer (Request-Scoped)

### 8.1 Design Principle

The `ProtocolFormatter` layer converts protocol-neutral IR (StreamEvent,
AdapterResult) to protocol-specific responses. Formatters are **request-scoped**:
created once per request and owned by the router.

This replaces the old `ProtocolTranslator` which performed bidirectional
conversion. The new design has **unidirectional flow**: IR → protocol response.

### 8.2 ProtocolFormatter Interface

```python
class ProtocolFormatter(ABC):
    """Base class for protocol-specific formatters.

    Converts protocol-neutral IR to protocol responses.
    Created once per request by the router.
    """

    @abstractmethod
    def format_stream_event(self, event: StreamEvent) -> ProtocolChunk | None:
        """Convert IR StreamEvent to protocol-specific SSE chunk.

        May return None to suppress events (e.g., tool deltas in some protocols).
        Returns dict or Pydantic model ready for SSE serialization.
        """

    @abstractmethod
    def format_response(self, result: AdapterResult) -> ProtocolResponse:
        """Convert complete IR result to protocol-specific response.

        Handles all AdapterResult subtypes:
        - TextResult → ChatCompletionResponse / MessagesResponse
        - EmbeddingResult → EmbeddingResponse
        - AudioResult → raw bytes Response
        - TranscriptionResult → TranscriptionResponse
        """

    @abstractmethod
    def format_error(self, error: Exception) -> ProtocolError:
        """Convert exception to protocol-specific error response."""
```

### 8.3 OpenAIFormatter

```python
class OpenAIFormatter(ProtocolFormatter):
    """Formats IR as OpenAI-compatible responses.

    Used by: /v1/chat/completions, /v1/completions, /v1/embeddings
    """

    def format_stream_event(self, event: StreamEvent) -> dict | None:
        """StreamEvent → ChatCompletionChunk dict.

        Structure:
        {
            "id": "chatcmpl-...",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": event.content,           # Regular content
                    "reasoning_content": event.reasoning_content,  # Thinking
                    "tool_calls": [event.tool_call_delta],  # Tool deltas
                },
                "finish_reason": null,
            }],
        }
        """
        if not event.content and not event.reasoning_content and not event.tool_call_delta:
            return None  # Suppress empty events

        return {
            "id": self._request_id,
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": self._build_delta(event),
                "finish_reason": "stop" if event.is_complete else None,
            }],
        }

    def format_response(self, result: AdapterResult) -> dict | Response:
        """AdapterResult → OpenAI response.

        TextResult → ChatCompletionResponse
        EmbeddingResult → EmbeddingResponse
        AudioResult → Response(content=bytes, media_type="audio/wav")
        TranscriptionResult → TranscriptionResponse
        """
        if isinstance(result, TextResult):
            return {
                "id": self._request_id,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.content,
                        "reasoning_content": result.reasoning_content,
                        "tool_calls": result.tool_calls,
                    },
                    "finish_reason": result.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            }
        elif isinstance(result, EmbeddingResult):
            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": emb, "index": i}
                    for i, emb in enumerate(result.embeddings)
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "total_tokens": result.prompt_tokens,
                },
            }
        # ... other result types
```

### 8.4 AnthropicFormatter

```python
class AnthropicFormatter(ProtocolFormatter):
    """Formats IR as Anthropic-compatible responses.

    Used by: /v1/messages

    Anthropic has a more complex streaming protocol with multiple event types:
    - message_start
    - content_block_start
    - content_block_delta (multiple, for thinking + content)
    - content_block_stop
    - message_delta (usage)
    - message_stop
    """

    def format_stream_event(self, event: StreamEvent) -> dict | None:
        """StreamEvent → Anthropic SSE event.

        Anthropic streams content blocks separately:
        - Thinking content → content_block_delta with type="thinking"
        - Regular content → content_block_delta with type="text"
        - Tool calls → content_block_delta with type="tool_use"
        """
        events = []

        # Send message_start on first event
        if self._is_first_event:
            events.append({
                "type": "message_start",
                "message": {"id": self._request_id, "role": "assistant"},
            })
            self._is_first_event = False

        # Thinking content block
        if event.reasoning_content:
            if not self._thinking_block_started:
                events.append({
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "thinking"},
                })
                self._thinking_block_started = True

            events.append({
                "type": "content_block_delta",
                "index": self._block_index,
                "delta": {"type": "thinking_delta", "thinking": event.reasoning_content},
            })

        # Regular content block
        if event.content:
            if not self._content_block_started:
                events.append({
                    "type": "content_block_start",
                    "index": self._block_index + 1,
                    "content_block": {"type": "text"},
                })
                self._content_block_started = True

            events.append({
                "type": "content_block_delta",
                "index": self._block_index + 1,
                "delta": {"type": "text_delta", "text": event.content},
            })

        return events  # May return multiple SSE events for one StreamEvent

    def format_response(self, result: AdapterResult) -> dict:
        """TextResult → Anthropic MessagesResponse.

        Structure:
        {
            "id": "msg_...",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": result.reasoning_content},
                {"type": "text", "text": result.content},
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
            ],
            "usage": {...},
        }
        """
        content_blocks = []

        if result.reasoning_content:
            content_blocks.append({
                "type": "thinking",
                "thinking": result.reasoning_content,
            })

        if result.content:
            content_blocks.append({
                "type": "text",
                "text": result.content,
            })

        if result.tool_calls:
            for tc in result.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        return {
            "id": self._request_id,
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "usage": {
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
            },
        }
```

### 8.5 Router Integration

Routers determine which formatter to use based on the endpoint:

```python
# api/v1/chat.py (OpenAI endpoint)
@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    formatter = OpenAIFormatter(request_id=generate_id())

    async for event in generate_chat_completion_stream(
        model_id=request.model,
        messages=request.messages,
        tools=request.tools,
        # ...
    ):
        # event is StreamEvent (IR)
        chunk = formatter.format_stream_event(event)
        if chunk:
            yield f"data: {json.dumps(chunk)}\n\n"

    # Final response
    result = await finalize_stream()  # returns TextResult (IR)
    final_chunk = formatter.format_response(result)
    yield f"data: {json.dumps(final_chunk)}\n\n"


# api/v1/messages.py (Anthropic endpoint)
@router.post("/messages")
async def create_message(request: AnthropicMessagesRequest):
    formatter = AnthropicFormatter(request_id=generate_id())

    async for event in generate_chat_completion_stream(
        model_id=request.model,
        messages=request.messages,  # Already converted to internal format
        # ...
    ):
        # event is StreamEvent (IR)
        events = formatter.format_stream_event(event)  # May return multiple events
        for sse_event in events:
            yield f"event: {sse_event['type']}\ndata: {json.dumps(sse_event)}\n\n"

    # Final response
    result = await finalize_stream()  # returns TextResult (IR)
    final_message = formatter.format_response(result)
    yield f"event: message_stop\ndata: {json.dumps(final_message)}\n\n"
```

### 8.6 What Replaced ProtocolTranslator

**Old architecture:**
- `ProtocolTranslator.anthropic_to_internal()` converted Anthropic requests → OpenAI format
- `ProtocolTranslator.internal_to_anthropic_response()` converted OpenAI responses → Anthropic format
- Bidirectional conversion, tightly coupled to OpenAI as "internal" format

**New architecture:**
- Anthropic router converts request → generic message dict (minimal, protocol-agnostic)
- Service layer works with generic message dicts (not OpenAI-specific)
- Adapters produce protocol-neutral IR (StreamEvent, TextResult)
- **ProtocolFormatter** converts IR → protocol responses (unidirectional)
- OpenAI is not privileged; it's one formatter among equals

---

## 10. Inference Service Layer (Model-Type Agnostic Server)

### 10.1 Design Principle

The service layer is a **thin orchestrator** that coordinates the adapter
pipeline without knowing model-specific details. It delegates all
family-specific logic to adapters.

### 10.2 Universal Inference Flow

```python
async def generate_chat_completion(
    model_id: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    formatter: ProtocolFormatter,
    **kwargs,
):
    """Universal inference service for all text/vision models.

    Model-type agnostic: delegates everything to the adapter.
    Works with any ProtocolFormatter (OpenAI, Anthropic, etc.).
    """
    # 1. Get model from pool (adapter already attached)
    loaded = await pool.get_model(model_id)
    adapter = loaded.adapter  # Pre-configured at load time

    # 2. INPUT PIPELINE (delegated to adapter)
    prepared = adapter.prepare_input(
        messages=messages,
        tools=tools,
        enable_thinking=enable_thinking,
    )

    # 3. GENERATION (on Metal thread, model-type aware)
    if loaded.model_type == ModelType.VISION:
        raw_stream = await _generate_vision(loaded, prepared, **kwargs)
    else:  # TEXT_GEN
        raw_stream = await _generate_text(loaded, prepared, **kwargs)

    # 4. OUTPUT PIPELINE (delegated to adapter + formatter)
    stream_processor = adapter.create_stream_processor()

    async for token in raw_stream:
        # Adapter layer: token → StreamEvent (IR)
        event = stream_processor.feed(token)

        # Formatter layer: StreamEvent → protocol chunk
        chunk = formatter.format_stream_event(event)
        if chunk:
            yield chunk

    # 5. FINALIZE (adapter extracts, formatter formats)
    result = stream_processor.finalize()  # → TextResult (IR)
    final_chunk = formatter.format_response(result)
    yield final_chunk
```

**What the service layer does:**
- Get model from pool
- Call `adapter.prepare_input()` → get PreparedInput
- Route to appropriate MLX library based on `model_type`
- Pipe raw stream through `adapter.create_stream_processor()`
- Pipe IR events through `formatter`
- No family detection, no parser selection, no protocol logic

**What the service layer does NOT do:**
- Message conversion (adapter's job)
- Chat template application (adapter's job)
- Tool formatting (adapter's job)
- Stop token computation (adapter's job, pre-computed)
- Tool call extraction (adapter's parser's job)
- Thinking extraction (adapter's parser's job)
- Protocol formatting (formatter's job)

### 10.3 Service Layer per Model Type

While the main `generate_chat_completion()` handles TEXT_GEN and VISION,
specialized services exist for other model types:

```python
# services/embeddings.py
async def generate_embeddings(
    model_id: str,
    texts: list[str],
    formatter: ProtocolFormatter,
    **kwargs,
):
    loaded = await pool.get_model(model_id)
    adapter = loaded.adapter  # EmbeddingAdapter

    prepared = adapter.prepare_input(texts)
    embeddings = await _embed_on_metal_thread(loaded, prepared)
    result = adapter.process_complete(embeddings)  # → EmbeddingResult (IR)

    return formatter.format_response(result)


# services/audio.py
async def generate_speech(
    model_id: str,
    text: str,
    voice: str,
    formatter: ProtocolFormatter,
    **kwargs,
):
    loaded = await pool.get_model(model_id)
    adapter = loaded.adapter  # KokoroAdapter

    prepared = adapter.prepare_input(text, voice=voice)
    audio_data = await _tts_on_metal_thread(loaded, prepared)
    result = adapter.process_complete(audio_data)  # → AudioResult (IR)

    return formatter.format_response(result)
```

All services follow the same pattern: **get → prepare → generate → process → format**.

---

## 11. Model Lifecycle

### 11.1 Classification Axes

Model identification has two independent axes, both resolved **once** during
probing and stored in the database:

| Axis       | Determines              | Source             | Stored As                    |
|------------|--------------------------|--------------------|-----------------------------|
| **Type**   | Which MLX library/loader | `config.json`      | `ModelCapabilities.model_type`  |
| **Family** | Which adapter subclass   | Model ID + config  | `ModelCapabilities.model_family`|

Type determines the MLX library (mlx-lm, mlx-vlm, etc.).
Family determines the adapter class, default parsers, chat template, stop tokens.

### 11.2 Probe Phase (One-Time)

Probing runs once per model (typically on download). It determines the model's
full configuration and validates that the inference pipeline will work.

```
Probe(model_id):
    1. detect_model_type(config.json) → model_type
    2. detect_model_family(model_id + config) → model_family
    3. Load model via pool
    4. Instantiate family adapter with default parsers
    5. Test tool support (2-attempt, adapter-driven):
       a. Attempt 1 — Template delivery:
          If adapter.supports_native_tools() or has_native_tool_support(tokenizer):
          → Generate with tools= param via adapter
          → Validate: adapter's tool_parser first, then sweep ALL parsers
          → If match: tool_format="template", record parser_id
       b. Attempt 2 — Adapter delivery:
          If adapter.format_tools_for_prompt() returns content:
          → Inject as system message, generate via adapter
          → Validate: adapter's tool_parser first, then sweep ALL parsers
          → If match: tool_format="adapter", record parser_id
       c. No match: scan for unknown XML tags (WARNING), tool_format=None
    6. Test thinking support (generation-based):
       → Generate with enable_thinking=True via adapter
       → Validate: adapter's thinking_parser first, then sweep ALL parsers
       → Template check is authoritative (even if no tags in output)
    7. Store to DB:
       model_type, model_family, tool_parser_id, thinking_parser_id,
       supports_native_tools, supports_thinking, tool_format,
       practical_max_tokens, ...
    8. Unload model (if not preloaded)
```

The key insight: the probe uses the **same parsers** that inference will use.
If the probe's `tool_parser.validates()` passes, inference's
`tool_parser.extract()` will work — they are the same method on the same class.

### 11.3 Load Phase (On-Demand)

When a model is needed for inference, the pool loads it and creates a fully
configured adapter:

```
pool.get_model(model_id):
    1. Check cache → if hot, return immediately (adapter already attached)
    2. Read ModelCapabilities from DB:
       → model_type, model_family, tool_parser_id, thinking_parser_id
    3. If no DB entry → fallback to detect_model_type() + detect_model_family()
       → Notify UI: "Model not probed — capabilities may be incomplete"
    4. Use model_type to select loader:
       TEXT_GEN → mlx_lm.load()
       VISION  → mlx_vlm.load()
       EMBEDDINGS → mlx_embeddings.load()
       AUDIO   → mlx_audio.load_model()
    5. Create adapter (in parallel with model loading):
       → Resolve parser IDs to instances via registry
       → create_adapter(family, tokenizer, capabilities)
       → Adapter pre-computes stop tokens in __init__
    6. Attach adapter to LoadedModel:
       loaded.adapter = adapter
    7. Return LoadedModel (ready for inference)
```

### 11.4 Inference Phase (Zero Detection)

At inference time, the service layer reads everything from the LoadedModel.
No detection, no per-request adapter creation, no trial calls:

```
generate_chat_completion(model_id, messages, tools, formatter, ...):
    loaded = pool.get_model(model_id)    → LoadedModel with adapter
    adapter = loaded.adapter             → already initialized

    # 1. INPUT PIPELINE (delegated to adapter)
    prepared = adapter.prepare_input(
        messages=messages,
        tools=tools,
        enable_thinking=enable_thinking,
    )
    # prepared.prompt — formatted prompt string
    # prepared.stop_tokens — pre-computed, aggregated
    # prepared.metadata — any additional context

    # 2. GENERATION (Metal thread, model-type aware)
    raw_stream = await _generate(loaded, prepared)

    # 3. OUTPUT PIPELINE (adapter + formatter)
    stream_processor = adapter.create_stream_processor()

    async for token in raw_stream:
        event = stream_processor.feed(token)  # → StreamEvent (IR)
        chunk = formatter.format_stream_event(event)  # → protocol chunk
        yield chunk

    # 4. FINALIZE
    result = stream_processor.finalize()  # → TextResult (IR)
    final_chunk = formatter.format_response(result)  # → protocol response
    yield final_chunk
```

**Key insight:** The service layer never calls `convert_messages()`,
`apply_chat_template()`, `format_tools_for_prompt()`, `tool_parser.extract()`,
or `thinking_parser.extract()` directly. All of that is encapsulated in the
adapter's `prepare_input()` and the stream processor's `feed()` / `finalize()`.

### 11.5 Pool Management

`ModelPoolManager` maintains a bounded set of hot models:

- **On-demand loading**: Models loaded when first requested
- **LRU eviction**: Least-recently-used models evicted at memory/count limits.
  When a model is evicted, its adapter is destroyed with it.
- **Type-aware loading**: Uses `ModelType` to select the correct loader
- **Adapter-aware loading**: Creates and attaches adapter during load
- **Preload protection**: Pinned models exempt from eviction
- **LoRA support**: LoRA adapter loading via `get_model_with_adapter()`

---

## 12. Data Model (ModelCapabilities)

The `ModelCapabilities` table stores everything needed to configure a model
without runtime detection. It is the single source of truth for model behavior.

```
ModelCapabilities (SQLModel, table=True):
    model_id: str (PK)              # e.g., "mlx-community/Qwen3-0.6B-4bit"

    # Classification
    model_type: str | None          # TEXT_GEN, VISION, EMBEDDINGS, AUDIO
    model_family: str | None        # qwen, glm4, llama, gemma, mistral, default

    # Parser configuration
    tool_parser_id: str | None      # hermes_json, glm4_native, llama_xml, null
    thinking_parser_id: str | None  # think_tag, null

    # Capabilities (text-gen)
    supports_native_tools: bool | None  # True if any tool delivery method works
    supports_thinking: bool | None      # model produces thinking blocks
    tool_format: str | None             # "template" | "adapter" | None
    practical_max_tokens: int | None    # estimated from KV cache + memory

    # Capabilities (vision)
    supports_multi_image: bool | None
    supports_video: bool | None

    # Capabilities (embeddings)
    embedding_dimensions: int | None
    max_sequence_length: int | None
    is_normalized: bool | None

    # Capabilities (audio)
    supports_tts: bool | None
    supports_stt: bool | None

    # Metadata
    probed_at: datetime
    probe_version: int = 3          # Schema version for future migrations
```

---

## 13. Singleton Services

Services follow the `get_*()` / `reset_*()` pattern. `reset_*()` exists
for test isolation.

| Accessor                    | Returns               | Scope              |
|-----------------------------|-----------------------|--------------------|
| `get_model_pool()`          | `ModelPoolManager`    | Global             |
| `get_settings()`            | `MLXServerSettings`   | Global             |
| `get_router()`              | `BackendRouter`       | Global             |
| `audit_service`             | `AuditService`        | Module-level       |

**Note on non-singletons:**
- **Adapters**: Model-scoped, one per LoadedModel. Created at load, destroyed at eviction.
- **StreamProcessors**: Request-scoped, one per streaming request. Created by adapter factory.
- **ProtocolFormatters**: Request-scoped, one per request. Created by router.
- **Parsers**: Stateless strategy objects. May be shared across adapters (e.g., all Qwen models
  share the same `HermesJsonParser()` instance if using default parsers).

---

## 14. Cross-Cutting Concerns

### 14.1 Observability

- **LogFire**: Optional spans wrapping inference calls (standalone mode only)
- **Audit logging**: `AuditService` tracks requests with timing, tokens, backend
- **Structured logging**: `loguru` with DEBUG-level prompt/response dumps

### 14.2 Error Handling

- RFC 7807 `ProblemDetail` responses for structured errors
- `TimeoutHTTPException` with per-endpoint configurable timeouts
- Circuit breaker on cloud backend clients

### 14.3 Configuration

`MLXServerSettings` via pydantic-settings (`MLX_SERVER_` prefix):
- Model pool limits (memory GB, max model count)
- Per-endpoint timeout settings
- Feature flags: cloud routing, batching
- Audit log retention

---

## 15. Deployment Modes

| Mode          | Flag                    | Behavior |
|---------------|-------------------------|----------|
| **Embedded**  | `embedded_mode=True`    | Mounted at `/v1` inside MLX Manager. Shares DB. No lifespan, no LogFire. |
| **Standalone**| `embedded_mode=False`   | Independent FastAPI app. Own lifespan, LogFire, DB. |

---

## 16. Experimental Features

Behind configuration flags, not production-ready:

- **Cloud routing** (`enable_cloud_routing`): Rule-based dispatch to cloud
  backends (OpenAI, Anthropic) with circuit breaker and local fallback.
- **Continuous batching** (`enable_batching`): PagedAttention-inspired request
  scheduling for concurrent text inference.

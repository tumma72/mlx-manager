# MLX Server — Architecture Blueprint

This document defines the target architecture of the `mlx_server` component: an
embedded inference server providing OpenAI and Anthropic-compatible APIs for MLX
models on Apple Silicon. It is the authoritative reference for how the component
**should** work. Deviations from this document are tracked separately as
compliance issues.

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
      glm4.py             # GLM4Adapter (defaults: Glm4NativeParser, ThinkTagParser)
      llama.py            # LlamaAdapter (defaults: LlamaXmlParser, ThinkTagParser)
      gemma.py            # GemmaAdapter (defaults: NullToolParser, NullThinkingParser)
      mistral.py          # MistralAdapter (defaults: NullToolParser, NullThinkingParser)

  services/               # Inference orchestration
    inference.py          # Text chat/completion generation (mlx-lm)
    vision.py             # Vision generation (mlx-vlm)
    embeddings.py         # Embedding generation (mlx-embeddings)
    audio.py              # TTS / STT (mlx-audio)
    streaming.py          # Single-pass StreamingProcessor (parser-driven)
    protocol.py           # ProtocolTranslator (Anthropic ↔ OpenAI internal)
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
│  schemas, converts to internal dict format preserving   │
│  ALL fields (content, tool_calls, tool_call_id, role),  │
│  dispatches to services, formats responses.             │
├─────────────────────────────────────────────────────────┤
│                   Service Layer                         │
│  Family-agnostic inference orchestration. Reads the     │
│  adapter from the LoadedModel (no per-request lookup).  │
│  Owns protocol translation, structured output.          │
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
- Parsers are injected into adapters; they never call adapters or services.
- Adapters never call services or API endpoints.

---

## 4. Request Flow

### 4.1 Chat Completion (text) — all protocols converge

```
POST /v1/chat/completions (OpenAI)         POST /v1/messages (Anthropic)
    │                                           │
    ▼                                           ▼
api/v1/chat.py                             api/v1/messages.py
    │                                           │
    ├── Validate against schema                 ├── Validate against schema
    ├── Convert ChatMessage → dict              ├── ProtocolTranslator
    │   preserving ALL fields:                  │   .anthropic_to_internal()
    │   role, content, tool_calls,              │
    │   tool_call_id, name                      │
    │                                           │
    └──────────────┬────────────────────────────┘
                   │
                   ▼  list[dict] + tools + params
          services/inference.py
          ::generate_chat_completion()
                   │
                   ├── pool.get_model()         → LoadedModel (with adapter)
                   │
                   │   NO per-request detection. Adapter is already
                   │   initialized on LoadedModel with the correct
                   │   parsers, stop tokens, and capabilities.
                   │
                   │  ┌─── Adapter Pipeline ───────────────────────────┐
                   │  │                                                │
                   ├──│  1. adapter.convert_messages()                 │
                   │  │     Transform tool/special roles to format     │
                   │  │     the tokenizer can handle                   │
                   │  │                                                │
                   ├──│  2. If tools AND adapter.supports_native_tools:│
                   │  │       pass tools= to apply_chat_template()    │
                   │  │     Elif tools AND enable_prompt_injection:    │
                   │  │       adapter.format_tools_for_prompt()        │
                   │  │                                                │
                   ├──│  3. adapter.apply_chat_template()              │
                   │  │     Messages → prompt string                   │
                   │  │                                                │
                   ├──│  4. adapter.get_stop_tokens()                  │
                   │  │     Pre-computed at load time (cached)         │
                   │  │                                                │
                   │  └────────────────────────────────────────────────┘
                   │
                   ▼  prompt string + stop tokens
          Metal Thread (stream_generate / generate)
                   │
                   ▼  token stream
          StreamingProcessor (single-pass, parser-driven)
                   │
                   │   Uses adapter.get_stream_markers() which
                   │   combines markers from both tool_parser
                   │   and thinking_parser. ONE pass over the
                   │   stream handles all extraction.
                   │
                   ├── reasoning_content (via thinking_parser markers)
                   ├── content (regular text, markers filtered out)
                   └── tool_calls (via tool_parser.extract() in finalize)
                   │
                   ▼
          OpenAI-compatible response dict
```

### 4.2 Vision

```
POST /v1/chat/completions (with image content blocks)
    │
    ▼
api/v1/chat.py → loaded.model_type == VISION
    │
    ├── preprocess_images() → PIL Image list
    │
    ▼
services/vision.py::generate_vision_completion()
    │
    ├── pool.get_model() (VISION type → mlx-vlm)
    ├── Metal thread generation
    └── OpenAI-compatible response dict
```

### 4.3 Embeddings

```
POST /v1/embeddings
    │
    ▼
services/embeddings.py::generate_embeddings()
    │
    ├── pool.get_model() (EMBEDDINGS type → mlx-embeddings)
    ├── Metal thread embedding
    └── OpenAI-compatible embedding response
```

### 4.4 Audio (TTS / STT)

```
POST /v1/audio/speech        → services/audio.py::generate_speech()
POST /v1/audio/transcriptions → services/audio.py::transcribe_audio()
    │
    ├── pool.get_model() (AUDIO type → mlx-audio)
    ├── Metal thread generation
    └── Audio bytes / transcription text
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

## 6. Adapter Architecture (Composition + Strategy)

### 6.1 Stateful Adapter

Each `LoadedModel` holds a dedicated `ModelAdapter` instance. The adapter is
**created once at model load time** and **destroyed when the model is evicted**.
There is no per-request adapter detection or creation.

```python
class ModelAdapter(ABC):
    """Stateful adapter, one instance per loaded model.

    Composes a tool call parser and thinking parser via dependency injection.
    Pre-computes stop tokens at init time. Lives as long as the model.
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

    # --- Identity ---

    @property
    @abstractmethod
    def family(self) -> str: ...

    # --- Chat Template ---

    @abstractmethod
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> str: ...

    # --- Stop Tokens (pre-computed) ---

    def get_stop_tokens(self) -> list[int]:
        """Return pre-computed stop tokens. No runtime lookup."""
        return self._stop_token_ids

    @abstractmethod
    def _compute_stop_tokens(self) -> list[int]:
        """Compute stop tokens once at init. Called by __init__."""

    # --- Tool Support ---

    def supports_native_tools(self) -> bool:
        """Whether tokenizer accepts tools= parameter natively."""
        if self.capabilities:
            return bool(self.capabilities.supports_native_tools)
        return False

    @abstractmethod
    def format_tools_for_prompt(self, tools: list[dict]) -> str:
        """Format tool definitions for system prompt injection fallback."""

    # --- Message Conversion ---

    @abstractmethod
    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Transform tool/special roles to tokenizer-compatible format."""

    # --- Streaming Integration ---

    def get_stream_markers(self) -> list[tuple[str, str]]:
        """Combined markers from both parsers for single-pass streaming."""
        return (
            self.tool_parser.stream_markers
            + self.thinking_parser.stream_markers
        )
```

### 6.2 Family Subclasses

Subclasses provide **sensible defaults** for known families. The parsers can
be overridden at instantiation time (e.g., when probe discovers a model uses
a different parser than the family default).

```
ModelAdapter (abstract, stateful)
  ├── QwenAdapter         # Defaults: HermesJsonParser, ThinkTagParser
  ├── GLM4Adapter         # Defaults: Glm4NativeParser, ThinkTagParser
  ├── LlamaAdapter        # Defaults: LlamaXmlParser, NullThinkingParser
  ├── GemmaAdapter        # Defaults: NullToolParser, NullThinkingParser
  ├── MistralAdapter      # Defaults: NullToolParser, NullThinkingParser
  └── DefaultAdapter      # Defaults: NullToolParser, NullThinkingParser
```

Example subclass:

```python
class QwenAdapter(ModelAdapter):
    """Adapter for Qwen model family.

    Defaults to HermesJsonParser for tools and ThinkTagParser for thinking.
    Qwen3 models support enable_thinking toggle via tokenizer template.
    """

    def __init__(self, tokenizer, tool_parser=None, thinking_parser=None, **kw):
        super().__init__(
            tokenizer=tokenizer,
            tool_parser=tool_parser or HermesJsonParser(),
            thinking_parser=thinking_parser or ThinkTagParser(),
            **kw,
        )

    @property
    def family(self) -> str:
        return "qwen"

    def apply_chat_template(self, messages, *, tools=None, enable_thinking=None):
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if tools and self.supports_native_tools():
            kwargs["tools"] = tools
        if enable_thinking is not None and self.thinking_parser.supports_toggle:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)
```

### 6.3 Adapter Factory

The adapter factory creates adapter instances during model loading:

```python
def create_adapter(
    family: str,
    tokenizer: Any,
    capabilities: ModelCapabilities | None = None,
) -> ModelAdapter:
    """Create an adapter instance for a loaded model.

    If capabilities specify parser IDs (from probe), those parsers are
    injected. Otherwise, the family's defaults are used.
    """
    adapter_class = FAMILY_REGISTRY.get(family, DefaultAdapter)

    kwargs = {"tokenizer": tokenizer, "capabilities": capabilities}

    # Override parsers if DB specifies them
    if capabilities and capabilities.tool_parser_id:
        kwargs["tool_parser"] = resolve_tool_parser(capabilities.tool_parser_id)
    if capabilities and capabilities.thinking_parser_id:
        kwargs["thinking_parser"] = resolve_thinking_parser(
            capabilities.thinking_parser_id
        )

    return adapter_class(**kwargs)
```

### 6.4 Message Conversion Contract

`convert_messages()` must handle these OpenAI message types:

1. **`role: "tool"`** — Tool result messages. Convert to a format the tokenizer
   can process (typically `role: "user"` with structured text).
2. **`role: "assistant"` with `tool_calls`** — Assistant requesting tool use.
   Convert `tool_calls` list to inline text representation.
3. All other roles — Pass through unchanged.

This method is called **before** tool prompt injection and chat template
application. It ensures the tokenizer never sees message structures it cannot
handle.

---

## 7. Streaming Architecture (Single-Pass)

### 7.1 Design Principle

The streaming processor performs a **single pass** over the model's token
stream. It receives combined markers from the adapter's tool_parser and
thinking_parser, and routes content accordingly. There is no double parsing.

### 7.2 Flow

```
Token Stream (from Metal thread)
    │
    ▼
StreamingProcessor.feed(token)
    │
    ├── Check token against ALL markers (tool + thinking combined)
    │   via adapter.get_stream_markers()
    │
    ├── If inside thinking markers → yield as reasoning_content
    ├── If inside tool markers → buffer silently (accumulate)
    ├── If regular content → yield as content
    │
    ▼
StreamingProcessor.finalize()
    │
    ├── adapter.tool_parser.extract(accumulated_text) → tool_calls
    ├── adapter.thinking_parser.extract(accumulated_text) → reasoning
    ├── adapter.thinking_parser.remove(remaining_text) → clean content
    │
    ▼
ParseResult(content, tool_calls, reasoning)
```

### 7.3 Canonical Types

There is **one** set of Pydantic models for tool calls, used by parsers and
API schemas:

- `schemas/openai.py::ToolCall` — The canonical tool call type
- `schemas/openai.py::FunctionCall` — Function name + arguments string

Parsers produce these types directly. No bridging or conversion.

---

## 8. Metal Thread Affinity

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

---

## 9. Protocol Translation

The `ProtocolTranslator` enables a single inference pipeline to serve multiple
API protocols:

```
Anthropic Request → anthropic_to_internal() → list[dict] (OpenAI-format messages)
                                                    │
                                                    ▼
                                          generate_chat_completion()
                                                    │
                                                    ▼
Anthropic Response ← internal_to_anthropic_response() ← OpenAI result dict
```

The frontend chat endpoint (`mlx_manager/routers/chat.py`) also calls
`generate_chat_completion()` directly with the same dict message format,
converting the streaming response into UI-specific SSE events.

---

## 10. Model Lifecycle

### 10.1 Classification Axes

Model identification has two independent axes, both resolved **once** during
probing and stored in the database:

| Axis       | Determines              | Source             | Stored As                    |
|------------|--------------------------|--------------------|-----------------------------|
| **Type**   | Which MLX library/loader | `config.json`      | `ModelCapabilities.model_type`  |
| **Family** | Which adapter subclass   | Model ID + config  | `ModelCapabilities.model_family`|

Type determines the MLX library (mlx-lm, mlx-vlm, etc.).
Family determines the adapter class, default parsers, chat template, stop tokens.

### 10.2 Probe Phase (One-Time)

Probing runs once per model (typically on download). It determines the model's
full configuration and validates that the inference pipeline will work.

```
Probe(model_id):
    1. detect_model_type(config.json) → model_type
    2. detect_model_family(model_id + config) → model_family
    3. Load model via pool
    4. Instantiate family adapter with default parsers
    5. Test tool support:
       a. HAPPY PATH: Generate with adapter's default tool_parser
          → adapter.tool_parser.validates(output, "get_weather")
          → If passes: record tool_parser_id = adapter.tool_parser.parser_id
       b. FALLBACK: Default parser failed → iterate ALL registered
          ToolCallParsers, test each one
          → If any passes: record that parser's ID
          → If none pass: record tool_parser_id = "null"
    6. Test thinking support (same pattern: default first, then all)
    7. Store to DB:
       model_type, model_family, tool_parser_id, thinking_parser_id,
       supports_native_tools, supports_thinking, practical_max_tokens, ...
    8. Unload model (if not preloaded)
```

The key insight: the probe uses the **same parsers** that inference will use.
If the probe's `tool_parser.validates()` passes, inference's
`tool_parser.extract()` will work — they are the same method on the same class.

### 10.3 Load Phase (On-Demand)

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

### 10.4 Inference Phase (Zero Detection)

At inference time, the service layer reads everything from the LoadedModel.
No detection, no per-request adapter creation, no trial calls:

```
generate_chat_completion(model_id, messages, tools, ...):
    loaded = pool.get_model(model_id)    → LoadedModel with adapter
    adapter = loaded.adapter             → already initialized

    # 1. Convert messages
    converted = adapter.convert_messages(messages)

    # 2. Handle tools
    if tools and adapter.supports_native_tools():
        prompt = adapter.apply_chat_template(converted, tools=tools)
    elif tools and enable_prompt_injection:
        tool_prompt = adapter.format_tools_for_prompt(tools)
        # inject tool_prompt into system message
        prompt = adapter.apply_chat_template(injected_messages)
    else:
        prompt = adapter.apply_chat_template(converted)

    # 3. Stop tokens (pre-computed, no lookup)
    stop_tokens = adapter.get_stop_tokens()

    # 4. Generate + single-pass stream processing
    markers = adapter.get_stream_markers()
    stream_processor = StreamingProcessor(markers=markers)
    # ... generate and feed tokens ...

    # 5. Finalize: extract tool calls and thinking via adapter's parsers
    result = stream_processor.finalize(adapter.tool_parser, adapter.thinking_parser)
```

### 10.5 Pool Management

`ModelPoolManager` maintains a bounded set of hot models:

- **On-demand loading**: Models loaded when first requested
- **LRU eviction**: Least-recently-used models evicted at memory/count limits.
  When a model is evicted, its adapter is destroyed with it.
- **Type-aware loading**: Uses `ModelType` to select the correct loader
- **Adapter-aware loading**: Creates and attaches adapter during load
- **Preload protection**: Pinned models exempt from eviction
- **LoRA support**: LoRA adapter loading via `get_model_with_adapter()`

---

## 11. Data Model (ModelCapabilities)

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
    supports_native_tools: bool | None  # tokenizer accepts tools= parameter
    supports_thinking: bool | None      # model produces thinking blocks
    tool_format: str | None             # "native" | "injection" | None
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

## 12. Singleton Services

Services follow the `get_*()` / `reset_*()` pattern. `reset_*()` exists
for test isolation.

| Accessor                    | Returns               | Scope              |
|-----------------------------|-----------------------|--------------------|
| `get_model_pool()`          | `ModelPoolManager`    | Global             |
| `get_translator()`          | `ProtocolTranslator`  | Global             |
| `get_settings()`            | `MLXServerSettings`   | Global             |
| `get_router()`              | `BackendRouter`       | Global             |
| `audit_service`             | `AuditService`        | Module-level       |

**Note**: Adapters and parsers are **not** singletons. Each loaded model has
its own adapter instance. Parser instances may be shared across adapters of
the same type (they are stateless strategy objects).

---

## 13. Cross-Cutting Concerns

### 13.1 Observability

- **LogFire**: Optional spans wrapping inference calls (standalone mode only)
- **Audit logging**: `AuditService` tracks requests with timing, tokens, backend
- **Structured logging**: `loguru` with DEBUG-level prompt/response dumps

### 13.2 Error Handling

- RFC 7807 `ProblemDetail` responses for structured errors
- `TimeoutHTTPException` with per-endpoint configurable timeouts
- Circuit breaker on cloud backend clients

### 13.3 Configuration

`MLXServerSettings` via pydantic-settings (`MLX_SERVER_` prefix):
- Model pool limits (memory GB, max model count)
- Per-endpoint timeout settings
- Feature flags: cloud routing, batching
- Audit log retention

---

## 14. Deployment Modes

| Mode          | Flag                    | Behavior |
|---------------|-------------------------|----------|
| **Embedded**  | `embedded_mode=True`    | Mounted at `/v1` inside MLX Manager. Shares DB. No lifespan, no LogFire. |
| **Standalone**| `embedded_mode=False`   | Independent FastAPI app. Own lifespan, LogFire, DB. |

---

## 15. Experimental Features

Behind configuration flags, not production-ready:

- **Cloud routing** (`enable_cloud_routing`): Rule-based dispatch to cloud
  backends (OpenAI, Anthropic) with circuit breaker and local fallback.
- **Continuous batching** (`enable_batching`): PagedAttention-inspired request
  scheduling for concurrent text inference.

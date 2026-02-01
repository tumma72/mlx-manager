# Phase 14: Model Adapter Enhancements - Research

**Researched:** 2026-02-01
**Domain:** MLX model adapters for tool calling, reasoning mode, message conversion, structured output, and LoRA support
**Confidence:** MEDIUM (verified with multiple sources but some implementation details require validation)

## Summary

This phase focuses on achieving feature parity with reference MLX servers (mlx-openai-server, mlx-omni-server, vllm-mlx) by extending the existing model adapter architecture. The current adapters handle chat templates and stop tokens; this phase adds tool calling, reasoning mode, message conversion, structured output, and LoRA adapter support.

The key insight is that each model family (Llama, Qwen, GLM4) has its own unique format for tool calls and reasoning tokens. Rather than a monolithic parser, the pattern is to use model-specific parsers that understand each format and normalize to OpenAI-compatible output. The mlx-lm library provides native LoRA support via the `adapter_path` parameter in the `load()` function.

**Primary recommendation:** Extend the existing `ModelAdapter` protocol with optional methods for tool call parsing, reasoning extraction, message conversion, and structured output validation. Each model-family adapter implements only the methods relevant to its capabilities.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx-lm | 0.21+ | Model loading with LoRA | Native adapter_path support, Apple-official |
| pydantic | 2.x | Schema validation | Already used, excellent JSON Schema support |
| jsonschema | 4.x | JSON Schema validation | Standard for structured output validation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| outlines | latest | Constrained decoding | If grammar-based structured output needed |
| lm-format-enforcer | latest | JSON schema enforcement | Alternative to outlines for constrained decoding |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| jsonschema | pydantic-only | pydantic can validate but jsonschema is more flexible for runtime schema loading |
| Custom parser | outlines | Outlines is heavier but guarantees valid output; custom parser is lighter but post-hoc validation |

**Installation:**
```bash
pip install jsonschema
# outlines is optional, only if constrained decoding is needed
```

## Architecture Patterns

### Recommended Project Structure
```
mlx_server/
├── models/
│   └── adapters/
│       ├── base.py          # ModelAdapter protocol (extended)
│       ├── registry.py      # Adapter detection and registration
│       ├── llama.py         # Llama 3.x adapter (tool calling, reasoning)
│       ├── qwen.py          # Qwen adapter (tool calling, Hermes style)
│       ├── glm4.py          # GLM4 adapter (XML-style tool calling)
│       └── parsers/         # NEW: Tool call parsers
│           ├── __init__.py
│           ├── base.py      # ToolCallParser protocol
│           ├── llama.py     # Llama tool parser (<function=name>...</function>)
│           ├── qwen.py      # Qwen tool parser (Hermes-style JSON)
│           └── glm4.py      # GLM4 tool parser (XML arg_key/arg_value)
├── services/
│   ├── inference.py         # Extended with tool call handling
│   └── structured_output.py # NEW: JSON Schema validation service
└── schemas/
    └── openai.py            # Extended with tool/function schemas
```

### Pattern 1: Extended ModelAdapter Protocol
**What:** Add optional methods to the existing ModelAdapter protocol for new capabilities
**When to use:** All adapters, but each implements only relevant methods
**Example:**
```python
# Source: Architecture pattern from mlx-omni-server
from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class ModelAdapter(Protocol):
    """Extended protocol for model-specific handling."""

    @property
    def family(self) -> str: ...

    def apply_chat_template(self, tokenizer: Any, messages: list[dict],
                           add_generation_prompt: bool = True) -> str: ...

    def get_stop_tokens(self, tokenizer: Any) -> list[int]: ...

    # NEW: Tool calling support (optional)
    def supports_tool_calling(self) -> bool:
        """Return True if this model family supports tool calling."""
        return False

    def parse_tool_calls(self, text: str) -> list[dict] | None:
        """Parse tool calls from model output. Returns None if no tool calls."""
        return None

    def format_tools_for_prompt(self, tools: list[dict]) -> str:
        """Format tool definitions for inclusion in prompt."""
        return ""

    # NEW: Reasoning mode support (optional)
    def supports_reasoning_mode(self) -> bool:
        """Return True if this model supports thinking/reasoning mode."""
        return False

    def extract_reasoning(self, text: str) -> tuple[str | None, str]:
        """Extract reasoning content from response.
        Returns (reasoning_content, final_content)."""
        return None, text

    # NEW: Message conversion (optional)
    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages to model-specific format."""
        return messages
```

### Pattern 2: Tool Call Parser Registry
**What:** Separate parser classes for each model family's tool call format
**When to use:** When model outputs tool calls that need parsing
**Example:**
```python
# Source: mlx-omni-server pattern
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ToolCall:
    """Parsed tool call from model output."""
    id: str
    name: str
    arguments: dict[str, Any]

class ToolCallParser(ABC):
    """Base class for model-specific tool call parsers."""

    @abstractmethod
    def parse(self, text: str) -> list[ToolCall]:
        """Parse tool calls from model output text."""
        ...

    @abstractmethod
    def format_tools(self, tools: list[dict]) -> str:
        """Format tool definitions for prompt injection."""
        ...

class LlamaToolParser(ToolCallParser):
    """Parser for Llama 3.x tool calling format.

    Formats:
    - Custom: <function=name>{"param": "value"}</function><|eot_id|>
    - Python: <|python_tag|>tool.call(query="...")<|eom_id|>
    - Array:  [func_name(param1='value1')]<|eot_id|>
    """

    def parse(self, text: str) -> list[ToolCall]:
        calls = []
        # Check for XML-style function calls
        import re
        pattern = r'<function=(\w+)>(.*?)</function>'
        for match in re.finditer(pattern, text):
            name = match.group(1)
            args_json = match.group(2)
            calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=json.loads(args_json)
            ))
        return calls
```

### Pattern 3: Reasoning Content Extraction
**What:** Extract `<think>` or `<reasoning>` tags and separate from final content
**When to use:** When processing output from reasoning models
**Example:**
```python
# Source: mlx-omni-server ThinkingDecoder pattern
import re

class ReasoningExtractor:
    """Extract reasoning/thinking content from model output."""

    THINKING_PATTERNS = [
        (r'<think>(.*?)</think>', 'think'),
        (r'<thinking>(.*?)</thinking>', 'thinking'),
        (r'<reasoning>(.*?)</reasoning>', 'reasoning'),
    ]

    def extract(self, text: str) -> tuple[str | None, str]:
        """Extract reasoning and return (reasoning, final_content)."""
        reasoning_parts = []
        content = text

        for pattern, tag_name in self.THINKING_PATTERNS:
            matches = re.findall(pattern, text, re.DOTALL)
            reasoning_parts.extend(matches)
            # Remove the tags from content
            content = re.sub(pattern, '', content, flags=re.DOTALL)

        reasoning = '\n'.join(reasoning_parts) if reasoning_parts else None
        return reasoning, content.strip()
```

### Pattern 4: Structured Output Validation
**What:** Validate model output against provided JSON Schema
**When to use:** When response_format specifies json_schema
**Example:**
```python
# Source: OpenAI Structured Outputs pattern
from jsonschema import validate, ValidationError

class StructuredOutputValidator:
    """Validate model output against JSON Schema."""

    def validate(self, output: str, schema: dict) -> tuple[bool, dict | str]:
        """Validate output against schema.
        Returns (success, parsed_json_or_error_message)."""
        try:
            parsed = json.loads(output)
            validate(instance=parsed, schema=schema)
            return True, parsed
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except ValidationError as e:
            return False, f"Schema validation failed: {e.message}"
```

### Pattern 5: LoRA Adapter Loading
**What:** Load LoRA adapters alongside base models using mlx-lm's native support
**When to use:** When model pool needs to load adapters
**Example:**
```python
# Source: mlx-lm documentation
from mlx_lm import load

def load_model_with_adapter(
    model_id: str,
    adapter_path: str | None = None
) -> tuple[Any, Any]:
    """Load model with optional LoRA adapter.

    Args:
        model_id: HuggingFace model ID
        adapter_path: Path to adapter directory (containing adapter_config.json)

    Returns:
        (model, tokenizer) tuple
    """
    return load(model_id, adapter_path=adapter_path)
```

### Anti-Patterns to Avoid
- **Monolithic parser:** Don't try to parse all model formats in one class; use registry pattern
- **Hardcoded token IDs:** Token IDs vary by model/version; always look up from tokenizer
- **Regex-only parsing:** Complex tool call formats need proper parsing; regex is fragile
- **Ignoring streaming:** Tool calls and reasoning must work with streaming output

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON Schema validation | Custom validator | jsonschema library | Edge cases in JSON Schema spec are complex |
| Constrained decoding | Token masking logic | outlines or lm-format-enforcer | FSM compilation is non-trivial |
| Tool call ID generation | Sequential counters | uuid.uuid4().hex[:8] | Must be unique across requests |
| Message format conversion | Ad-hoc transforms | Adapter pattern with registry | Model-specific quirks are numerous |

**Key insight:** Tool calling formats vary significantly between model families. The Llama format uses XML-style `<function=name>`, Qwen uses Hermes-style JSON, and GLM4 uses XML with `<arg_key>/<arg_value>` tags. Trying to unify parsing is error-prone; use model-specific parsers.

## Common Pitfalls

### Pitfall 1: Incomplete Stop Token Handling for Tool Calls
**What goes wrong:** Model generates tool call but continues past the closing tag
**Why it happens:** Tool call delimiters aren't in stop token list
**How to avoid:** Add tool call end tokens (`</function>`, `</tool_call>`, `<|eom_id|>`) to stop tokens when tools are enabled
**Warning signs:** Model output contains complete tool call plus extra text

### Pitfall 2: Streaming Tool Call Parsing
**What goes wrong:** Tool call parsed as partial fragments during streaming
**Why it happens:** Parser runs on incomplete output
**How to avoid:** Buffer output and only parse tool calls when end delimiter detected
**Warning signs:** Malformed tool call objects in streaming responses

### Pitfall 3: Type Coercion in Tool Arguments
**What goes wrong:** Integer arguments passed as strings break function execution
**Why it happens:** Model outputs `"5"` instead of `5` for integer parameters
**How to avoid:** Use JSON Schema to coerce types after parsing; validate against tool parameter schema
**Warning signs:** TypeError when calling functions with parsed arguments

### Pitfall 4: Reasoning Tag Stripping Order
**What goes wrong:** Reasoning content bleeds into final response
**Why it happens:** Nested or overlapping tags not handled correctly
**How to avoid:** Extract reasoning first, then process remaining content
**Warning signs:** `<think>` fragments in API response content

### Pitfall 5: LoRA Adapter Path Format
**What goes wrong:** `adapter_config.json not found` error
**Why it happens:** mlx-lm expects directory containing config, not the .npz file directly
**How to avoid:** Always pass directory path, not file path; validate config exists before loading
**Warning signs:** FileNotFoundError during adapter loading

### Pitfall 6: GLM4 Duplicate Tool Call Tags
**What goes wrong:** Parser extracts same tool call multiple times
**Why it happens:** GLM4 sometimes outputs `<tool_call><tool_call><tool_call>` (known bug)
**How to avoid:** Deduplicate parsed tool calls by content; handle malformed markers gracefully
**Warning signs:** Repeated identical tool calls in parsed output

## Code Examples

### Example 1: Qwen Tool Call Parsing (Hermes Style)
```python
# Source: mlx-omni-server pattern, Qwen documentation
import json
import re
import uuid

class QwenToolParser:
    """Parse Qwen3 Hermes-style tool calls.

    Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    """

    TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)

    def parse(self, text: str) -> list[dict]:
        """Parse tool calls from Qwen output."""
        calls = []
        for match in self.TOOL_CALL_PATTERN.finditer(text):
            try:
                data = json.loads(match.group(1))
                calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data.get("arguments", {}))
                    }
                })
            except json.JSONDecodeError:
                continue
        return calls
```

### Example 2: Llama Tool Call Formatting
```python
# Source: Meta Llama documentation, llama-models prompt_format.md
class LlamaToolFormatter:
    """Format tools for Llama 3.x models."""

    def format_tools_system_prompt(self, tools: list[dict]) -> str:
        """Generate system prompt section for tools."""
        if not tools:
            return ""

        tool_docs = []
        for tool in tools:
            func = tool.get("function", {})
            doc = f"""
{func.get("name", "unknown")}:
  description: {func.get("description", "")}
  parameters: {json.dumps(func.get("parameters", {}))}
"""
            tool_docs.append(doc.strip())

        return f"""You have access to the following functions:

{chr(10).join(tool_docs)}

To call a function, respond with:
<function=function_name>{{"param": "value"}}</function>
"""
```

### Example 3: Extended OpenAI Schema for Tool Calling
```python
# Source: OpenAI API Reference
from pydantic import BaseModel, Field
from typing import Literal, Any

class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None

class Tool(BaseModel):
    """Tool definition."""
    type: Literal["function"] = "function"
    function: FunctionDefinition

class ToolCall(BaseModel):
    """Tool call in assistant response."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

class FunctionCall(BaseModel):
    """Function call details."""
    name: str
    arguments: str  # JSON string

class ChatCompletionRequestWithTools(ChatCompletionRequest):
    """Extended request with tool support."""
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict | None = None
    response_format: ResponseFormat | None = None

class ResponseFormat(BaseModel):
    """Structured output format."""
    type: Literal["text", "json_object", "json_schema"]
    json_schema: dict | None = None  # For type="json_schema"
```

### Example 4: LoRA Adapter Integration
```python
# Source: mlx-lm documentation, DeepWiki
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AdapterConfig:
    """Configuration for a LoRA adapter."""
    adapter_path: str
    model_id: str  # Base model this adapter was trained for
    description: str | None = None

class ModelPoolWithAdapters:
    """Extended model pool with adapter support."""

    async def get_model_with_adapter(
        self,
        model_id: str,
        adapter_path: str | None = None
    ) -> LoadedModel:
        """Load model with optional LoRA adapter.

        Args:
            model_id: Base model HuggingFace ID
            adapter_path: Path to adapter directory
        """
        from mlx_lm import load

        # Validate adapter path if provided
        if adapter_path:
            adapter_dir = Path(adapter_path)
            if not adapter_dir.is_dir():
                raise ValueError(f"Adapter path must be directory: {adapter_path}")
            config_file = adapter_dir / "adapter_config.json"
            if not config_file.exists():
                raise ValueError(f"Missing adapter_config.json in: {adapter_path}")

        model, tokenizer = load(model_id, adapter_path=adapter_path)

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            adapter_path=adapter_path
        )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual JSON mode | Structured Outputs with json_schema | OpenAI Aug 2024 | Guaranteed schema adherence |
| No reasoning visibility | `<think>` tags with reasoning_content field | 2025 with DeepSeek-R1, Qwen3 | Transparent chain-of-thought |
| Single tool format | Model-specific tool parsers | 2024-2025 | Each model family needs its parser |
| Post-hoc JSON validation | Constrained decoding | 2024 with outlines/guidance | Output is valid by construction |

**Deprecated/outdated:**
- **JSON mode without schema:** Still works but Structured Outputs is preferred
- **ReAct-style tool prompting:** Reasoning models may behave unexpectedly with stopword-based templates

## Open Questions

1. **Constrained Decoding Priority**
   - What we know: outlines/lm-format-enforcer can guarantee valid JSON during generation
   - What's unclear: Performance impact on MLX, compatibility with streaming
   - Recommendation: Start with post-hoc validation; add constrained decoding in future phase

2. **Multi-Turn Tool Calling**
   - What we know: Tool results need special message role (ipython for Llama, tool for OpenAI)
   - What's unclear: Exact format for each model family's tool result handling
   - Recommendation: Research further when implementing; may need additional adapter methods

3. **Adapter Compatibility Validation**
   - What we know: LoRA adapters are model-specific
   - What's unclear: How to validate adapter is compatible with base model before loading
   - Recommendation: Parse adapter_config.json and compare base_model field

## Sources

### Primary (HIGH confidence)
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm) - Model loading, LoRA support, stream_generate API
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) - Tool calling and structured output schemas
- [Meta Llama Models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md) - Llama tool calling format

### Secondary (MEDIUM confidence)
- [mlx-omni-server GitHub](https://github.com/madroidmaq/mlx-omni-server) - Architecture patterns for tool calling and reasoning mode
- [mlx-openai-server GitHub](https://github.com/cubist38/mlx-openai-server) - Tool parser configuration patterns
- [Qwen Function Calling docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html) - Qwen tool call format
- [vLLM Tool Calling docs](https://docs.vllm.ai/en/latest/features/tool_calling/) - Parser configurations for various models

### Tertiary (LOW confidence)
- [DeepWiki mlx-omni-server](https://deepwiki.com/madroidmaq/mlx-omni-server/1-overview) - Detailed architecture analysis
- [DeepWiki mlx-lm Python API](https://deepwiki.com/ml-explore/mlx-lm/3.2-python-api) - Additional load() parameters

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - mlx-lm load() with adapter_path verified, jsonschema is standard
- Architecture: MEDIUM - Patterns derived from multiple sources, not directly tested
- Pitfalls: MEDIUM - Gathered from GitHub issues and documentation, some from community reports

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (30 days - MLX ecosystem is rapidly evolving)

"""Base probe classes providing shared hierarchy and generative probing.

BaseProbe is the minimal ABC matching the ProbeStrategy protocol.
GenerativeProbe extends it with shared thinking/tool verification
logic that both TextGenProbe and VisionProbe can reuse.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
    from mlx_manager.mlx_server.models.pool import LoadedModel


class BaseProbe(ABC):
    """Abstract base for all probe strategies.

    Provides the minimal interface matching the ProbeStrategy protocol.
    """

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """The model type this strategy handles."""
        ...

    @abstractmethod
    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        """Run type-specific probe steps."""
        ...  # pragma: no cover
        if False:  # pragma: no cover
            yield ProbeStep(step="", status="")


# ---------------------------------------------------------------------------
# Constants shared by generative probes (text-gen + vision)
# ---------------------------------------------------------------------------

_TOOL_PROBE_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Tokyo",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

_TOOL_PROBE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

_KNOWN_XML_TAGS: frozenset[str] = frozenset(
    {
        "think",
        "thinking",
        "reasoning",
        "reflection",
        "tool_call",
        "tool_response",
        "function_call",
        "output",
        "result",
        "code",
        "step",
        "answer",
        "solution",
    }
)


class GenerativeProbe(BaseProbe):
    """ABC for probes that can generate text and test thinking/tool capabilities.

    Subclasses implement ``_generate()`` using their backend-specific
    generation path (mlx-lm for text, mlx-vlm for vision).
    """

    @abstractmethod
    async def _generate(
        self,
        loaded: LoadedModel,
        messages: list[dict],
        tools: list[dict] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Generate a response using the model.

        Subclasses provide backend-specific generation logic.
        """
        ...

    # ------------------------------------------------------------------
    # Shared thinking/tool verification
    # ------------------------------------------------------------------

    async def _verify_thinking_support(
        self, loaded: LoadedModel, adapter: ModelAdapter
    ) -> tuple[bool, str]:
        """Verify thinking support via generation and parser validation.

        Returns (supports_thinking, thinking_parser_id).
        """
        from mlx_manager.mlx_server.parsers import THINKING_PARSERS
        from mlx_manager.mlx_server.utils.template_tools import has_thinking_support

        tokenizer = loaded.tokenizer
        template_supports = has_thinking_support(tokenizer)

        if not template_supports:
            return (False, "null")

        try:
            messages = [{"role": "user", "content": "What is 2 + 2?"}]
            output = await self._generate(loaded, messages, enable_thinking=True)

            # Try adapter's thinking_parser first
            thinking_parser = adapter.thinking_parser
            if thinking_parser.parser_id != "null":
                thinking_content = thinking_parser.extract(output)
                if thinking_content is not None:
                    logger.info(
                        "Thinking probe: verified via adapter parser (parser={})",
                        thinking_parser.parser_id,
                    )
                    return (True, thinking_parser.parser_id)

            # Sweep all registered thinking parsers
            for parser_id, parser_cls in THINKING_PARSERS.items():
                if parser_id == "null":
                    continue
                if parser_id == thinking_parser.parser_id:
                    continue
                parser = parser_cls()
                thinking_content = parser.extract(output)
                if thinking_content is not None:
                    logger.info(
                        "Thinking probe: verified via sweep parser (parser={})",
                        parser_id,
                    )
                    return (True, parser_id)

            # Template supports it but no tags found -- still report supported
            logger.info(
                "Thinking probe: template supports thinking but no tags in output; "
                "reporting supported (template authoritative)"
            )
            fallback_id = (
                thinking_parser.parser_id if thinking_parser.parser_id != "null" else "think_tag"
            )
            return (True, fallback_id)

        except Exception as e:
            logger.debug("Thinking generation failed, falling back to template check: {}", e)
            return (True, "think_tag")

    async def _verify_tool_support(
        self, loaded: LoadedModel, adapter: ModelAdapter
    ) -> tuple[str | None, str | None]:
        """2-attempt adapter-driven tool verification.

        Returns (tool_format, tool_parser_id) or (None, None).
        """
        from mlx_manager.mlx_server.utils.template_tools import has_native_tool_support

        tokenizer = loaded.tokenizer
        last_output: str | None = None

        # Attempt 1: Template delivery
        if adapter.supports_native_tools() or has_native_tool_support(tokenizer):
            try:
                last_output = await self._generate(
                    loaded, _TOOL_PROBE_MESSAGES, tools=_TOOL_PROBE_TOOL
                )
                parser_id = _validate_tool_output(last_output, "get_weather", adapter)
                if parser_id:
                    logger.info(
                        "Tool probe: template delivery verified (parser={})",
                        parser_id,
                    )
                    return ("template", parser_id)
                logger.debug(
                    "Template delivery produced output but no parser matched: {}",
                    last_output[:200],
                )
            except Exception as e:
                logger.debug("Template delivery generation failed: {}", e)

        # Attempt 2: Adapter delivery
        tool_prompt = adapter.format_tools_for_prompt(_TOOL_PROBE_TOOL)
        if tool_prompt:
            try:
                messages_with_tools = [
                    {"role": "system", "content": tool_prompt},
                    *_TOOL_PROBE_MESSAGES,
                ]
                last_output = await self._generate(loaded, messages_with_tools)
                parser_id = _validate_tool_output(last_output, "get_weather", adapter)
                if parser_id:
                    logger.info(
                        "Tool probe: adapter delivery verified (parser={})",
                        parser_id,
                    )
                    return ("adapter", parser_id)
                logger.debug(
                    "Adapter delivery produced output but no parser matched: {}",
                    last_output[:200],
                )
            except Exception as e:
                logger.debug("Adapter delivery generation failed: {}", e)

        # No match -- scan for partial tool markers as diagnostic
        try:
            if last_output is None:
                last_output = await self._generate(loaded, _TOOL_PROBE_MESSAGES)

            # Check for partial/corrupted tool call markers
            tool_markers = [
                "<tool_call>",
                "</tool_call>",
                "<function=",
                "</function>",
                "get_weather",
                '"name"',
            ]
            found_markers = [m for m in tool_markers if m in last_output]
            if found_markers:
                logger.warning(
                    "Tool probe: model attempted tool use (found markers: {}) "
                    "but output could not be parsed. Raw output: {}",
                    found_markers,
                    last_output[:300],
                )
            else:
                unknown_tags = _detect_unknown_xml_tags(last_output)
                if unknown_tags:
                    logger.warning(
                        "Tool probe: no parser matched but found unknown XML tags: {}",
                        unknown_tags,
                    )
                else:
                    logger.info("Tool probe: no tool support detected")
        except Exception:
            logger.info("Tool probe: no tool support detected")

        return (None, None)

    async def _probe_generative_capabilities(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        """Shared orchestrator for family detection + thinking + tools.

        Yields ProbeSteps and populates result with discovered capabilities.
        """
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        # Detect model family (if not already set)
        if result.model_family is None:
            result.model_family = detect_model_family(model_id)

        tokenizer = loaded.tokenizer
        adapter = loaded.adapter

        # Guard: adapter must exist for generation-based probing
        if adapter is None:
            logger.warning(
                "No adapter available for {}; skipping thinking and tool probes",
                model_id,
            )
            yield ProbeStep(step="test_thinking", status="skipped")
            yield ProbeStep(step="test_tools", status="skipped")
            return

        # Test thinking support
        if tokenizer is not None:
            yield ProbeStep(step="test_thinking", status="running")
            try:
                supports_thinking, thinking_parser_id = await self._verify_thinking_support(
                    loaded, adapter
                )
                result.supports_thinking = supports_thinking
                result.thinking_parser_id = thinking_parser_id
                yield ProbeStep(
                    step="test_thinking",
                    status="completed",
                    capability="supports_thinking",
                    value=supports_thinking,
                )
            except Exception as e:
                logger.warning("Thinking test failed for {}: {}", model_id, e)
                yield ProbeStep(step="test_thinking", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_thinking", status="skipped")

        # Generation-based tool verification (2-attempt)
        if tokenizer is not None:
            yield ProbeStep(step="test_tools", status="running")
            try:
                tool_format, tool_parser_id = await self._verify_tool_support(loaded, adapter)
                result.supports_native_tools = tool_format is not None
                result.tool_format = tool_format
                result.tool_parser_id = tool_parser_id
                yield ProbeStep(
                    step="test_tools",
                    status="completed",
                    capability="tool_format",
                    value=tool_format,
                )
            except Exception as e:
                logger.warning("Tool verification failed for {}: {}", model_id, e)
                yield ProbeStep(step="test_tools", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_tools", status="skipped")


# ---------------------------------------------------------------------------
# Module-level helpers (shared by generative probes)
# ---------------------------------------------------------------------------


def _validate_tool_output(output: str, expected_fn: str, adapter: ModelAdapter) -> str | None:
    """Validate tool output using adapter's parser first, then sweep all parsers."""
    adapter_parser = adapter.tool_parser
    adapter_parser_id: str | None = None
    if adapter_parser.parser_id != "null":
        adapter_parser_id = adapter_parser.parser_id
        if adapter_parser.validates(output, expected_fn):
            return adapter_parser_id

    sweep_result = _find_matching_parser(output, expected_fn, exclude_parser_id=adapter_parser_id)
    return sweep_result


def _find_matching_parser(
    output: str,
    expected_function: str,
    exclude_parser_id: str | None = None,
) -> str | None:
    """Try ALL registered parsers to find one that validates the output."""
    from mlx_manager.mlx_server.parsers import TOOL_PARSERS

    for parser_id, parser_cls in TOOL_PARSERS.items():
        if parser_id == "null":
            continue
        if parser_id == exclude_parser_id:
            continue
        parser = parser_cls()
        if parser.validates(output, expected_function):
            return parser_id
    return None


def _detect_unknown_xml_tags(output: str) -> set[str]:
    """Scan output for XML-style tags not in the known set."""
    tag_pattern = re.compile(r"<([a-zA-Z_][\w-]*)(?:\s[^>]*)?>")
    found_tags: set[str] = set()
    for match in tag_pattern.finditer(output):
        tag_name = match.group(1).lower()
        close_pattern = re.compile(rf"</{re.escape(match.group(1))}>", re.IGNORECASE)
        if close_pattern.search(output):
            found_tags.add(tag_name)

    unknown = found_tags - _KNOWN_XML_TAGS
    return unknown

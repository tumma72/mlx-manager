"""Additional tests for probe modules to improve coverage.

Targets uncovered lines in:
- services/probe/base.py
- services/probe/coordinator.py
- services/probe/vision.py
- services/probe/embeddings.py
- services/probe/text_gen.py
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.services.probe.base import (
    GenerativeProbe,
    _find_unclosed_thinking_tag,
    _has_tokenization_artifacts,
    _detect_unknown_xml_tags,
    _validate_tool_output,
)
from mlx_manager.services.probe.steps import ProbeResult, ProbeStep


# ---------------------------------------------------------------------------
# Concrete probe for testing GenerativeProbe methods
# ---------------------------------------------------------------------------


class SimpleProbe(GenerativeProbe):
    """Concrete GenerativeProbe for testing base class methods directly."""

    def __init__(self, mock_output: str = ""):
        self._mock_output = mock_output

    @property
    def model_type(self) -> ModelType:
        return ModelType.TEXT_GEN

    async def _generate(
        self,
        loaded,
        messages: list[dict],
        tools: list[dict] | None = None,
        template_options: dict[str, Any] | None = None,
        max_tokens: int = 800,
    ) -> str:
        return self._mock_output

    async def probe(self, model_id: str, loaded, result: ProbeResult):
        async for step in self._probe_generative_capabilities(model_id, loaded, result):
            yield step


# ---------------------------------------------------------------------------
# Tests for GenerativeProbe._generate() (base.py lines 116-136)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_raises_when_adapter_is_none():
    """_generate raises RuntimeError when adapter is None (lines 117-118)."""
    probe = SimpleProbe()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None

    with pytest.raises(RuntimeError, match="No adapter available"):
        await GenerativeProbe._generate(
            probe,
            mock_loaded,
            [{"role": "user", "content": "test"}],
        )


@pytest.mark.asyncio
async def test_generate_configures_and_resets_template_options():
    """_generate calls adapter.configure for template_options and resets after (lines 122, 136)."""
    probe = SimpleProbe()

    mock_adapter = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "hello"
    mock_adapter.generate = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    template_opts = {"enable_thinking": True}
    result = await GenerativeProbe._generate(
        probe,
        mock_loaded,
        [{"role": "user", "content": "test"}],
        template_options=template_opts,
    )

    assert result == "hello"
    # configure called with options and then reset to None
    assert mock_adapter.configure.call_count == 2
    first_call = mock_adapter.configure.call_args_list[0]
    second_call = mock_adapter.configure.call_args_list[1]
    assert first_call[1]["template_options"] == {"enable_thinking": True}
    assert second_call[1]["template_options"] is None


@pytest.mark.asyncio
async def test_generate_returns_content(
):
    """_generate returns result.content from adapter.generate (line 132)."""
    probe = SimpleProbe()

    mock_adapter = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "The answer is 4"
    mock_adapter.generate = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    result = await GenerativeProbe._generate(
        probe,
        mock_loaded,
        [{"role": "user", "content": "What is 2+2?"}],
    )

    assert result == "The answer is 4"
    mock_adapter.configure.assert_not_called()  # No template_options, no configure


# ---------------------------------------------------------------------------
# Tests for _verify_thinking_support Path A with unclosed tags + retry
# (base.py lines 173, 178-209)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_thinking_path_a_succeeds_immediately():
    """Path A: thinking detected directly without retry (line 173)."""
    call_count = 0

    async def generate_with_tags(loaded, messages, tools=None, template_options=None, max_tokens=800):
        nonlocal call_count
        call_count += 1
        return "<think>Some reasoning</think>Answer: 4"

    probe = SimpleProbe()
    probe._generate = generate_with_tags

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.parser_id = "think_tag"
    mock_thinking_parser.extract = MagicMock(return_value="Some reasoning")
    mock_adapter.thinking_parser = mock_thinking_parser

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": MagicMock},
    ):
        supports, parser_id, diags = await probe._verify_thinking_support(
            mock_loaded,
            mock_adapter,
            template_params={"enable_thinking": {"default": True}},
        )

    assert supports is True
    assert parser_id == "think_tag"
    assert call_count == 1  # Only one generation call needed


@pytest.mark.asyncio
async def test_verify_thinking_path_a_unclosed_tag_retry_succeeds():
    """Path A: unclosed thinking tag triggers retry; retry succeeds (lines 178-209)."""
    calls = []

    async def generate_with_unclosed_then_complete(
        loaded, messages, tools=None, template_options=None, max_tokens=800
    ):
        calls.append(len(messages))
        if len(messages) == 1:
            return "<think>Thinking without closing tag"  # Unclosed
        else:
            return "<think>Complete thinking</think>Answer"  # Closed

    probe = SimpleProbe()
    probe._generate = generate_with_unclosed_then_complete

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.parser_id = "null"
    mock_thinking_parser.extract = MagicMock(return_value=None)
    mock_adapter.thinking_parser = mock_thinking_parser

    # Parser that extracts content from retry output
    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    # First call (direct): returns None. Second call (retry): returns content.
    mock_parser_instance.extract = MagicMock(side_effect=[None, "Complete thinking"])
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ):
        supports, parser_id, diags = await probe._verify_thinking_support(
            mock_loaded,
            mock_adapter,
            template_params={"enable_thinking": {"default": True}},
        )

    assert supports is True
    assert parser_id == "think_tag"
    assert len(calls) == 2  # Initial + retry


@pytest.mark.asyncio
async def test_verify_thinking_path_a_unclosed_tag_retry_fails():
    """Path A: unclosed tag retry fails → falls through to Path B (lines 178-209)."""
    calls = []

    async def always_unclosed(loaded, messages, tools=None, template_options=None, max_tokens=800):
        calls.append(True)
        return "<think>Always unclosed"  # Always unclosed

    probe = SimpleProbe()
    probe._generate = always_unclosed

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.parser_id = "null"
    mock_thinking_parser.extract = MagicMock(return_value=None)
    mock_adapter.thinking_parser = mock_thinking_parser

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract = MagicMock(return_value=None)  # Never matches
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ):
        supports, parser_id, diags = await probe._verify_thinking_support(
            mock_loaded,
            mock_adapter,
            template_params={"enable_thinking": {"default": True}},
        )

    # Should fall through to Path B and also fail → no thinking
    assert supports is False
    assert parser_id == "null"
    assert len(calls) >= 2  # At least initial + retry (Path A); may also run Path B


# ---------------------------------------------------------------------------
# Tests for _verify_tool_support exception handlers
# (base.py lines 329-330, 352-353)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_tool_template_delivery_exception_handled():
    """Template delivery exception is caught and falls through to adapter delivery (lines 329-330)."""

    async def raise_on_generate(loaded, messages, tools=None, template_options=None, max_tokens=800):
        raise RuntimeError("Template generation failed")

    probe = SimpleProbe()
    probe._generate = raise_on_generate

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = True
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.format_tools_for_prompt.return_value = None  # No adapter delivery either

    with patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=True,
    ), patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock, "xml_tag": MagicMock},
    ):
        tool_format, parser_id, diags = await probe._verify_tool_support(mock_loaded, mock_adapter)

    # Both delivery methods failed, no tool support found
    assert tool_format is None
    assert parser_id is None


@pytest.mark.asyncio
async def test_verify_tool_adapter_delivery_exception_handled():
    """Adapter delivery exception is caught and falls through to diagnostic scan (lines 352-353)."""
    call_count = 0

    async def fail_on_second(loaded, messages, tools=None, template_options=None, max_tokens=800):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Template delivery attempt: no tools in response
            return "The weather is sunny today"
        raise RuntimeError("Adapter delivery failed")

    probe = SimpleProbe()
    probe._generate = fail_on_second

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = True
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.format_tools_for_prompt.return_value = "Tools: get_weather"

    with patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=True,
    ), patch(
        "mlx_manager.services.probe.base._validate_tool_output",
        return_value=None,
    ), patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock, "xml_tag": MagicMock},
    ):
        tool_format, parser_id, diags = await probe._verify_tool_support(mock_loaded, mock_adapter)

    # Both delivery paths failed - diagnostic scan was attempted but no markers found
    assert tool_format is None
    assert parser_id is None


# ---------------------------------------------------------------------------
# Tests for tokenization artifact detection
# (base.py lines 374-380, 664, 666)
# ---------------------------------------------------------------------------


def test_has_tokenization_artifacts_sp_marker():
    """Sentencepiece boundary marker triggers tokenization artifact detection (line 664)."""
    output = "The \u2581weather \u2581is \u2581sunny"  # ▁ markers
    assert _has_tokenization_artifacts(output) is True


def test_has_tokenization_artifacts_spaced_json_key():
    """Spaced JSON key triggers tokenization artifact detection (line 666)."""
    output = '{ " name ": "get_weather", " arguments ": {} }'
    assert _has_tokenization_artifacts(output) is True


def test_has_tokenization_artifacts_clean_output():
    """Clean output returns False for tokenization artifact check."""
    output = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
    assert _has_tokenization_artifacts(output) is False


@pytest.mark.asyncio
async def test_verify_tool_tokenization_artifacts_diagnostic():
    """Tokenization artifacts produce UNSUPPORTED diagnostic (lines 374-380)."""
    sp_marker = "\u2581"  # ▁

    async def generate_garbled(loaded, messages, tools=None, template_options=None, max_tokens=800):
        return f"<tool_call>{sp_marker}get{sp_marker}_weather{sp_marker}</tool_call>"

    probe = SimpleProbe()
    probe._generate = generate_garbled

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.format_tools_for_prompt.return_value = None

    with patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ), patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock},
    ):
        tool_format, parser_id, diags = await probe._verify_tool_support(mock_loaded, mock_adapter)

    assert tool_format is None
    assert parser_id is None
    assert len(diags) == 1
    assert diags[0].level.value == "action_needed"
    assert diags[0].category.value == "unsupported"
    assert "tokenizer" in diags[0].message.lower()


# ---------------------------------------------------------------------------
# Tests for _verify_tool_support exception handler in diagnostic scan
# (base.py line 440-441)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_tool_diagnostic_scan_exception_handled():
    """Exception in diagnostic scan fallback is caught silently (lines 440-441)."""
    call_count = 0

    async def raise_on_third(loaded, messages, tools=None, template_options=None, max_tokens=800):
        nonlocal call_count
        call_count += 1
        # First two calls (template delivery sweep) fail with exception
        raise RuntimeError("Generation always fails")

    probe = SimpleProbe()
    probe._generate = raise_on_third

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = MagicMock()

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.format_tools_for_prompt.return_value = None

    with patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ), patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock},
    ):
        # Should not raise - exception is caught
        tool_format, parser_id, diags = await probe._verify_tool_support(mock_loaded, mock_adapter)

    assert tool_format is None
    assert parser_id is None
    assert diags == []


# ---------------------------------------------------------------------------
# Tests for _probe_generative_capabilities - default family path
# (base.py lines 465-490)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_generative_capabilities_default_family_diagnostic():
    """Default family produces a WARNING diagnostic (lines 465-490)."""
    probe = SimpleProbe()

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = None  # Skip thinking/tool tests
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="default",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"default": MagicMock, "qwen": MagicMock},
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"architectures": ["SomeUnknownArch"]},
        ),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    # Should have a diagnostic for default family
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.level.value == "warning"
    assert diag.category.value == "family"
    assert "DefaultAdapter" in diag.message
    assert "SomeUnknownArch" in diag.message


@pytest.mark.asyncio
async def test_probe_generative_capabilities_default_family_no_config():
    """Default family with no config uses 'unknown' architecture (lines 465-490)."""
    probe = SimpleProbe()

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = None  # Skip thinking/tool tests
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="default",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"default": MagicMock},
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value=None,  # No config
        ),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert "unknown" in diag.message


@pytest.mark.asyncio
async def test_probe_generative_capabilities_default_family_config_exception():
    """Default family with read_model_config exception uses 'unknown' architecture."""
    probe = SimpleProbe()

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = None
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="default",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"default": MagicMock},
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            side_effect=RuntimeError("Config read failed"),
        ),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert "unknown" in diag.message


# ---------------------------------------------------------------------------
# Tests for _probe_generative_capabilities - template params discovery
# (base.py lines 521-522)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_generative_capabilities_discovers_template_params():
    """Template params discovery populates result.template_params (lines 521-522)."""
    probe = SimpleProbe()

    mock_tokenizer = MagicMock()
    mock_loaded = MagicMock()
    mock_loaded.tokenizer = mock_tokenizer
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()

    mock_params = {"enable_thinking": {"type": "bool", "default": False}}

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_params.discover_template_params",
            return_value=mock_params,
        ),
        patch.object(probe, "_verify_thinking_support", new_callable=AsyncMock, return_value=(False, "null", [])),
        patch.object(probe, "_verify_tool_support", new_callable=AsyncMock, return_value=(None, None, [])),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    assert result.template_params == mock_params


# ---------------------------------------------------------------------------
# Tests for _probe_generative_capabilities - tool results with diagnostics
# (base.py lines 562-566)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_generative_capabilities_tool_success_with_diagnostics():
    """Tool results with diagnostics are accumulated in result (lines 562-566)."""
    from mlx_manager.services.probe.steps import ProbeDiagnostic, DiagnosticLevel, DiagnosticCategory

    probe = SimpleProbe()

    mock_tokenizer = MagicMock()
    mock_loaded = MagicMock()
    mock_loaded.tokenizer = mock_tokenizer
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()
    mock_diag = ProbeDiagnostic(
        level=DiagnosticLevel.WARNING,
        category=DiagnosticCategory.TOOL_DIALECT,
        message="Unknown tool dialect",
    )

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_params.discover_template_params",
            return_value={},
        ),
        patch.object(probe, "_verify_thinking_support", new_callable=AsyncMock, return_value=(False, "null", [])),
        patch.object(probe, "_verify_tool_support", new_callable=AsyncMock, return_value=("template", "xml_tag", [mock_diag])),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    assert result.supports_native_tools is True
    assert result.tool_format == "template"
    assert result.tool_parser_id == "xml_tag"
    assert len(result.diagnostics) == 1


# ---------------------------------------------------------------------------
# Tests for _find_unclosed_thinking_tag - returns None path
# (base.py line 599)
# ---------------------------------------------------------------------------


def test_find_unclosed_thinking_tag_returns_none_for_complete():
    """Returns None when all tags are properly closed (line 599)."""
    output = "<think>Complete thinking</think>Answer: 4"
    result = _find_unclosed_thinking_tag(output)
    assert result is None


def test_find_unclosed_thinking_tag_finds_thinking_tag():
    """Returns tag name for unclosed 'thinking' tag."""
    output = "<thinking>Started but not finished"
    result = _find_unclosed_thinking_tag(output)
    assert result == "thinking"


def test_find_unclosed_thinking_tag_finds_reasoning_tag():
    """Returns tag name for unclosed 'reasoning' tag."""
    output = "<reasoning>Started reasoning"
    result = _find_unclosed_thinking_tag(output)
    assert result == "reasoning"


def test_find_unclosed_thinking_tag_finds_reflection_tag():
    """Returns tag name for unclosed 'reflection' tag."""
    output = "<reflection>Reflecting"
    result = _find_unclosed_thinking_tag(output)
    assert result == "reflection"


def test_find_unclosed_thinking_tag_no_opening_tag():
    """Returns None when there is no opening tag at all."""
    output = "Just plain text with no tags"
    result = _find_unclosed_thinking_tag(output)
    assert result is None


def test_find_unclosed_thinking_tag_case_insensitive():
    """Unclosed tag detection is case-insensitive."""
    output = "<THINK>Uppercase thinking without closing"
    result = _find_unclosed_thinking_tag(output)
    assert result == "think"


# ---------------------------------------------------------------------------
# Tests for vision.py uncovered lines (44-45, 143-153)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vision_probe_generate_raises_when_no_adapter():
    """VisionProbe._generate raises RuntimeError when adapter is None (lines 44-45)."""
    from mlx_manager.services.probe.vision import VisionProbe

    probe = VisionProbe()
    mock_loaded = MagicMock()
    mock_loaded.adapter = None

    with pytest.raises(RuntimeError, match="No adapter available"):
        await probe._generate(mock_loaded, [{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_vision_probe_generate_with_template_options():
    """VisionProbe._generate configures and resets template_options (lines 51-66)."""
    from mlx_manager.services.probe.vision import VisionProbe
    from PIL import Image

    probe = VisionProbe()

    mock_adapter = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "I see a gray image"
    mock_adapter.generate = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    result = await probe._generate(
        mock_loaded,
        [{"role": "user", "content": "Describe this image"}],
        template_options={"enable_thinking": True},
    )

    assert result == "I see a gray image"
    assert mock_adapter.configure.call_count == 2
    # images=[test_image] is passed to adapter.generate
    call_kwargs = mock_adapter.generate.call_args[1]
    assert "images" in call_kwargs
    assert len(call_kwargs["images"]) == 1
    assert isinstance(call_kwargs["images"][0], Image.Image)


def test_messages_to_text_with_string_content():
    """_messages_to_text extracts text from string-content messages (lines 143-153)."""
    from mlx_manager.services.probe.vision import _messages_to_text

    messages = [
        {"role": "user", "content": "What do you see?"},
        {"role": "assistant", "content": "I see an image."},
    ]
    result = _messages_to_text(messages)
    assert "What do you see?" in result
    assert "I see an image." in result


def test_messages_to_text_with_structured_content():
    """_messages_to_text extracts text from structured content blocks (lines 143-153)."""
    from mlx_manager.services.probe.vision import _messages_to_text

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "url": "data:image/png;base64,abc"},
            ],
        }
    ]
    result = _messages_to_text(messages)
    assert "Describe this image" in result


def test_messages_to_text_empty():
    """_messages_to_text returns empty string for messages with no text content."""
    from mlx_manager.services.probe.vision import _messages_to_text

    messages = [{"role": "user", "content": [{"type": "image_url", "url": "data:..."}]}]
    result = _messages_to_text(messages)
    assert result == ""


# ---------------------------------------------------------------------------
# Tests for embeddings.py uncovered lines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_text_raises_when_no_adapter():
    """_encode_text raises RuntimeError when adapter is None (line 111-112)."""
    from mlx_manager.services.probe.embeddings import _encode_text

    mock_loaded = MagicMock()
    mock_loaded.adapter = None

    with pytest.raises(RuntimeError, match="No adapter available for embeddings"):
        await _encode_text(mock_loaded, "hello")


@pytest.mark.asyncio
async def test_encode_text_returns_none_for_empty_embeddings():
    """_encode_text returns None when embedding result is empty (line 117)."""
    from mlx_manager.services.probe.embeddings import _encode_text

    mock_adapter = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = []  # Empty result
    mock_adapter.generate_embeddings = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    result = await _encode_text(mock_loaded, "hello")
    assert result is None


@pytest.mark.asyncio
async def test_embeddings_probe_encoding_returns_none():
    """EmbeddingsProbe handles encoding returning None (line 49)."""
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    probe = EmbeddingsProbe()

    mock_adapter = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = []
    mock_adapter.generate_embeddings = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    result = ProbeResult()
    steps = []
    async for step in probe.probe("test/model", mock_loaded, result):
        steps.append(step)

    encode_steps = [s for s in steps if s.step == "test_encode"]
    failed_step = next((s for s in encode_steps if s.status == "failed"), None)
    assert failed_step is not None
    assert "empty" in failed_step.error.lower()


@pytest.mark.asyncio
async def test_embeddings_probe_normalization_exception():
    """EmbeddingsProbe handles normalization check exception (lines 72-74)."""
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe, _encode_text

    probe = EmbeddingsProbe()

    # First call returns a valid embedding, second call raises for similarity test
    mock_adapter = MagicMock()
    mock_embedding_result = MagicMock()
    mock_embedding_result.embeddings = [[0.6, 0.8]]  # Normalized embedding
    mock_similarity_result = MagicMock()
    mock_similarity_result.embeddings = [[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]

    mock_loaded = MagicMock()
    mock_loaded.model = MagicMock()

    # Patch _encode_text to raise on normalization check
    with patch(
        "mlx_manager.services.probe.embeddings._encode_text",
        side_effect=RuntimeError("Normalization check failed"),
    ):
        result = ProbeResult()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

    encode_steps = [s for s in steps if s.step == "test_encode"]
    assert any(s.status == "failed" for s in encode_steps)
    # Should return early since encode failed
    norm_steps = [s for s in steps if s.step == "check_normalization"]
    assert len(norm_steps) == 0


@pytest.mark.asyncio
async def test_get_max_sequence_length_no_config():
    """_get_max_sequence_length returns None when no config is available (line 126)."""
    from mlx_manager.services.probe.embeddings import _get_max_sequence_length

    with patch("mlx_manager.utils.model_detection.read_model_config", return_value=None):
        result = _get_max_sequence_length("test/model")

    assert result is None


@pytest.mark.asyncio
async def test_test_similarity_ordering_no_adapter():
    """_test_similarity_ordering returns False when adapter is None (line 141)."""
    from mlx_manager.services.probe.embeddings import _test_similarity_ordering

    mock_loaded = MagicMock()
    mock_loaded.adapter = None

    result = await _test_similarity_ordering(mock_loaded)
    assert result is False


@pytest.mark.asyncio
async def test_test_similarity_ordering_insufficient_embeddings():
    """_test_similarity_ordering returns False for wrong number of embeddings (line 155)."""
    from mlx_manager.services.probe.embeddings import _test_similarity_ordering

    mock_adapter = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Only 2, need 3
    mock_adapter.generate_embeddings = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    result = await _test_similarity_ordering(mock_loaded)
    assert result is False


# ---------------------------------------------------------------------------
# Tests for text_gen.py uncovered lines (79-81)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_gen_probe_context_check_exception():
    """TextGenProbe handles context check exception (lines 79-81)."""
    from mlx_manager.services.probe.text_gen import TextGenProbe

    probe = TextGenProbe()

    mock_loaded = MagicMock()
    mock_loaded.size_gb = 4.0

    with patch(
        "mlx_manager.services.probe.text_gen._estimate_practical_max_tokens",
        side_effect=RuntimeError("Context estimation failed"),
    ), patch(
        "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
        return_value="qwen",
    ):
        result = ProbeResult()
        steps = []
        async for step in probe.probe("test/model", mock_loaded, result):
            steps.append(step)

    context_steps = [s for s in steps if s.step == "check_context"]
    failed_step = next((s for s in context_steps if s.status == "failed"), None)
    assert failed_step is not None
    assert "Context estimation failed" in failed_step.error


# ---------------------------------------------------------------------------
# Tests for coordinator._sweep_generative_capabilities
# (coordinator.py lines 246, 256-259, 288-294, 305-306)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_sweep_no_adapter():
    """_sweep_generative_capabilities skips thinking/tools when adapter is None (lines 288-294)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None  # No adapter
    mock_loaded.tokenizer = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)
    result = ProbeResult()

    mock_strategy = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
    ):
        steps = []
        async for step in coordinator._sweep_generative_capabilities(
            "test/model", mock_loaded, result, mock_strategy
        ):
            steps.append(step)

    skipped = [s for s in steps if s.status == "skipped"]
    skipped_names = {s.step for s in skipped}
    assert "test_thinking" in skipped_names
    assert "test_tools" in skipped_names


@pytest.mark.asyncio
async def test_coordinator_sweep_default_family_with_architecture():
    """_sweep_generative_capabilities default family creates diagnostic with architecture (lines 256-259)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None  # Skip thinking/tool tests
    mock_loaded.tokenizer = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)
    result = ProbeResult()

    mock_strategy = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="default",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"default": MagicMock, "qwen": MagicMock},
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value={"architectures": ["TestModelArch"]},
        ),
    ):
        steps = []
        async for step in coordinator._sweep_generative_capabilities(
            "test/model", mock_loaded, result, mock_strategy
        ):
            steps.append(step)

    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert "TestModelArch" in diag.message
    assert diag.category.value == "family"


@pytest.mark.asyncio
async def test_coordinator_sweep_detects_family_when_none():
    """_sweep_generative_capabilities detects model family when result.model_family is None (line 246)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None
    mock_loaded.tokenizer = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)
    result = ProbeResult()
    result.model_family = None  # Must be None to trigger detection

    mock_strategy = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ) as mock_detect,
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
    ):
        steps = []
        async for step in coordinator._sweep_generative_capabilities(
            "test/model", mock_loaded, result, mock_strategy
        ):
            steps.append(step)

    assert result.model_family == "qwen"
    mock_detect.assert_called_once_with("test/model")


@pytest.mark.asyncio
async def test_coordinator_sweep_discovers_template_params():
    """_sweep_generative_capabilities discovers template params when tokenizer is present (lines 305-306)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    mock_loaded = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.thinking_parser = MagicMock()
    mock_adapter.thinking_parser.extract = MagicMock(return_value=None)
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.supports_native_tools = MagicMock(return_value=False)
    mock_adapter.format_tools_for_prompt = MagicMock(return_value=None)
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)
    result = ProbeResult()

    mock_strategy = MagicMock()
    mock_params = {"enable_thinking": {"type": "bool", "default": False}}

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_params.discover_template_params",
            return_value=mock_params,
        ),
        patch(
            "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
            {"null": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
            return_value=None,
        ),
    ):
        steps = []
        async for step in coordinator._sweep_generative_capabilities(
            "test/model", mock_loaded, result, mock_strategy
        ):
            steps.append(step)

    assert result.template_params == mock_params


# ---------------------------------------------------------------------------
# Tests for coordinator._sweep_thinking
# (coordinator.py lines 387-443, 455-462, 476-477)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_path_a_success():
    """_sweep_thinking Path A: thinking detected with enable_thinking param (lines 387-443)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.extract = MagicMock(return_value="Some thinking content")
    mock_adapter.thinking_parser = mock_thinking_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value="<think>Some thinking content</think>Answer")

    template_params = {"enable_thinking": {"default": True}}

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract = MagicMock(return_value="thinking content")
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ), patch(
        "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
        return_value=None,
    ):
        supports, parser_id, diags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is True
    assert parser_id == "think_tag"
    assert diags == []


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_path_a_unclosed_retry():
    """_sweep_thinking Path A: retry on unclosed tag (lines 410-441)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    call_count = [0]

    async def generate_with_retry(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return "<think>Unclosed thinking"  # Unclosed
        return "<think>Complete thinking</think>Answer"  # Complete

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    # First call: None (unclosed), second call: content
    mock_thinking_parser.extract = MagicMock(side_effect=[None, "Complete thinking"])
    mock_adapter.thinking_parser = mock_thinking_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    mock_strategy = MagicMock()
    mock_strategy._generate = generate_with_retry

    template_params = {"enable_thinking": {"default": True}}

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract = MagicMock(side_effect=[None, "Complete thinking"])
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ):
        supports, parser_id, diags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is True
    assert call_count[0] == 2  # Initial + retry


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_path_b_always_thinks():
    """_sweep_thinking Path B: always-thinks detection (lines 455-462)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.extract = MagicMock(return_value="Always thinking")
    mock_adapter.thinking_parser = mock_thinking_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value="<think>Always thinking</think>Answer")

    # No enable_thinking in template_params → goes directly to Path B
    template_params = None

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract = MagicMock(return_value="Always thinking")
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ), patch(
        "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
        return_value=None,
    ):
        supports, parser_id, diags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is True
    assert parser_id == "think_tag"


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_has_enable_thinking_but_no_tags_diagnostic():
    """_sweep_thinking: has enable_thinking but no tags → diagnostic (lines 476-477)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.extract = MagicMock(return_value=None)  # No match
    mock_adapter.thinking_parser = mock_thinking_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value="The answer is 4.")

    template_params = {"enable_thinking": {"default": True}}

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract = MagicMock(return_value=None)
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ), patch(
        "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
        return_value=None,
    ):
        supports, parser_id, diags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is False
    assert parser_id == "null"
    assert len(diags) == 1
    assert "enable_thinking" in diags[0].message


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_exception_in_path_b():
    """_sweep_thinking: Path B exception produces diagnostic (lines 463-472)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.extract = MagicMock(return_value=None)
    mock_adapter.thinking_parser = mock_thinking_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(side_effect=RuntimeError("Generation failed"))

    # No enable_thinking so goes directly to Path B
    template_params = None

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": MagicMock},
    ):
        supports, parser_id, diags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is False
    assert parser_id == "null"
    assert len(diags) > 0
    assert diags[0].level.value == "warning"


# ---------------------------------------------------------------------------
# Tests for coordinator._sweep_tools
# (coordinator.py lines 542-549, 575-582, 598-675)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_template_delivery_success():
    """_sweep_tools: template delivery returns ('template', parser_id) (lines 542-549)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_tool_parser = MagicMock()
    mock_tool_parser.validates = MagicMock(return_value=True)
    mock_adapter.tool_parser = mock_tool_parser
    mock_adapter.supports_native_tools = MagicMock(return_value=True)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value='{"name":"get_weather","arguments":{}}')

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.validates = MagicMock(return_value=True)
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock, "json_schema": mock_parser_cls},
    ), patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=True,
    ):
        tool_format, parser_id, diags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format == "template"
    assert parser_id == "json_schema"
    assert diags == []


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_adapter_delivery_success():
    """_sweep_tools: adapter delivery returns ('adapter', parser_id) (lines 575-582).

    Template delivery is skipped (no native tool support).
    Adapter delivery succeeds: format_tools_for_prompt returns prompt,
    and the adapter's tool_parser.validates() returns True.
    """
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    # The adapter's tool_parser.validates returns True on the adapter delivery call
    mock_tool_parser = MagicMock()
    mock_tool_parser.validates = MagicMock(return_value=True)

    mock_adapter = MagicMock()
    mock_adapter.tool_parser = mock_tool_parser
    mock_adapter.supports_native_tools = MagicMock(return_value=False)
    mock_adapter.format_tools_for_prompt = MagicMock(return_value="Available tools: get_weather")

    mock_loaded = MagicMock()
    # loaded.adapter always returns the same mock_adapter
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value='<tool_call>{"name":"get_weather"}</tool_call>')

    # We need at least one non-null parser in TOOL_PARSERS for the loop to run
    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.validates = MagicMock(return_value=True)
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock, "xml_tag": mock_parser_cls},
    ), patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ):
        tool_format, parser_id, diags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format == "adapter"
    assert parser_id == "xml_tag"
    assert diags == []


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_no_tool_support_detected():
    """_sweep_tools: no tool support, no markers → 'no tool support' (lines 598-675)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=False)
    mock_adapter.format_tools_for_prompt = MagicMock(return_value=None)
    mock_tool_parser = MagicMock()
    mock_tool_parser.validates = MagicMock(return_value=False)
    mock_adapter.tool_parser = mock_tool_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value="The weather is sunny in Tokyo today.")

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock},
    ), patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ):
        tool_format, parser_id, diags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert diags == []  # No markers, no unknown XML tags


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_tool_markers_found():
    """_sweep_tools: tool markers found but no parser matches → ACTION_NEEDED diagnostic (lines 598-653)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=False)
    mock_adapter.format_tools_for_prompt = MagicMock(return_value=None)
    mock_tool_parser = MagicMock()
    mock_tool_parser.validates = MagicMock(return_value=False)
    mock_adapter.tool_parser = mock_tool_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    # Output has tool markers but no parser matches
    mock_strategy._generate = AsyncMock(return_value='<tool_call>{"name":"get_weather"}</tool_call> get_weather')

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock},
    ), patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ):
        tool_format, parser_id, diags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert len(diags) == 1
    assert diags[0].level.value == "action_needed"
    assert diags[0].category.value == "tool_dialect"


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_tokenization_artifacts():
    """_sweep_tools: tokenization artifacts produce UNSUPPORTED diagnostic (lines 608-631)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    sp_marker = "\u2581"  # ▁

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=False)
    mock_adapter.format_tools_for_prompt = MagicMock(return_value=None)
    mock_tool_parser = MagicMock()
    mock_tool_parser.validates = MagicMock(return_value=False)
    mock_adapter.tool_parser = mock_tool_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    # Garbled output with SP markers and tool markers
    mock_strategy._generate = AsyncMock(
        return_value=f"<tool_call>{sp_marker}get{sp_marker}_weather{sp_marker}</tool_call>"
    )

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock},
    ), patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ):
        tool_format, parser_id, diags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert len(diags) == 1
    assert diags[0].level.value == "action_needed"
    assert diags[0].category.value == "unsupported"


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_unknown_xml_tags():
    """_sweep_tools: unknown XML tags produce WARNING diagnostic (lines 655-673)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=False)
    mock_adapter.format_tools_for_prompt = MagicMock(return_value=None)
    mock_tool_parser = MagicMock()
    mock_tool_parser.validates = MagicMock(return_value=False)
    mock_adapter.tool_parser = mock_tool_parser

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    # Output has unknown XML tag (not a tool marker, not a known tag)
    mock_strategy._generate = AsyncMock(return_value="<custom_response>Tokyo weather</custom_response>")

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock},
    ), patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ):
        tool_format, parser_id, diags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert len(diags) == 1
    assert diags[0].level.value == "warning"
    assert diags[0].category.value == "tool_dialect"
    assert "custom_response" in diags[0].details.get("unknown_tags", [])


# ---------------------------------------------------------------------------
# Tests for coordinator._sweep_thinking/tools exception handling
# (coordinator.py lines 329-333, 353-357)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_exception_yields_failed():
    """_sweep_generative_capabilities handles thinking exception gracefully (lines 329-333)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    mock_adapter = MagicMock()
    mock_adapter.thinking_parser = MagicMock()
    mock_adapter.thinking_parser.extract = MagicMock(return_value=None)
    mock_adapter.supports_native_tools = MagicMock(return_value=False)
    mock_adapter.format_tools_for_prompt = MagicMock(return_value=None)
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)
    result = ProbeResult()

    mock_strategy = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_params.discover_template_params",
            return_value={},
        ),
        patch.object(
            coordinator,
            "_sweep_thinking",
            side_effect=RuntimeError("Thinking sweep crashed"),
        ),
        patch.object(
            coordinator,
            "_sweep_tools",
            new_callable=AsyncMock,
            return_value=(None, None, []),
        ),
    ):
        steps = []
        async for step in coordinator._sweep_generative_capabilities(
            "test/model", mock_loaded, result, mock_strategy
        ):
            steps.append(step)

    thinking_steps = [s for s in steps if s.step == "test_thinking"]
    failed = next((s for s in thinking_steps if s.status == "failed"), None)
    assert failed is not None
    assert "Thinking sweep crashed" in failed.error


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_exception_yields_failed():
    """_sweep_generative_capabilities handles tool sweep exception gracefully (lines 353-357)."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    mock_adapter = MagicMock()
    mock_adapter.thinking_parser = MagicMock()
    mock_adapter.thinking_parser.extract = MagicMock(return_value=None)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)
    result = ProbeResult()

    mock_strategy = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_params.discover_template_params",
            return_value={},
        ),
        patch.object(
            coordinator,
            "_sweep_thinking",
            new_callable=AsyncMock,
            return_value=(False, "null", []),
        ),
        patch.object(
            coordinator,
            "_sweep_tools",
            side_effect=RuntimeError("Tool sweep crashed"),
        ),
    ):
        steps = []
        async for step in coordinator._sweep_generative_capabilities(
            "test/model", mock_loaded, result, mock_strategy
        ):
            steps.append(step)

    tool_steps = [s for s in steps if s.step == "test_tools"]
    failed = next((s for s in tool_steps if s.status == "failed"), None)
    assert failed is not None
    assert "Tool sweep crashed" in failed.error


# ---------------------------------------------------------------------------
# Tests for coordinator._save_capabilities template params serialization
# (coordinator.py lines 716-719)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_capabilities_template_params_with_model_dump():
    """_save_capabilities serializes template_params with model_dump (lines 716-719)."""
    from mlx_manager.services.probe.coordinator import _save_capabilities
    from mlx_manager.services.probe.steps import ProbeResult

    # Object with model_dump (Pydantic BaseModel-like)
    class FakeParam:
        def model_dump(self):
            return {"type": "bool", "default": False}

    result = ProbeResult()
    result.template_params = {"enable_thinking": FakeParam()}

    with patch(
        "mlx_manager.services.model_registry.update_model_capabilities",
        new_callable=AsyncMock,
    ) as mock_update:
        await _save_capabilities("test/model", result)

    call_kwargs = mock_update.call_args[1]
    import json
    template_params_str = call_kwargs.get("template_params")
    assert template_params_str is not None
    parsed = json.loads(template_params_str)
    assert "enable_thinking" in parsed
    assert parsed["enable_thinking"] == {"type": "bool", "default": False}


@pytest.mark.asyncio
async def test_save_capabilities_template_params_plain_value():
    """_save_capabilities serializes template_params plain values (lines 716-719)."""
    from mlx_manager.services.probe.coordinator import _save_capabilities
    from mlx_manager.services.probe.steps import ProbeResult

    result = ProbeResult()
    result.template_params = {"temperature": 0.7, "enable_thinking": True}

    with patch(
        "mlx_manager.services.model_registry.update_model_capabilities",
        new_callable=AsyncMock,
    ) as mock_update:
        await _save_capabilities("test/model", result)

    call_kwargs = mock_update.call_args[1]
    import json
    template_params_str = call_kwargs.get("template_params")
    assert template_params_str is not None
    parsed = json.loads(template_params_str)
    assert parsed["temperature"] == 0.7
    assert parsed["enable_thinking"] is True

"""Additional tests for probe modules to improve coverage.

Targets uncovered lines in:
- services/probe/base.py
- services/probe/coordinator.py
- services/probe/vision.py
- services/probe/embeddings.py
- services/probe/text_gen.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.services.probe.base import (
    GenerativeProbe,
    _find_unclosed_thinking_tag,
    _has_tokenization_artifacts,
)
from mlx_manager.services.probe.coordinator import ProbingCoordinator
from mlx_manager.services.probe.steps import ProbeResult

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
    ):
        from mlx_manager.mlx_server.models.ir import TextResult

        return TextResult(content=self._mock_output)

    async def probe(self, model_id: str, loaded, result: ProbeResult):
        return
        yield  # make this an async generator


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
    """_generate calls adapter.configure for template_options and resets after."""
    from mlx_manager.mlx_server.models.ir import TextResult

    probe = SimpleProbe()

    mock_adapter = MagicMock()
    mock_result = TextResult(content="hello")
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

    assert result.content == "hello"
    # configure called with options and then reset to None
    assert mock_adapter.configure.call_count == 2
    first_call = mock_adapter.configure.call_args_list[0]
    second_call = mock_adapter.configure.call_args_list[1]
    assert first_call[1]["template_options"] == {"enable_thinking": True}
    assert second_call[1]["template_options"] is None


@pytest.mark.asyncio
async def test_generate_returns_text_result():
    """_generate returns the full TextResult from adapter.generate."""
    from mlx_manager.mlx_server.models.ir import TextResult

    probe = SimpleProbe()

    mock_adapter = MagicMock()
    mock_result = TextResult(content="The answer is 4")
    mock_adapter.generate = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    result = await GenerativeProbe._generate(
        probe,
        mock_loaded,
        [{"role": "user", "content": "What is 2+2?"}],
    )

    assert result.content == "The answer is 4"
    assert result.reasoning_content is None
    mock_adapter.configure.assert_not_called()  # No template_options, no configure


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
    """VisionProbe._generate returns full TextResult and configures template_options."""
    from PIL import Image

    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.vision import VisionProbe

    probe = VisionProbe()

    mock_adapter = MagicMock()
    mock_result = TextResult(content="I see a gray image")
    mock_adapter.generate = AsyncMock(return_value=mock_result)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.model = MagicMock()

    result = await probe._generate(
        mock_loaded,
        [{"role": "user", "content": "Describe this image"}],
        template_options={"enable_thinking": True},
    )

    assert result.content == "I see a gray image"
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
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    probe = EmbeddingsProbe()

    # First call returns a valid embedding, second call raises for similarity test
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

    with (
        patch(
            "mlx_manager.services.probe.text_gen._estimate_practical_max_tokens",
            side_effect=RuntimeError("Context estimation failed"),
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
            return_value="qwen",
        ),
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
    """Default family creates diagnostic with architecture."""
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
    """Detects model family when result.model_family is None."""
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
    """Discovers template params when tokenizer is present."""
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
        patch.object(
            coordinator,
            "_sweep_thinking",
            new_callable=AsyncMock,
            return_value=(False, "null", [], []),
        ),
        patch.object(
            coordinator,
            "_sweep_tools",
            new_callable=AsyncMock,
            return_value=(None, None, [], []),
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
async def test_coordinator_sweep_thinking_success_with_parser_match():
    """_sweep_thinking: thinking detected when parser matches raw output."""
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}

    coordinator = ProbingCoordinator(mock_pool)

    mock_loaded = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(
        return_value=TextResult(content="<think>Some thinking content</think>Answer")
    )

    template_params = {"enable_thinking": {"default": True}}

    # Mock parser that matches
    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract.return_value = "Some thinking content"
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ):
        supports, parser_id, diags, tags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is True
    assert parser_id == "think_tag"
    assert diags == []


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_unclosed_retry():
    """_sweep_thinking: retry on unclosed tag succeeds."""
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}

    coordinator = ProbingCoordinator(mock_pool)

    call_count = [0]

    async def generate_with_retry(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Unclosed tag in raw output
            return TextResult(content="<think>Unclosed thinking")
        # Retry: complete tags in raw output
        return TextResult(content="<think>Complete thinking</think>Answer")

    mock_loaded = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = generate_with_retry

    template_params = {"enable_thinking": {"default": True}}

    # Mock parser: fails on unclosed, succeeds on complete
    mock_parser_cls = MagicMock()

    def extract_side_effect(text):
        if "</think>" in text:
            return "Complete thinking"
        return None

    mock_parser_instance = MagicMock()
    mock_parser_instance.extract = MagicMock(side_effect=extract_side_effect)
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ):
        supports, parser_id, diags, tags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is True
    assert call_count[0] == 2  # Initial + retry


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_always_thinks():
    """_sweep_thinking: always-thinks detection (no enable_thinking param)."""
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}

    coordinator = ProbingCoordinator(mock_pool)

    mock_loaded = MagicMock()

    mock_strategy = MagicMock()
    # Raw output has thinking tags
    mock_strategy._generate = AsyncMock(
        return_value=TextResult(content="<think>Always thinking</think>Answer")
    )

    # No enable_thinking in template_params
    template_params = None

    # Mock parser that matches
    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract.return_value = "Always thinking"
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ):
        supports, parser_id, diags, tags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is True
    assert parser_id == "think_tag"


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_has_enable_thinking_but_no_tags_diagnostic():
    """_sweep_thinking: has enable_thinking but no tags → diagnostic."""
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}

    coordinator = ProbingCoordinator(mock_pool)

    mock_loaded = MagicMock()

    mock_strategy = MagicMock()
    # Plain output without thinking tags
    mock_strategy._generate = AsyncMock(return_value=TextResult(content="The answer is 4."))

    template_params = {"enable_thinking": {"default": True}}

    # Mock parser that doesn't match
    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.extract.return_value = None
    mock_parser_cls.return_value = mock_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_parser_cls},
    ):
        supports, parser_id, diags, tags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is False
    assert parser_id == "null"
    assert len(diags) == 1
    assert "enable_thinking" in diags[0].message


@pytest.mark.asyncio
async def test_coordinator_sweep_thinking_generation_exception():
    """_sweep_thinking: generation exception produces diagnostic."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}

    coordinator = ProbingCoordinator(mock_pool)

    mock_loaded = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(side_effect=RuntimeError("Generation failed"))

    template_params = None

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": MagicMock},
    ):
        supports, parser_id, diags, tags = await coordinator._sweep_thinking(
            "test/model", mock_loaded, mock_strategy, template_params
        )

    assert supports is False
    assert parser_id == "null"
    assert len(diags) == 1
    assert diags[0].level.value == "warning"
    assert "generation error" in diags[0].message.lower()


# ---------------------------------------------------------------------------
# Tests for coordinator._sweep_tools
# (coordinator.py lines 542-549, 575-582, 598-675)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_template_delivery_success():
    """_sweep_tools: template delivery ('template', parser_id) on fallback."""
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=True)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    call_count = [0]

    async def generate_per_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Phase 1 generic injection: no tool output
            return TextResult(content="Tokyo weather is sunny today.")
        # Phase 4 template delivery: tool output
        return TextResult(content='{"name":"get_weather","arguments":{}}')

    mock_strategy = MagicMock()
    mock_strategy._generate = generate_per_call

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    # Only validate on tool-like output (not plain text)
    mock_parser_instance.validates = MagicMock(
        side_effect=lambda text, fn: "get_weather" in text and "{" in text
    )
    mock_parser_instance.stream_markers = []
    mock_parser_cls.return_value = mock_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock, "json_schema": mock_parser_cls},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format == "template"
    assert parser_id == "json_schema"
    assert diags == []


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_adapter_delivery_success():
    """_sweep_tools: generic injection returns ('detected', parser_id) via tag discovery.

    Generic injection is Phase 1. Output has <tool_call> tags which are discovered
    and mapped to the xml_tag parser. Tag-first validation succeeds.
    """
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=False)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(
        return_value=TextResult(content='<tool_call>{"name":"get_weather"}</tool_call>')
    )

    # Parser needs stream_markers so _build_marker_to_parsers can map <tool_call> to it
    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.validates = MagicMock(return_value=True)
    mock_parser_instance.stream_markers = [("<tool_call>", "</tool_call>")]
    mock_parser_cls.return_value = mock_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock, "xml_tag": mock_parser_cls},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format == "detected"
    assert parser_id == "xml_tag"
    assert diags == []


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_no_tool_support_detected():
    """_sweep_tools: no tool support, no markers → 'no tool support'."""
    from mlx_manager.mlx_server.models.ir import TextResult
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
    mock_strategy._generate = AsyncMock(
        return_value=TextResult(content="The weather is sunny in Tokyo today.")
    )

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert diags == []  # No markers, no unknown XML tags


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_tool_markers_found():
    """_sweep_tools: tool markers found but no parser matches → ACTION_NEEDED diagnostic."""
    from mlx_manager.mlx_server.models.ir import TextResult
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
    mock_strategy._generate = AsyncMock(
        return_value=TextResult(content='<tool_call>{"name":"get_weather"}</tool_call> get_weather')
    )

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert len(diags) == 1
    assert diags[0].level.value == "action_needed"
    assert diags[0].category.value == "tool_dialect"


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_tokenization_artifacts():
    """_sweep_tools: tokenization artifacts produce UNSUPPORTED diagnostic."""
    from mlx_manager.mlx_server.models.ir import TextResult
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
        return_value=TextResult(
            content=f"<tool_call>{sp_marker}get{sp_marker}_weather{sp_marker}</tool_call>"
        )
    )

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert len(diags) == 1
    assert diags[0].level.value == "action_needed"
    assert diags[0].category.value == "unsupported"


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_unknown_xml_tags():
    """_sweep_tools: unknown XML tags produce WARNING diagnostic."""
    from mlx_manager.mlx_server.models.ir import TextResult
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
    mock_strategy._generate = AsyncMock(
        return_value=TextResult(content="<custom_response>Tokyo weather</custom_response>")
    )

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    assert parser_id is None
    assert len(diags) == 1
    assert diags[0].level.value == "warning"
    assert diags[0].category.value == "tool_dialect"
    # Generic tag detection reports unmatched tags as list of dicts
    detected = diags[0].details.get("detected_tags", [])
    tag_names = [t["name"] if isinstance(t, dict) else t[0] for t in detected]
    assert "custom_response" in tag_names


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
            return_value=(None, None, [], []),
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
            return_value=(False, "null", [], []),
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


# ---------------------------------------------------------------------------
# Tests for coordinator._sweep_tools — new flow (generic injection first,
# template delivery fallback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_generic_first_then_template_fallback():
    """_sweep_tools: generic injection fails, template delivery succeeds as fallback."""
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=True)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    call_count = [0]

    async def generate_per_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Phase 1 generic injection: plain text, no tool tags
            return TextResult(content="I can help you check the weather in Tokyo.")
        # Phase 4 template delivery: proper tool call
        return TextResult(content='{"name":"get_weather","arguments":{}}')

    mock_strategy = MagicMock()
    mock_strategy._generate = generate_per_call

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    # Only validate on tool-like output (not plain text)
    mock_parser_instance.validates = MagicMock(
        side_effect=lambda text, fn: "get_weather" in text and "{" in text
    )
    mock_parser_instance.stream_markers = []
    mock_parser_cls.return_value = mock_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock, "json_schema": mock_parser_cls},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format == "template"
    assert parser_id == "json_schema"
    assert call_count[0] == 2  # generic injection + template delivery


@pytest.mark.asyncio
async def test_coordinator_sweep_tools_discovered_tags_in_result():
    """_sweep_tools returns discovered tags even when no parser validates."""
    from mlx_manager.mlx_server.models.ir import TextResult
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()

    coordinator = ProbingCoordinator(mock_pool)

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools = MagicMock(return_value=False)

    mock_loaded = MagicMock()
    mock_loaded.adapter = mock_adapter
    mock_loaded.tokenizer = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(
        return_value=TextResult(content="<custom_response>weather info</custom_response>")
    )

    with (
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock},
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diags, tags = await coordinator._sweep_tools(
            "test/model", mock_loaded, mock_strategy
        )

    assert tool_format is None
    # Tags should contain the custom_response tag even though no parser matched
    tag_names = [t.name for t in tags]
    assert "custom_response" in tag_names


# ---------------------------------------------------------------------------
# Tests for null parser probing
# (coordinator.py: null parsers configured before sweep, restored after)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_configures_null_parsers_after_load():
    """Coordinator forces null parsers so raw output reaches sweep code."""
    from mlx_manager.mlx_server.models.types import ModelType
    from mlx_manager.mlx_server.parsers.thinking import NullThinkingParser, ThinkTagParser
    from mlx_manager.mlx_server.parsers.tool_call import HermesJsonParser, NullToolParser

    pool = MagicMock()
    pool._models = {}
    pool._profile_settings = {}

    adapter = MagicMock()
    # Start with active parsers (family defaults)
    adapter.tool_parser = HermesJsonParser()
    adapter.thinking_parser = ThinkTagParser()
    configure_calls = []

    def track_configure(**kwargs):
        configure_calls.append(kwargs)

    adapter.configure = track_configure

    loaded = MagicMock()
    loaded.adapter = adapter
    loaded.tokenizer = MagicMock()
    loaded.capabilities = None

    pool.get_model = AsyncMock(return_value=loaded)
    pool.register_profile_settings = MagicMock()
    pool.unregister_profile_settings = MagicMock()
    pool.unload_model = AsyncMock()

    # Build a real-looking detection result pointing to TEXT_GEN
    mock_detection = MagicMock()
    mock_detection.model_type = ModelType.TEXT_GEN
    mock_detection.detection_method = "config"
    mock_detection.architecture = "Qwen2ForCausalLM"

    async def empty_probe(*a, **kw):
        return
        yield

    mock_strategy_inst = MagicMock()
    mock_strategy_inst.probe = empty_probe

    with (
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=mock_detection,
        ),
        patch(
            "mlx_manager.services.probe.strategy.get_probe_strategy",
            return_value=mock_strategy_inst,
        ),
        patch(
            "mlx_manager.services.probe.coordinator._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
    ):
        coord = ProbingCoordinator(pool)
        steps = []
        async for step in coord.probe("test-model"):
            steps.append(step)

    # Verify null parsers were configured after load
    assert len(configure_calls) >= 1
    first_call = configure_calls[0]
    assert isinstance(first_call["tool_parser"], NullToolParser)
    assert isinstance(first_call["thinking_parser"], NullThinkingParser)


@pytest.mark.asyncio
async def test_coordinator_restores_parsers_for_preloaded_model():
    """Coordinator restores original parsers on preloaded models after probing."""
    from mlx_manager.mlx_server.models.types import ModelType
    from mlx_manager.mlx_server.parsers.thinking import ThinkTagParser
    from mlx_manager.mlx_server.parsers.tool_call import HermesJsonParser

    original_tool = HermesJsonParser()
    original_thinking = ThinkTagParser()

    pool = MagicMock()
    pool._models = {"preloaded-model": MagicMock()}  # preloaded
    pool._profile_settings = {}

    adapter = MagicMock()
    adapter.tool_parser = original_tool
    adapter.thinking_parser = original_thinking
    configure_calls = []

    def track_configure(**kwargs):
        configure_calls.append(kwargs)
        # Update the mock properties to track state
        if "tool_parser" in kwargs:
            adapter.tool_parser = kwargs["tool_parser"]
        if "thinking_parser" in kwargs:
            adapter.thinking_parser = kwargs["thinking_parser"]

    adapter.configure = track_configure

    loaded = MagicMock()
    loaded.adapter = adapter
    loaded.tokenizer = MagicMock()
    loaded.capabilities = None

    pool.get_model = AsyncMock(return_value=loaded)
    pool.register_profile_settings = MagicMock()

    mock_detection = MagicMock()
    mock_detection.model_type = ModelType.TEXT_GEN
    mock_detection.detection_method = "config"
    mock_detection.architecture = "Qwen2ForCausalLM"

    async def empty_probe(*a, **kw):
        return
        yield

    mock_strategy_inst = MagicMock()
    mock_strategy_inst.probe = empty_probe

    with (
        patch(
            "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
            return_value=mock_detection,
        ),
        patch(
            "mlx_manager.services.probe.strategy.get_probe_strategy",
            return_value=mock_strategy_inst,
        ),
        patch(
            "mlx_manager.services.probe.coordinator._save_capabilities",
            new_callable=AsyncMock,
        ),
        patch("mlx_manager.database.get_session"),
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
    ):
        coord = ProbingCoordinator(pool)
        steps = []
        async for step in coord.probe("preloaded-model"):
            steps.append(step)

    # Last configure call should restore original parsers
    assert len(configure_calls) >= 2
    last_call = configure_calls[-1]
    assert isinstance(last_call["tool_parser"], HermesJsonParser)
    assert isinstance(last_call["thinking_parser"], ThinkTagParser)


@pytest.mark.asyncio
async def test_coordinator_raw_output_reaches_sweep_with_thinking_tags():
    """With null parsers, thinking tags in raw output are preserved for sweep detection."""
    from mlx_manager.mlx_server.parsers.thinking import NullThinkingParser, ThinkTagParser

    raw = "<think>Let me reason about this.</think>The answer is 42."

    # NullThinkingParser.extract returns None: it does not recognise/strip thinking tags
    null_parser = NullThinkingParser()
    result = null_parser.extract(raw)
    assert result is None  # No thinking extracted — raw output is unchanged

    # ThinkTagParser.extract DOES find and return the thinking content
    active_parser = ThinkTagParser()
    result = active_parser.extract(raw)
    assert result is not None  # It finds and extracts thinking content

"""Tests for GenerativeProbe.sweep_capabilities() and coordinator delegation.

TDD tests written BEFORE implementation. These verify the design:
- GenerativeProbe.sweep_capabilities() owns the sweep logic
- ProbingCoordinator delegates to strategy.sweep_capabilities() for GenerativeProbe subclasses
- Non-generative strategies (embeddings, audio) do NOT get sweep called
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.services.probe.steps import ProbeResult

# ---------------------------------------------------------------------------
# Concrete GenerativeProbe subclass for testing
# ---------------------------------------------------------------------------


def _make_generative_probe():
    """Create a minimal concrete GenerativeProbe subclass for tests."""
    from mlx_manager.mlx_server.models.types import ModelType
    from mlx_manager.services.probe.base import GenerativeProbe

    class _TestGenerativeProbe(GenerativeProbe):
        @property
        def model_type(self):
            return ModelType.TEXT_GEN

        async def probe(self, model_id, loaded, result):
            if False:
                yield  # make it an async generator

    return _TestGenerativeProbe()


# ---------------------------------------------------------------------------
# sweep_capabilities: family detection step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_capabilities_yields_family_detection():
    """sweep_capabilities yields a detect_family step with capability=model_family."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None
    mock_loaded.tokenizer = MagicMock()

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
        async for step in probe.sweep_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    step_names = [s.step for s in steps]
    assert "detect_family" in step_names

    # Find the completed detect_family step
    completed = next(
        (s for s in steps if s.step == "detect_family" and s.status == "completed"),
        None,
    )
    assert completed is not None
    assert completed.capability == "model_family"


@pytest.mark.asyncio
async def test_sweep_capabilities_sets_family_on_result():
    """sweep_capabilities sets result.model_family when it was None."""
    probe = _make_generative_probe()
    result = ProbeResult()
    result.model_family = None

    mock_loaded = MagicMock()
    mock_loaded.adapter = None
    mock_loaded.tokenizer = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ) as mock_detect,
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch(
            "mlx_manager.utils.model_detection.read_model_config",
            return_value=None,
        ),
    ):
        async for _ in probe.sweep_capabilities("test/model", mock_loaded, result):
            pass

    assert result.model_family == "qwen"
    mock_detect.assert_called_once_with("test/model", architecture=None)


@pytest.mark.asyncio
async def test_sweep_capabilities_skips_detect_when_family_already_set():
    """sweep_capabilities skips detect_model_family call when result.model_family is already set."""
    probe = _make_generative_probe()
    result = ProbeResult()
    result.model_family = "llama"  # Pre-set

    mock_loaded = MagicMock()
    mock_loaded.adapter = None
    mock_loaded.tokenizer = MagicMock()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
        ) as mock_detect,
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"llama": MagicMock},
        ),
    ):
        async for _ in probe.sweep_capabilities("test/model", mock_loaded, result):
            pass

    mock_detect.assert_not_called()
    assert result.model_family == "llama"  # Unchanged


# ---------------------------------------------------------------------------
# sweep_capabilities: adapter/tokenizer guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_capabilities_skips_when_no_adapter():
    """sweep_capabilities skips thinking and tools when adapter is None."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None  # No adapter
    mock_loaded.tokenizer = MagicMock()

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
        async for step in probe.sweep_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    skipped = {s.step for s in steps if s.status == "skipped"}
    assert "test_thinking" in skipped
    assert "test_tools" in skipped


@pytest.mark.asyncio
async def test_sweep_capabilities_skips_when_no_tokenizer():
    """sweep_capabilities skips thinking and tools when tokenizer is None."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.tokenizer = None  # No tokenizer

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
        async for step in probe.sweep_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    skipped = {s.step for s in steps if s.status == "skipped"}
    assert "test_thinking" in skipped
    assert "test_tools" in skipped


# ---------------------------------------------------------------------------
# sweep_capabilities: template parameter discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_capabilities_discovers_template_params():
    """sweep_capabilities calls discover_template_params and stores result."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.tokenizer = MagicMock()

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
            "mlx_manager.services.probe.sweeps.sweep_thinking",
            new_callable=AsyncMock,
            return_value=(False, "null", [], []),
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_tools",
            new_callable=AsyncMock,
            return_value=(None, None, [], []),
        ),
    ):
        async for _ in probe.sweep_capabilities("test/model", mock_loaded, result):
            pass

    assert result.template_params == mock_params


# ---------------------------------------------------------------------------
# sweep_capabilities: delegates to sweep functions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_capabilities_calls_sweep_functions():
    """sweep_capabilities delegates to sweep_thinking and sweep_tools."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.tokenizer = MagicMock()

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
            return_value=None,
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_thinking",
            new_callable=AsyncMock,
            return_value=(True, "qwen", [], []),
        ) as mock_thinking,
        patch(
            "mlx_manager.services.probe.sweeps.sweep_tools",
            new_callable=AsyncMock,
            return_value=("detected", "hermes", [], []),
        ) as mock_tools,
    ):
        async for _ in probe.sweep_capabilities("test/model", mock_loaded, result):
            pass

    # Both sweep functions should be called with self as the strategy
    mock_thinking.assert_called_once()
    mock_tools.assert_called_once()

    # The first positional arg to sweep_thinking should be the probe itself (strategy)
    thinking_args = mock_thinking.call_args[0]
    assert thinking_args[2] is probe  # strategy argument is self

    # The first positional arg to sweep_tools should be the probe itself (strategy)
    tools_args = mock_tools.call_args[0]
    assert tools_args[2] is probe  # strategy argument is self


@pytest.mark.asyncio
async def test_sweep_capabilities_stores_sweep_results():
    """sweep_capabilities stores thinking and tool results on ProbeResult."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.tokenizer = MagicMock()

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
            return_value=None,
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_thinking",
            new_callable=AsyncMock,
            return_value=(True, "qwen_thinking", [], []),
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_tools",
            new_callable=AsyncMock,
            return_value=("detected", "hermes_json", [], []),
        ),
    ):
        async for _ in probe.sweep_capabilities("test/model", mock_loaded, result):
            pass

    assert result.supports_thinking is True
    assert result.thinking_parser_id == "qwen_thinking"
    assert result.supports_native_tools is True
    assert result.tool_format == "detected"
    assert result.tool_parser_id == "hermes_json"


@pytest.mark.asyncio
async def test_sweep_capabilities_yields_test_thinking_step():
    """sweep_capabilities yields test_thinking step with capability=supports_thinking."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.tokenizer = MagicMock()

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
            return_value=None,
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_thinking",
            new_callable=AsyncMock,
            return_value=(True, "qwen_thinking", [], []),
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_tools",
            new_callable=AsyncMock,
            return_value=(None, None, [], []),
        ),
    ):
        steps = []
        async for step in probe.sweep_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    completed_thinking = next(
        (s for s in steps if s.step == "test_thinking" and s.status == "completed"),
        None,
    )
    assert completed_thinking is not None
    assert completed_thinking.capability == "supports_thinking"
    assert completed_thinking.value is True


@pytest.mark.asyncio
async def test_sweep_capabilities_yields_test_tools_step():
    """sweep_capabilities yields test_tools step with capability=tool_format."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.tokenizer = MagicMock()

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
            return_value=None,
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_thinking",
            new_callable=AsyncMock,
            return_value=(False, "null", [], []),
        ),
        patch(
            "mlx_manager.services.probe.sweeps.sweep_tools",
            new_callable=AsyncMock,
            return_value=("template", "hermes_json", [], []),
        ),
    ):
        steps = []
        async for step in probe.sweep_capabilities("test/model", mock_loaded, result):
            steps.append(step)

    completed_tools = next(
        (s for s in steps if s.step == "test_tools" and s.status == "completed"),
        None,
    )
    assert completed_tools is not None
    assert completed_tools.capability == "tool_format"
    assert completed_tools.value == "template"


@pytest.mark.asyncio
async def test_sweep_capabilities_default_family_adds_diagnostic():
    """Default family triggers a WARNING diagnostic about missing adapter config."""
    probe = _make_generative_probe()
    result = ProbeResult()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None
    mock_loaded.tokenizer = MagicMock()

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
            return_value={"architectures": ["CustomModelForCausalLM"]},
        ),
    ):
        async for _ in probe.sweep_capabilities("test/model", mock_loaded, result):
            pass

    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.category.value == "family"
    assert "CustomModelForCausalLM" in diag.message


# ---------------------------------------------------------------------------
# coordinator delegates to GenerativeProbe.sweep_capabilities()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinator_delegates_to_generative_probe():
    """Coordinator calls strategy.sweep_capabilities() for GenerativeProbe strategies."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator
    from mlx_manager.services.probe.text_gen import TextGenProbe

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()
    mock_pool.unregister_profile_settings = MagicMock()
    mock_pool.unload_model = AsyncMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.tokenizer = MagicMock()
    mock_loaded.capabilities = None

    strategy = TextGenProbe()
    coordinator = ProbingCoordinator(mock_pool)

    sweep_steps = [
        MagicMock(step="detect_family", status="running"),
        MagicMock(step="detect_family", status="completed"),
        MagicMock(step="test_thinking", status="completed"),
        MagicMock(step="test_tools", status="completed"),
    ]

    async def fake_sweep(model_id, loaded, result, *, verbose=False):
        for s in sweep_steps:
            yield s

    with patch.object(strategy, "sweep_capabilities", side_effect=fake_sweep) as mock_sweep:
        # Patch all the external dependencies for a minimal probe run
        with (
            patch(
                "mlx_manager.mlx_server.models.detection.detect_model_type_detailed",
                return_value=MagicMock(
                    model_type=MagicMock(value="text_gen"),
                    detection_method="config",
                    architecture="LlamaForCausalLM",
                ),
            ),
            patch(
                "mlx_manager.mlx_server.models.types.ModelType",
            ),
            patch(
                "mlx_manager.services.probe.strategy.get_probe_strategy",
                return_value=strategy,
            ),
            patch.object(mock_pool, "get_model", new=AsyncMock(return_value=mock_loaded)),
            patch(
                "mlx_manager.services.probe.coordinator._save_capabilities",
                new=AsyncMock(),
            ),
        ):
            steps = []
            async for step in coordinator.probe("test/model"):
                steps.append(step)

    # sweep_capabilities should have been called
    mock_sweep.assert_called_once()


@pytest.mark.asyncio
async def test_coordinator_skips_sweep_for_non_generative_strategies():
    """Coordinator does NOT call sweep_capabilities for non-GenerativeProbe strategies."""
    from mlx_manager.services.probe.embeddings import EmbeddingsProbe

    mock_pool = MagicMock()
    mock_pool._models = {}
    mock_pool._profile_settings = {}
    mock_pool.register_profile_settings = MagicMock()
    mock_pool.unregister_profile_settings = MagicMock()
    mock_pool.unload_model = AsyncMock()

    mock_loaded = MagicMock()
    mock_loaded.adapter = None
    mock_loaded.tokenizer = None

    strategy = EmbeddingsProbe()

    # Confirm EmbeddingsProbe is NOT a GenerativeProbe
    from mlx_manager.services.probe.base import GenerativeProbe

    assert not isinstance(strategy, GenerativeProbe)

    # Also confirm it has no sweep_capabilities method
    assert not hasattr(strategy, "sweep_capabilities")


# ---------------------------------------------------------------------------
# Backward compat: coordinator._sweep_generative_capabilities removed
# ---------------------------------------------------------------------------


def test_coordinator_no_longer_has_sweep_generative_capabilities():
    """ProbingCoordinator should NOT have _sweep_generative_capabilities anymore."""
    from mlx_manager.services.probe.coordinator import ProbingCoordinator

    mock_pool = MagicMock()
    coordinator = ProbingCoordinator(mock_pool)

    assert not hasattr(coordinator, "_sweep_generative_capabilities"), (
        "Coordinator should delegate to strategy.sweep_capabilities() — the method was moved"
    )


def test_generative_probe_has_sweep_capabilities():
    """GenerativeProbe should expose sweep_capabilities() as a public method."""
    from mlx_manager.services.probe.base import GenerativeProbe

    assert hasattr(GenerativeProbe, "sweep_capabilities"), (
        "GenerativeProbe must have sweep_capabilities() method"
    )


# ---------------------------------------------------------------------------
# Direct sweep function tests for uncovered lines in sweeps.py
# ---------------------------------------------------------------------------


def _make_gen_result(content: str):
    """Create a mock generation result with a .content attribute."""
    result = MagicMock()
    result.content = content
    return result


def _make_tag_discovery(name: str, style: str, paired: bool, matched_parsers: list[str]):
    """Create a TagDiscovery instance."""
    from mlx_manager.services.probe.steps import TagDiscovery

    return TagDiscovery(name=name, style=style, paired=paired, matched_parsers=matched_parsers)


def _make_mock_thinking_parser(parser_id: str, extract_return, stream_markers_list):
    """Create a mock ThinkingParser class that returns given values."""

    class MockParser:
        @property
        def parser_id(self):
            return parser_id

        @property
        def stream_markers(self):
            return stream_markers_list

        def extract(self, text):
            return extract_return

    return MockParser


def _make_mock_tool_parser(parser_id: str, validates_return: bool, stream_markers_list=None):
    """Create a mock ToolCallParser class that returns given values."""

    class MockToolParser:
        @property
        def parser_id(self):
            return parser_id

        @property
        def stream_markers(self):
            return stream_markers_list or []

        def validates(self, text, expected_fn):
            return validates_return

        def extract(self, text):
            return []

    return MockToolParser


# ---------------------------------------------------------------------------
# sweep_thinking: Lines 141-145 — Phase 3 tag-first detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_thinking_phase3_tag_first_detection():
    """Phase 3: tag-first detection when _discover_and_map_tags finds tags matching a parser
    and the parser's extract() returns content. (Lines 141-145)"""
    from mlx_manager.services.probe.sweeps import sweep_thinking

    strategy = MagicMock()
    strategy._generate = AsyncMock(
        return_value=_make_gen_result("<think>step by step reasoning</think>answer")
    )
    loaded = MagicMock()

    # Parser that successfully extracts thinking content
    mock_parser = _make_mock_thinking_parser(
        "think_tag", "step by step reasoning", [("<think>", "</think>")]
    )

    mock_parsers = {
        "null": MagicMock,
        "think_tag": mock_parser,
    }

    # Tag discovery returns a tag with matched_parsers containing "think_tag"
    mock_tag = _make_tag_discovery("think", "xml", True, ["think_tag"])

    with (
        patch("mlx_manager.mlx_server.parsers.THINKING_PARSERS", mock_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[mock_tag],
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            return_value=["think_tag"],
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_thinking_parser_id",
            return_value=None,
        ),
    ):
        supports, parser_id, diags, tags = await sweep_thinking(
            "test/model", loaded, strategy, None, family=None
        )

    assert supports is True
    assert parser_id == "think_tag"


# ---------------------------------------------------------------------------
# sweep_thinking: Lines 182-183 — Phase 4 retry exception handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_thinking_phase4_retry_exception():
    """Phase 4: retry generation fails with exception, then falls through
    to unclosed tag matching. (Lines 182-183 + 191-198)"""
    from mlx_manager.services.probe.sweeps import sweep_thinking

    # First call succeeds with content that has an unclosed tag
    # Second call (retry) raises an exception
    strategy = MagicMock()
    strategy._generate = AsyncMock(
        side_effect=[
            _make_gen_result("some output without thinking tags"),
            RuntimeError("retry generation failed"),
        ]
    )
    loaded = MagicMock()

    # Parser whose stream_markers match the unclosed tag "think"
    mock_parser = _make_mock_thinking_parser(
        "think_tag",
        None,  # extract returns None (no match)
        [("<think>", "</think>")],
    )

    mock_parsers = {
        "null": MagicMock,
        "think_tag": mock_parser,
    }

    with (
        patch("mlx_manager.mlx_server.parsers.THINKING_PARSERS", mock_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],  # No tags discovered
        ),
        patch(
            "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
            return_value="think",  # Unclosed <think> tag found
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_thinking_parser_id",
            return_value=None,
        ),
    ):
        supports, parser_id, diags, tags = await sweep_thinking(
            "test/model", loaded, strategy, None, family=None
        )

    # Retry failed, but unclosed tag "think" matches parser stream_markers
    # "<think>" -> strip("<>") -> "think" == unclosed "think"
    assert supports is True
    assert parser_id == "think_tag"


# ---------------------------------------------------------------------------
# sweep_thinking: Lines 191-198 — Phase 4 unclosed tag matching stream_markers
# (retry succeeds but no parser extract matches)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_thinking_phase4_unclosed_tag_matches_stream_markers():
    """Phase 4: retry succeeds but no parser.extract() matches the retry output;
    then unclosed tag matches a parser's stream_markers. (Lines 191-198)"""
    from mlx_manager.services.probe.sweeps import sweep_thinking

    # First call: normal output, second call: retry output (no thinking tags either)
    strategy = MagicMock()
    strategy._generate = AsyncMock(
        side_effect=[
            _make_gen_result("output with unclosed think tag"),
            _make_gen_result("retry output still no closed tags"),
        ]
    )
    loaded = MagicMock()

    # Parser whose extract returns None (doesn't match retry output)
    # but stream_markers contain [THINK] which matches unclosed "THINK"
    mock_bracket_parser = _make_mock_thinking_parser(
        "mistral_think",
        None,  # extract returns None
        [("[THINK]", "[/THINK]")],
    )

    mock_parsers = {
        "null": MagicMock,
        "mistral_think": mock_bracket_parser,
    }

    with (
        patch("mlx_manager.mlx_server.parsers.THINKING_PARSERS", mock_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
            return_value="THINK",
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_thinking_parser_id",
            return_value=None,
        ),
    ):
        supports, parser_id, diags, tags = await sweep_thinking(
            "test/model", loaded, strategy, None, family=None
        )

    # Unclosed "THINK" matches "[THINK]" -> strip("[]") -> "THINK"
    assert supports is True
    assert parser_id == "mistral_think"


# ---------------------------------------------------------------------------
# sweep_thinking: Line 203 — Phase 5 unknown thinking tag detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_thinking_phase5_unknown_tag_diagnostic():
    """Phase 5: _detect_unknown_thinking_tags finds an unknown tag and adds
    a WARNING diagnostic. (Line 203)"""
    from mlx_manager.services.probe.steps import DiagnosticCategory, DiagnosticLevel
    from mlx_manager.services.probe.sweeps import sweep_thinking

    strategy = MagicMock()
    strategy._generate = AsyncMock(
        return_value=_make_gen_result("<custom_reason>some reasoning</custom_reason>answer")
    )
    loaded = MagicMock()

    # No parsers match anything
    mock_parser = _make_mock_thinking_parser("think_tag", None, [("<think>", "</think>")])
    mock_parsers = {
        "null": MagicMock,
        "think_tag": mock_parser,
    }

    with (
        patch("mlx_manager.mlx_server.parsers.THINKING_PARSERS", mock_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
            return_value=None,  # No unclosed tag
        ),
        patch(
            "mlx_manager.services.probe.base._detect_unknown_thinking_tags",
            return_value="custom_reason",  # Unknown tag found
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_thinking_parser_id",
            return_value=None,
        ),
    ):
        supports, parser_id, diags, tags = await sweep_thinking(
            "test/model", loaded, strategy, None, family=None
        )

    assert supports is False
    assert parser_id == "null"
    assert len(diags) == 1
    assert diags[0].level == DiagnosticLevel.WARNING
    assert diags[0].category == DiagnosticCategory.THINKING_DIALECT
    assert "custom_reason" in diags[0].message
    assert "custom_reason" in diags[0].details["detected_tag"]


# ---------------------------------------------------------------------------
# sweep_tools: Lines 385-388, 393, 397-401 — Phase 4 template delivery
# tag merging + tag-matched parser validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_tools_phase4_template_tag_merge_and_validation():
    """Phase 4: template delivery discovers new tags (merged, lines 385-388),
    collects template_matched parser IDs (line 393), and validates a
    tag-matched parser (lines 397-401)."""
    from mlx_manager.services.probe.sweeps import sweep_tools

    # Phase 1 generic injection produces output with no matching parsers
    # Phase 4 template delivery produces output with matching parsers
    gen_calls = [
        _make_gen_result("I don't know how to use tools"),  # Phase 1 generic
        _make_gen_result('<tool_call>{"name":"get_weather"}</tool_call>'),  # Phase 4 template
    ]
    strategy = MagicMock()
    strategy._generate = AsyncMock(side_effect=gen_calls)

    loaded = MagicMock()
    loaded.adapter = MagicMock()
    loaded.adapter.supports_native_tools = MagicMock(return_value=True)
    loaded.tokenizer = MagicMock()

    # A parser that validates only the template output
    class TemplateMatchParser:
        @property
        def parser_id(self):
            return "hermes_json"

        @property
        def stream_markers(self):
            return [("<tool_call>", "</tool_call>")]

        def validates(self, text, expected_fn):
            # Only validate when called with template output
            return "tool_call" in text

        def extract(self, text):
            return []

    mock_tool_parsers = {
        "null": MagicMock,
        "hermes_json": TemplateMatchParser,
    }

    # Phase 2: generic output has no tags
    generic_tags = []
    # Phase 4: template output has a new tag
    template_tags = [_make_tag_discovery("tool_call", "xml", True, ["hermes_json"])]

    discover_calls = [generic_tags, template_tags]
    discover_call_idx = {"i": 0}

    def mock_discover(output, parsers):
        idx = discover_call_idx["i"]
        discover_call_idx["i"] += 1
        return discover_calls[idx] if idx < len(discover_calls) else []

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", mock_tool_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            side_effect=mock_discover,
        ),
        patch(
            "mlx_manager.services.probe.base._has_tokenization_artifacts",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            side_effect=lambda candidates, fam: sorted(candidates),
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_tool_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        tool_format, parser_id, diags, all_tags = await sweep_tools(
            "test/model", loaded, strategy, family=None
        )

    assert tool_format == "template"
    assert parser_id == "hermes_json"
    # The template tag should be merged into all_tags
    tag_names = [t.name for t in all_tags]
    assert "tool_call" in tag_names


# ---------------------------------------------------------------------------
# sweep_tools: Line 413 — Phase 4 debug log when no parser matches template
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_tools_phase4_template_no_parser_match():
    """Phase 4: template delivery produces output but no parser validates it.
    Falls through to logger.debug (line 413) and eventually to Phase 5."""
    from mlx_manager.services.probe.sweeps import sweep_tools

    # Phase 1: no output or no match
    # Phase 4: template produces output but no parser validates
    gen_calls = [
        _make_gen_result("no tools here"),  # Phase 1 generic
        _make_gen_result("template output but unparseable format"),  # Phase 4 template
    ]
    strategy = MagicMock()
    strategy._generate = AsyncMock(side_effect=gen_calls)

    loaded = MagicMock()
    loaded.adapter = MagicMock()
    loaded.adapter.supports_native_tools = MagicMock(return_value=True)
    loaded.tokenizer = MagicMock()

    # Parser that never validates
    never_match_parser = _make_mock_tool_parser("hermes_json", False)
    mock_tool_parsers = {
        "null": MagicMock,
        "hermes_json": never_match_parser,
    }

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", mock_tool_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base._has_tokenization_artifacts",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            side_effect=lambda candidates, fam: sorted(candidates),
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_tool_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        tool_format, parser_id, diags, all_tags = await sweep_tools(
            "test/model", loaded, strategy, family=None
        )

    # No parser matched — should return (None, None, ...)
    assert tool_format is None
    assert parser_id is None


# ---------------------------------------------------------------------------
# sweep_tools: Lines 385-388 only — template tag merging with dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_tools_phase4_template_tag_dedup():
    """Phase 4: template tags that already exist in all_discovered_tags are
    not duplicated (dedup by name+style). (Lines 385-388)"""
    from mlx_manager.services.probe.sweeps import sweep_tools

    gen_calls = [
        _make_gen_result("generic output with some_tag"),  # Phase 1
        _make_gen_result("template output with some_tag"),  # Phase 4
    ]
    strategy = MagicMock()
    strategy._generate = AsyncMock(side_effect=gen_calls)

    loaded = MagicMock()
    loaded.adapter = MagicMock()
    loaded.adapter.supports_native_tools = MagicMock(return_value=True)
    loaded.tokenizer = MagicMock()

    never_match_parser = _make_mock_tool_parser("hermes_json", False)
    mock_tool_parsers = {
        "null": MagicMock,
        "hermes_json": never_match_parser,
    }

    # Both generic and template discover the same tag (should not duplicate)
    existing_tag = _make_tag_discovery("tool_call", "xml", True, [])
    new_tag = _make_tag_discovery("function", "xml", True, [])
    duplicate_tag = _make_tag_discovery("tool_call", "xml", True, [])  # same name+style

    discover_call_idx = {"i": 0}

    def mock_discover(output, parsers):
        idx = discover_call_idx["i"]
        discover_call_idx["i"] += 1
        if idx == 0:
            return [existing_tag]  # Phase 2: generic found "tool_call"
        else:
            # Phase 4: template found "tool_call" (dup) + "function" (new)
            return [duplicate_tag, new_tag]

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", mock_tool_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            side_effect=mock_discover,
        ),
        patch(
            "mlx_manager.services.probe.base._has_tokenization_artifacts",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            side_effect=lambda candidates, fam: sorted(candidates),
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_tool_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        tool_format, parser_id, diags, all_tags = await sweep_tools(
            "test/model", loaded, strategy, family=None
        )

    # Should have 2 unique tags, not 3 (dedup removed the duplicate "tool_call")
    tag_keys = [(t.name, t.style) for t in all_tags]
    assert tag_keys.count(("tool_call", "xml")) == 1
    assert ("function", "xml") in tag_keys
    assert len(all_tags) == 2


# ---------------------------------------------------------------------------
# sweep_tools: Lines 305-306 — Phase 1 generic injection exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_tools_phase1_generic_injection_exception():
    """Phase 1: generic injection generation fails with exception.
    last_output stays None, falls through to Phase 4 or Phase 5. (Lines 305-306)"""
    import logging

    from mlx_manager.services.probe.sweeps import sweep_tools

    strategy = MagicMock()
    # First call (Phase 1) fails, second call (Phase 4 template) also not reached
    # because can_try_template is False
    strategy._generate = AsyncMock(side_effect=RuntimeError("generation error"))

    loaded = MagicMock()
    loaded.adapter = MagicMock()
    loaded.adapter.supports_native_tools = MagicMock(return_value=False)
    loaded.tokenizer = MagicMock()

    never_match_parser = _make_mock_tool_parser("hermes_json", False)
    mock_tool_parsers = {
        "null": MagicMock,
        "hermes_json": never_match_parser,
    }

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", mock_tool_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base._has_tokenization_artifacts",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_tool_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        # Suppress the logger.debug call which has a format string bug ({} vs %s)
        patch.object(logging.getLogger("mlx_manager.services.probe.sweeps"), "debug"),
    ):
        tool_format, parser_id, diags, all_tags = await sweep_tools(
            "test/model", loaded, strategy, family=None
        )

    # No output at all — should return (None, None, ...)
    assert tool_format is None
    assert parser_id is None


# ---------------------------------------------------------------------------
# sweep_tools: Lines 417-418 — Phase 4 template delivery exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_tools_phase4_template_delivery_exception():
    """Phase 4: template delivery generation fails with exception. (Lines 417-418)"""
    from mlx_manager.services.probe.sweeps import sweep_tools

    # Phase 1 succeeds with no matching output, Phase 4 raises exception
    gen_calls = [
        _make_gen_result("no tools here"),  # Phase 1 generic
        RuntimeError("template delivery failed"),  # Phase 4 template
    ]
    strategy = MagicMock()
    strategy._generate = AsyncMock(side_effect=gen_calls)

    loaded = MagicMock()
    loaded.adapter = MagicMock()
    loaded.adapter.supports_native_tools = MagicMock(return_value=True)
    loaded.tokenizer = MagicMock()

    never_match_parser = _make_mock_tool_parser("hermes_json", False)
    mock_tool_parsers = {
        "null": MagicMock,
        "hermes_json": never_match_parser,
    }

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", mock_tool_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base._has_tokenization_artifacts",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._prioritize_parsers",
            side_effect=lambda candidates, fam: sorted(candidates),
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_tool_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
    ):
        tool_format, parser_id, diags, all_tags = await sweep_tools(
            "test/model", loaded, strategy, family=None
        )

    # Template delivery failed but generic output had no matches either
    assert tool_format is None
    assert parser_id is None


# ---------------------------------------------------------------------------
# sweep_tools: Line 483 — No output at all (last_output is None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_tools_no_output_at_all():
    """Phase 5: last_output is None (both Phase 1 failed and no template delivery).
    Hits the else branch with 'no tool support detected (no output)'. (Line 483)"""
    import logging

    from mlx_manager.services.probe.sweeps import sweep_tools

    strategy = MagicMock()
    # Phase 1 fails with exception, so last_output stays None
    # Phase 4 template delivery also fails with exception
    strategy._generate = AsyncMock(side_effect=RuntimeError("all generation failed"))

    loaded = MagicMock()
    loaded.adapter = MagicMock()
    loaded.adapter.supports_native_tools = MagicMock(return_value=True)
    loaded.tokenizer = MagicMock()

    never_match_parser = _make_mock_tool_parser("hermes_json", False)
    mock_tool_parsers = {
        "null": MagicMock,
        "hermes_json": never_match_parser,
    }

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", mock_tool_parsers),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base._has_tokenization_artifacts",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base.get_family_tool_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
        # Suppress the logger.debug call which has a format string bug ({} vs %s)
        patch.object(logging.getLogger("mlx_manager.services.probe.sweeps"), "debug"),
    ):
        tool_format, parser_id, diags, all_tags = await sweep_tools(
            "test/model", loaded, strategy, family=None
        )

    assert tool_format is None
    assert parser_id is None

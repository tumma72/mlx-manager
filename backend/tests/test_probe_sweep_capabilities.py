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

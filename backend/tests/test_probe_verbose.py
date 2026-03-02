"""Tests for verbose flag utilization in the probe pipeline.

TDD tests written BEFORE implementation. These verify:
- StepContext adds elapsed_ms to details when verbose=True
- StepContext does NOT add elapsed_ms when verbose=False (default)
- timing is added even for failed steps when verbose=True
- sweep_thinking includes raw output and parser trial details in diagnostics when verbose=True
- sweep_tools includes parser trial details in diagnostics when verbose=True
- ProbingCoordinator threads verbose flag through to probe_step calls
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.services.probe.steps import (
    StepContext,
    probe_step,
)

# ============================================================================
# StepContext timing
# ============================================================================


def test_probe_step_no_timing_by_default():
    """StepContext does not include elapsed_ms in details when verbose=False."""
    ctx = StepContext("check_tools", "supports_native_tools")
    ctx.value = True

    result = ctx.result

    # Default behavior: no timing info added
    assert result.details is None


def test_probe_step_no_timing_when_verbose_false():
    """StepContext with explicit verbose=False does not include elapsed_ms."""
    ctx = StepContext("check_tools", "supports_native_tools", verbose=False)
    ctx.value = True

    result = ctx.result

    assert result.details is None


def test_probe_step_timing_when_verbose():
    """StepContext with verbose=True includes elapsed_ms in details."""
    ctx = StepContext("check_tools", "supports_native_tools", verbose=True)
    ctx.value = True

    result = ctx.result

    assert result.details is not None
    assert "elapsed_ms" in result.details
    assert isinstance(result.details["elapsed_ms"], int)
    assert result.details["elapsed_ms"] >= 0


def test_probe_step_timing_preserves_existing_details():
    """When verbose=True and details already set, elapsed_ms is added to existing dict."""
    ctx = StepContext("check_tools", "supports_native_tools", verbose=True)
    ctx.value = True
    ctx.details = {"detection_method": "config", "architecture": "LlamaForCausalLM"}

    result = ctx.result

    assert result.details is not None
    assert "elapsed_ms" in result.details
    assert "detection_method" in result.details
    assert "architecture" in result.details
    assert result.details["detection_method"] == "config"


def test_probe_step_timing_on_failure_verbose():
    """When verbose=True, elapsed_ms is included even for failed steps."""
    ctx = StepContext("check_tools", verbose=True)
    ctx.fail("something went wrong")

    result = ctx.result

    assert result.status == "failed"
    assert result.error == "something went wrong"
    # Timing included even on failure
    assert result.details is not None
    assert "elapsed_ms" in result.details


def test_probe_step_no_timing_on_failure_default():
    """With default verbose=False, failed steps do NOT get elapsed_ms."""
    ctx = StepContext("check_tools")
    ctx.fail("something went wrong")

    result = ctx.result

    assert result.status == "failed"
    assert result.details is None


async def test_probe_step_context_manager_verbose_true():
    """probe_step context manager with verbose=True yields timing in result."""
    async with probe_step("check_tools", "supports_native_tools", verbose=True) as ctx:
        ctx.value = True

    result = ctx.result
    assert result.details is not None
    assert "elapsed_ms" in result.details
    assert isinstance(result.details["elapsed_ms"], int)


async def test_probe_step_context_manager_verbose_false_default():
    """probe_step context manager without verbose=True does not include timing."""
    async with probe_step("check_tools", "supports_native_tools") as ctx:
        ctx.value = True

    result = ctx.result
    assert result.details is None


async def test_probe_step_context_manager_verbose_exception():
    """probe_step with verbose=True includes timing even when exception is raised."""
    async with probe_step("check_tools", verbose=True) as ctx:
        raise ValueError("inference failed")

    result = ctx.result
    assert result.status == "failed"
    assert result.details is not None
    assert "elapsed_ms" in result.details


# ============================================================================
# sweep_thinking verbose diagnostics
# ============================================================================


@pytest.mark.asyncio
async def test_sweep_thinking_verbose_includes_raw_output():
    """sweep_thinking with verbose=True includes raw_output_sample in an INFO diagnostic."""
    from mlx_manager.services.probe.sweeps import sweep_thinking

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()

    mock_result = MagicMock()
    mock_result.content = (
        "Let me think... <think>3 foxes × 2 = 6 chickens eaten</think>"
        " 5 - 6 = -1... wait, -1 chickens, so 0 remain."
    )
    mock_result.reasoning_content = None

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value=mock_result)

    from mlx_manager.services.probe.steps import DiagnosticLevel

    # With no thinking parsers matching, we get no thinking support
    # But we want to verify that with verbose=True an INFO diagnostic is added
    with (
        patch(
            "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
            {},  # Empty parsers so none match
        ),
        patch(
            "mlx_manager.services.probe.base._discover_and_map_tags",
            return_value=[],
        ),
        patch(
            "mlx_manager.services.probe.base._find_unclosed_thinking_tag",
            return_value=None,
        ),
        patch(
            "mlx_manager.services.probe.base._detect_unknown_thinking_tags",
            return_value=None,
        ),
    ):
        supports, parser_id, diagnostics, tags = await sweep_thinking(
            "test/model",
            mock_loaded,
            mock_strategy,
            None,
            "default",
            verbose=True,
        )

    # When verbose=True, at least one diagnostic with raw output sample should appear
    raw_diags = [
        d
        for d in diagnostics
        if d.level == DiagnosticLevel.INFO and "raw_output_sample" in d.details
    ]
    assert len(raw_diags) >= 1
    assert raw_diags[0].details["raw_output_sample"] is not None


@pytest.mark.asyncio
async def test_sweep_thinking_verbose_false_no_raw_output():
    """sweep_thinking with verbose=False does NOT add raw output diagnostics."""
    from mlx_manager.services.probe.steps import DiagnosticLevel
    from mlx_manager.services.probe.sweeps import sweep_thinking

    mock_loaded = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Some output here"
    mock_result.reasoning_content = None

    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value=mock_result)

    with (
        patch("mlx_manager.mlx_server.parsers.THINKING_PARSERS", {}),
        patch("mlx_manager.services.probe.base._discover_and_map_tags", return_value=[]),
        patch("mlx_manager.services.probe.base._find_unclosed_thinking_tag", return_value=None),
        patch("mlx_manager.services.probe.base._detect_unknown_thinking_tags", return_value=None),
    ):
        supports, parser_id, diagnostics, tags = await sweep_thinking(
            "test/model",
            mock_loaded,
            mock_strategy,
            None,
            "default",
            # no verbose param → defaults to False
        )

    # No raw output diagnostics in non-verbose mode
    raw_diags = [
        d
        for d in diagnostics
        if d.level == DiagnosticLevel.INFO and "raw_output_sample" in d.details
    ]
    assert len(raw_diags) == 0


# ============================================================================
# sweep_tools verbose diagnostics
# ============================================================================


@pytest.mark.asyncio
async def test_sweep_tools_verbose_includes_parser_trials():
    """sweep_tools with verbose=True includes parser_trials in an INFO diagnostic."""
    from mlx_manager.services.probe.steps import DiagnosticLevel
    from mlx_manager.services.probe.sweeps import sweep_tools

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.adapter.supports_native_tools = MagicMock(return_value=False)
    mock_loaded.tokenizer = MagicMock()

    mock_result = MagicMock()
    mock_result.content = "The weather in Tokyo is sunny."
    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value=mock_result)

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", {}),
        patch("mlx_manager.services.probe.base._discover_and_map_tags", return_value=[]),
        patch("mlx_manager.services.probe.base._has_tokenization_artifacts", return_value=False),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diagnostics, tags = await sweep_tools(
            "test/model",
            mock_loaded,
            mock_strategy,
            "default",
            verbose=True,
        )

    # When verbose=True, at least one INFO diagnostic with parser_trials should appear
    trial_diags = [
        d for d in diagnostics if d.level == DiagnosticLevel.INFO and "parser_trials" in d.details
    ]
    assert len(trial_diags) >= 1


@pytest.mark.asyncio
async def test_sweep_tools_verbose_false_no_parser_trials():
    """sweep_tools with verbose=False does NOT add parser_trial diagnostics."""
    from mlx_manager.services.probe.steps import DiagnosticLevel
    from mlx_manager.services.probe.sweeps import sweep_tools

    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.adapter.supports_native_tools = MagicMock(return_value=False)
    mock_loaded.tokenizer = MagicMock()

    mock_result = MagicMock()
    mock_result.content = "The weather in Tokyo is sunny."
    mock_strategy = MagicMock()
    mock_strategy._generate = AsyncMock(return_value=mock_result)

    with (
        patch("mlx_manager.mlx_server.parsers.TOOL_PARSERS", {}),
        patch("mlx_manager.services.probe.base._discover_and_map_tags", return_value=[]),
        patch("mlx_manager.services.probe.base._has_tokenization_artifacts", return_value=False),
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
    ):
        tool_format, parser_id, diagnostics, tags = await sweep_tools(
            "test/model",
            mock_loaded,
            mock_strategy,
            "default",
            # no verbose → defaults to False
        )

    trial_diags = [
        d for d in diagnostics if d.level == DiagnosticLevel.INFO and "parser_trials" in d.details
    ]
    assert len(trial_diags) == 0


# ============================================================================
# coordinator verbose threading
# ============================================================================


@pytest.mark.asyncio
async def test_coordinator_passes_verbose_flag():
    """ProbingCoordinator.probe(verbose=True) passes verbose to sweep_capabilities."""
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

    sweep_calls: list[dict] = []

    async def fake_sweep(model_id, loaded, result, *, verbose=False):
        sweep_calls.append({"model_id": model_id, "verbose": verbose})
        return
        yield  # make async generator

    with patch.object(strategy, "sweep_capabilities", side_effect=fake_sweep):
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
                "mlx_manager.services.probe.strategy.get_probe_strategy",
                return_value=strategy,
            ),
            patch.object(mock_pool, "get_model", new=AsyncMock(return_value=mock_loaded)),
            patch(
                "mlx_manager.services.probe.coordinator._save_capabilities",
                new=AsyncMock(),
            ),
        ):
            async for _ in coordinator.probe("test/model", verbose=True):
                pass

    # sweep_capabilities should have been called with verbose=True
    assert len(sweep_calls) == 1
    assert sweep_calls[0]["verbose"] is True


@pytest.mark.asyncio
async def test_coordinator_passes_verbose_false_by_default():
    """ProbingCoordinator.probe() passes verbose=False by default to sweep_capabilities."""
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

    sweep_calls: list[dict] = []

    async def fake_sweep(model_id, loaded, result, *, verbose=False):
        sweep_calls.append({"model_id": model_id, "verbose": verbose})
        return
        yield

    with patch.object(strategy, "sweep_capabilities", side_effect=fake_sweep):
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
                "mlx_manager.services.probe.strategy.get_probe_strategy",
                return_value=strategy,
            ),
            patch.object(mock_pool, "get_model", new=AsyncMock(return_value=mock_loaded)),
            patch(
                "mlx_manager.services.probe.coordinator._save_capabilities",
                new=AsyncMock(),
            ),
        ):
            async for _ in coordinator.probe("test/model"):  # No verbose= → default False
                pass

    assert len(sweep_calls) == 1
    assert sweep_calls[0]["verbose"] is False


# ============================================================================
# Backward compatibility: existing tests still pass
# ============================================================================


def test_step_context_backward_compat_no_verbose():
    """StepContext still works identically when verbose is not passed (backward compat)."""
    ctx = StepContext("check_tools", "supports_native_tools")
    ctx.value = True
    ctx.details = {"some": "detail"}

    result = ctx.result

    # No elapsed_ms added — backward compat
    assert result.status == "completed"
    assert result.details == {"some": "detail"}
    assert "elapsed_ms" not in result.details


async def test_probe_step_backward_compat():
    """probe_step without verbose param works identically to before (backward compat)."""
    async with probe_step("check_tools", "supports_native_tools") as ctx:
        ctx.value = True

    result = ctx.result
    assert result.status == "completed"
    assert result.value is True
    assert result.details is None

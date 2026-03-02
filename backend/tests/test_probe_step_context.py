"""Tests for StepContext and probe_step async context manager.

Tests the step lifecycle helpers added to mlx_manager.services.probe.steps
to eliminate the repetitive try/yield-running/yield-completed/except/yield-failed
boilerplate in probe strategies.
"""

import json

from mlx_manager.services.probe.steps import (
    DiagnosticCategory,
    DiagnosticLevel,
    ProbeDiagnostic,
    ProbeStep,
    StepContext,
    probe_step,
)

# ============================================================================
# StepContext basics
# ============================================================================


def test_step_context_running_returns_running_step():
    """ctx.running produces ProbeStep(step=name, status='running')."""
    ctx = StepContext("check_tools")

    result = ctx.running

    assert isinstance(result, ProbeStep)
    assert result.step == "check_tools"
    assert result.status == "running"


def test_step_context_result_completed():
    """After setting ctx.value, ctx.result produces completed ProbeStep."""
    ctx = StepContext("check_tools", capability="supports_native_tools")
    ctx.value = True

    result = ctx.result

    assert isinstance(result, ProbeStep)
    assert result.step == "check_tools"
    assert result.status == "completed"
    assert result.capability == "supports_native_tools"
    assert result.value is True
    assert result.error is None


def test_step_context_result_failed():
    """After ctx.fail('msg'), ctx.result produces failed ProbeStep with error."""
    ctx = StepContext("check_tools", capability="supports_native_tools")
    ctx.fail("tool call detection failed")

    result = ctx.result

    assert isinstance(result, ProbeStep)
    assert result.step == "check_tools"
    assert result.status == "failed"
    assert result.error == "tool call detection failed"
    assert result.capability is None
    assert result.value is None


def test_step_context_result_with_details():
    """ctx.details and ctx.diagnostics appear in ctx.result."""
    ctx = StepContext("check_thinking", capability="supports_thinking")
    ctx.value = True
    ctx.details = {"token": "<think>"}
    diag = ProbeDiagnostic(
        level=DiagnosticLevel.INFO,
        category=DiagnosticCategory.THINKING_DIALECT,
        message="Thinking dialect detected",
    )
    ctx.diagnostics = [diag]

    result = ctx.result

    assert result.details == {"token": "<think>"}
    assert result.diagnostics == [diag]
    assert result.status == "completed"


def test_step_context_no_capability():
    """When capability=None, result has capability=None."""
    ctx = StepContext("check_type")
    ctx.value = "lm"

    result = ctx.result

    assert result.capability is None
    assert result.value == "lm"
    assert result.status == "completed"


def test_step_context_default_state():
    """Fresh context has value=None, not failed."""
    ctx = StepContext("check_something", capability="some_cap")

    assert ctx.value is None
    assert ctx._failed is False
    assert ctx._error is None

    result = ctx.result
    assert result.status == "completed"
    assert result.value is None


# ============================================================================
# probe_step context manager
# ============================================================================


async def test_probe_step_success():
    """Normal exit produces completed result."""
    async with probe_step("check_tools", "supports_native_tools") as ctx:
        ctx.value = True

    result = ctx.result
    assert result.status == "completed"
    assert result.value is True
    assert result.capability == "supports_native_tools"


async def test_probe_step_catches_exception():
    """Exception inside 'async with' sets failed state."""
    async with probe_step("check_tools", "supports_native_tools") as ctx:
        raise ValueError("Something went wrong")

    result = ctx.result
    assert result.status == "failed"
    assert result.error == "Something went wrong"


async def test_probe_step_explicit_fail():
    """Calling ctx.fail() inside block produces failed result."""
    async with probe_step("check_tools", "supports_native_tools") as ctx:
        ctx.fail("boolean check failed")

    result = ctx.result
    assert result.status == "failed"
    assert result.error == "boolean check failed"


async def test_probe_step_exception_preserves_error_message():
    """The exception's str() is captured as the error message."""
    async with probe_step("check_thinking") as ctx:
        raise RuntimeError("model returned unexpected output: <|fim|>")

    assert ctx.result.error == "model returned unexpected output: <|fim|>"


async def test_probe_step_exception_does_not_propagate():
    """Exceptions inside probe_step are caught and do not propagate."""
    raised = False
    try:
        async with probe_step("check_something") as ctx:
            raise Exception("internal error")
    except Exception:
        raised = True

    assert not raised, "Exception should have been caught by probe_step"
    assert ctx.result.status == "failed"


async def test_probe_step_in_async_generator():
    """Full integration: used inside an async generator that yields running/result."""

    async def run_probe():
        async with probe_step("check_tools", "supports_native_tools") as ctx:
            yield ctx.running  # Emit "running" step
            # Simulate work
            ctx.value = True
        yield ctx.result  # Emit "completed" step

    steps = [step async for step in run_probe()]

    assert len(steps) == 2
    assert steps[0].status == "running"
    assert steps[0].step == "check_tools"
    assert steps[1].status == "completed"
    assert steps[1].value is True
    assert steps[1].capability == "supports_native_tools"


async def test_probe_step_in_async_generator_with_exception():
    """Integration: exception in async generator body is caught by probe_step."""

    async def run_probe():
        async with probe_step("check_tools", "supports_native_tools") as ctx:
            yield ctx.running
            raise RuntimeError("inference failed")
        yield ctx.result

    steps = [step async for step in run_probe()]

    assert len(steps) == 2
    assert steps[0].status == "running"
    assert steps[1].status == "failed"
    assert steps[1].error == "inference failed"


# ============================================================================
# Integration with ProbeStep.to_sse()
# ============================================================================


async def test_step_context_result_to_sse():
    """Verify that ctx.result.to_sse() produces correct JSON."""
    async with probe_step("check_tools", "supports_native_tools") as ctx:
        ctx.value = True

    sse_output = ctx.result.to_sse()

    # Should be formatted as SSE data event
    assert sse_output.startswith("data: ")
    assert sse_output.endswith("\n\n")

    # Parse the JSON payload
    json_str = sse_output[len("data: ") :].strip()
    data = json.loads(json_str)

    assert data["step"] == "check_tools"
    assert data["status"] == "completed"
    assert data["capability"] == "supports_native_tools"
    assert data["value"] is True


async def test_step_context_failed_result_to_sse():
    """Verify that failed ctx.result.to_sse() produces correct JSON."""
    async with probe_step("check_thinking") as ctx:
        raise ValueError("thinking not detected")

    sse_output = ctx.result.to_sse()

    json_str = sse_output[len("data: ") :].strip()
    data = json.loads(json_str)

    assert data["step"] == "check_thinking"
    assert data["status"] == "failed"
    assert data["error"] == "thinking not detected"
    # No capability, value, or details in failed result
    assert "capability" not in data
    assert "value" not in data


# ============================================================================
# Public exports verification
# ============================================================================


def test_step_context_importable_from_probe_package():
    """StepContext and probe_step are importable from the probe package."""
    from mlx_manager.services.probe import StepContext as ImportedStepContext
    from mlx_manager.services.probe import probe_step as ps

    assert ImportedStepContext is StepContext
    assert ps is probe_step

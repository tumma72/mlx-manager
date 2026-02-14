"""Tests for probe diagnostic models, serialization, and report generation."""

from __future__ import annotations

import json

from mlx_manager.services.probe.report import generate_support_report
from mlx_manager.services.probe.steps import (
    DiagnosticCategory,
    DiagnosticLevel,
    ProbeDiagnostic,
    ProbeResult,
    ProbeStep,
)

# ---------------------------------------------------------------------------
# ProbeDiagnostic model tests
# ---------------------------------------------------------------------------


def test_probe_diagnostic_creation():
    """Test ProbeDiagnostic can be created with all fields."""
    diag = ProbeDiagnostic(
        level=DiagnosticLevel.WARNING,
        category=DiagnosticCategory.FAMILY,
        message="No dedicated adapter",
        details={"architecture": "SomeArch"},
    )
    assert diag.level == DiagnosticLevel.WARNING
    assert diag.category == DiagnosticCategory.FAMILY
    assert diag.message == "No dedicated adapter"
    assert diag.details["architecture"] == "SomeArch"


def test_probe_diagnostic_default_details():
    """Test ProbeDiagnostic has empty dict as default details."""
    diag = ProbeDiagnostic(
        level=DiagnosticLevel.INFO,
        category=DiagnosticCategory.TYPE,
        message="Test message",
    )
    assert diag.details == {}


def test_probe_diagnostic_model_dump():
    """Test ProbeDiagnostic serialization via model_dump."""
    diag = ProbeDiagnostic(
        level=DiagnosticLevel.ACTION_NEEDED,
        category=DiagnosticCategory.TOOL_DIALECT,
        message="Unknown tool dialect",
        details={"found_markers": ["<tool_call>"]},
    )
    data = diag.model_dump()
    assert data["level"] == "action_needed"
    assert data["category"] == "tool_dialect"
    assert data["message"] == "Unknown tool dialect"
    assert data["details"]["found_markers"] == ["<tool_call>"]


def test_diagnostic_level_values():
    """Test all diagnostic level enum values."""
    assert DiagnosticLevel.INFO.value == "info"
    assert DiagnosticLevel.WARNING.value == "warning"
    assert DiagnosticLevel.ACTION_NEEDED.value == "action_needed"


def test_diagnostic_category_values():
    """Test all diagnostic category enum values."""
    assert DiagnosticCategory.FAMILY.value == "family"
    assert DiagnosticCategory.TOOL_DIALECT.value == "tool_dialect"
    assert DiagnosticCategory.THINKING_DIALECT.value == "thinking_dialect"
    assert DiagnosticCategory.TYPE.value == "type"
    assert DiagnosticCategory.UNSUPPORTED.value == "unsupported"


# ---------------------------------------------------------------------------
# ProbeStep with diagnostics tests
# ---------------------------------------------------------------------------


def test_probe_step_without_diagnostics():
    """Test ProbeStep serialization without diagnostics."""
    step = ProbeStep(step="detect_type", status="completed")
    assert step.diagnostics is None
    data = json.loads(step.to_sse().removeprefix("data: ").strip())
    assert "diagnostics" not in data


def test_probe_step_with_diagnostics():
    """Test ProbeStep serialization with diagnostics."""
    diag = ProbeDiagnostic(
        level=DiagnosticLevel.WARNING,
        category=DiagnosticCategory.TYPE,
        message="Type defaulted",
        details={"detection_method": "default"},
    )
    step = ProbeStep(
        step="detect_type",
        status="completed",
        diagnostics=[diag],
    )
    sse = step.to_sse()
    data = json.loads(sse.removeprefix("data: ").strip())
    assert "diagnostics" in data
    assert len(data["diagnostics"]) == 1
    assert data["diagnostics"][0]["level"] == "warning"
    assert data["diagnostics"][0]["category"] == "type"
    assert data["diagnostics"][0]["message"] == "Type defaulted"


def test_probe_step_empty_diagnostics_not_serialized():
    """Test that empty diagnostics list is not included in SSE."""
    step = ProbeStep(step="test_tools", status="completed", diagnostics=[])
    sse = step.to_sse()
    data = json.loads(sse.removeprefix("data: ").strip())
    # Empty list is falsy, so should not be included
    assert "diagnostics" not in data


# ---------------------------------------------------------------------------
# ProbeResult diagnostics accumulation tests
# ---------------------------------------------------------------------------


def test_probe_result_default_diagnostics():
    """Test ProbeResult has empty diagnostics list by default."""
    result = ProbeResult()
    assert result.diagnostics == []


def test_probe_result_accumulate_diagnostics():
    """Test diagnostics can be accumulated on ProbeResult."""
    result = ProbeResult()
    diag1 = ProbeDiagnostic(
        level=DiagnosticLevel.WARNING,
        category=DiagnosticCategory.FAMILY,
        message="No adapter",
    )
    diag2 = ProbeDiagnostic(
        level=DiagnosticLevel.ACTION_NEEDED,
        category=DiagnosticCategory.TOOL_DIALECT,
        message="Unknown dialect",
    )
    result.diagnostics.append(diag1)
    result.diagnostics.append(diag2)
    assert len(result.diagnostics) == 2


def test_probe_result_diagnostics_in_model_dump():
    """Test diagnostics are serialized in ProbeResult.model_dump()."""
    result = ProbeResult(model_type="text_gen")
    result.diagnostics.append(
        ProbeDiagnostic(
            level=DiagnosticLevel.INFO,
            category=DiagnosticCategory.TYPE,
            message="test",
        )
    )
    data = result.model_dump()
    assert "diagnostics" in data
    assert len(data["diagnostics"]) == 1
    assert data["diagnostics"][0]["level"] == "info"


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


def test_generate_report_basic():
    """Test basic report generation structure."""
    result = ProbeResult(model_type="text_gen", model_family="qwen")
    steps = [
        ProbeStep(
            step="detect_type",
            status="completed",
            capability="model_type",
            value="text_gen",
            details={"detection_method": "architecture", "architecture": "Qwen2ForCausalLM"},
        ),
        ProbeStep(step="load_model", status="completed"),
    ]
    report = generate_support_report("mlx-community/Qwen3-0.6B-4bit-DWQ", result, steps)

    assert "# Probe Diagnostic Report" in report
    assert "mlx-community/Qwen3-0.6B-4bit-DWQ" in report
    assert "text_gen" in report
    assert "qwen" in report
    assert "Qwen2ForCausalLM" in report
    assert "architecture" in report.lower()


def test_generate_report_with_diagnostics():
    """Test report includes diagnostics section when present."""
    result = ProbeResult(model_type="text_gen", model_family="default")
    result.diagnostics.append(
        ProbeDiagnostic(
            level=DiagnosticLevel.WARNING,
            category=DiagnosticCategory.FAMILY,
            message="No dedicated adapter",
            details={"architecture": "UnknownArch"},
        )
    )
    result.diagnostics.append(
        ProbeDiagnostic(
            level=DiagnosticLevel.ACTION_NEEDED,
            category=DiagnosticCategory.TOOL_DIALECT,
            message="Unknown tool dialect",
            details={
                "found_markers": ["<tool_call>"],
                "raw_output_sample": "<tool_call>test</tool_call>",
            },
        )
    )
    steps = [
        ProbeStep(step="detect_type", status="completed"),
    ]
    report = generate_support_report("test/model", result, steps)

    assert "## Diagnostics" in report
    assert "### Action Needed" in report
    assert "### Warnings" in report
    assert "Unknown tool dialect" in report
    assert "No dedicated adapter" in report
    assert "## Raw Output Samples" in report


def test_generate_report_without_diagnostics():
    """Test report omits diagnostics section when no diagnostics present."""
    result = ProbeResult(model_type="text_gen", model_family="qwen")
    steps = [
        ProbeStep(step="detect_type", status="completed"),
    ]
    report = generate_support_report("test/model", result, steps)

    assert "## Diagnostics" not in report


def test_generate_report_environment_section():
    """Test report includes environment info."""
    result = ProbeResult()
    steps = []
    report = generate_support_report("test/model", result, steps)

    assert "## Environment" in report
    assert "OS" in report
    assert "Python" in report


def test_generate_report_capabilities():
    """Test report shows detected capabilities."""
    result = ProbeResult(
        model_type="text_gen",
        supports_native_tools=True,
        supports_thinking=True,
        tool_format="template",
    )
    steps = []
    report = generate_support_report("test/model", result, steps)

    assert "Native Tools" in report
    assert "Thinking/Reasoning" in report
    assert "template" in report


def test_generate_report_skips_probe_complete_step():
    """Test that probe_complete step is not shown in report step list."""
    result = ProbeResult(model_type="text_gen")
    steps = [
        ProbeStep(step="detect_type", status="completed"),
        ProbeStep(step="probe_complete", status="completed"),
    ]
    report = generate_support_report("test/model", result, steps)

    assert "detect_type" in report
    assert "probe_complete" not in report

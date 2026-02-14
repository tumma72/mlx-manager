"""Probe diagnostic report generation.

Produces a markdown report suitable for GitHub issues when a model
has unresolved diagnostics (unknown family, unrecognized tool dialect, etc.).
"""

from __future__ import annotations

import platform
from typing import Any

from .steps import DiagnosticLevel, ProbeDiagnostic, ProbeResult, ProbeStep


def generate_support_report(
    model_id: str,
    result: ProbeResult,
    steps: list[ProbeStep],
) -> str:
    """Generate a markdown diagnostic report for a probe run.

    The report includes model information, detected capabilities,
    diagnostics with details, raw output samples, and environment info.
    Suitable for pasting into a GitHub issue.

    Args:
        model_id: HuggingFace model ID that was probed
        result: Accumulated probe results
        steps: All probe steps from the run

    Returns:
        Markdown-formatted report string
    """
    sections: list[str] = []

    # Header
    sections.append(f"# Probe Diagnostic Report: `{model_id}`\n")

    # Model info
    sections.append("## Model Information\n")
    sections.append(f"- **Model ID:** `{model_id}`")
    sections.append(f"- **Detected Type:** {result.model_type or 'unknown'}")
    sections.append(f"- **Model Family:** {result.model_family or 'unknown'}")

    # Architecture from detect_type step
    architecture = _extract_architecture(steps)
    if architecture:
        sections.append(f"- **Architecture:** `{architecture}`")

    detection_method = _extract_detection_method(steps)
    if detection_method:
        sections.append(f"- **Detection Method:** {detection_method}")
    sections.append("")

    # Capabilities summary
    sections.append("## Detected Capabilities\n")
    caps = _format_capabilities(result)
    if caps:
        for key, value in caps:
            sections.append(f"- **{key}:** {value}")
    else:
        sections.append("- No capabilities detected")
    sections.append("")

    # Diagnostics
    diagnostics = result.diagnostics
    if diagnostics:
        sections.append("## Diagnostics\n")
        action_needed = [d for d in diagnostics if d.level == DiagnosticLevel.ACTION_NEEDED]
        warnings = [d for d in diagnostics if d.level == DiagnosticLevel.WARNING]
        infos = [d for d in diagnostics if d.level == DiagnosticLevel.INFO]

        if action_needed:
            sections.append("### Action Needed\n")
            for d in action_needed:
                sections.append(f"- **[{d.category.value}]** {d.message}")
                _append_details(sections, d)

        if warnings:
            sections.append("### Warnings\n")
            for d in warnings:
                sections.append(f"- **[{d.category.value}]** {d.message}")
                _append_details(sections, d)

        if infos:
            sections.append("### Info\n")
            for d in infos:
                sections.append(f"- **[{d.category.value}]** {d.message}")
        sections.append("")

    # Raw output samples from diagnostics
    raw_samples = _extract_raw_samples(diagnostics)
    if raw_samples:
        sections.append("## Raw Output Samples\n")
        for label, sample in raw_samples:
            sections.append(f"### {label}\n")
            sections.append(f"```\n{sample}\n```\n")

    # Probe steps summary
    sections.append("## Probe Steps\n")
    for step in steps:
        if step.step == "probe_complete" or step.status == "running":
            continue
        icon = {"completed": "pass", "failed": "FAIL", "skipped": "skip"}.get(
            step.status, step.status
        )
        detail = ""
        if step.capability and step.value is not None:
            detail = f" = {step.value}"
        if step.error:
            detail = f" - {step.error}"
        sections.append(f"- [{icon}] `{step.step}`{detail}")
    sections.append("")

    # Environment
    sections.append("## Environment\n")
    env_info = _get_environment_info()
    for key, value in env_info:
        sections.append(f"- **{key}:** {value}")
    sections.append("")

    return "\n".join(sections)


def _extract_architecture(steps: list[ProbeStep]) -> str:
    """Extract architecture from detect_type step details."""
    for step in steps:
        if step.step == "detect_type" and step.details:
            return str(step.details.get("architecture", ""))
    return ""


def _extract_detection_method(steps: list[ProbeStep]) -> str:
    """Extract detection method from detect_type step details."""
    for step in steps:
        if step.step == "detect_type" and step.details:
            return str(step.details.get("detection_method", ""))
    return ""


def _format_capabilities(result: ProbeResult) -> list[tuple[str, Any]]:
    """Format non-None capabilities as (label, value) pairs."""
    caps: list[tuple[str, Any]] = []
    field_labels = {
        "supports_native_tools": "Native Tools",
        "supports_thinking": "Thinking/Reasoning",
        "tool_format": "Tool Format",
        "tool_parser_id": "Tool Parser",
        "thinking_parser_id": "Thinking Parser",
        "practical_max_tokens": "Max Tokens",
        "supports_multi_image": "Multi-Image",
        "supports_video": "Video",
        "embedding_dimensions": "Embedding Dimensions",
        "max_sequence_length": "Max Sequence Length",
        "is_normalized": "Normalized",
        "supports_tts": "Text-to-Speech",
        "supports_stt": "Speech-to-Text",
    }
    for field, label in field_labels.items():
        value = getattr(result, field, None)
        if value is not None:
            caps.append((label, value))
    return caps


def _append_details(sections: list[str], diagnostic: ProbeDiagnostic) -> None:
    """Append diagnostic details as indented sub-items."""
    for key, value in diagnostic.details.items():
        if key == "raw_output_sample":
            continue  # Handled separately
        sections.append(f"  - {key}: `{value}`")


def _extract_raw_samples(diagnostics: list[ProbeDiagnostic]) -> list[tuple[str, str]]:
    """Extract raw output samples from diagnostics."""
    samples: list[tuple[str, str]] = []
    for d in diagnostics:
        raw = d.details.get("raw_output_sample")
        if raw:
            samples.append((f"{d.category.value} diagnostic", raw))
    return samples


def _get_environment_info() -> list[tuple[str, str]]:
    """Gather environment information for the report."""
    info: list[tuple[str, str]] = []

    info.append(("OS", f"{platform.system()} {platform.release()}"))
    info.append(("Python", platform.python_version()))
    info.append(("Architecture", platform.machine()))

    # MLX library versions
    for lib_name in ("mlx", "mlx_lm", "mlx_vlm", "mlx_embeddings", "mlx_audio"):
        try:
            import importlib.metadata

            version = importlib.metadata.version(lib_name.replace("_", "-"))
            info.append((lib_name, version))
        except Exception:
            pass

    return info

"""Tests for include_report query parameter in the probe API endpoint.

TDD tests written BEFORE implementation (RED phase).

Verifies:
- SSE stream does NOT include probe_report event by default (include_report=False)
- SSE stream DOES include probe_report event with markdown when include_report=true
- The report event has proper markdown content (contains model_id and headings)
- The report event appears BEFORE the [DONE] sentinel
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

from mlx_manager.services.auth_service import create_access_token
from mlx_manager.services.probe.steps import ProbeResult, ProbeStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_token() -> str:
    """Generate a JWT token for SSE endpoint tests (query-param auth)."""
    return create_access_token(data={"sub": "test@example.com"})


def _make_probe_steps(model_id: str = "test/model") -> list[ProbeStep]:
    """Build a minimal list of ProbeStep objects as a mock probe run would yield."""
    result = ProbeResult(
        model_type="text_gen",
        model_family="default",
        supports_native_tools=False,
        supports_thinking=False,
    )
    return [
        ProbeStep(step="load_model", status="completed"),
        ProbeStep(
            step="detect_type",
            status="completed",
            capability="model_type",
            value="text_gen",
        ),
        ProbeStep(
            step="probe_complete",
            status="completed",
            details={"result": result.model_dump()},
        ),
    ]


async def _fake_probe_model(model_id: str, **kwargs) -> AsyncGenerator[ProbeStep, None]:
    """Async generator that yields minimal steps like probe_model()."""
    for step in _make_probe_steps(model_id):
        yield step


def _parse_sse_events(content: str) -> list[dict]:
    """Parse SSE content string into a list of decoded JSON event dicts.

    Handles the ``data: {...}`` SSE format.  Skips the ``data: [DONE]`` sentinel.
    """
    events = []
    for line in content.splitlines():
        if line.startswith("data: ") and not line.startswith("data: [DONE]"):
            payload = line[len("data: "):]
            events.append(json.loads(payload))
    return events


# The router uses a lazy import:
#   from mlx_manager.services.probe import probe_model
# so we patch at the package level where that name is exported.
_PROBE_MODEL_PATCH = "mlx_manager.services.probe.probe_model"


# ---------------------------------------------------------------------------
# Test 1: No probe_report event by default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_endpoint_no_report_by_default(auth_client):
    """SSE stream must NOT include a probe_report event when include_report is not set."""
    model_id = "test/model"
    token = _sse_token()

    with patch(_PROBE_MODEL_PATCH, side_effect=_fake_probe_model):
        response = await auth_client.post(
            f"/api/models/probe/{model_id}?token={token}",
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse_events(response.text)
    step_names = [e["step"] for e in events]
    assert "probe_report" not in step_names, (
        f"probe_report event must not appear by default, got steps: {step_names}"
    )


# ---------------------------------------------------------------------------
# Test 2: probe_report event present when include_report=true
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_endpoint_with_include_report(auth_client):
    """SSE stream must include a probe_report event when include_report=true."""
    model_id = "test/model"
    token = _sse_token()

    with patch(_PROBE_MODEL_PATCH, side_effect=_fake_probe_model):
        response = await auth_client.post(
            f"/api/models/probe/{model_id}?token={token}&include_report=true",
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse_events(response.text)
    step_names = [e["step"] for e in events]
    assert "probe_report" in step_names, (
        f"probe_report event must appear when include_report=true, got steps: {step_names}"
    )


# ---------------------------------------------------------------------------
# Test 3: Report event contains meaningful markdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_report_event_contains_markdown(auth_client):
    """The probe_report event must carry a markdown string with recognisable content."""
    model_id = "mlx-community/some-model"
    token = _sse_token()

    async def fake_probe(mid: str, **kwargs) -> AsyncGenerator[ProbeStep, None]:
        for step in _make_probe_steps(mid):
            yield step

    with patch(_PROBE_MODEL_PATCH, side_effect=fake_probe):
        response = await auth_client.post(
            f"/api/models/probe/{model_id}?token={token}&include_report=true",
        )

    events = _parse_sse_events(response.text)
    report_events = [e for e in events if e.get("step") == "probe_report"]
    assert len(report_events) == 1, "Expected exactly one probe_report event"

    report_event = report_events[0]
    assert report_event.get("status") == "completed"
    assert "details" in report_event
    assert "report" in report_event["details"]

    report_md: str = report_event["details"]["report"]
    assert isinstance(report_md, str)
    assert len(report_md) > 0, "Report markdown must not be empty"

    # Must contain a recognisable section heading
    assert "## " in report_md or "# " in report_md, (
        "Report must contain markdown headings"
    )
    # Must reference the probed model
    assert "some-model" in report_md or model_id in report_md, (
        f"Report should reference the model ID '{model_id}'"
    )


# ---------------------------------------------------------------------------
# Test 4: Report event appears BEFORE [DONE] sentinel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_report_event_before_done(auth_client):
    """The probe_report SSE event must appear before the [DONE] sentinel."""
    model_id = "test/model"
    token = _sse_token()

    with patch(_PROBE_MODEL_PATCH, side_effect=_fake_probe_model):
        response = await auth_client.post(
            f"/api/models/probe/{model_id}?token={token}&include_report=true",
        )

    # Split raw SSE text into non-empty lines preserving order
    lines = [line for line in response.text.splitlines() if line.strip()]

    done_index = next(
        (i for i, line in enumerate(lines) if "data: [DONE]" in line),
        None,
    )
    assert done_index is not None, "Stream must end with [DONE]"

    report_index = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("data: ") and '"probe_report"' in line
        ),
        None,
    )
    assert report_index is not None, "probe_report event must be present in stream"
    assert report_index < done_index, (
        f"probe_report (line {report_index}) must appear before [DONE] (line {done_index})"
    )


# ---------------------------------------------------------------------------
# Test 5: include_report=false is treated as not including the report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_endpoint_include_report_false(auth_client):
    """include_report=false must behave identically to not passing the parameter."""
    model_id = "test/model"
    token = _sse_token()

    with patch(_PROBE_MODEL_PATCH, side_effect=_fake_probe_model):
        response = await auth_client.post(
            f"/api/models/probe/{model_id}?token={token}&include_report=false",
        )

    events = _parse_sse_events(response.text)
    step_names = [e["step"] for e in events]
    assert "probe_report" not in step_names, (
        f"probe_report must not appear when include_report=false, got: {step_names}"
    )

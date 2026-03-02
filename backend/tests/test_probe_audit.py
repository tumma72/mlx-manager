"""Tests for --audit flag on the probe CLI command (Family-Parser Audit).

TDD tests written BEFORE implementation (RED phase).

Verifies:
- _audit_parsers() prints "No cached models" when scan_cache_dir returns no repos
- _audit_parsers() shows family, parser IDs, and stored parsers in a Rich table
- _audit_parsers() detects MISMATCH when stored parser != family parser
- _audit_parsers() skips models without config.json
- probe --audit flag skips normal probe flow and calls _audit_parsers()
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: mock HF cache repo
# ---------------------------------------------------------------------------


def _make_repo(repo_id: str) -> MagicMock:
    """Build a minimal mock HF cache repo object."""
    repo = MagicMock()
    repo.repo_id = repo_id
    return repo


def _make_cache_info(*repo_ids: str) -> MagicMock:
    """Build a mock HF scan_cache_dir result."""
    cache = MagicMock()
    cache.repos = [_make_repo(r) for r in repo_ids]
    return cache


# ---------------------------------------------------------------------------
# Test 1: empty cache prints advisory message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_with_no_cached_models():
    """_audit_parsers() prints 'No cached models' when no repos are in cache."""
    from mlx_manager.cli import _audit_parsers

    with (
        patch("mlx_manager.cli._init_probe_runtime", new=AsyncMock()),
        patch(
            "mlx_manager.cli.scan_cache_dir",
            return_value=_make_cache_info(),
        ),
        patch.object(
            __import__("mlx_manager.cli", fromlist=["console"]).console,
            "print",
        ) as mock_print,
    ):
        await _audit_parsers()

    # Verify advisory message was printed
    printed_args = [str(a) for call_ in mock_print.call_args_list for a in call_.args]
    assert any("no cached" in arg.lower() or "no models" in arg.lower() for arg in printed_args)


# ---------------------------------------------------------------------------
# Test 2: table shows family and parser columns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_shows_family_and_parsers():
    """_audit_parsers() outputs a Rich Table with Model, Family, parser columns."""
    from mlx_manager.cli import _audit_parsers

    model_id = "mlx-community/Qwen3-0.6B-4bit-DWQ"

    printed_objects: list = []

    def capture_print(*args, **kwargs):
        printed_objects.extend(args)

    with (
        patch("mlx_manager.cli._init_probe_runtime", new=AsyncMock()),
        patch(
            "mlx_manager.cli.scan_cache_dir",
            return_value=_make_cache_info(model_id),
        ),
        patch(
            "mlx_manager.cli.read_model_config",
            return_value={"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        ),
        patch(
            "mlx_manager.cli.detect_model_type",
            return_value=MagicMock(value="text_gen"),
        ),
        patch(
            "mlx_manager.cli.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.cli.get_family_tool_parser_id",
            return_value="hermes_json",
        ),
        patch(
            "mlx_manager.cli.get_family_thinking_parser_id",
            return_value="think_tag",
        ),
        patch(
            "mlx_manager.cli._get_stored_parsers",
            new=AsyncMock(return_value=(None, None)),
        ),
        patch.object(
            __import__("mlx_manager.cli", fromlist=["console"]).console,
            "print",
            side_effect=capture_print,
        ),
    ):
        await _audit_parsers()

    # At least one call must include a Rich Table object
    from rich.table import Table

    tables = [obj for obj in printed_objects if isinstance(obj, Table)]
    assert len(tables) == 1, "Expected exactly one Rich Table to be printed"

    table = tables[0]
    # Verify columns include key headings
    column_headers = [col.header for col in table.columns]
    assert "Model" in column_headers
    assert "Family" in column_headers


# ---------------------------------------------------------------------------
# Test 3: MISMATCH detected when stored parser != family parser
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_detects_mismatch():
    """_audit_parsers() marks row as MISMATCH when stored tool parser differs."""
    from mlx_manager.cli import _audit_parsers

    model_id = "mlx-community/Qwen3-0.6B-4bit-DWQ"

    # Stored parser says "llama_xml" but family config says "hermes_json"
    stored_tool = "llama_xml"
    stored_think = None

    printed_objects: list = []

    def capture_print(*args, **kwargs):
        printed_objects.extend(args)

    with (
        patch("mlx_manager.cli._init_probe_runtime", new=AsyncMock()),
        patch(
            "mlx_manager.cli.scan_cache_dir",
            return_value=_make_cache_info(model_id),
        ),
        patch(
            "mlx_manager.cli.read_model_config",
            return_value={"model_type": "qwen3"},
        ),
        patch(
            "mlx_manager.cli.detect_model_type",
            return_value=MagicMock(value="text_gen"),
        ),
        patch(
            "mlx_manager.cli.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.cli.get_family_tool_parser_id",
            return_value="hermes_json",
        ),
        patch(
            "mlx_manager.cli.get_family_thinking_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.cli._get_stored_parsers",
            new=AsyncMock(return_value=(stored_tool, stored_think)),
        ),
        patch.object(
            __import__("mlx_manager.cli", fromlist=["console"]).console,
            "print",
            side_effect=capture_print,
        ),
    ):
        await _audit_parsers()

    # A mismatch summary line should be printed
    printed_strs = [str(obj) for obj in printed_objects]
    assert any("mismatch" in s.lower() for s in printed_strs), (
        f"Expected 'mismatch' in output. Got: {printed_strs}"
    )


# ---------------------------------------------------------------------------
# Test 4: models without config.json are skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_skips_models_without_config():
    """_audit_parsers() skips models when read_model_config returns None."""
    from mlx_manager.cli import _audit_parsers

    model_id = "mlx-community/SomeModelWithoutConfig"

    printed_objects: list = []

    def capture_print(*args, **kwargs):
        printed_objects.extend(args)

    with (
        patch("mlx_manager.cli._init_probe_runtime", new=AsyncMock()),
        patch(
            "mlx_manager.cli.scan_cache_dir",
            return_value=_make_cache_info(model_id),
        ),
        patch(
            "mlx_manager.cli.read_model_config",
            return_value=None,  # No config.json
        ),
        patch.object(
            __import__("mlx_manager.cli", fromlist=["console"]).console,
            "print",
            side_effect=capture_print,
        ),
    ):
        await _audit_parsers()

    # When all models are skipped, no table rows should appear
    # The table should be empty or only a summary message should be printed
    from rich.table import Table

    tables = [obj for obj in printed_objects if isinstance(obj, Table)]
    if tables:
        table = tables[0]
        # Row count should be 0 since the model was skipped
        assert table.row_count == 0, f"Expected 0 rows, got {table.row_count}"


# ---------------------------------------------------------------------------
# Test 5: --audit flag calls _audit_parsers and skips normal probe flow
# ---------------------------------------------------------------------------


def test_audit_flag_skips_normal_probe():
    """probe --audit triggers _audit_parsers() and does NOT call _probe_single/_probe_all."""
    from typer.testing import CliRunner

    from mlx_manager.cli import app

    runner = CliRunner()

    with (
        patch("mlx_manager.cli._audit_parsers", new=AsyncMock()) as mock_audit,
        patch("mlx_manager.cli._probe_single", new=AsyncMock()) as mock_single,
        patch("mlx_manager.cli._probe_all", new=AsyncMock()) as mock_all,
    ):
        result = runner.invoke(app, ["probe", "--audit"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    mock_audit.assert_called_once()
    mock_single.assert_not_called()
    mock_all.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: _get_stored_parsers returns None, None for unknown model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_stored_parsers_unknown_model():
    """_get_stored_parsers() returns (None, None) when model not in DB."""
    from mlx_manager.cli import _get_stored_parsers

    mock_model = None  # Not in DB

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_model

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("mlx_manager.cli.get_session", return_value=mock_session):
        tool_id, think_id = await _get_stored_parsers("mlx-community/UnknownModel")

    assert tool_id is None
    assert think_id is None


# ---------------------------------------------------------------------------
# Test 7: _get_stored_parsers returns parser IDs from capabilities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_stored_parsers_with_capabilities():
    """_get_stored_parsers() extracts tool_parser_id and thinking_parser_id from DB."""
    from mlx_manager.cli import _get_stored_parsers

    mock_caps = MagicMock()
    mock_caps.tool_parser_id = "hermes_json"
    mock_caps.thinking_parser_id = "think_tag"

    mock_model = MagicMock()
    mock_model.capabilities = mock_caps

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_model

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("mlx_manager.cli.get_session", return_value=mock_session):
        tool_id, think_id = await _get_stored_parsers("mlx-community/Qwen3-0.6B-4bit-DWQ")

    assert tool_id == "hermes_json"
    assert think_id == "think_tag"


# ---------------------------------------------------------------------------
# Test 8: OK match when stored and family parsers agree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_shows_ok_when_parsers_match():
    """_audit_parsers() marks row as OK when stored parsers match family config."""
    from mlx_manager.cli import _audit_parsers

    model_id = "mlx-community/Qwen3-0.6B-4bit-DWQ"

    printed_objects: list = []

    def capture_print(*args, **kwargs):
        printed_objects.extend(args)

    with (
        patch("mlx_manager.cli._init_probe_runtime", new=AsyncMock()),
        patch(
            "mlx_manager.cli.scan_cache_dir",
            return_value=_make_cache_info(model_id),
        ),
        patch(
            "mlx_manager.cli.read_model_config",
            return_value={"model_type": "qwen3"},
        ),
        patch(
            "mlx_manager.cli.detect_model_type",
            return_value=MagicMock(value="text_gen"),
        ),
        patch(
            "mlx_manager.cli.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.cli.get_family_tool_parser_id",
            return_value="hermes_json",
        ),
        patch(
            "mlx_manager.cli.get_family_thinking_parser_id",
            return_value=None,
        ),
        patch(
            "mlx_manager.cli._get_stored_parsers",
            # stored tool matches family tool, stored think is None (matches no family think)
            new=AsyncMock(return_value=("hermes_json", None)),
        ),
        patch.object(
            __import__("mlx_manager.cli", fromlist=["console"]).console,
            "print",
            side_effect=capture_print,
        ),
    ):
        await _audit_parsers()

    # Summary line should say all probed models match
    printed_strs = [str(obj) for obj in printed_objects]
    assert any("match" in s.lower() or "ok" in s.lower() for s in printed_strs), (
        f"Expected 'match'/'ok' in output. Got: {printed_strs}"
    )

"""Probe orchestrator — thin wrapper delegating to ProbingCoordinator.

Public API: probe_model() — used by CLI (cli.py) and API (routers/models.py).
The coordinator handles the full lifecycle through the Profile → Pool → Adapter path.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from .steps import ProbeStep


async def probe_model(model_id: str, *, verbose: bool = False) -> AsyncGenerator[ProbeStep, None]:
    """Probe a model's capabilities using the ProbingCoordinator.

    Loads the model through the normal Profile path, detects its type,
    runs type-specific checks and parser sweeps, stores results in the DB,
    and cleans up.

    Yields ProbeStep objects for progressive SSE streaming.

    Args:
        model_id: HuggingFace model path (e.g. "mlx-community/Qwen3-0.6B-4bit-DWQ")
        verbose: If True, include diagnostic details in ProbeStep.details

    Yields:
        ProbeStep objects describing each step's progress
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool

    from .coordinator import ProbingCoordinator

    pool = get_model_pool()
    coordinator = ProbingCoordinator(pool)

    async for step in coordinator.probe(model_id, verbose=verbose):
        yield step

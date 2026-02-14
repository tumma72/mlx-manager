"""MLX Manager CLI."""

import threading
import webbrowser

import typer
from rich.console import Console
from rich.table import Table

from mlx_manager import __version__
from mlx_manager.config import DEFAULT_PORT

app = typer.Typer(
    name="mlx-manager",
    help="MLX Model Manager - Manage MLX models on Apple Silicon",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (dev)"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser on start"),
):
    """Start the MLX Manager web server."""
    import uvicorn

    console.print(f"[green]Starting MLX Manager on http://{host}:{port}[/green]")

    if open_browser:

        def open_delayed():
            import time

            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_delayed, daemon=True).start()

    uvicorn.run(
        "mlx_manager.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def install_service(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to bind to"),
):
    """Install MLX Manager as a launchd service (runs at login)."""
    from mlx_manager.services.manager_launchd import install_manager_service

    plist_path = install_manager_service(host=host, port=port)
    console.print(f"[green]Service installed: {plist_path}[/green]")
    console.print("MLX Manager will start automatically at login.")


@app.command()
def uninstall_service():
    """Remove MLX Manager launchd service."""
    from mlx_manager.services.manager_launchd import uninstall_manager_service

    if uninstall_manager_service():
        console.print("[green]Service uninstalled successfully[/green]")
    else:
        console.print("[yellow]Service was not installed[/yellow]")


@app.command()
def status():
    """Show status of running MLX servers."""
    import httpx

    try:
        response = httpx.get(f"http://127.0.0.1:{DEFAULT_PORT}/api/servers", timeout=5.0)
        servers = response.json()

        if not servers:
            console.print("[yellow]No servers running[/yellow]")
            return

        table = Table(title="Running MLX Servers")
        table.add_column("Profile", style="cyan")
        table.add_column("Port", style="green")
        table.add_column("PID", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Memory", style="blue")

        for server in servers:
            table.add_row(
                server["profile_name"],
                str(server["port"]),
                str(server["pid"]),
                server["health_status"],
                f"{server['memory_mb']:.1f} MB",
            )

        console.print(table)
    except httpx.ConnectError:
        console.print("[red]MLX Manager is not running[/red]")
        console.print("Start it with: mlx-manager serve")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def menubar():
    """Launch the status bar app."""
    from mlx_manager.menubar import run_menubar

    run_menubar()


@app.command()
def version():
    """Show version information."""
    console.print(f"MLX Manager v{__version__}")


@app.command()
def probe(
    model_id: str = typer.Argument(
        None,
        help="HuggingFace model ID to probe (e.g. mlx-community/Qwen3-0.6B-4bit-DWQ)",
    ),
    all_models: bool = typer.Option(False, "--all", help="Probe all cached models"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or markdown"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show diagnostic details"),
    force: bool = typer.Option(False, "--force", help="Clear cached capabilities before probing"),
    save: str = typer.Option(None, "--save", help="Write raw outputs to directory"),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
    report: bool = typer.Option(False, "--report", help="Generate a markdown diagnostic report"),
):
    """Probe model capabilities (tools, thinking, embeddings, audio, etc.)."""
    import asyncio

    if not model_id and not all_models:
        console.print("[red]Provide a model_id or use --all to probe all cached models[/red]")
        raise typer.Exit(1)

    if all_models:
        asyncio.run(_probe_all(format, verbose=verbose, force=force))
    else:
        asyncio.run(
            _probe_single(
                model_id,
                format,
                verbose=verbose,
                force=force,
                save=save,
                json_output=json_output,
                report=report,
            )
        )


async def _init_probe_runtime() -> None:
    """Initialize the database and model pool for CLI probing."""
    from mlx_manager.database import init_db
    from mlx_manager.mlx_server.models import pool
    from mlx_manager.mlx_server.models.pool import ModelPoolManager
    from mlx_manager.mlx_server.utils.memory import auto_detect_memory_limit, set_memory_limit

    await init_db()

    if pool.model_pool is None:
        max_memory_gb = auto_detect_memory_limit()
        set_memory_limit(max_memory_gb)
        pool.model_pool = ModelPoolManager(
            max_memory_gb=max_memory_gb,
            max_models=2,  # Conservative for CLI usage
        )


async def _clear_cached_capabilities(model_id: str) -> None:
    """Clear cached capabilities for a model before re-probing."""
    from mlx_manager.database import get_session
    from mlx_manager.models import Model
    from mlx_manager.models.capabilities import ModelCapabilities

    async with get_session() as session:
        from sqlmodel import select

        result = await session.execute(select(Model).where(Model.repo_id == model_id))
        model = result.scalar_one_or_none()
        if model:
            cap_result = await session.execute(
                select(ModelCapabilities).where(ModelCapabilities.model_id == model.id)
            )
            caps = cap_result.scalar_one_or_none()
            if caps:
                await session.delete(caps)
                await session.commit()


async def _probe_single(
    model_id: str,
    format: str,
    *,
    verbose: bool = False,
    force: bool = False,
    save: str | None = None,
    json_output: bool = False,
    report: bool = False,
) -> None:
    """Probe a single model and display results."""
    import json as json_mod

    from mlx_manager.services.probe import ProbeStep, probe_model

    await _init_probe_runtime()

    if force:
        await _clear_cached_capabilities(model_id)
        if not json_output:
            console.print(f"[yellow]Cleared cached capabilities for {model_id}[/yellow]")

    if not json_output:
        console.print(f"\n[bold]Probing [cyan]{model_id}[/cyan]...[/bold]\n")

    steps: list[ProbeStep] = []
    async for step in probe_model(model_id, verbose=verbose):
        steps.append(step)
        if not json_output:
            _print_step(step, verbose=verbose)

    if json_output:
        output: dict = {
            "model_id": model_id,
            "steps": [
                {
                    "step": s.step,
                    "status": s.status,
                    **({"capability": s.capability} if s.capability else {}),
                    **({"value": s.value} if s.value is not None else {}),
                    **({"error": s.error} if s.error else {}),
                    **({"details": s.details} if verbose and s.details else {}),
                    **(
                        {"diagnostics": [d.model_dump() for d in s.diagnostics]}
                        if s.diagnostics
                        else {}
                    ),
                }
                for s in steps
            ],
        }
        print(json_mod.dumps(output, indent=2))
    else:
        # Print diagnostic summary
        _print_diagnostic_summary(steps)
        console.print()

    # Generate report if requested
    if report:
        _print_report(model_id, steps)

    if save:
        _save_probe_outputs(model_id, steps, save)


async def _probe_all(format: str, *, verbose: bool = False, force: bool = False) -> None:
    """Probe all cached models and display results as a table."""
    from huggingface_hub import scan_cache_dir

    from mlx_manager.mlx_server.models.detection import detect_model_type
    from mlx_manager.services.probe import probe_model
    from mlx_manager.utils.model_detection import read_model_config

    await _init_probe_runtime()

    cache_info = scan_cache_dir()
    model_ids = sorted(repo.repo_id for repo in cache_info.repos)

    if not model_ids:
        console.print("[yellow]No cached models found[/yellow]")
        return

    console.print(f"[bold]Found {len(model_ids)} cached models[/bold]\n")

    rows: list[dict] = []
    for model_id in model_ids:
        # Quick config check — skip models without config.json
        config = read_model_config(model_id)
        if config is None:
            rows.append(
                {
                    "model_id": model_id,
                    "model_type": "?",
                    "status": "no config.json",
                    "capabilities": {},
                }
            )
            continue

        model_type = detect_model_type(model_id, config)
        if force:
            await _clear_cached_capabilities(model_id)

        console.print(f"[bold]{model_id}[/bold] ({model_type.value})")

        capabilities: dict = {}
        status = "ok"
        try:
            async for step in probe_model(model_id, verbose=verbose):
                _print_step(step, indent=2, verbose=verbose)
                if step.capability and step.value is not None:
                    capabilities[step.capability] = step.value
                if step.status == "failed" and step.step == "load_model":
                    status = f"load failed: {step.error or 'unknown'}"
        except Exception as e:
            status = f"error: {e}"

        rows.append(
            {
                "model_id": model_id,
                "model_type": model_type.value,
                "status": status,
                "capabilities": capabilities,
            }
        )
        console.print()

    # Output summary
    if format == "markdown":
        _print_markdown_table(rows)
    else:
        _print_rich_table(rows)


def _print_step(step, indent: int = 0, *, verbose: bool = False) -> None:
    """Print a single probe step with status icon and diagnostic badges."""
    # Skip probe_complete in normal output (it's a meta-step)
    if step.step == "probe_complete":
        return

    prefix = " " * indent
    icons = {
        "running": "[yellow]⟳[/yellow]",
        "completed": "[green]✓[/green]",
        "failed": "[red]✗[/red]",
        "skipped": "[dim]-[/dim]",
    }
    icon = icons.get(step.status, "?")
    detail = ""
    if step.capability and step.value is not None:
        detail = f" → {step.value}"
    if step.error:
        detail = f" [red]{step.error}[/red]"

    # Diagnostic badge
    badge = ""
    if step.diagnostics:
        action_count = sum(1 for d in step.diagnostics if d.level.value == "action_needed")
        warn_count = sum(1 for d in step.diagnostics if d.level.value == "warning")
        if action_count:
            badge = f" [red]({action_count} action needed)[/red]"
        elif warn_count:
            badge = f" [yellow]({warn_count} warning{'s' if warn_count > 1 else ''})[/yellow]"

    console.print(f"{prefix}{icon} {step.step}{detail}{badge}")

    if verbose and step.details:
        for key, value in step.details.items():
            if key == "result":
                continue  # Skip full result dict in verbose output
            console.print(f"{prefix}  [dim]{key}:[/dim] {value}")

    if verbose and step.diagnostics:
        for diag in step.diagnostics:
            level_color = {
                "action_needed": "red",
                "warning": "yellow",
                "info": "blue",
            }.get(diag.level.value, "dim")
            console.print(
                f"{prefix}  [{level_color}][{diag.level.value}][/{level_color}] {diag.message}"
            )
            for key, value in diag.details.items():
                if key == "raw_output_sample":
                    console.print(f"{prefix}    [dim]{key}:[/dim] {str(value)[:200]}...")
                else:
                    console.print(f"{prefix}    [dim]{key}:[/dim] {value}")


def _print_diagnostic_summary(steps: list) -> None:
    """Print a summary line of all diagnostics after probe steps."""
    all_diags = []
    for step in steps:
        if step.diagnostics:
            all_diags.extend(step.diagnostics)
    if not all_diags:
        return
    action_count = sum(1 for d in all_diags if d.level.value == "action_needed")
    warn_count = sum(1 for d in all_diags if d.level.value == "warning")
    info_count = sum(1 for d in all_diags if d.level.value == "info")
    parts = []
    if action_count:
        parts.append(f"[red]{action_count} action needed[/red]")
    if warn_count:
        parts.append(f"[yellow]{warn_count} warning{'s' if warn_count > 1 else ''}[/yellow]")
    if info_count:
        parts.append(f"[blue]{info_count} info[/blue]")
    console.print(f"\n  {len(all_diags)} diagnostics: {', '.join(parts)}")


def _print_report(model_id: str, steps: list) -> None:
    """Generate and print a markdown diagnostic report."""
    from mlx_manager.services.probe import ProbeResult, generate_support_report

    # Extract ProbeResult from probe_complete step
    result = ProbeResult()
    for step in steps:
        if step.step == "probe_complete" and step.details and "result" in step.details:
            result = ProbeResult(**step.details["result"])
            break

    report = generate_support_report(model_id, result, steps)
    console.print("\n[bold]Diagnostic Report[/bold]\n")
    console.print(report)


def _save_probe_outputs(model_id: str, steps: list, save_dir: str) -> None:
    """Write probe step data to files for offline analysis."""
    import json as json_mod
    from pathlib import Path

    model_dir = Path(save_dir) / model_id.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_id": model_id,
        "steps": [
            {
                "step": s.step,
                "status": s.status,
                **({"capability": s.capability} if s.capability else {}),
                **({"value": s.value} if s.value is not None else {}),
                **({"error": s.error} if s.error else {}),
                **({"details": s.details} if s.details else {}),
            }
            for s in steps
        ],
    }
    summary_path = model_dir / "probe_summary.json"
    summary_path.write_text(json_mod.dumps(summary, indent=2))

    # Write individual detail files for steps that have details
    for step in steps:
        if step.details:
            for key, value in step.details.items():
                detail_path = model_dir / f"{step.step}_{key}.txt"
                detail_path.write_text(str(value))

    console.print(f"[green]Saved probe outputs to {model_dir}[/green]")


def _print_rich_table(rows: list[dict]) -> None:
    """Display probe results as a Rich table."""
    table = Table(title="Model Probe Results")
    table.add_column("Model", style="cyan", no_wrap=True, max_width=50)
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Capabilities", style="yellow")

    for row in rows:
        caps = row["capabilities"]
        cap_strs = []
        for k, v in caps.items():
            if isinstance(v, bool):
                cap_strs.append(f"{k}={'Y' if v else 'N'}")
            else:
                cap_strs.append(f"{k}={v}")
        table.add_row(
            row["model_id"],
            row["model_type"],
            row["status"],
            ", ".join(cap_strs) if cap_strs else "-",
        )
    console.print(table)


def _print_markdown_table(rows: list[dict]) -> None:
    """Output probe results as a markdown table to stdout."""
    print("| Model | Type | Status | Capabilities |")
    print("|-------|------|--------|-------------|")
    for row in rows:
        caps = row["capabilities"]
        cap_strs = []
        for k, v in caps.items():
            if isinstance(v, bool):
                cap_strs.append(f"{k}={'Yes' if v else 'No'}")
            else:
                cap_strs.append(f"{k}={v}")
        caps_str = ", ".join(cap_strs) if cap_strs else "-"
        print(f"| {row['model_id']} | {row['model_type']} | {row['status']} | {caps_str} |")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

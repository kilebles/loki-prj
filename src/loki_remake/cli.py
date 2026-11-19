"""Typer CLI entrypoint for LOKI Remake."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from loki_remake.config import get_settings
from loki_remake.gpuniq.client import GPUniqClient
from loki_remake.gpuniq.prompter import normalize_ascii_punctuation, validate_minimax_prompt
from loki_remake.pipeline import prepare_job
from loki_remake.schemas import RemakeMode, RenderTier

app = typer.Typer(help="LOKI Remake: Viggle-Remake / MiniMax-H3 preparation and GPUniq tooling.")
console = Console()


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)


@app.command()
def balance() -> None:
    """Show GPUniq account and workspace balances."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    with GPUniqClient(
        api_key=settings.gpuniq_api_key.get_secret_value(),
        base_url=settings.gpuniq_base_url,
    ) as client:
        data = client.get_balances()

    acct = data["account"]
    console.print(
        f"[bold]GPUniq Account[/bold]: balance=${acct['balance_usd']:.4f} | "
        f"free=${acct['free_usd']:.4f} | reserved=${acct['reserved_usd']:.4f}"
    )

    table = Table(title="Workspaces")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Balance (USD)", justify="right")
    for ws in data.get("workspaces", []):
        table.add_row(str(ws["id"]), ws["name"], ws["kind"], f"${ws['balance_usd']:.4f}")
    console.print(table)


@app.command()
def gpus(
    min_vram: Annotated[int, typer.Option(help="Minimum VRAM per GPU in GB")] = 80,
) -> None:
    """List available Dex-Cloud GPUs on GPUniq."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    with GPUniqClient(
        api_key=settings.gpuniq_api_key.get_secret_value(),
        base_url=settings.gpuniq_base_url,
    ) as client:
        data = client.list_cloud_gpus(min_vram_gb=min_vram)

    rows = [*data.get("featured", []), *data.get("others", [])]
    table = Table(title=f"GPUniq Cloud GPUs (>= {min_vram} GB VRAM)")
    table.add_column("GPU")
    table.add_column("Avail", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("$/hr", justify="right")

    for gpu in rows:
        table.add_row(
            str(gpu["gpu_name"]),
            str(gpu["available_count"]),
            f"{gpu['vram_gb']:.1f} GB",
            f"{gpu['ram_gb']:.1f} GB",
            f"${gpu['total_price_per_hour']:.4f}",
        )
    console.print(table)


@app.command()
def prepare(
    video: Annotated[Path, typer.Option(exists=True, readable=True, help="Input video file")],
    mode: Annotated[RemakeMode, typer.Option(help="Remake mode: swap or edit")],
    instruction: Annotated[str, typer.Option(help="Target edit or character swap instruction")],
    ref: Annotated[Path | None, typer.Option(exists=True, help="Optional reference image")] = None,
    tier: Annotated[RenderTier, typer.Option(help="Canvas short edge: 480 or 736")] = 480,
    max_refs: Annotated[int, typer.Option(min=1, max=4, help="Max keyframes to paint")] = 2,
    out: Annotated[Path | None, typer.Option(help="Output job directory")] = None,
) -> None:
    """Normalize video, extract shots/keyframes, paint refs, and build MiniMax-H3 prompt."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    job = prepare_job(
        video_path=video,
        mode=mode,
        instruction=instruction,
        settings=settings,
        user_ref_path=ref,
        tier=tier,
        max_refs=max_refs,
        out_dir=out,
    )

    console.print(f"[green]Prepared job {job.job_id}[/green]")
    console.print(
        f"Normalized: {job.normalized.video_path} "
        f"({job.normalized.width}x{job.normalized.height}, {job.normalized.num_frames} frames, "
        f"{job.normalized.duration_sec}s)"
    )
    console.print(f"Shots detected: {len(job.shots)} | Painted refs: {len(job.refs)}")
    console.print(f"Prompt saved: {job.prompt_path}")
    console.print(f"Prep cost: ${job.prep_cost_usd:.4f} | Timings: {job.timings_sec}")


@app.command("validate-prompt")
def validate_prompt_cmd(
    prompt: Annotated[Path, typer.Option(exists=True, readable=True, help="Prompt text file")],
    frames: Annotated[int, typer.Option(help="Target frame count (17n+5, 90..345)")],
    pics: Annotated[int, typer.Option(help="Number of <Picture N> references")],
    shots: Annotated[int, typer.Option(help="Number of [Shot N] sections")],
    mode: Annotated[RemakeMode, typer.Option(help="Remake mode: swap or edit")] = "edit",
) -> None:
    """Validate an existing six-section MiniMax-H3 prompt file."""
    raw = prompt.read_text(encoding="utf-8")
    cleaned = normalize_ascii_punctuation(raw)
    result = validate_minimax_prompt(
        cleaned,
        mode=mode,
        num_frames=frames,
        num_pics=pics,
        num_shots=shots,
    )
    for w in result.warnings:
        console.print(f"[yellow]WARNING:[/yellow] {w}")
    if not result.valid:
        for err in result.errors:
            console.print(f"[red]ERROR:[/red] {err}")
        raise typer.Exit(code=1)
    console.print("[green]Prompt is valid.[/green]")


if __name__ == "__main__":
    app()

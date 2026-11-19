"""Typer CLI entrypoint for LOKI Remake."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from loki_remake.config import get_settings
from loki_remake.engine.viggle_bf16 import run_remote_render
from loki_remake.gpuniq.client import GPUniqAPIError, GPUniqClient
from loki_remake.gpuniq.ops import (
    clear_active_gpu,
    ensure_ssh_key,
    extract_ssh_connection,
    load_active_gpu,
    pick_best_h200,
    rank_pro6000_gpus,
    rank_upscaler_gpus,
    rsync_from_remote,
    rsync_to_remote,
    run_ssh,
    save_active_gpu,
    utc_now_iso,
    wait_for_ssh_ready,
)
from loki_remake.gpuniq.prompter import normalize_ascii_punctuation, validate_minimax_prompt
from loki_remake.pipeline import prepare_job
from loki_remake.schemas import GPUInstanceState, RemakeMode, RenderTier, UpscaleResult
from loki_remake.video.upscale import upscale_via_topaz

app = typer.Typer(help="LOKI Remake: Viggle-Remake / MiniMax-H3 preparation and GPUniq tooling.")
console = Console()


def _client_from_settings() -> GPUniqClient:
    settings = get_settings()
    return GPUniqClient(
        api_key=settings.gpuniq_api_key.get_secret_value(),
        base_url=settings.gpuniq_base_url,
    )


def _require_active_gpu() -> GPUInstanceState:
    state = load_active_gpu()
    if state is None:
        console.print("[red]No active GPU saved in work/active_gpu.json. Run gpu-up first.[/red]")
        raise typer.Exit(code=1)
    return state


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
    paint_model: Annotated[
        str | None, typer.Option(help="Override painter image model slug")
    ] = None,
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
        paint_model=paint_model,
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


@app.command("gpu-offers")
def gpu_offers(
    gpu: Annotated[str, typer.Option(help="GPU model filter (e.g. H200)")] = "H200",
    min_ram: Annotated[float, typer.Option(help="Minimum host RAM in GB")] = 150.0,
    min_down: Annotated[float, typer.Option(help="Minimum download speed in Mbps")] = 500.0,
    max_price: Annotated[float, typer.Option(help="Maximum price per hour in USD")] = 6.0,
) -> None:
    """List matching 1x GPU offers on GPUniq Marketplace."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    models = ["H200", "H200 NVL"] if gpu.upper() == "H200" else [gpu]
    with _client_from_settings() as client:
        data = client.list_marketplace(
            gpu_model=models,
            max_gpu_count=1,
            min_ram_gb=min_ram,
            min_disk_gb=250,
            min_inet_speed_mbps=min_down,
            max_price_per_hour=max_price,
            provider="vast",
            sort_by="price-low",
            page_size=20,
        )

    table = Table(title=f"GPUniq Marketplace 1x {gpu}")
    table.add_column("Agent ID")
    table.add_column("Model")
    table.add_column("VRAM", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Disk", justify="right")
    table.add_column("Down", justify="right")
    table.add_column("Rel", justify="right")
    table.add_column("$/hr", justify="right")
    table.add_column("Loc")

    for a in data.get("agents", []):
        if int(a.get("gpu_count") or 0) != 1:
            continue
        table.add_row(
            str(a["id"]),
            str(a.get("gpu_model", "")),
            f"{a.get('vram_gb', 0)} GB",
            f"{a.get('ram_gb', 0)} GB",
            f"{a.get('storage_gb', 0)} GB",
            f"{a.get('down_mbps', 0)} Mbps",
            f"{float(a.get('reliability') or 0):.3f}",
            f"${float(a.get('price_per_hour') or 0):.4f}",
            str(a.get("location", "")),
        )
    console.print(table)


@app.command("gpu-up")
def gpu_up(
    agent_id: Annotated[int | None, typer.Option(help="Explicit Marketplace agent_id")] = None,
    auto_h200: Annotated[bool, typer.Option(help="Auto-select best fast 1x H200 offer")] = False,
    auto_pro6000: Annotated[
        bool, typer.Option(help="Auto-select best fast 1x RTX PRO 6000 96GB offer")
    ] = False,
    disk: Annotated[int, typer.Option(help="Container disk size in GB")] = 250,
) -> None:
    """Rent a 1x GPU on GPUniq Marketplace with minute billing and wait for SSH."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    if agent_id is None and not auto_h200 and not auto_pro6000:
        console.print("[red]Pass --agent-id <ID>, --auto-h200, or --auto-pro6000[/red]")
        raise typer.Exit(code=2)

    priv_key = Path.home() / ".ssh" / "id_ed25519"
    pub_key = Path.home() / ".ssh" / "id_ed25519.pub"

    with _client_from_settings() as client:
        ssh_key_id = ensure_ssh_key(client, pubkey_path=pub_key)
        min_disk = max(250.0, float(disk))
        if agent_id is not None:
            candidates = [client.get_agent(agent_id)]
        elif auto_pro6000:
            candidates = rank_pro6000_gpus(client, min_disk_gb=max(280.0, float(disk)))
        else:
            candidates = [pick_best_h200(client, min_disk_gb=min_disk)]

        order: dict[str, object] | None = None
        selected_offer: dict[str, object] | None = None
        last_err: Exception | None = None
        for offer in candidates:
            selected_id = int(offer["id"])
            gpu_model = str(offer.get("gpu_model") or "GPU")
            price_hr = float(offer.get("price_per_hour") or 0.0)
            console.print(
                f"Ordering agent_id={selected_id} ({gpu_model}, "
                f"{offer.get('ram_gb')}GB RAM, {offer.get('down_mbps')} Mbps) at ${price_hr:.4f}/hr"
            )
            try:
                order = client.create_marketplace_order(
                    agent_id=selected_id,
                    gpu_required=1,
                    pricing_type="minute",
                    docker_image="vastai/pytorch:cuda-12.9.1-auto",
                    ssh_key_ids=[ssh_key_id],
                    disk_gb=disk,
                )
                selected_offer = offer
                break
            except GPUniqAPIError as exc:
                last_err = exc
                console.print(f"[yellow]Offer agent_id={selected_id} skipped: {exc}[/yellow]")
                continue

        if order is None or selected_offer is None:
            raise RuntimeError(f"All GPU order attempts failed: {last_err}")

        selected_id = int(selected_offer["id"])
        gpu_model = str(selected_offer.get("gpu_model") or "GPU")
        price_hr = float(selected_offer.get("price_per_hour") or 0.0)
        raw_task = (
            order.get("order_id")
            or order.get("task_id")
            or order.get("instance_id")
            or order.get("id")
        )
        if raw_task is None:
            raise RuntimeError(f"Order response did not include task_id: {order}")
        task_id = int(raw_task)
        effective_price = float(order.get("estimated_cost") or price_hr)

        # Save provisioning state immediately so gpu-down can always recover on timeout.
        provisional = GPUInstanceState(
            task_id=task_id,
            agent_id=selected_id,
            gpu_model=gpu_model,
            price_per_hour=effective_price,
            ssh_host="",
            ssh_port=22,
            ssh_user="root",
            ssh_key_path=priv_key,
            status="provisioning",
            created_at=utc_now_iso(),
        )
        save_active_gpu(provisional)
        console.print(f"Order created: task_id={task_id}. Waiting for SSH readiness...")

        host, port, user = wait_for_ssh_ready(client, task_id, priv_key)
        ready_state = provisional.model_copy(
            update={"ssh_host": host, "ssh_port": port, "ssh_user": user, "status": "running"}
        )
        save_active_gpu(ready_state)

    console.print(f"[green]GPU ready:[/green] task_id={task_id} -> ssh -p {port} {user}@{host}")


@app.command("gpu-status")
def gpu_status() -> None:
    """List active GPUniq instances and local saved GPU state."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    with _client_from_settings() as client:
        data = client.list_instances()

    instances = data.get("instances", [])
    total = data.get("total_count", len(instances))
    console.print(f"[bold]Active GPUniq instances:[/bold] {total}")
    for inst in instances:
        host, port, user = extract_ssh_connection(inst)
        gpu_name = (
            inst.get("gpu_model")
            or inst.get("agent", {}).get("gpu", {}).get("model")
            or "unknown"
        )
        console.print(
            f"  task_id={inst.get('id') or inst.get('task_id')} | "
            f"status={inst.get('status')} | gpu={gpu_name} | "
            f"ssh={user}@{host}:{port}"
        )

    saved = load_active_gpu()
    if saved:
        console.print(
            f"[dim]Local state: task_id={saved.task_id}, {saved.gpu_model}, "
            f"{saved.ssh_user}@{saved.ssh_host}:{saved.ssh_port} ({saved.status})[/dim]"
        )


@app.command("gpu-setup")
def gpu_setup() -> None:
    """Upload and run infra/setup_gpu.sh on the active rented GPU."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    state = _require_active_gpu()
    script_path = Path("infra/setup_gpu.sh")
    if not script_path.exists():
        raise FileNotFoundError(f"Setup script not found: {script_path}")

    rsync_to_remote(
        state.ssh_host,
        state.ssh_port,
        state.ssh_user,
        state.ssh_key_path,
        script_path,
        "/tmp/setup_gpu.sh",
    )
    rc, _ = run_ssh(
        state.ssh_host,
        state.ssh_port,
        state.ssh_user,
        state.ssh_key_path,
        "chmod +x /tmp/setup_gpu.sh && bash /tmp/setup_gpu.sh",
        stream=True,
        timeout=1080.0,
        idle_timeout=180.0,
    )
    if rc != 0:
        console.print(f"[red]gpu-setup failed with exit code {rc}[/red]")
        raise typer.Exit(code=rc)
    console.print("[green]GPU environment and model weights verified ready.[/green]")


@app.command("gpu-render")
def gpu_render(
    job_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, help="Prepared job dir")],
    tier: Annotated[RenderTier | None, typer.Option(help="Override short edge: 480 or 736")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
) -> None:
    """Run MiniMax-H3 + Viggle-Remake bf16 render on the active GPU and download MP4."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    state = _require_active_gpu()

    res = run_remote_render(state, job_dir, short_edge=tier, seed=seed)
    console.print(f"[green]Rendered {res.out_video_path}[/green]")
    console.print(
        f"Geometry: {res.width}x{res.height}, {res.num_frames} frames, audio={res.has_audio} | "
        f"render={res.render_sec:.1f}s, wall={res.wall_sec:.1f}s | "
        f"peak VRAM={res.peak_alloc_gib:.1f}/{res.peak_reserved_gib:.1f} GiB | "
        f"cost=${res.render_cost_usd:.4f} (infer=${res.inference_only_cost_usd:.4f})"
    )


@app.command("gpu-stop")
def gpu_stop(
    task_id: Annotated[
        int | None, typer.Option(help="Instance task_id (defaults to active)")
    ] = None,
) -> None:
    """Stop (pause billing on) a running GPUniq instance."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    target_id = task_id or _require_active_gpu().task_id
    with _client_from_settings() as client:
        client.stop_instance(target_id)
    saved = load_active_gpu()
    if saved and saved.task_id == target_id:
        save_active_gpu(saved.model_copy(update={"status": "stopped"}))
    console.print(f"[yellow]Stopped instance task_id={target_id}[/yellow]")


@app.command("gpu-down")
def gpu_down(
    task_id: Annotated[
        int | None, typer.Option(help="Instance task_id (defaults to active)")
    ] = None,
) -> None:
    """Permanently delete a GPUniq instance and clear local active state."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    target_id = task_id or _require_active_gpu().task_id
    with _client_from_settings() as client:
        client.delete_instance(target_id)
    saved = load_active_gpu()
    if saved and saved.task_id == target_id:
        clear_active_gpu()
    console.print(f"[green]Deleted instance task_id={target_id}[/green]")


@app.command("topaz-models")
def topaz_models() -> None:
    """List Topaz Video AI models exposed by GPUniq Media API."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    with _client_from_settings() as client:
        data = client.list_topaz_models()

    table = Table(title=f"Topaz Video Models (${data.get('usd_per_credit', 0.14)}/credit)")
    table.add_column("Slug")
    table.add_column("Family")
    table.add_column("Description")
    for m in data.get("video_models", []):
        table.add_row(str(m["slug"]), str(m.get("family", "")), str(m.get("description", "")))
    console.print(table)


@app.command("upscale-topaz")
def upscale_topaz(
    video: Annotated[Path, typer.Option(exists=True, readable=True, help="Input MP4 file")],
    model: Annotated[str, typer.Option(help="Topaz video model slug")] = "topaz-video-proteus",
    out: Annotated[Path | None, typer.Option(help="Output upscaled MP4 path")] = None,
    width: Annotated[int, typer.Option(help="Target output width")] = 1280,
    height: Annotated[int, typer.Option(help="Target output height")] = 736,
    video_url: Annotated[str | None, typer.Option(help="Pre-hosted HTTPS URL")] = None,
    allow_temp_upload: Annotated[
        bool, typer.Option(help="Fallback to expiring temp HTTPS host if data URI rejected")
    ] = False,
) -> None:
    """Upscale a video via GPUniq Topaz Video AI API and preserve source audio."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    slug_short = model.replace("topaz-video-", "")
    out_path = out or video.with_name(f"{video.stem}_{slug_short}.mp4")

    with _client_from_settings() as client:
        res = upscale_via_topaz(
            client,
            video,
            out_path,
            model=model,
            output_width=width,
            output_height=height,
            video_url=video_url,
            allow_temp_upload=allow_temp_upload,
        )

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(res.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Topaz upscaled {res.out_video_path}[/green]")
    console.print(
        f"Geometry: {res.width}x{res.height} @ {res.fps}fps "
        f"({res.num_frames}f, audio={res.has_audio}) | "
        f"wall={res.wall_sec:.1f}s | credits={res.credits} | cost=${res.upscale_cost_usd:.4f}"
    )


@app.command("gpu-upscale-bench")
def gpu_upscale_bench(
    video: Annotated[Path, typer.Option(exists=True, readable=True, help="Input MP4 file")],
    out_dir: Annotated[Path, typer.Option(help="Output directory for upscaled MP4s")] = Path(
        "work/smoke_test"
    ),
    width: Annotated[int, typer.Option(help="Target width")] = 1280,
    height: Annotated[int, typer.Option(help="Target height")] = 736,
    disk: Annotated[int, typer.Option(help="Container disk size in GB")] = 120,
    agent_id: Annotated[int | None, typer.Option(help="Explicit Marketplace agent_id")] = None,
) -> None:
    """Rent a cheap 1x GPU, benchmark self-hosted upscalers, and always tear down."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_script = Path("infra/setup_upscaler_gpu.sh")
    runner_script = Path("infra/run_gpu_upscalers.py")
    if not setup_script.exists() or not runner_script.exists():
        raise FileNotFoundError("Missing infra/setup_upscaler_gpu.sh or infra/run_gpu_upscalers.py")

    priv_key = Path.home() / ".ssh" / "id_ed25519"
    pub_key = Path.home() / ".ssh" / "id_ed25519.pub"
    task_id: int | None = None

    with _client_from_settings() as client:
        ssh_key_id = ensure_ssh_key(client, pubkey_path=pub_key)
        candidates = (
            [client.get_agent(agent_id)]
            if agent_id is not None
            else rank_upscaler_gpus(client, min_disk_gb=float(disk))
        )

        order: dict[str, object] | None = None
        selected_offer: dict[str, object] | None = None
        last_order_err: Exception | None = None
        for offer in candidates[:8]:
            selected_id = int(offer["id"])
            gpu_model = str(offer.get("gpu_model") or "GPU")
            price_hr = float(offer.get("price_per_hour") or 0.0)
            console.print(
                f"Ordering upscaler GPU agent_id={selected_id} ({gpu_model}, "
                f"{offer.get('vram_gb')}GB VRAM, "
                f"{offer.get('down_mbps')} Mbps) at ${price_hr:.4f}/hr"
            )
            try:
                order = client.create_marketplace_order(
                    agent_id=selected_id,
                    gpu_required=1,
                    pricing_type="minute",
                    docker_image="vastai/pytorch:cuda-12.9.1-auto",
                    ssh_key_ids=[ssh_key_id],
                    disk_gb=disk,
                )
                selected_offer = offer
                break
            except GPUniqAPIError as exc:
                last_order_err = exc
                console.print(f"[yellow]Offer agent_id={selected_id} skipped: {exc}[/yellow]")
                continue

        if order is None or selected_offer is None:
            raise RuntimeError(f"All upscaler GPU order attempts failed: {last_order_err}")

        selected_id = int(selected_offer["id"])
        gpu_model = str(selected_offer.get("gpu_model") or "GPU")
        price_hr = float(selected_offer.get("price_per_hour") or 0.0)
        raw_task = (
            order.get("order_id")
            or order.get("task_id")
            or order.get("instance_id")
            or order.get("id")
        )
        if raw_task is None:
            raise RuntimeError(f"Order response did not include task_id: {order}")
        task_id = int(raw_task)
        effective_price = float(order.get("estimated_cost") or price_hr)

        state = GPUInstanceState(
            task_id=task_id,
            agent_id=selected_id,
            gpu_model=gpu_model,
            price_per_hour=effective_price,
            ssh_host="",
            ssh_port=22,
            ssh_user="root",
            ssh_key_path=priv_key,
            status="provisioning",
            created_at=utc_now_iso(),
        )
        save_active_gpu(state)
        console.print(f"Created task_id={task_id}. Waiting for SSH...")

        try:
            host, port, user = wait_for_ssh_ready(client, task_id, priv_key)
            state = state.model_copy(
                update={"ssh_host": host, "ssh_port": port, "ssh_user": user, "status": "running"}
            )
            save_active_gpu(state)

            rsync_to_remote(host, port, user, priv_key, setup_script, "/tmp/setup_upscaler_gpu.sh")
            rc, _ = run_ssh(
                host,
                port,
                user,
                priv_key,
                "chmod +x /tmp/setup_upscaler_gpu.sh && bash /tmp/setup_upscaler_gpu.sh",
                stream=True,
            )
            if rc != 0:
                raise RuntimeError(f"setup_upscaler_gpu.sh failed with exit code {rc}")

            remote_in = f"/workspace/in/{video.name}"
            rsync_to_remote(host, port, user, priv_key, video, remote_in)
            rsync_to_remote(
                host, port, user, priv_key, runner_script, "/workspace/run_gpu_upscalers.py"
            )

            run_cmd = (
                "PY_BIN=$(cat /workspace/upscaler_python.txt) && "
                f"\"$PY_BIN\" /workspace/run_gpu_upscalers.py "
                f"--input {remote_in} --out-dir /workspace/out "
                f"--target-width {width} --target-height {height}"
            )
            rc, _ = run_ssh(host, port, user, priv_key, run_cmd, stream=True)
            if rc != 0:
                raise RuntimeError(f"run_gpu_upscalers.py failed with exit code {rc}")

            rsync_from_remote(
                host,
                port,
                user,
                priv_key,
                "/workspace/out/remote_upscale_results.json",
                out_dir / "remote_upscale_results.json",
            )
            summary_file = out_dir / "remote_upscale_results.json"
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            upscale_results: list[UpscaleResult] = []

            for item in summary.get("results", []):
                fname = str(item["out_filename"])
                rsync_from_remote(
                    host, port, user, priv_key, f"/workspace/out/{fname}", out_dir / fname
                )
                local_out = out_dir / fname
                infer_sec = float(item["inference_sec"])
                wall_sec = float(item["wall_sec"])
                gpu_cost_infer = round(effective_price * (infer_sec / 3600.0), 4)
                upscale_results.append(
                    UpscaleResult(
                        name=str(item["name"]),
                        method=str(item["method"]),
                        backend="self_hosted_gpu",
                        source_video_path=video,
                        out_video_path=local_out,
                        width=int(item["width"]),
                        height=int(item["height"]),
                        fps=float(item["fps"]),
                        num_frames=int(item["num_frames"]),
                        has_audio=bool(item["has_audio"]),
                        wall_sec=wall_sec,
                        inference_sec=infer_sec,
                        peak_vram_gib=float(item.get("peak_vram_gib") or 0.0),
                        upscale_cost_usd=gpu_cost_infer,
                        gpu_model=gpu_model,
                        gpu_price_per_hour=effective_price,
                        notes=str(item.get("notes") or f"dl={item.get('download_sec', 0)}s"),
                    )
                )

            summary["gpu_price_per_hour"] = effective_price
            summary["gpu_model_ordered"] = gpu_model
            summary["typed_results"] = [r.model_dump(mode="json") for r in upscale_results]
            summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            console.print(f"[green]Saved self-hosted upscaler results to {summary_file}[/green]")
        finally:
            if task_id is not None:
                try:
                    client.delete_instance(task_id)
                    clear_active_gpu()
                    console.print(f"[green]Auto-deleted upscaler GPU task_id={task_id}[/green]")
                except Exception as del_exc:
                    console.print(
                        f"[red]WARNING: failed to auto-delete task_id={task_id}: {del_exc}[/red]"
                    )
                    raise


if __name__ == "__main__":
    app()

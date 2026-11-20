"""SSH-tunneled client for the resident GPU worker daemon (infra/gpu_worker.py)."""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from typing import Any

from loguru import logger

from loki_remake.gpuniq.ops import rsync_from_remote, rsync_to_remote, run_ssh
from loki_remake.schemas import GPUInstanceState, PreparedJob, RenderTier

_WORKER_SCRIPTS = (
    "gpu_worker.py",
    "sample_fp8.py",
    "face_enhance.py",
    "restoreformer_arch.py",
)


def _ssh_curl_json(
    gpu: GPUInstanceState,
    *,
    method: str,
    path: str,
    port: int = 8000,
    body_remote_file: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call the resident worker's localhost HTTP endpoint over SSH and parse JSON."""
    if body_remote_file is not None:
        curl_cmd = (
            f"curl -sS --max-time {int(timeout)} -X {method} "
            f"-H 'Content-Type: application/json' "
            f"--data-binary @{shlex.quote(body_remote_file)} "
            f"http://127.0.0.1:{port}{path}"
        )
    elif method == "POST":
        curl_cmd = (
            f"curl -sS --max-time {int(timeout)} -X POST "
            f"-H 'Content-Type: application/json' -d '{{}}' "
            f"http://127.0.0.1:{port}{path}"
        )
    else:
        curl_cmd = f"curl -sS --max-time {int(timeout)} http://127.0.0.1:{port}{path}"

    rc, out = run_ssh(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        curl_cmd,
        stream=False,
        timeout=timeout + 15.0,
    )
    if rc != 0 or not out.strip():
        raise RuntimeError(f"Worker HTTP {method} {path} failed (rc={rc}): {out.strip()}")
    try:
        return json.loads(out.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Worker returned non-JSON for {method} {path}: {out[:400]}") from exc


def get_remote_worker_status(gpu: GPUInstanceState, *, port: int = 8000) -> dict[str, Any]:
    """Query /health on the remote resident GPU worker."""
    return _ssh_curl_json(gpu, method="GET", path="/health", port=port, timeout=15.0)


def stop_remote_worker(gpu: GPUInstanceState, *, port: int = 8000) -> dict[str, Any]:
    """Gracefully stop the remote resident GPU worker via POST /shutdown."""
    return _ssh_curl_json(gpu, method="POST", path="/shutdown", port=port, timeout=15.0)


def start_remote_worker(
    gpu: GPUInstanceState,
    *,
    mode: str = "swap",
    quant: str = "fp8",
    port: int = 8000,
    wait_ready: bool = True,
    timeout_sec: float = 600.0,
    infra_dir: Path = Path("infra"),
    remote_root: str = "/workspace",
) -> dict[str, Any]:
    """Sync worker scripts and launch the resident GPU daemon (idempotent if already up)."""
    # Check if already running and healthy
    try:
        health = get_remote_worker_status(gpu, port=port)
        if health.get("status") in ("ready", "busy", "loading"):
            logger.info("Resident GPU worker already running on port {}: {}", port, health)
            return health
    except RuntimeError:
        pass

    remote_app = f"{remote_root}/app"
    run_ssh(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        f"mkdir -p {shlex.quote(remote_app)}",
        stream=False,
    )
    for name in _WORKER_SCRIPTS:
        local_script = infra_dir / name
        if not local_script.exists():
            raise FileNotFoundError(f"Missing worker script: {local_script}")
        rsync_to_remote(
            gpu.ssh_host,
            gpu.ssh_port,
            gpu.ssh_user,
            gpu.ssh_key_path,
            local_script,
            f"{remote_app}/{name}",
        )

    launch_cmd = (
        f"nohup {remote_root}/venv/bin/python -u {remote_app}/gpu_worker.py "
        f"--host 127.0.0.1 --port {port} "
        f"--preload-mode {shlex.quote(mode)} --quant {shlex.quote(quant)} "
        f"> {remote_root}/worker.log 2>&1 </dev/null & "
        f"echo $! > {remote_root}/worker.pid && cat {remote_root}/worker.pid"
    )
    rc, pid_out = run_ssh(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        launch_cmd,
        stream=False,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to launch gpu_worker.py (rc={rc}): {pid_out}")
    pid = pid_out.strip().splitlines()[-1]
    logger.info("Launched resident gpu_worker.py (pid={}, mode={}, quant={})", pid, mode, quant)

    if not wait_ready:
        return {"status": "starting", "pid": pid, "port": port}

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            health = get_remote_worker_status(gpu, port=port)
            if health.get("status") == "ready":
                return health
            if health.get("status") == "error":
                raise RuntimeError(f"Worker failed during warmup: {health.get('last_error')}")
        except RuntimeError:
            # Verify process is still alive
            chk_rc, _ = run_ssh(
                gpu.ssh_host,
                gpu.ssh_port,
                gpu.ssh_user,
                gpu.ssh_key_path,
                f"kill -0 {shlex.quote(pid)} 2>/dev/null",
                stream=False,
                timeout=15.0,
            )
            if chk_rc != 0:
                _, log_tail = run_ssh(
                    gpu.ssh_host,
                    gpu.ssh_port,
                    gpu.ssh_user,
                    gpu.ssh_key_path,
                    f"tail -n 40 {remote_root}/worker.log",
                    stream=False,
                    timeout=15.0,
                )
                raise RuntimeError(f"gpu_worker.py exited prematurely:\n{log_tail}") from None
        time.sleep(5.0)

    raise TimeoutError(f"Resident GPU worker did not become ready within {timeout_sec:.0f}s")


def render_via_remote_worker(
    gpu: GPUInstanceState,
    job_dir: Path,
    *,
    short_edge: RenderTier | None = None,
    seed: int = 0,
    restorer: str = "none",
    fidelity: float = 0.65,
    grain_strength: float = 0.35,
    color_match: bool = True,
    semantic_mask: bool = True,
    upscale_to: str | None = None,
    port: int = 8000,
    remote_root: str = "/workspace",
) -> dict[str, Any]:
    """Upload job_dir, render via warm resident GPU worker (zero cold start), download outputs."""
    job_file = job_dir / "job.json"
    if not job_file.exists():
        raise FileNotFoundError(f"Missing prepared job manifest: {job_file}")

    job = PreparedJob.model_validate_json(job_file.read_text(encoding="utf-8"))
    tier: RenderTier = short_edge or job.tier
    remote_job_dir = f"{remote_root}/jobs/{job.job_id}"

    run_ssh(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        f"mkdir -p {shlex.quote(remote_job_dir)}",
        stream=False,
    )
    rsync_to_remote(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        job_dir,
        remote_job_dir,
    )

    remote_job_p = Path(remote_job_dir)
    remote_out = remote_job_p / f"out_{tier}_worker.mp4"
    payload: dict[str, Any] = {
        "mode": job.mode,
        "video": str(remote_job_p / job.normalized.video_path.name),
        "audio": (
            str(remote_job_p / job.normalized.audio_path.name)
            if job.normalized.audio_path is not None
            else None
        ),
        "prompt": str(remote_job_p / job.prompt_path.name),
        "refs": [
            {"path": str(remote_job_p / "painted" / r.painted_path.name), "pin": r.pin_frame}
            for r in job.refs
        ],
        "frames": job.normalized.num_frames,
        "short_edge": tier,
        "seed": seed,
        "out": str(remote_out),
    }

    if restorer != "none":
        payload["enhance"] = {
            "restorer": restorer,
            "fidelity": fidelity,
            "grain_strength": grain_strength,
            "color_match": color_match,
            "semantic_mask": semantic_mask,
            "upscale_to": upscale_to,
            "out": str(remote_job_p / f"out_{tier}_worker_{restorer}.mp4"),
            "stats_out": str(remote_job_p / f"out_{tier}_worker_{restorer}_stats.json"),
        }

    req_local = job_dir / f"worker_req_{tier}.json"
    req_local.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    remote_req = f"{remote_job_dir}/worker_req_{tier}.json"
    rsync_to_remote(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        req_local,
        remote_req,
    )

    t0 = time.perf_counter()
    resp = _ssh_curl_json(
        gpu,
        method="POST",
        path="/render",
        port=port,
        body_remote_file=remote_req,
        timeout=1800.0,
    )
    wall_sec = time.perf_counter() - t0
    if not resp.get("ok"):
        raise RuntimeError(
            f"Worker render failed: {resp.get('error')}\n{resp.get('traceback', '')}"
        )

    local_out = job_dir / remote_out.name
    rsync_from_remote(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        str(remote_out),
        local_out,
    )

    local_enhanced: Path | None = None
    if resp.get("enhanced_out"):
        enh_remote = Path(str(resp["enhanced_out"]))
        local_enhanced = job_dir / enh_remote.name
        rsync_from_remote(
            gpu.ssh_host,
            gpu.ssh_port,
            gpu.ssh_user,
            gpu.ssh_key_path,
            str(enh_remote),
            local_enhanced,
        )

    total_sec = float(resp.get("total_sec") or wall_sec)
    resp["wall_sec"] = round(wall_sec, 2)
    resp["local_out"] = str(local_out)
    resp["local_enhanced_out"] = str(local_enhanced) if local_enhanced else None
    resp["gpu_cost_usd"] = round(gpu.price_per_hour * (total_sec / 3600.0), 4)

    meta_out = job_dir / f"worker_result_{tier}.json"
    meta_out.write_text(json.dumps(resp, indent=2) + "\n", encoding="utf-8")
    return resp

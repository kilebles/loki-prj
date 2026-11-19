"""Remote bf16 render runner wrapping Viggle-Remake's inference/sample.py."""

from __future__ import annotations

import re
import shlex
import time
from pathlib import Path

from loguru import logger

from loki_remake.gpuniq.ops import rsync_from_remote, rsync_to_remote, run_ssh
from loki_remake.schemas import GPUInstanceState, PreparedJob, RenderResult, RenderTier
from loki_remake.video.normalize import probe_video

_SAMPLE_SUMMARY_RE = re.compile(
    r"render\s+(?P<render_sec>\d+(?:\.\d+)?)s\s+at\s+"
    r"(?P<width>\d+)x(?P<height>\d+)x(?P<frames>\d+),\s+"
    r"peak\s+(?P<alloc>\d+(?:\.\d+)?)\s+GiB\s+allocated\s*/\s*"
    r"(?P<reserved>\d+(?:\.\d+)?)\s+GiB\s+reserved"
)


def build_sample_command(
    job: PreparedJob,
    *,
    base_dir: Path,
    lora_repo: Path,
    out_video: Path,
    video_path: Path | None = None,
    audio_path: Path | None = None,
    prompt_path: Path | None = None,
    ref_paths: list[Path] | None = None,
    short_edge: RenderTier | None = None,
    seed: int = 0,
    attn_backend: str = "_native_cudnn",
    python_bin: str = "/workspace/venv/bin/python",
) -> list[str]:
    """Construct exact sample.py CLI arguments from a PreparedJob."""
    tier: RenderTier = short_edge or job.tier
    cond_video = video_path or job.normalized.video_path
    cond_audio = audio_path if audio_path is not None else job.normalized.audio_path
    prompt_file = prompt_path or job.prompt_path
    refs = ref_paths if ref_paths is not None else [r.painted_path for r in job.refs]

    if len(refs) != len(job.refs):
        raise ValueError(f"Expected {len(job.refs)} ref paths, got {len(refs)}")

    cmd = [
        python_bin,
        str(lora_repo / "inference" / "sample.py"),
        "--mode",
        job.mode,
        "--video",
        str(cond_video),
        "--prompt",
        str(prompt_file),
        "--frames",
        str(job.normalized.num_frames),
        "--short-edge",
        str(tier),
        "--seed",
        str(seed),
        "--base",
        str(base_dir),
        "--lora-repo",
        str(lora_repo),
        "--attn-backend",
        attn_backend,
        "--out",
        str(out_video),
    ]
    if cond_audio is not None:
        cmd.extend(["--audio", str(cond_audio)])

    for ref_p, ref_meta in zip(refs, job.refs, strict=True):
        cmd.extend(["--ref", str(ref_p), "--pin", str(ref_meta.pin_frame)])

    return cmd


def parse_sample_output(stdout: str) -> dict[str, float]:
    """Parse sample.py summary line into render timing, geometry, and VRAM stats."""
    matches = list(_SAMPLE_SUMMARY_RE.finditer(stdout))
    if not matches:
        raise ValueError("Could not find sample.py render summary line in output")
    m = matches[-1]
    return {
        "render_sec": float(m.group("render_sec")),
        "width": float(m.group("width")),
        "height": float(m.group("height")),
        "num_frames": float(m.group("frames")),
        "peak_alloc_gib": float(m.group("alloc")),
        "peak_reserved_gib": float(m.group("reserved")),
    }


def run_remote_render(
    gpu: GPUInstanceState,
    job_dir: Path,
    *,
    short_edge: RenderTier | None = None,
    seed: int = 0,
    attn_backend: str = "_native_cudnn",
    remote_root: str = "/workspace",
) -> RenderResult:
    """Upload prepared job, execute sample.py on rented GPUniq GPU, and download output."""
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
    remote_video = remote_job_p / job.normalized.video_path.name
    remote_audio = (
        remote_job_p / job.normalized.audio_path.name
        if job.normalized.audio_path is not None
        else None
    )
    remote_prompt = remote_job_p / job.prompt_path.name
    remote_refs = [remote_job_p / "painted" / r.painted_path.name for r in job.refs]
    remote_out = remote_job_p / f"out_{tier}.mp4"

    argv = build_sample_command(
        job,
        base_dir=Path(f"{remote_root}/models/MiniMax-H3"),
        lora_repo=Path(f"{remote_root}/models/Viggle-Remake"),
        out_video=remote_out,
        video_path=remote_video,
        audio_path=remote_audio,
        prompt_path=remote_prompt,
        ref_paths=remote_refs,
        short_edge=tier,
        seed=seed,
        attn_backend=attn_backend,
        python_bin=f"{remote_root}/venv/bin/python",
    )
    shell_cmd = " ".join(shlex.quote(arg) for arg in argv)
    logger.info("Starting remote render (job={}, tier={}): {}", job.job_id, tier, shell_cmd)

    t0 = time.monotonic()
    rc, output = run_ssh(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        shell_cmd,
        stream=True,
    )
    wall_sec = round(time.monotonic() - t0, 3)
    if rc != 0:
        raise RuntimeError(f"Remote sample.py failed with exit code {rc}")

    stats = parse_sample_output(output)
    local_out = job_dir / f"out_{tier}.mp4"
    rsync_from_remote(
        gpu.ssh_host,
        gpu.ssh_port,
        gpu.ssh_user,
        gpu.ssh_key_path,
        str(remote_out),
        local_out,
    )

    out_probe = probe_video(local_out)
    render_sec = stats["render_sec"]
    render_cost = round((wall_sec / 3600.0) * gpu.price_per_hour, 6)
    inference_only_cost = round((render_sec / 3600.0) * gpu.price_per_hour, 6)

    result = RenderResult(
        job_id=job.job_id,
        mode=job.mode,
        tier=tier,
        width=out_probe.width,
        height=out_probe.height,
        num_frames=out_probe.frame_count,
        seed=seed,
        attn_backend=attn_backend,
        wall_sec=wall_sec,
        render_sec=render_sec,
        peak_alloc_gib=stats["peak_alloc_gib"],
        peak_reserved_gib=stats["peak_reserved_gib"],
        gpu_model=gpu.gpu_model,
        gpu_price_per_hour=gpu.price_per_hour,
        render_cost_usd=render_cost,
        inference_only_cost_usd=inference_only_cost,
        out_video_path=local_out,
        has_audio=out_probe.has_audio,
    )
    (job_dir / f"render_result_{tier}.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result

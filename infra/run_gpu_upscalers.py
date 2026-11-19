#!/usr/bin/env python3
"""Remote GPU benchmark runner for self-hosted video upscalers (Real-ESRGAN & SeedVR2)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_REALESRGAN_V3_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
)
_PEAK_VRAM_RE = re.compile(r"peak[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(?:GB|GiB)", re.IGNORECASE)


def _probe_basic(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    streams = json.loads(proc.stdout).get("streams", [])
    vs = next(s for s in streams if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    r_fps = vs.get("r_frame_rate") or "24/1"
    num, _, den = r_fps.partition("/")
    fps = float(num) / float(den) if den and float(den) else 24.0
    return {
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "fps": round(fps, 4),
        "num_frames": int(vs.get("nb_frames") or 0),
        "has_audio": has_audio,
    }


def _mux_audio_and_fit(
    video_in: Path,
    audio_src: Path,
    out_path: Path,
    target_w: int,
    target_h: int,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = _probe_basic(video_in)
    src_info = _probe_basic(audio_src)

    need_scale = (info["width"], info["height"]) != (target_w, target_h)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_in)]
    if src_info["has_audio"]:
        cmd.extend(["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0"])
    if need_scale:
        cmd.extend(["-vf", f"scale={target_w}:{target_h}:flags=lanczos"])
    cmd.extend(["-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    if src_info["has_audio"]:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)
    return _probe_basic(out_path)


def _ensure_file_download(url: str, dest: Path) -> float:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return 0.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"Downloading {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return round(time.perf_counter() - t0, 2)


def run_realesrgan(
    *,
    name: str,
    method: str,
    input_path: Path,
    out_path: Path,
    target_w: int,
    target_h: int,
    weight_path: Path,
    download_url: str | None = None,
) -> dict[str, Any]:
    """Run a Real-ESRGAN spandrel model in FP16 and downscale smoothly to target size."""
    import spandrel

    dl_sec = _ensure_file_download(download_url, weight_path) if download_url else 0.0
    if not weight_path.exists():
        raise FileNotFoundError(f"Real-ESRGAN weight missing: {weight_path}")

    t_wall0 = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_desc = spandrel.ModelLoader().load_from_file(weight_path)
    model = model_desc.model.eval().to(device="cuda", dtype=torch.float16)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    raw_video = out_path.with_name(f"{out_path.stem}_raw.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{target_w}x{target_h}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        str(raw_video),
    ]
    enc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert enc.stdin is not None

    t_infer0 = time.perf_counter()
    frames_done = 0
    try:
        with torch.inference_mode():
            while True:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor = (
                    torch.from_numpy(rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device="cuda", dtype=torch.float16)
                    .div_(255.0)
                )
                up = model(tensor).clamp_(0.0, 1.0)
                resized = torch.nn.functional.interpolate(
                    up,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp_(0.0, 1.0)
                out_np = (
                    resized.squeeze(0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                enc.stdin.write(np.ascontiguousarray(out_np).tobytes())
                frames_done += 1
        torch.cuda.synchronize()
    finally:
        cap.release()
        enc.stdin.close()
        rc = enc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder failed with exit code {rc}")

    infer_sec = time.perf_counter() - t_infer0
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
    del model
    torch.cuda.empty_cache()

    probe = _mux_audio_and_fit(raw_video, input_path, out_path, target_w, target_h)
    raw_video.unlink(missing_ok=True)
    wall_sec = round(dl_sec + (time.perf_counter() - t_wall0), 2)

    print(
        f"[{name}] done: frames={frames_done} infer={infer_sec:.2f}s peak={peak_vram_gib:.2f} GiB",
        flush=True,
    )
    return {
        "name": name,
        "method": method,
        "backend": "self_hosted_gpu",
        "out_filename": out_path.name,
        "frames": frames_done,
        "download_sec": dl_sec,
        "inference_sec": round(infer_sec, 2),
        "wall_sec": wall_sec,
        "peak_vram_gib": round(peak_vram_gib, 2),
        **probe,
    }


def _pre_download_seedvr2(seedvr2_dir: Path, model_dir: Path, dit_model: str) -> float:
    """Download DiT + VAE via huggingface_hub (hf_xet) and validate SeedVR2 cache before timing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    code = f"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, {str(seedvr2_dir)!r})
from src.utils.model_registry import MODEL_REGISTRY, DEFAULT_VAE
from src.utils.downloads import is_file_validated_cached, validate_file

model_dir = Path({str(model_dir)!r})
for fname in [{dit_model!r}, DEFAULT_VAE]:
    info = MODEL_REGISTRY[fname]
    dest = model_dir / fname
    if not dest.exists():
        print(f"hf_hub_download {{info.repo}}/{{fname}}...", flush=True)
        hf_hub_download(repo_id=info.repo, filename=fname, local_dir=str(model_dir))
    if not is_file_validated_cached(str(dest), str(model_dir)):
        print(f"Validating {{fname}}...", flush=True)
        assert validate_file(str(dest), info.sha256, str(model_dir)), f"Hash mismatch for {{fname}}"
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    return round(time.perf_counter() - t0, 2)


def _run_streaming_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    return rc, "".join(lines)


def _build_seedvr2_cmd(
    *,
    seedvr2_dir: Path,
    input_path: Path,
    raw_out: Path,
    model_dir: Path,
    dit_model: str,
    target_w: int,
    target_h: int,
    batch_size: int,
    low_vram_fallback: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(seedvr2_dir / "inference_cli.py"),
        str(input_path),
        "--output",
        str(raw_out),
        "--output_format",
        "mp4",
        "--model_dir",
        str(model_dir),
        "--dit_model",
        dit_model,
        "--resolution",
        str(min(target_w, target_h)),
        "--batch_size",
        str(batch_size),
        "--temporal_overlap",
        "4",
        "--prepend_frames",
        "0",
        "--video_backend",
        "ffmpeg",
        "--color_correction",
        "lab",
        "--debug",
    ]
    if low_vram_fallback:
        cmd.extend(
            [
                "--vae_encode_tiled",
                "--vae_decode_tiled",
                "--dit_offload_device",
                "cpu",
                "--vae_offload_device",
                "cpu",
            ]
        )
    return cmd


def run_seedvr2(
    *,
    name: str,
    method: str,
    dit_model: str,
    input_path: Path,
    out_path: Path,
    target_w: int,
    target_h: int,
    seedvr2_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    """Pre-download SeedVR2 weights via hf_xet, run inference_cli.py, and capture timing & VRAM."""
    dl_sec = _pre_download_seedvr2(seedvr2_dir, model_dir, dit_model)
    raw_out = out_path.with_name(f"{out_path.stem}_raw.mp4")
    if raw_out.exists():
        raw_out.unlink()

    cmd = _build_seedvr2_cmd(
        seedvr2_dir=seedvr2_dir,
        input_path=input_path,
        raw_out=raw_out,
        model_dir=model_dir,
        dit_model=dit_model,
        target_w=target_w,
        target_h=target_h,
        batch_size=33,
        low_vram_fallback=False,
    )

    t_infer0 = time.perf_counter()
    rc, output_log = _run_streaming_cmd(cmd)
    fallback_note = ""
    if rc != 0:
        fallback_note = "retried with batch_size=17 + tiled VAE + CPU offload"
        print(f"[{name}] Retrying with {fallback_note}...", flush=True)
        retry_cmd = _build_seedvr2_cmd(
            seedvr2_dir=seedvr2_dir,
            input_path=input_path,
            raw_out=raw_out,
            model_dir=model_dir,
            dit_model=dit_model,
            target_w=target_w,
            target_h=target_h,
            batch_size=17,
            low_vram_fallback=True,
        )
        t_infer0 = time.perf_counter()
        rc, output_log = _run_streaming_cmd(retry_cmd)
        if rc != 0:
            raise RuntimeError(f"SeedVR2 failed ({dit_model}) with exit code {rc}")

    infer_sec = time.perf_counter() - t_infer0
    if not raw_out.exists():
        raise FileNotFoundError(f"SeedVR2 did not create expected output {raw_out}")

    peak_matches = [float(m) for m in _PEAK_VRAM_RE.findall(output_log)]
    peak_vram_gib = round(max(peak_matches), 2) if peak_matches else 0.0

    probe = _mux_audio_and_fit(raw_out, input_path, out_path, target_w, target_h)
    raw_out.unlink(missing_ok=True)
    # Remove large DiT checkpoint after inference to keep container overlay storage bounded.
    (model_dir / dit_model).unlink(missing_ok=True)
    return {
        "name": name,
        "method": method,
        "backend": "self_hosted_gpu",
        "out_filename": out_path.name,
        "download_sec": dl_sec,
        "inference_sec": round(infer_sec, 2),
        "wall_sec": round(dl_sec + infer_sec, 2),
        "peak_vram_gib": peak_vram_gib,
        "notes": fallback_note or f"dl={dl_sec}s",
        **probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark self-hosted video upscalers on GPU.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-width", type=int, default=1280)
    parser.add_argument("--target-height", type=int, default=736)
    parser.add_argument("--seedvr2-dir", type=Path, default=Path("/workspace/SeedVR2"))
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models/SEEDVR2"))
    parser.add_argument(
        "--realesrgan-weights",
        type=Path,
        default=Path("/workspace/models/realesrgan/RealESRGAN_x2plus.pth"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    vram_total_gib = round(props.total_memory / (1024**3), 1)
    print(f"Running 5-model benchmark on {gpu_name} ({vram_total_gib} GiB)", flush=True)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    jobs: list[tuple[str, Any]] = [
        (
            "Real-ESRGAN x2plus",
            lambda: run_realesrgan(
                name="Real-ESRGAN x2plus",
                method="realesrgan-x2plus",
                input_path=args.input,
                out_path=args.out_dir / "out_480_realesrgan_x2.mp4",
                target_w=args.target_width,
                target_h=args.target_height,
                weight_path=args.realesrgan_weights,
            ),
        ),
        (
            "Real-ESRGAN general-x4v3",
            lambda: run_realesrgan(
                name="Real-ESRGAN general-x4v3",
                method="realesrgan-general-x4v3",
                input_path=args.input,
                out_path=args.out_dir / "out_480_realesrgan_v3.mp4",
                target_w=args.target_width,
                target_h=args.target_height,
                weight_path=args.realesrgan_weights.with_name("realesr-general-x4v3.pth"),
                download_url=_REALESRGAN_V3_URL,
            ),
        ),
        (
            "SeedVR2-3B FP16",
            lambda: run_seedvr2(
                name="SeedVR2-3B FP16",
                method="seedvr2-3b-fp16",
                dit_model="seedvr2_ema_3b_fp16.safetensors",
                input_path=args.input,
                out_path=args.out_dir / "out_480_seedvr2_3b.mp4",
                target_w=args.target_width,
                target_h=args.target_height,
                seedvr2_dir=args.seedvr2_dir,
                model_dir=args.model_dir,
            ),
        ),
        (
            "SeedVR2-7B FP8",
            lambda: run_seedvr2(
                name="SeedVR2-7B FP8",
                method="seedvr2-7b-fp8",
                dit_model="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
                input_path=args.input,
                out_path=args.out_dir / "out_480_seedvr2_7b.mp4",
                target_w=args.target_width,
                target_h=args.target_height,
                seedvr2_dir=args.seedvr2_dir,
                model_dir=args.model_dir,
            ),
        ),
        (
            "SeedVR2-7B Sharp FP8",
            lambda: run_seedvr2(
                name="SeedVR2-7B Sharp FP8",
                method="seedvr2-7b-sharp-fp8",
                dit_model="seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
                input_path=args.input,
                out_path=args.out_dir / "out_480_seedvr2_7b_sharp.mp4",
                target_w=args.target_width,
                target_h=args.target_height,
                seedvr2_dir=args.seedvr2_dir,
                model_dir=args.model_dir,
            ),
        ),
    ]

    summary_path = args.out_dir / "remote_upscale_results.json"
    for idx, (label, runner) in enumerate(jobs, start=1):
        print(f"\n=== [{idx}/{len(jobs)}] {label} ===", flush=True)
        try:
            results.append(runner())
        except Exception as exc:
            print(f"ERROR in {label}: {exc}", flush=True)
            errors.append({"name": label, "error": str(exc)})
        summary_path.write_text(
            json.dumps(
                {
                    "gpu_name": gpu_name,
                    "vram_total_gib": vram_total_gib,
                    "results": results,
                    "errors": errors,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"\nSaved {summary_path} ({len(results)} ok, {len(errors)} failed)", flush=True)


if __name__ == "__main__":
    main()

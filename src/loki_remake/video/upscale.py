"""Video upscaling backends (GPUniq Topaz API + audio muxing) and visual comparison grids."""

from __future__ import annotations

import base64
import math
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import httpx
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from loki_remake.gpuniq.client import GPUniqAPIError, GPUniqClient
from loki_remake.schemas import UpscaleResult, VideoProbe
from loki_remake.video.normalize import probe_video

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def ensure_audio_and_h264(
    video_path: Path,
    audio_source_path: Path,
    out_path: Path,
    *,
    target_size: tuple[int, int] | None = None,
) -> VideoProbe:
    """Normalize upscaled MP4 to H.264 yuv420p, optional exact target_size, and mux source audio."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v_probe = probe_video(video_path)
    src_probe = probe_video(audio_source_path)

    needs_scale = target_size is not None and (v_probe.width, v_probe.height) != target_size
    tmp_path = out_path.with_suffix(".tmp.mp4")

    cmd: list[str] = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path)]
    if src_probe.has_audio:
        cmd.extend(["-i", str(audio_source_path), "-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0"])

    if needs_scale and target_size is not None:
        tw, th = target_size
        cmd.extend(["-vf", f"scale={tw}:{th}:flags=lanczos"])

    cmd.extend(["-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    if src_probe.has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

    cmd.append(str(tmp_path))
    subprocess.run(cmd, check=True)
    tmp_path.replace(out_path)
    return probe_video(out_path)


def upload_temp_https(
    video_path: Path,
    *,
    http_client: httpx.Client | None = None,
) -> str:
    """Upload a temporary public clip to an expiring HTTPS host and verify accessibility."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for upload: {video_path}")

    owns = http_client is None
    client = http_client or httpx.Client(timeout=120.0, follow_redirects=True)
    errors: list[str] = []

    try:
        # 1. uguu.se (3h auto-expiring static nginx host with Content-Type: video/mp4)
        try:
            with video_path.open("rb") as fh:
                resp = client.post(
                    "https://uguu.se/upload",
                    files={"files[]": (video_path.name, fh, "video/mp4")},
                )
            payload = resp.json()
            files_list = payload.get("files") or []
            url = str(files_list[0].get("url") or "").strip() if files_list else ""
            if resp.is_success and url.startswith("https://"):
                check = client.head(url, timeout=15.0)
                if check.status_code == 200:
                    logger.info("Uploaded temporary clip to {}", url)
                    return url
            errors.append(f"uguu HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as exc:
            errors.append(f"uguu error: {exc}")

        # 2. Fallback: litterbox.catbox.moe (1h auto-expiring public temporary file host)
        try:
            with video_path.open("rb") as fh:
                resp = client.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data={"reqtype": "fileupload", "time": "1h"},
                    files={"fileToUpload": (video_path.name, fh, "video/mp4")},
                )
            url = resp.text.strip()
            if resp.is_success and url.startswith("https://"):
                check = client.head(url, timeout=15.0)
                if check.status_code in (200, 405):
                    logger.info("Uploaded temporary clip to {}", url)
                    return url
            errors.append(f"litterbox HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as exc:
            errors.append(f"litterbox error: {exc}")
    finally:
        if owns:
            client.close()

    raise RuntimeError("Failed to upload temporary clip to HTTPS hosts: " + "; ".join(errors))


def _is_url_rejection(exc: GPUniqAPIError) -> bool:
    msg = (str(exc) + " " + (exc.error_code or "")).lower()
    if exc.status_code in (400, 413, 422):
        return True
    return any(tok in msg for tok in ("invalid_request", "video_url", "fetch", "url", "data:"))


def upscale_via_topaz(
    client: GPUniqClient,
    video_path: Path,
    out_path: Path,
    *,
    model: str,
    name: str | None = None,
    output_width: int = 1280,
    output_height: int = 736,
    video_url: str | None = None,
    allow_temp_upload: bool = False,
    poll_interval_sec: float = 3.0,
    timeout_sec: float = 600.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    download_client: httpx.Client | None = None,
) -> UpscaleResult:
    """Upscale a video via GPUniq Topaz Video AI API and preserve audio."""
    probe = probe_video(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload_base: dict[str, Any] = {
        "model": model,
        "source_width": probe.width,
        "source_height": probe.height,
        "source_duration": round(probe.duration_sec, 2),
        "source_frame_rate": max(1, round(probe.fps)),
        "output_width": output_width,
        "output_height": output_height,
    }

    t0 = time.monotonic()
    if video_url is not None:
        media_url = video_url
        kickoff = client.create_topaz_video_job({**payload_base, "video_url": media_url})
    else:
        b64_video = base64.b64encode(video_path.read_bytes()).decode("ascii")
        data_uri = f"data:video/mp4;base64,{b64_video}"
        try:
            kickoff = client.create_topaz_video_job({**payload_base, "video_url": data_uri})
        except GPUniqAPIError as exc:
            if not allow_temp_upload or not _is_url_rejection(exc):
                raise
            logger.warning("Topaz rejected data URI ({}); uploading temp HTTPS URL", exc)
            https_url = upload_temp_https(video_path, http_client=download_client)
            kickoff = client.create_topaz_video_job({**payload_base, "video_url": https_url})

    job_id = str(kickoff["job_id"])
    est_cost = kickoff.get("estimated_cost_usd")
    logger.info(
        "Topaz job started: model={} job_id={} estimated_cost_usd={}",
        model,
        job_id,
        est_cost,
    )

    deadline = time.monotonic() + timeout_sec
    completed_data: dict[str, Any] | None = None
    while True:
        status_data = client.get_topaz_video_job(job_id)
        status = str(status_data.get("status") or "").lower()
        if status == "completed":
            completed_data = status_data
            break
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"Topaz job {job_id} ({model}) failed: {status_data}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Topaz job {job_id} ({model}) timed out after {timeout_sec}s")
        sleep_fn(poll_interval_sec)

    video_info = completed_data.get("video") or {}
    result_url = str(video_info.get("url") or "")
    if not result_url:
        raise RuntimeError(f"Topaz job {job_id} completed without video.url: {completed_data}")

    owns_dl = download_client is None
    dl_client = download_client or httpx.Client(timeout=180.0, follow_redirects=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        raw_dl_path = Path(tmp.name)

    try:
        dl_resp = dl_client.get(result_url)
        dl_resp.raise_for_status()
        raw_dl_path.write_bytes(dl_resp.content)

        raw_probe = probe_video(raw_dl_path)
        if not raw_probe.has_audio or (raw_probe.width, raw_probe.height) != (
            output_width,
            output_height,
        ):
            final_probe = ensure_audio_and_h264(
                raw_dl_path,
                video_path,
                out_path,
                target_size=(output_width, output_height),
            )
        else:
            raw_dl_path.replace(out_path)
            final_probe = raw_probe
    finally:
        if raw_dl_path.exists():
            raw_dl_path.unlink(missing_ok=True)
        if owns_dl:
            dl_client.close()

    wall_sec = round(time.monotonic() - t0, 2)
    credits_raw = completed_data.get("credits")
    credits_val = float(credits_raw) if credits_raw is not None else None
    cost_usd = float(completed_data.get("cost_usd") or 0.0)

    return UpscaleResult(
        name=name or model,
        method=model,
        backend="topaz_api",
        source_video_path=video_path,
        out_video_path=out_path,
        width=final_probe.width,
        height=final_probe.height,
        fps=final_probe.fps,
        num_frames=final_probe.frame_count,
        has_audio=final_probe.has_audio,
        wall_sec=wall_sec,
        inference_sec=wall_sec,
        credits=credits_val,
        upscale_cost_usd=cost_usd,
        notes=f"job_id={job_id}, est=${est_cost}",
    )


def _extract_frame_pil(
    video_path: Path, frame_index: int, target_size: tuple[int, int]
) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idx = max(0, min(total - 1, frame_index))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise ValueError(f"Could not read frame {idx} from {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img
    finally:
        cap.release()


def _make_grid_canvas(
    tiles: list[tuple[Image.Image, str, str]],
    *,
    cols: int,
    banner_h: int = 56,
    pad: int = 12,
) -> Image.Image:
    if not tiles:
        raise ValueError("Cannot build comparison grid with zero tiles")
    tile_w, tile_h = tiles[0][0].size
    rows = math.ceil(len(tiles) / cols)
    cell_h = tile_h + banner_h
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * cell_h + (rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(max(15, tile_w // 26))
    sub_font = _load_font(max(12, tile_w // 34))

    for i, (tile_img, title, subtitle) in enumerate(tiles):
        if tile_img.size != (tile_w, tile_h):
            tile_img = tile_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        r = i // cols
        c = i % cols
        x = pad + c * (tile_w + pad)
        y = pad + r * (cell_h + pad)
        canvas.paste(tile_img, (x, y))
        draw.rectangle([x, y + tile_h, x + tile_w - 1, y + cell_h - 1], fill=(28, 33, 45))
        draw.rectangle([x, y, x + tile_w - 1, y + cell_h - 1], outline=(71, 85, 105), width=2)
        draw.text((x + 10, y + tile_h + 7), title, fill=(248, 250, 252), font=title_font)
        draw.text((x + 10, y + tile_h + 30), subtitle, fill=(148, 163, 184), font=sub_font)

    return canvas


def build_comparison_grids(
    variants: list[dict[str, Any]],
    *,
    frame_index: int = 200,
    target_size: tuple[int, int] = (1280, 736),
    crop_box: tuple[int, int, int, int] = (460, 55, 920, 455),
    crops_out_path: Path,
    full_out_path: Path,
    full_subset_indices: list[int] | None = None,
) -> None:
    """Render zoomed crop and full-frame visual comparison grids across all variants."""
    if not variants:
        raise ValueError("variants must not be empty")

    tw, th = target_size
    x0, y0, x1, y1 = crop_box
    x0 = max(0, min(tw - 2, x0))
    y0 = max(0, min(th - 2, y0))
    x1 = max(x0 + 1, min(tw, x1))
    y1 = max(y0 + 1, min(th, y1))

    full_frames: list[Image.Image] = []
    crop_tiles: list[tuple[Image.Image, str, str]] = []

    for v in variants:
        v_frame = int(v.get("frame_index", frame_index))
        frame = _extract_frame_pil(Path(v["path"]), v_frame, target_size)
        full_frames.append(frame)
        cropped = frame.crop((x0, y0, x1, y1))
        crop_tiles.append((cropped, str(v["title"]), str(v["subtitle"])))

    crops_cols = min(4, len(crop_tiles))
    crops_grid = _make_grid_canvas(crop_tiles, cols=crops_cols, banner_h=58, pad=12)
    crops_out_path.parent.mkdir(parents=True, exist_ok=True)
    crops_grid.save(crops_out_path, format="JPEG", quality=93)

    if full_subset_indices is None:
        subset_indices = list(range(min(4, len(variants))))
    else:
        subset_indices = [idx for idx in full_subset_indices if 0 <= idx < len(variants)]

    full_tile_size = (tw // 2, th // 2)
    full_tiles: list[tuple[Image.Image, str, str]] = []
    for idx in subset_indices:
        v = variants[idx]
        thumb = full_frames[idx].resize(full_tile_size, Image.Resampling.LANCZOS)
        full_tiles.append((thumb, str(v["title"]), str(v["subtitle"])))

    full_grid = _make_grid_canvas(
        full_tiles, cols=2 if len(full_tiles) > 1 else 1, banner_h=54, pad=12
    )
    full_out_path.parent.mkdir(parents=True, exist_ok=True)
    full_grid.save(full_out_path, format="JPEG", quality=93)

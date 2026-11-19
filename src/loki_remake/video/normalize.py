"""Video probing and normalization for MiniMax-H3 / Viggle-Remake."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from loguru import logger

from loki_remake.schemas import NormalizedVideo, RenderTier, VideoProbe

MIN_FRAMES = 90
MAX_FRAMES = 345
TARGET_FPS = 24
_MAX_ASPECT = 2.31
_MIN_ASPECT = 1.0 / _MAX_ASPECT

_SUPPORTED_RATIOS: list[tuple[str, float]] = [
    ("1:1", 1.0),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:3", 4 / 3),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
]


def _parse_rate(rate_str: str | None) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        den = float(den_s)
        return float(num_s) / den if den else 0.0
    return float(rate_str)


def probe_video(path: Path) -> VideoProbe:
    """Inspect video/audio streams using ffprobe."""
    if not path.exists():
        raise FileNotFoundError(f"Source video not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)

    streams = payload.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v_stream is None:
        raise ValueError(f"No video stream found in {path}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    width = int(v_stream["width"])
    height = int(v_stream["height"])

    # Honor rotation metadata so portrait phone videos report oriented dimensions.
    for sd in v_stream.get("side_data_list", []):
        rotation = abs(int(sd.get("rotation", 0)))
        if rotation in (90, 270):
            width, height = height, width
            break
    else:
        rot_tag = abs(int(v_stream.get("tags", {}).get("rotate", 0)))
        if rot_tag in (90, 270):
            width, height = height, width

    fps = _parse_rate(v_stream.get("r_frame_rate")) or _parse_rate(
        v_stream.get("avg_frame_rate")
    )
    if fps <= 0:
        fps = float(TARGET_FPS)

    fmt_dur = payload.get("format", {}).get("duration")
    stream_dur = v_stream.get("duration")
    duration_sec = float(stream_dur or fmt_dur or 0.0)

    nb_frames_raw = v_stream.get("nb_frames")
    if nb_frames_raw and str(nb_frames_raw).isdigit():
        frame_count = int(nb_frames_raw)
    else:
        frame_count = max(1, round(duration_sec * fps))

    if duration_sec <= 0 and frame_count > 0:
        duration_sec = frame_count / fps

    return VideoProbe(
        width=width,
        height=height,
        fps=round(fps, 4),
        duration_sec=round(duration_sec, 4),
        frame_count=frame_count,
        has_audio=has_audio,
    )


def snap_frame_count(raw_frames: int) -> int:
    """Snap frame count to MiniMax-H3's 17n+5 grid within [90, 345]."""
    if raw_frames <= MIN_FRAMES:
        return MIN_FRAMES
    if raw_frames >= MAX_FRAMES:
        return MAX_FRAMES

    n = math.ceil((raw_frames - 5) / 17)
    snapped = 17 * n + 5
    return max(MIN_FRAMES, min(MAX_FRAMES, snapped))


def _nearest_supported_ratio(width: int, height: int) -> str:
    aspect = width / height
    return min(_SUPPORTED_RATIOS, key=lambda item: abs(math.log(aspect / item[1])))[0]


def compute_canvas(src_w: int, src_h: int, short_edge: RenderTier) -> tuple[int, int, str]:
    """Compute target (width, height) multiples of 32 and closest image ratio slug."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source dimensions: {src_w}x{src_h}")

    raw_aspect = src_w / src_h
    clamped_aspect = max(_MIN_ASPECT, min(_MAX_ASPECT, raw_aspect))

    if src_w >= src_h:
        h = int(short_edge)
        # Snap landscape 16:9 exactly to 832x480 or 1280x736 matching Viggle-Remake
        if abs(clamped_aspect - 16 / 9) < 0.03:
            w = short_edge * 16 // 9 // 32 * 32
        else:
            w = max(32, round((short_edge * clamped_aspect) / 32) * 32)
    else:
        w = int(short_edge)
        inv_aspect = 1.0 / clamped_aspect
        if abs(inv_aspect - 16 / 9) < 0.03:
            h = short_edge * 16 // 9 // 32 * 32
        else:
            h = max(32, round((short_edge * inv_aspect) / 32) * 32)

    ratio_slug = _nearest_supported_ratio(w, h)
    return w, h, ratio_slug


def normalize_video(
    source: Path,
    out_dir: Path,
    short_edge: RenderTier = 480,
    max_frames: int = MAX_FRAMES,
) -> NormalizedVideo:
    """Normalize source video to 24 fps CFR, multiple-of-32 canvas, and 17n+5 frame grid."""
    if max_frames < MIN_FRAMES or max_frames > MAX_FRAMES or (max_frames - 5) % 17 != 0:
        raise ValueError(
            f"max_frames must be 17n+5 in [{MIN_FRAMES}, {MAX_FRAMES}], got {max_frames}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_video(source)
    w, h, ratio_slug = compute_canvas(probe.width, probe.height, short_edge)

    est_24fps_frames = max(1, math.floor(probe.duration_sec * TARGET_FPS + 1e-6))
    capped_frames = min(est_24fps_frames, max_frames)
    target_frames = snap_frame_count(capped_frames)
    padded_frames = max(0, target_frames - est_24fps_frames)

    video_out = out_dir / "cond.mp4"
    vf_parts = [
        f"fps={TARGET_FPS}",
        f"scale={w}:{h}:force_original_aspect_ratio=increase",
        f"crop={w}:{h}",
    ]
    if padded_frames > 0:
        # Clone tail frame so frame count meets 17n+5 when source is shorter than target_frames.
        vf_parts.append(f"tpad=stop_mode=clone:stop={padded_frames + 2}")

    video_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        ",".join(vf_parts),
        "-fps_mode",
        "cfr",
        "-frames:v",
        str(target_frames),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        str(video_out),
    ]
    logger.debug("Running ffmpeg normalize: {}", " ".join(video_cmd))
    subprocess.run(video_cmd, check=True)

    actual_probe = probe_video(video_out)
    audio_out: Path | None = None
    target_dur = target_frames / float(TARGET_FPS)

    if probe.has_audio:
        audio_candidate = out_dir / "audio.m4a"
        audio_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-t",
            f"{target_dur:.4f}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(audio_candidate),
        ]
        subprocess.run(audio_cmd, check=True)
        audio_out = audio_candidate

    return NormalizedVideo(
        source_path=source,
        video_path=video_out,
        audio_path=audio_out,
        width=actual_probe.width,
        height=actual_probe.height,
        fps=TARGET_FPS,
        num_frames=actual_probe.frame_count,
        duration_sec=round(actual_probe.frame_count / float(TARGET_FPS), 4),
        padded_frames=padded_frames,
        aspect_ratio=ratio_slug,
    )

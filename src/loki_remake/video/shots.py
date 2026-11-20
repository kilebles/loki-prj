"""Shot boundary detection, sharpest keyframe extraction, and contact sheet builder."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from loki_remake.schemas import ShotInfo

_MIN_SHOT_FRAMES = 12
_HIST_CORR_CUT_THRESHOLD = 0.55
_LUMA_JUMP_THRESHOLD = 28.0
_LUMA_CUT_THRESHOLD = 45.0


def format_timestamp(frame_idx: int, fps: float = 24.0) -> str:
    """Format a 0-based frame index into MM:SS.mmm."""
    total_ms = max(0, round((frame_idx / fps) * 1000))
    minutes, rem_ms = divmod(total_ms, 60_000)
    seconds, ms = divmod(rem_ms, 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def _hsv_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _read_frames(video_path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video with OpenCV: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Zero frames decoded from {video_path}")
    return frames


def _detect_cut_indices(frames: list[np.ndarray]) -> list[int]:
    if len(frames) < _MIN_SHOT_FRAMES * 2:
        return []

    cuts: list[int] = []
    prev_hist = _hsv_histogram(frames[0])
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    last_cut = 0

    for idx in range(1, len(frames)):
        hist = _hsv_histogram(frames[idx])
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY).astype(np.float32)

        corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
        luma_diff = float(np.mean(np.abs(gray - prev_gray)))

        if idx - last_cut >= _MIN_SHOT_FRAMES and (
            corr < _HIST_CORR_CUT_THRESHOLD
            or luma_diff > _LUMA_CUT_THRESHOLD
            or (corr < 0.75 and luma_diff > _LUMA_JUMP_THRESHOLD)
        ):
            cuts.append(idx)
            last_cut = idx

        prev_hist = hist
        prev_gray = gray

    # Drop trailing stub shorter than half a second by merging into previous shot.
    if cuts and (len(frames) - cuts[-1]) < _MIN_SHOT_FRAMES:
        cuts.pop()

    return cuts


def _pick_sharpest_in_span(
    frames: list[np.ndarray], start: int, end: int
) -> tuple[int, float]:
    span_len = end - start + 1
    inner_start = start + 2 if span_len > 8 else start
    inner_end = end - 2 if span_len > 8 else end

    best_idx = inner_start
    best_score = -1.0
    step = 1 if (inner_end - inner_start) <= 60 else 2
    for idx in range(inner_start, inner_end + 1, step):
        score = _sharpness(frames[idx])
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def _build_contact_sheet(
    frames: list[np.ndarray],
    shots: list[ShotInfo],
    out_path: Path,
    fps: float = 24.0,
    max_tiles: int = 16,
) -> Path:
    n = len(frames)
    tile_count = min(max_tiles, n)
    indices = sorted({round(i * (n - 1) / max(1, tile_count - 1)) for i in range(tile_count)})

    def _shot_for(frame_idx: int) -> int:
        for s in shots:
            if s.start_frame <= frame_idx <= s.end_frame:
                return s.index
        return shots[-1].index if shots else 1

    src_h, src_w = frames[0].shape[:2]
    tile_w = 280
    tile_h = max(120, round(tile_w * src_h / src_w))
    cols = min(4, len(indices))
    rows = math.ceil(len(indices) / cols)
    pad = 8
    label_h = 24

    sheet = Image.new(
        "RGB",
        (cols * (tile_w + pad) + pad, rows * (tile_h + label_h + pad) + pad),
        (18, 18, 22),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for pos, frame_idx in enumerate(indices):
        r, c = divmod(pos, cols)
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + label_h + pad)

        rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        tile_img = Image.fromarray(rgb).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(tile_img, (x, y))

        shot_num = _shot_for(frame_idx)
        label = f"[Shot {shot_num}] {format_timestamp(frame_idx, fps)} f={frame_idx}"
        draw.rectangle([x, y + tile_h, x + tile_w, y + tile_h + label_h], fill=(32, 34, 42))
        draw.text((x + 6, y + tile_h + 5), label, fill=(235, 235, 240), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=90)
    return out_path


def detect_shots_and_keyframes(
    video_path: Path,
    num_frames: int,
    out_dir: Path,
    max_refs: int = 2,
    fps: float = 24.0,
) -> tuple[list[ShotInfo], list[tuple[int, int, Path]], Path]:
    """
    Detect hard cuts, extract sharpest keyframes per shot, and build a contact sheet.

    Returns:
        - shots: all detected ShotInfo items (1-based index)
        - selected_pins: list of (shot_index, frame_idx, keyframe_path) up to min(max_refs, 4)
        - contact_sheet_path: JPEG contact sheet for VLM prompting
    """
    max_refs = max(1, min(max_refs, 4))
    frames = _read_frames(video_path, max_frames=num_frames)
    n = len(frames)

    cuts = _detect_cut_indices(frames)
    boundaries = [0, *cuts, n]

    keyframes_dir = out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    shots: list[ShotInfo] = []
    for shot_idx, (start_f, end_excl) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True), start=1
    ):
        end_f = end_excl - 1
        kf_idx, score = _pick_sharpest_in_span(frames, start_f, end_f)
        kf_path = keyframes_dir / f"shot_{shot_idx}_f{kf_idx}.png"
        cv2.imwrite(str(kf_path), frames[kf_idx])

        shots.append(
            ShotInfo(
                index=shot_idx,
                start_frame=start_f,
                end_frame=end_f,
                start_sec=round(start_f / fps, 4),
                end_sec=round(end_f / fps, 4),
                keyframe_index=kf_idx,
                keyframe_sec=round(kf_idx / fps, 4),
                sharpness_score=round(score, 2),
                keyframe_path=kf_path,
            )
        )

    # Select up to max_refs pins across shots. On a single long take (>=175 frames),
    # pick an additional sharp frame in the last third so identity holds over 14.4s.
    selected_pins: list[tuple[int, int, Path]] = []
    if len(shots) == 1 and n >= 175 and max_refs >= 2:
        first_end = min(n - 1, n // 3)
        late_start = max(first_end + 1, (2 * n) // 3)
        f1, _ = _pick_sharpest_in_span(frames, 0, first_end)
        f2, _ = _pick_sharpest_in_span(frames, late_start, n - 1)
        p1 = keyframes_dir / f"shot_1_pin1_f{f1}.png"
        p2 = keyframes_dir / f"shot_1_pin2_f{f2}.png"
        cv2.imwrite(str(p1), frames[f1])
        cv2.imwrite(str(p2), frames[f2])
        selected_pins = [(1, f1, p1), (1, f2, p2)]
    else:
        for s in shots[:max_refs]:
            selected_pins.append((s.index, s.keyframe_index, s.keyframe_path))

    contact_sheet_path = _build_contact_sheet(
        frames=frames,
        shots=shots,
        out_path=out_dir / "contact_sheet.jpg",
        fps=fps,
    )
    return shots, selected_pins, contact_sheet_path

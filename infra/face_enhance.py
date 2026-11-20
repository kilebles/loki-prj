#!/usr/bin/env python3
"""Commercial-safe face restoration (YuNet + Umeyama + GFPGANv1.4/RestoreFormer++ + MediaPipe)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

# Canonical 512x512 FFHQ 5-point landmarks matching facexlib/GFPGAN/RestoreFormer alignment.
_FFHQ_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def _umeyama_similarity_2d(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Closed-form 2D similarity (no reflection) via Umeyama SVD; immune to LMEDS snaps."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = float(np.sum(sc * sc) / max(len(src), 1))
    if var_s < 1e-8:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    cov = (dc.T @ sc) / len(src)
    u_mat, s_vals, vt_mat = np.linalg.svd(cov)
    sign_diag = np.ones(2, dtype=np.float64)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0.0:
        sign_diag[1] = -1.0
    rot = u_mat @ np.diag(sign_diag) @ vt_mat
    scale = float(np.sum(s_vals * sign_diag) / var_s)
    trans = mu_d - scale * (rot @ mu_s)

    mat = np.zeros((2, 3), dtype=np.float32)
    mat[:, :2] = (scale * rot).astype(np.float32)
    mat[:, 2] = trans.astype(np.float32)
    return mat


def _affine_to_params(mat: np.ndarray) -> np.ndarray:
    sc = max(float(np.hypot(mat[0, 0], mat[1, 0])), 1e-6)
    return np.array(
        [
            float(mat[0, 2]),
            float(mat[1, 2]),
            float(np.log(sc)),
            float(mat[0, 0] / sc),
            float(mat[1, 0] / sc),
        ],
        dtype=np.float64,
    )


def _params_to_affine(params: np.ndarray) -> np.ndarray:
    tx, ty, log_s, cos_t, sin_t = [float(x) for x in params]
    norm = max(float(np.hypot(cos_t, sin_t)), 1e-6)
    c, s = cos_t / norm, sin_t / norm
    sc = float(np.exp(log_s))
    return np.array(
        [
            [sc * c, -sc * s, tx],
            [sc * s, sc * c, ty],
        ],
        dtype=np.float32,
    )


def _gaussian_smooth_1d(arr: np.ndarray, sigma: float = 2.0, radius: int = 5) -> np.ndarray:
    if len(arr) <= 2:
        return arr.copy()
    r = min(radius, len(arr) - 1)
    offsets = np.arange(-r, r + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    padded = np.pad(arr, [(r, r)] + [(0, 0)] * (arr.ndim - 1), mode="edge")
    out = np.zeros_like(arr, dtype=np.float64)
    for idx_w, w in enumerate(weights):
        out += w * padded[idx_w : idx_w + len(arr)]
    return out


def _smooth_track_landmarks(
    obs: list[tuple[int, np.ndarray, float]],
    *,
    shot_start: int,
    shot_end: int,
    max_gap: int = 5,
    min_len: int = 4,
    sigma: float = 2.0,
) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """Interpolate short gaps, reject outlier snaps, and smooth Umeyama transforms bidir."""
    if len(obs) < min_len:
        return {}

    obs_sorted = sorted(obs, key=lambda x: x[0])
    runs: list[list[tuple[int, np.ndarray, float]]] = [[obs_sorted[0]]]
    for item in obs_sorted[1:]:
        if item[0] - runs[-1][-1][0] <= max_gap + 1:
            runs[-1].append(item)
        else:
            runs.append([item])

    result: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for run in runs:
        if len(run) < min_len:
            continue

        # Step A: 5-point window-median outlier detection (clean neighbors are never corrupted),
        # then interpolate flagged landmark glitches linearly from nearest clean frames.
        obs_arr = np.stack([item[1] for item in run], axis=0).astype(np.float64)
        obs_frames = np.array([item[0] for item in run], dtype=np.float64)
        obs_iods = np.linalg.norm(obs_arr[:, 0] - obs_arr[:, 1], axis=1)
        ref_size = max(float(np.median(obs_iods)) * 2.2, 24.0)
        m = len(run)
        is_outlier = np.zeros((m, 5), dtype=bool)
        if m >= 4:
            for k in range(m):
                lo, hi = max(0, k - 2), min(m, k + 3)
                med = np.median(obs_arr[lo:hi], axis=0)
                thresh = (0.30 if k in (0, m - 1) else 0.20) * ref_size
                is_outlier[k] = np.linalg.norm(obs_arr[k] - med, axis=1) > thresh
        obs_cleaned = obs_arr.copy()
        for pt_i in range(5):
            good_idx = np.where(~is_outlier[:, pt_i])[0]
            if len(good_idx) >= 2 and bool(is_outlier[:, pt_i].any()):
                bad_idx = np.where(is_outlier[:, pt_i])[0]
                for dim in range(2):
                    obs_cleaned[bad_idx, pt_i, dim] = np.interp(
                        obs_frames[bad_idx],
                        obs_frames[good_idx],
                        obs_arr[good_idx, pt_i, dim],
                    )

        f_start = run[0][0]
        f_end = run[-1][0]
        n = f_end - f_start + 1
        dense_lm = np.zeros((n, 5, 2), dtype=np.float64)
        dense_area = np.zeros(n, dtype=np.float64)

        for k, (f_idx, _raw_lm, area) in enumerate(run):
            lm5 = obs_cleaned[k]
            pos = f_idx - f_start
            dense_lm[pos] = lm5
            dense_area[pos] = area
            if k + 1 < len(run):
                next_f, _next_raw, next_area = run[k + 1]
                next_lm = obs_cleaned[k + 1]
                gap = next_f - f_idx
                for step in range(1, gap):
                    alpha = step / float(gap)
                    dense_lm[pos + step] = (1.0 - alpha) * lm5 + alpha * next_lm
                    dense_area[pos + step] = (1.0 - alpha) * area + alpha * next_area

        # Step B: bidirectional Gaussian smoothing of landmarks + Umeyama similarity params.
        smooth_lm = _gaussian_smooth_1d(dense_lm, sigma=sigma, radius=5)
        raw_params = np.stack(
            [
                _affine_to_params(_umeyama_similarity_2d(smooth_lm[k], _FFHQ_TEMPLATE_512))
                for k in range(n)
            ],
            axis=0,
        )
        smooth_params = _gaussian_smooth_1d(raw_params, sigma=sigma, radius=5)

        for pos in range(n):
            f_idx = f_start + pos
            affine = _params_to_affine(smooth_params[pos])
            fade_in = 1.0 if f_start <= shot_start + 1 else min(1.0, (pos + 1) / 3.0)
            fade_out = 1.0 if f_end >= shot_end - 1 else min(1.0, (n - pos) / 3.0)
            fade = float(min(fade_in, fade_out))
            result[f_idx] = (smooth_lm[pos].astype(np.float32), affine, fade)

    return result


def _collect_smoothed_tracks(
    input_path: Path,
    detector: Any,
    *,
    score_thresh: float = 0.62,
) -> tuple[list[dict[int, tuple[float, np.ndarray, np.ndarray, float]]], int]:
    """Pass 1: detect hard shot cuts & build gap-filled smoothed face trajectories."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    shot_start = 0
    cuts = 0
    prev_small: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            hsv = cv2.cvtColor(
                cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV
            )
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_small is not None and prev_hist is not None:
                mae = float(np.abs(small - prev_small).mean())
                corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                if mae > 48.0 or (corr < 0.72 and mae > 20.0):
                    cuts += 1
                    for trk in active:
                        trk["shot_end"] = frame_idx - 1
                        finished.append(trk)
                    active = []
                    shot_start = frame_idx
            prev_small = small
            prev_hist = hist

            _, faces = detector.detect(frame_bgr)
            if faces is not None and len(faces) > 0:
                candidates = [row for row in faces if float(row[14]) >= score_thresh]
                candidates.sort(
                    key=lambda row: float(row[14]) * float(row[2] * row[3]),
                    reverse=True,
                )
                candidates = candidates[:4]

                used_ids: set[int] = set()
                for row in candidates:
                    bw, bh = float(row[2]), float(row[3])
                    area = bw * bh
                    lm5 = np.array(
                        [
                            [row[4], row[5]],
                            [row[6], row[7]],
                            [row[8], row[9]],
                            [row[10], row[11]],
                            [row[12], row[13]],
                        ],
                        dtype=np.float32,
                    )
                    max_dist = max(45.0, 0.55 * bw)
                    best_idx: int | None = None
                    best_dist = max_dist
                    for tidx, trk in enumerate(active):
                        if tidx in used_ids:
                            continue
                        if frame_idx - trk["last_f"] > 6:
                            continue
                        dist = float(np.mean(np.linalg.norm(lm5 - trk["last_lm"], axis=1)))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = tidx

                    if best_idx is not None:
                        active[best_idx]["obs"].append((frame_idx, lm5, area))
                        active[best_idx]["last_lm"] = lm5
                        active[best_idx]["last_f"] = frame_idx
                        active[best_idx]["mean_area"] = (
                            0.7 * active[best_idx]["mean_area"] + 0.3 * area
                        )
                        used_ids.add(best_idx)
                    else:
                        active.append(
                            {
                                "id": len(finished) + len(active),
                                "shot_start": shot_start,
                                "shot_end": frame_idx,
                                "obs": [(frame_idx, lm5, area)],
                                "last_lm": lm5,
                                "last_f": frame_idx,
                                "mean_area": area,
                            }
                        )
                        used_ids.add(len(active) - 1)

            frame_idx += 1
    finally:
        cap.release()

    for trk in active:
        trk["shot_end"] = frame_idx - 1
        finished.append(trk)

    per_frame: list[dict[int, tuple[float, np.ndarray, np.ndarray, float]]] = [
        {} for _ in range(frame_idx)
    ]
    for trk in finished:
        smoothed = _smooth_track_landmarks(
            trk["obs"],
            shot_start=int(trk["shot_start"]),
            shot_end=int(trk["shot_end"]),
        )
        tid = int(trk["id"])
        area = float(trk["mean_area"])
        for f_idx, (lm5, affine, fade) in smoothed.items():
            per_frame[f_idx][tid] = (area, lm5, affine, fade)

    return per_frame, cuts


def _make_feathered_mask(size: int = 512) -> np.ndarray:
    """Create a soft elliptical alpha mask inside 512x512 so pasted face has zero seam."""
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, int(size * 0.52))
    axes = (int(size * 0.40), int(size * 0.44))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), sigmaX=18, sigmaY=18)
    mask = np.clip(mask, 0.0, 1.0)
    return mask[:, :, None]


def _make_semantic_mask(
    crop_bgr: np.ndarray,
    seg_interp: Any | None,
    fallback_mask: np.ndarray,
) -> np.ndarray:
    """Occlusion-aware MediaPipe face mask (keeps hands, glasses, clothes, exterior hair)."""
    if seg_interp is None:
        return fallback_mask

    h, w = crop_bgr.shape[:2]
    in_det = seg_interp.get_input_details()[0]
    out_det = seg_interp.get_output_details()[0]
    in_h, in_w = int(in_det["shape"][1]), int(in_det["shape"][2])

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if np.issubdtype(in_det["dtype"], np.floating):
        arr = (resized.astype(np.float32) / 255.0)[None, ...]
    else:
        arr = resized.astype(in_det["dtype"])[None, ...]

    seg_interp.set_tensor(in_det["index"], arr)
    seg_interp.invoke()
    raw = seg_interp.get_tensor(out_det["index"])[0].astype(np.float32)
    if float(raw.min()) < -1e-4 or float(raw.max()) > 1.01:
        shifted = raw - np.max(raw, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        probs_small = exps / np.sum(exps, axis=-1, keepdims=True)
    else:
        probs_small = np.clip(raw, 0.0, 1.0)

    probs = cv2.resize(probs_small, (w, h), interpolation=cv2.INTER_LINEAR)
    probs = np.clip(probs, 0.0, 1.0)

    # Close interior holes (eyes, pupils, nostrils, mouth) in face_skin (class 3).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    face_closed = cv2.morphologyEx(probs[:, :, 3], cv2.MORPH_CLOSE, kernel)

    # Protect upper eyebrows gently without pulling stray exterior hair/bangs.
    inner_core = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        inner_core,
        (w // 2, int(h * 0.44)),
        (int(w * 0.27), int(h * 0.23)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    hair_brow = np.clip(probs[:, :, 1] * inner_core * 0.65, 0.0, 1.0)

    occluder = np.clip(probs[:, :, 0] + probs[:, :, 4] + probs[:, :, 5], 0.0, 1.0)
    sem = np.clip((face_closed + hair_brow) * (1.0 - occluder), 0.0, 1.0)
    sem = sem * fallback_mask[:, :, 0]
    sem = np.clip(cv2.GaussianBlur(sem, (21, 21), sigmaX=6, sigmaY=6), 0.0, 1.0)

    if float(sem.mean()) < 0.05:
        return fallback_mask
    return sem[:, :, None]


def _lab_color_match(
    src_crop_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    mask_2d: np.ndarray,
) -> np.ndarray:
    """Match mean and contrast of restored face to the local scene lighting in CIE Lab space."""
    valid = mask_2d > 0.45
    if int(valid.sum()) < 100:
        return restored_bgr

    src_lab = cv2.cvtColor(src_crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    res_lab = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        s_vals = src_lab[:, :, ch][valid]
        r_vals = res_lab[:, :, ch][valid]
        s_mean, s_std = float(s_vals.mean()), float(s_vals.std())
        r_mean, r_std = float(r_vals.mean()), float(r_vals.std())
        scale = float(np.clip(s_std / max(r_std, 1e-4), 0.85, 1.15))
        res_lab[:, :, ch] = np.clip((res_lab[:, :, ch] - r_mean) * scale + s_mean, 0.0, 255.0)

    return cv2.cvtColor(res_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _reinject_luminance_grain(
    src_crop_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Re-inject high-frequency luminance film/camera grain from source into restored face."""
    if strength <= 0.0:
        return restored_bgr

    src_ycrcb = cv2.cvtColor(src_crop_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    res_ycrcb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    y_src = src_ycrcb[:, :, 0]
    y_blur = cv2.GaussianBlur(y_src, (0, 0), sigmaX=1.4, sigmaY=1.4)
    high_freq = np.clip(y_src - y_blur, -12.0, 12.0)
    res_ycrcb[:, :, 0] = np.clip(res_ycrcb[:, :, 0] + float(strength) * high_freq, 0.0, 255.0)

    return cv2.cvtColor(res_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def _parse_size(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    w_s, _, h_s = spec.lower().partition("x")
    return int(w_s), int(h_s)


def _restore_face(model: torch.nn.Module, bgr_512: np.ndarray, restorer: str) -> np.ndarray:
    """Run GFPGANv1.4 or RestoreFormer++ in official [-1, 1] float32 normalization on CUDA."""
    import torch

    rgb = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = (
        torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32)
    )
    tensor = tensor * 2.0 - 1.0
    out = model(tensor, return_rgb=False)[0] if restorer == "gfpgan" else model(tensor)[0]
    out_01 = (out.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5
    out_rgb = out_01.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def _probe_basic(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
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


def _has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(proc.stdout).get("streams", [])
    return any(s.get("codec_type") == "audio" for s in streams)


def _mux_audio(video_raw: Path, audio_src: Path, out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_audio = _has_audio_stream(audio_src)
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video_raw)]
    if has_audio:
        cmd.extend(["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0"])
    cmd.extend(["-c:v", "copy", "-movflags", "+faststart"])
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.append(str(out_path))
    subprocess.run(cmd, stdin=subprocess.DEVNULL, check=True)
    return _probe_basic(out_path)


def enhance_video(
    *,
    input_path: Path,
    audio_source: Path,
    out_path: Path,
    upscale_to: tuple[int, int] | None,
    restorer: str,
    gfpgan_weights: Path,
    restoreformer_weights: Path,
    yunet_model: Path,
    seg_model: Path | None,
    use_semantic_mask: bool,
    color_match: bool,
    grain_strength: float,
    esrgan_weights: Path,
    fidelity: float,
    stats_out: Path | None,
    preloaded_face_model: Any | None = None,
    preloaded_esr_model: Any | None = None,
    preloaded_seg_interp: Any | None = None,
    preloaded_detector: Any | None = None,
) -> dict[str, Any]:
    import spandrel
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if preloaded_face_model is not None:
        face_model = preloaded_face_model
    elif restorer == "restoreformer":
        infra_dir = str(Path(__file__).resolve().parent)
        if infra_dir not in sys.path:
            sys.path.insert(0, infra_dir)
        from restoreformer_arch import load_restoreformer_pp

        face_model = load_restoreformer_pp(restoreformer_weights, device="cuda")
    else:
        gfp_desc = spandrel.ModelLoader().load_from_file(gfpgan_weights)
        face_model = gfp_desc.model.eval().to(device="cuda", dtype=torch.float32)

    seg_interp = preloaded_seg_interp
    semantic_active = seg_interp is not None
    if not semantic_active and use_semantic_mask and seg_model is not None and seg_model.exists():
        from ai_edge_litert.interpreter import Interpreter

        seg_interp = Interpreter(model_path=str(seg_model), num_threads=4)
        seg_interp.allocate_tensors()
        semantic_active = True

    esr_model = preloaded_esr_model
    if upscale_to is not None and esr_model is None:
        esr_desc = spandrel.ModelLoader().load_from_file(esrgan_weights)
        esr_model = esr_desc.model.eval().to(device="cuda", dtype=torch.float16)

    probe_cap = cv2.VideoCapture(str(input_path))
    if not probe_cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    fps = probe_cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe_cap.release()
    out_w, out_h = upscale_to if upscale_to is not None else (src_w, src_h)

    if preloaded_detector is not None:
        detector = preloaded_detector
        detector.setInputSize((out_w, out_h))
    else:
        detector = cv2.FaceDetectorYN.create(
            model=str(yunet_model),
            config="",
            input_size=(out_w, out_h),
            score_threshold=0.58,
            nms_threshold=0.30,
            top_k=5000,
        )

    t0 = time.perf_counter()
    # Build smoothed 2-pass trajectories (scaling landmarks if Real-ESRGAN upscales frames).
    if upscale_to is None:
        per_frame_tracks, cut_count = _collect_smoothed_tracks(input_path, detector)
    else:
        # Scale Pass-1 detections from src_w x src_h into out_w x out_h space.
        detector.setInputSize((src_w, src_h))
        per_frame_tracks, cut_count = _collect_smoothed_tracks(input_path, detector)
        sx, sy = out_w / float(src_w), out_h / float(src_h)
        scaled_tracks: list[dict[int, tuple[float, np.ndarray, np.ndarray, float]]] = []
        for fmap in per_frame_tracks:
            s_map: dict[int, tuple[float, np.ndarray, np.ndarray, float]] = {}
            for tid, (area, lm5, _aff, fade) in fmap.items():
                lm_scaled = lm5.copy()
                lm_scaled[:, 0] *= sx
                lm_scaled[:, 1] *= sy
                s_map[tid] = (
                    area * sx * sy,
                    lm_scaled,
                    _umeyama_similarity_2d(lm_scaled, _FFHQ_TEMPLATE_512),
                    fade,
                )
            scaled_tracks.append(s_map)
        per_frame_tracks = scaled_tracks

    feather_mask = _make_feathered_mask(512)
    raw_video = out_path.with_name(f"{out_path.stem}_raw.mp4")

    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{out_w}x{out_h}",
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
        ],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None

    cap = cv2.VideoCapture(str(input_path))
    prev_masks: dict[int, np.ndarray] = {}
    faces_detected = 0
    total_face_crops = 0
    frames_done = 0
    first_crop_checked = False

    try:
        with torch.inference_mode():
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                if esr_model is not None:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    tensor = (
                        torch.from_numpy(rgb)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device="cuda", dtype=torch.float16)
                        .div_(255.0)
                    )
                    up = esr_model(tensor).clamp_(0.0, 1.0)
                    resized = torch.nn.functional.interpolate(
                        up,
                        size=(out_h, out_w),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ).clamp_(0.0, 1.0)
                    out_rgb = (
                        resized.squeeze(0)
                        .permute(1, 2, 0)
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )
                    work_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                else:
                    work_bgr = frame_bgr

                fmap = (
                    per_frame_tracks[frames_done]
                    if frames_done < len(per_frame_tracks)
                    else {}
                )
                # Drop stale mask EMAs for tracks not in this frame.
                for stale_tid in [k for k in prev_masks if k not in fmap]:
                    del prev_masks[stale_tid]

                if fmap:
                    # Paste smaller background faces first, primary largest face last.
                    ordered = sorted(fmap.items(), key=lambda kv: kv[1][0])
                    frame_had_restore = False
                    for tid, (_area, _lm5, affine, fade) in ordered:
                        if fade <= 0.0:
                            continue
                        frame_had_restore = True
                        total_face_crops += 1
                        crop_bgr = cv2.warpAffine(
                            work_bgr,
                            affine,
                            (512, 512),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT_101,
                        )
                        restored_bgr = _restore_face(face_model, crop_bgr, restorer)

                        if not first_crop_checked:
                            mae = float(
                                np.mean(
                                    np.abs(
                                        restored_bgr.astype(np.float32)
                                        - crop_bgr.astype(np.float32)
                                    )
                                )
                            ) / 255.0
                            assert (
                                mae < 0.25
                            ), f"{restorer} normalization check failed: MAE={mae:.3f}"
                            first_crop_checked = True

                        raw_mask = _make_semantic_mask(crop_bgr, seg_interp, feather_mask)
                        if tid in prev_masks:
                            crop_mask = 0.60 * raw_mask + 0.40 * prev_masks[tid]
                        else:
                            crop_mask = raw_mask
                        prev_masks[tid] = crop_mask
                        eff_mask = crop_mask * float(fade)

                        if color_match:
                            restored_bgr = _lab_color_match(
                                crop_bgr, restored_bgr, crop_mask[:, :, 0]
                            )

                        blended_u8 = np.clip(
                            restored_bgr.astype(np.float32) * fidelity
                            + crop_bgr.astype(np.float32) * (1.0 - fidelity),
                            0.0,
                            255.0,
                        ).astype(np.uint8)
                        if grain_strength > 0.0:
                            blended_u8 = _reinject_luminance_grain(
                                crop_bgr, blended_u8, grain_strength
                            )
                        blended_crop = blended_u8.astype(np.float32)

                        inv_affine = cv2.invertAffineTransform(affine)
                        warped_face = cv2.warpAffine(
                            blended_crop,
                            inv_affine,
                            (out_w, out_h),
                            flags=cv2.INTER_LINEAR,
                        )
                        warped_mask = cv2.warpAffine(
                            eff_mask,
                            inv_affine,
                            (out_w, out_h),
                            flags=cv2.INTER_LINEAR,
                        )
                        if warped_mask.ndim == 2:
                            warped_mask = warped_mask[:, :, None]
                        work_bgr = np.clip(
                            warped_face * warped_mask
                            + work_bgr.astype(np.float32) * (1.0 - warped_mask),
                            0.0,
                            255.0,
                        ).astype(np.uint8)
                    if frame_had_restore:
                        faces_detected += 1

                enc.stdin.write(np.ascontiguousarray(work_bgr).tobytes())
                frames_done += 1
        torch.cuda.synchronize()
    finally:
        cap.release()
        enc.stdin.close()
        rc = enc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder failed with exit code {rc}")

    infer_sec = round(time.perf_counter() - t0, 2)
    peak_vram_gib = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    probe = _mux_audio(raw_video, audio_source, out_path)
    raw_video.unlink(missing_ok=True)

    summary = {
        "out_filename": out_path.name,
        "restorer": restorer,
        "semantic_mask": semantic_active,
        "color_match": color_match,
        "grain_strength": grain_strength,
        "shot_cuts": cut_count,
        "frames": frames_done,
        "faces_detected_frames": faces_detected,
        "face_crops_restored": total_face_crops,
        "fidelity": fidelity,
        "upscaled": upscale_to is not None,
        "inference_sec": infer_sec,
        "peak_vram_gib": peak_vram_gib,
        **probe,
    }
    print(
        f"[face_enhance:{restorer}] {out_path.name}: {frames_done}f "
        f"(faces={faces_detected}/{frames_done}, cuts={cut_count}, "
        f"sem={semantic_active}, lab={color_match}, grain={grain_strength}), "
        f"{infer_sec:.2f}s, peak={peak_vram_gib:.2f} GiB",
        flush=True,
    )
    if stats_out is not None:
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GFPGAN v1.4 / RestoreFormer++ + MediaPipe + YuNet + Umeyama post-processor."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audio-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upscale-to", type=str, default=None, help="Optional WxH e.g. 1280x736")
    parser.add_argument(
        "--restorer",
        type=str,
        choices=["gfpgan", "restoreformer"],
        default="gfpgan",
        help="Commercial-safe face restoration backbone",
    )
    parser.add_argument(
        "--gfpgan-weights", type=Path, default=Path("/workspace/models/face/GFPGANv1.4.pth")
    )
    parser.add_argument(
        "--restoreformer-weights",
        type=Path,
        default=Path("/workspace/models/face/RestoreFormer++.ckpt"),
    )
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=Path("/workspace/models/face/face_detection_yunet_2023mar.onnx"),
    )
    parser.add_argument(
        "--seg-model",
        type=Path,
        default=Path("/workspace/models/face/selfie_multiclass_256x256.tflite"),
    )
    parser.add_argument(
        "--no-semantic-mask",
        action="store_true",
        help="Force legacy elliptical feathered mask instead of MediaPipe semantic mask",
    )
    parser.add_argument(
        "--color-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match CIE Lab mean and contrast to local source face lighting",
    )
    parser.add_argument(
        "--grain-strength",
        type=float,
        default=0.35,
        help="High-pass luminance film grain re-injection strength (0 disables)",
    )
    parser.add_argument(
        "--esrgan-weights",
        type=Path,
        default=Path("/workspace/models/realesrgan/realesr-general-x4v3.pth"),
    )
    parser.add_argument("--fidelity", type=float, default=0.78)
    parser.add_argument("--stats-out", type=Path, default=None)
    args = parser.parse_args()

    enhance_video(
        input_path=args.input,
        audio_source=args.audio_source,
        out_path=args.out,
        upscale_to=_parse_size(args.upscale_to),
        restorer=args.restorer,
        gfpgan_weights=args.gfpgan_weights,
        restoreformer_weights=args.restoreformer_weights,
        yunet_model=args.yunet_model,
        seg_model=args.seg_model,
        use_semantic_mask=not args.no_semantic_mask,
        color_match=args.color_match,
        grain_strength=args.grain_strength,
        esrgan_weights=args.esrgan_weights,
        fidelity=args.fidelity,
        stats_out=args.stats_out,
    )


if __name__ == "__main__":
    main()

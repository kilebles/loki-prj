"""Keyframe face scale verification, rigid BG alignment, and zero-BG-warp normalization."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np
from ai_edge_litert.interpreter import Interpreter

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SELFIE_SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

_LOCK = threading.Lock()
_YUNET_CACHE: dict[str, cv2.FaceDetectorYN] = {}
_SEG_CACHE: dict[str, Interpreter] = {}


@dataclass(frozen=True)
class FaceMeasurement:
    """Detected face bounding box and 5-point landmarks from YuNet."""

    bbox: tuple[float, float, float, float]
    lm5: np.ndarray
    iod: float
    eye_center: tuple[float, float]
    score: float


@dataclass(frozen=True)
class ScaleReport:
    """Comparison of primary face scale between source and repainted keyframe."""

    src_face: FaceMeasurement
    painted_face: FaceMeasurement
    iod_ratio: float
    width_ratio: float
    height_ratio: float
    scale_ratio: float
    within_tolerance: bool
    normalized_applied: bool = False


def _download_if_missing(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.part")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def ensure_local_aux_models(cache_dir: Path = Path("work/.models")) -> tuple[Path, Path]:
    """Ensure local YuNet ONNX and MediaPipe Selfie Multiclass TFLite models exist."""
    yunet_path = _download_if_missing(
        YUNET_URL, cache_dir / "face_detection_yunet_2023mar.onnx"
    )
    tflite_path = _download_if_missing(
        SELFIE_SEG_URL, cache_dir / "selfie_multiclass_256x256.tflite"
    )
    return yunet_path, tflite_path


def detect_letterbox_bars(bgr: np.ndarray, threshold: float = 10.0) -> tuple[int, int]:
    """Detect contiguous black letterbox row bounds (top_bar, bot_bar)."""
    h = bgr.shape[0]
    means = bgr.mean(axis=(1, 2))
    top = 0
    while top < h // 3 and float(means[top]) < threshold:
        top += 1
    bot = h
    while bot > (2 * h) // 3 and float(means[bot - 1]) < threshold:
        bot -= 1
    return top, bot


def _restore_letterbox(out_bgr: np.ndarray, src_bgr: np.ndarray) -> np.ndarray:
    top_bar, bot_bar = detect_letterbox_bars(src_bgr)
    h = src_bgr.shape[0]
    if top_bar > 0:
        out_bgr[:top_bar] = src_bgr[:top_bar]
    if bot_bar < h:
        out_bgr[bot_bar:] = src_bgr[bot_bar:]
    return out_bgr


def detect_faces_yunet(
    img_bgr: np.ndarray,
    yunet_path: Path,
    *,
    score_threshold: float = 0.5,
) -> list[FaceMeasurement]:
    """Detect faces in BGR image sorted largest-first by area * score."""
    h, w = img_bgr.shape[:2]
    key = str(yunet_path.resolve())
    with _LOCK:
        det = _YUNET_CACHE.get(key)
        if det is None:
            det = cv2.FaceDetectorYN.create(key, "", (w, h), score_threshold, 0.3, 500)
            _YUNET_CACHE[key] = det
        else:
            det.setScoreThreshold(score_threshold)
            det.setInputSize((w, h))
        _, raw_faces = det.detect(img_bgr)

    if raw_faces is None or len(raw_faces) == 0:
        return []

    measurements: list[FaceMeasurement] = []
    for row in raw_faces:
        x, y, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        score = float(row[14])
        if bw <= 1.0 or bh <= 1.0 or score < score_threshold:
            continue
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
        iod = float(np.linalg.norm(lm5[0] - lm5[1]))
        ec = ((lm5[0] + lm5[1]) * 0.5).tolist()
        measurements.append(
            FaceMeasurement(
                bbox=(x, y, bw, bh),
                lm5=lm5,
                iod=iod,
                eye_center=(float(ec[0]), float(ec[1])),
                score=score,
            )
        )

    measurements.sort(key=lambda m: m.bbox[2] * m.bbox[3] * m.score, reverse=True)
    return measurements


def compare_face_scale(
    src_bgr: np.ndarray,
    painted_bgr: np.ndarray,
    yunet_path: Path,
    *,
    scale_tol: float = 0.07,
) -> ScaleReport | None:
    """Measure how much the painted primary face expanded/shrank relative to source."""
    src_faces = detect_faces_yunet(src_bgr, yunet_path)
    painted_faces = detect_faces_yunet(painted_bgr, yunet_path)
    if not src_faces or not painted_faces:
        return None

    src_face = src_faces[0]
    src_ec = np.array(src_face.eye_center, dtype=np.float32)

    def _dist(candidate: FaceMeasurement) -> float:
        cand_ec = np.array(candidate.eye_center, dtype=np.float32)
        return float(np.linalg.norm(cand_ec - src_ec)) / max(src_face.bbox[2], 1.0)

    painted_face = min(painted_faces, key=_dist)
    iod_ratio = painted_face.iod / max(src_face.iod, 1e-3)
    width_ratio = painted_face.bbox[2] / max(src_face.bbox[2], 1e-3)
    height_ratio = painted_face.bbox[3] / max(src_face.bbox[3], 1e-3)
    scale_ratio = float(np.median([iod_ratio, width_ratio, height_ratio]))

    return ScaleReport(
        src_face=src_face,
        painted_face=painted_face,
        iod_ratio=round(iod_ratio, 4),
        width_ratio=round(width_ratio, 4),
        height_ratio=round(height_ratio, 4),
        scale_ratio=round(scale_ratio, 4),
        within_tolerance=abs(scale_ratio - 1.0) <= scale_tol,
        normalized_applied=False,
    )


def segment_multiclass(img_bgr: np.ndarray, tflite_path: Path) -> np.ndarray:
    """Return HxWx6 probabilities (0:bg, 1:hair, 2:body_skin, 3:face_skin, 4:clothes, 5:others)."""
    h, w = img_bgr.shape[:2]
    key = str(tflite_path.resolve())
    with _LOCK:
        interp = _SEG_CACHE.get(key)
        if interp is None:
            interp = Interpreter(model_path=key, num_threads=4)
            interp.allocate_tensors()
            _SEG_CACHE[key] = interp

        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        in_shape = in_det["shape"]
        in_h, in_w = int(in_shape[1]), int(in_shape[2])

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        if np.issubdtype(in_det["dtype"], np.floating):
            arr = (resized.astype(np.float32) / 255.0)[None, ...]
        else:
            arr = resized.astype(in_det["dtype"])[None, ...]

        interp.set_tensor(in_det["index"], arr)
        interp.invoke()
        raw = interp.get_tensor(out_det["index"])[0].astype(np.float32)

    if float(raw.min()) < -1e-4 or float(raw.max()) > 1.01:
        shifted = raw - np.max(raw, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        probs_small = exps / np.sum(exps, axis=-1, keepdims=True)
    else:
        probs_small = np.clip(raw, 0.0, 1.0)

    probs = cv2.resize(probs_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(probs, 0.0, 1.0).astype(np.float32)


def align_painted_to_source_bg(
    src_bgr: np.ndarray,
    painted_bgr: np.ndarray,
    tflite_path: Path,
) -> tuple[np.ndarray, float]:
    """Rigidly undo any full-frame painter zoom/shift/letterbox strip via SIFT on background."""
    h, w = src_bgr.shape[:2]
    src_top, src_bot = detect_letterbox_bars(src_bgr)
    p_top, p_bot = detect_letterbox_bars(painted_bgr)

    src_probs = segment_multiclass(src_bgr, tflite_path)
    p_probs = segment_multiclass(painted_bgr, tflite_path)

    src_bg = (src_probs[:, :, 0] > 0.55).astype(np.uint8) * 255
    p_bg = (p_probs[:, :, 0] > 0.55).astype(np.uint8) * 255
    if src_top > 0:
        src_bg[:src_top] = 0
    if src_bot < h:
        src_bg[src_bot:] = 0
    if p_top > 0:
        p_bg[:p_top] = 0
    if p_bot < h:
        p_bg[p_bot:] = 0

    sift = cv2.SIFT_create(1500)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY), src_bg)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(painted_bgr, cv2.COLOR_BGR2GRAY), p_bg)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return _restore_letterbox(painted_bgr.copy(), src_bgr), 1.0

    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [
        m
        for pair in matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.72 * n.distance
    ]
    if len(good) < 15:
        return _restore_letterbox(painted_bgr.copy(), src_bgr), 1.0

    pts_src = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_p = np.float32([kp2[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        pts_p, pts_src, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if affine is None or inliers is None or int(inliers.sum()) < 15:
        return _restore_letterbox(painted_bgr.copy(), src_bgr), 1.0

    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
    if not (0.75 <= scale <= 1.32):
        return _restore_letterbox(painted_bgr.copy(), src_bgr), 1.0

    aligned = cv2.warpAffine(
        painted_bgr,
        affine,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid = cv2.warpAffine(
        np.full((h, w), 255, dtype=np.uint8),
        affine,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    aligned[valid == 0] = src_bgr[valid == 0]
    _restore_letterbox(aligned, src_bgr)

    bg_eval = src_bg > 0
    if int(bg_eval.sum()) > 100:
        mae_before = float(
            np.abs(painted_bgr.astype(np.float32) - src_bgr.astype(np.float32))[bg_eval].mean()
        )
        mae_after = float(
            np.abs(aligned.astype(np.float32) - src_bgr.astype(np.float32))[bg_eval].mean()
        )
        if mae_after > mae_before + 1.0:
            return _restore_letterbox(painted_bgr.copy(), src_bgr), 1.0

    return aligned, scale


def lock_background_to_source(
    src_bgr: np.ndarray,
    aligned_painted_bgr: np.ndarray,
    tflite_path: Path,
    yunet_path: Path,
) -> np.ndarray:
    """Lock non-subject BG, bystanders, and popped collars to exact source pixels."""
    h, w = src_bgr.shape[:2]
    src_faces = detect_faces_yunet(src_bgr, yunet_path)
    if not src_faces:
        return _restore_letterbox(aligned_painted_bgr.copy(), src_bgr)

    sf = src_faces[0]
    src_probs = segment_multiclass(src_bgr, tflite_path)
    aln_probs = segment_multiclass(aligned_painted_bgr, tflite_path)

    # Head + hair + neck classes only (1=hair, 2=body_skin/neck, 3=face_skin).
    s_head = np.clip(src_probs[:, :, 1] + src_probs[:, :, 2] + src_probs[:, :, 3], 0.0, 1.0)
    p_head = np.clip(aln_probs[:, :, 1] + aln_probs[:, :, 2] + aln_probs[:, :, 3], 0.0, 1.0)
    union_head = (np.maximum(s_head, p_head) > 0.25).astype(np.uint8)

    # Keep only connected component attached to primary face center (never grabs bystanders).
    _, labels = cv2.connectedComponents(union_head)
    cx = int(np.clip(round(sf.bbox[0] + sf.bbox[2] * 0.5), 0, w - 1))
    cy = int(np.clip(round(sf.bbox[1] + sf.bbox[3] * 0.5), 0, h - 1))
    target_lbl = int(labels[cy, cx])
    head_cc = (labels == target_lbl).astype(np.uint8) if target_lbl > 0 else union_head

    # Explicitly exclude bounding boxes of any secondary detected faces.
    for other in src_faces[1:]:
        ox, oy, ow, oh = other.bbox
        x1, y1 = max(0, int(ox - ow * 0.5)), max(0, int(oy - oh * 0.6))
        x2, y2 = min(w, int(ox + ow * 1.5)), min(h, int(oy + oh * 1.6))
        head_cc[y1:y2, x1:x2] = 0

    fy, fh = sf.bbox[1], sf.bbox[3]
    head_cc[int(min(h, fy + fh * 1.55)) :, :] = 0

    dil_k = max(11, (int(sf.bbox[2] * 0.12) | 1))
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
    head_dil = cv2.dilate(head_cc, dil_kernel).astype(np.float32)

    # Blur FIRST for smooth hair-to-sky transition, THEN apply crisp 3px collar guard below chin.
    blur_k = max(9, (int(sf.bbox[2] * 0.10) | 1))
    hair_alpha = np.clip(cv2.GaussianBlur(head_dil, (blur_k, blur_k), 0), 0.0, 1.0)

    yy = np.arange(h, dtype=np.float32)[:, None]
    below_nose = np.clip((yy - (fy + fh * 0.55)) / max(fh * 0.15, 6.0), 0.0, 1.0)
    clothes_bin = (src_probs[:, :, 4] > 0.30).astype(np.uint8)
    clothes_dil = cv2.dilate(clothes_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    clothes_sharp = cv2.GaussianBlur(
        np.maximum(np.clip(src_probs[:, :, 4] * 1.3, 0.0, 1.0), clothes_dil.astype(np.float32)),
        (5, 5),
        0,
    )
    alpha = np.clip(hair_alpha * (1.0 - clothes_sharp * below_nose), 0.0, 1.0)[:, :, None]

    out_bgr = np.clip(
        aligned_painted_bgr.astype(np.float32) * alpha
        + src_bgr.astype(np.float32) * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)
    return _restore_letterbox(out_bgr, src_bgr)


def normalize_head_scale(
    src_bgr: np.ndarray,
    painted_bgr: np.ndarray,
    *,
    yunet_path: Path,
    tflite_path: Path,
    keep_hair: bool = False,
    scale_tol: float = 0.09,
) -> tuple[np.ndarray, ScaleReport | None]:
    """Rigidly align BG (no non-rigid pixel warp) and lock non-subject background to source."""
    del keep_hair
    aligned_bgr, _ = align_painted_to_source_bg(src_bgr, painted_bgr, tflite_path)
    out_bgr = lock_background_to_source(src_bgr, aligned_bgr, tflite_path, yunet_path)
    post_report = compare_face_scale(src_bgr, out_bgr, yunet_path, scale_tol=scale_tol)
    if post_report is None:
        return out_bgr, None

    final_report = ScaleReport(
        src_face=post_report.src_face,
        painted_face=post_report.painted_face,
        iod_ratio=post_report.iod_ratio,
        width_ratio=post_report.width_ratio,
        height_ratio=post_report.height_ratio,
        scale_ratio=post_report.scale_ratio,
        within_tolerance=post_report.within_tolerance,
        normalized_applied=True,
    )
    return out_bgr, final_report

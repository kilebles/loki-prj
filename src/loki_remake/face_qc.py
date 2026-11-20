"""Keyframe face scale verification, semantic segmentation, and geometric head normalization."""

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
            interp = Interpreter(model_path=key)
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


def normalize_head_scale(
    src_bgr: np.ndarray,
    painted_bgr: np.ndarray,
    *,
    yunet_path: Path,
    tflite_path: Path,
    keep_hair: bool = False,
    scale_tol: float = 0.07,
) -> tuple[np.ndarray, ScaleReport | None]:
    """Deterministically rescale painted head to match source actor skull & eye scale."""
    del keep_hair  # Continuous local warp keeps clothes/neck anchored regardless of hair mode.
    report = compare_face_scale(src_bgr, painted_bgr, yunet_path, scale_tol=scale_tol)
    if report is None or report.within_tolerance:
        return painted_bgr, report

    h, w = painted_bgr.shape[:2]
    src_face = report.src_face
    painted_face = report.painted_face

    scale = float(np.clip(1.0 / max(report.scale_ratio, 1e-3), 0.76, 1.30))
    px = float(np.mean(painted_face.lm5[:, 0]))
    py = painted_face.eye_center[1] + painted_face.bbox[3] * 0.05
    sx = float(np.mean(src_face.lm5[:, 0]))
    sy = src_face.eye_center[1] + src_face.bbox[3] * 0.05

    probs = segment_multiclass(painted_bgr, tflite_path)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    rx = max(painted_face.bbox[2] * 0.95, 24.0)
    ry = max(painted_face.bbox[3] * 1.15, 24.0)
    dist = np.sqrt(((xx - sx) / rx) ** 2 + ((yy - sy) / ry) ** 2)
    t = np.clip((dist - 0.85) / (2.0 - 0.85), 0.0, 1.0)
    radial_w = 1.0 - (t * t * (3.0 - 2.0 * t))

    blur_k = max(9, (int(painted_face.bbox[2] * 0.10) | 1))
    clothes_blur = cv2.GaussianBlur(probs[:, :, 4], (blur_k, blur_k), 0)

    # Detect contiguous black letterbox bars on BOTH painted and source frames.
    def _letterbox_bounds(bgr: np.ndarray) -> tuple[int, int]:
        means = bgr.mean(axis=(1, 2))
        top = 0
        while top < h // 3 and float(means[top]) < 10.0:
            top += 1
        bot = h
        while bot > (2 * h) // 3 and float(means[bot - 1]) < 10.0:
            bot -= 1
        return top, bot

    top_bar, bot_bar = _letterbox_bounds(painted_bgr)
    src_top_bar, src_bot_bar = _letterbox_bounds(src_bgr)

    y_min = float(top_bar + 4) if top_bar > 0 else 0.0
    y_max = float(bot_bar - 5) if bot_bar < h else float(h - 1)
    row_active = np.ones((h, 1), dtype=np.float32)
    eff_top = max(top_bar, src_top_bar)
    eff_bot = min(bot_bar, src_bot_bar)
    if eff_top > 0:
        row_active[:eff_top, :] = 0.0
    if eff_bot < h:
        row_active[eff_bot:, :] = 0.0

    chin_y = painted_face.bbox[1] + painted_face.bbox[3] * 1.20
    chest_t = np.clip((yy - chin_y) / max(painted_face.bbox[3] * 0.55, 16.0), 0.0, 1.0)
    upper_body_guard = 1.0 - (chest_t * chest_t * (3.0 - 2.0 * chest_t))

    w_field = np.clip(
        radial_w * (1.0 - np.clip(clothes_blur * 1.3, 0.0, 1.0)) * upper_body_guard * row_active,
        0.0,
        1.0,
    )

    inv_s = 1.0 / scale
    raw_map_y = yy + w_field * ((py + (yy - sy) * inv_s) - yy)
    map_x = np.clip(xx + w_field * ((px + (xx - sx) * inv_s) - xx), 0.0, float(w - 1))
    map_y = np.clip(raw_map_y, y_min, y_max)

    out_bgr = cv2.remap(
        painted_bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # If shrinking tall hair pulled raw_map_y above the painted frame top, fill cleanly from
    # source background strictly inside source's active frame (never copying source bars/hair).
    if top_bar > 0 and float((raw_map_y < y_min).max()) > 0.0:
        src_probs = segment_multiclass(src_bgr, tflite_path)
        safe_sky = (
            (raw_map_y < y_min + 2.0)
            & (yy >= float(src_top_bar + 2 if src_top_bar > 0 else 0))
            & (yy < float(src_bot_bar - 2 if src_bot_bar < h else h))
            & (src_probs[:, :, 1] < 0.25)
            & (src_probs[:, :, 3] < 0.25)
        ).astype(np.float32)
        if float(safe_sky.max()) > 0.0:
            blend_top = cv2.GaussianBlur(safe_sky, (9, 9), 0)[:, :, None]
            out_bgr = np.clip(
                src_bgr.astype(np.float32) * blend_top
                + out_bgr.astype(np.float32) * (1.0 - blend_top),
                0.0,
                255.0,
            ).astype(np.uint8)
    if eff_top > 0:
        bar_src = src_bgr[:eff_top] if src_top_bar >= eff_top else painted_bgr[:eff_top]
        out_bgr[:eff_top] = bar_src
    if eff_bot < h:
        bar_bot = src_bgr[eff_bot:] if src_bot_bar <= eff_bot else painted_bgr[eff_bot:]
        out_bgr[eff_bot:] = bar_bot

    post_report = compare_face_scale(src_bgr, out_bgr, yunet_path, scale_tol=scale_tol)
    final_report = ScaleReport(
        src_face=report.src_face,
        painted_face=post_report.painted_face if post_report is not None else report.painted_face,
        iod_ratio=post_report.iod_ratio if post_report is not None else report.iod_ratio,
        width_ratio=post_report.width_ratio if post_report is not None else report.width_ratio,
        height_ratio=post_report.height_ratio if post_report is not None else report.height_ratio,
        scale_ratio=post_report.scale_ratio if post_report is not None else report.scale_ratio,
        within_tolerance=(
            post_report.within_tolerance if post_report is not None else report.within_tolerance
        ),
        normalized_applied=True,
    )
    return out_bgr, final_report

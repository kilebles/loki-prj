#!/usr/bin/env python3
"""Commercial-safe face restoration (YuNet + GFPGAN v1.4) + optional Real-ESRGAN general-x4v3."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import spandrel
import torch

# Canonical 512x512 FFHQ 5-point landmarks matching facexlib/GFPGAN alignment.
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


def _make_feathered_mask(size: int = 512) -> np.ndarray:
    """Create a soft elliptical alpha mask inside 512x512 so pasted face has zero seam."""
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, int(size * 0.52))
    axes = (int(size * 0.40), int(size * 0.44))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), sigmaX=18, sigmaY=18)
    mask = np.clip(mask, 0.0, 1.0)
    return mask[:, :, None]


def _parse_size(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    w_s, _, h_s = spec.lower().partition("x")
    return int(w_s), int(h_s)


def _restore_face_gfpgan(model: torch.nn.Module, bgr_512: np.ndarray) -> np.ndarray:
    """Run GFPGANv1.4 in official [-1, 1] normalization in float32 on CUDA."""
    rgb = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = (
        torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32)
    )
    tensor = tensor * 2.0 - 1.0
    out = model(tensor, return_rgb=False)[0]
    out_01 = (out.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5
    out_rgb = (
        out_01.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )
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
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_raw)]
    if has_audio:
        cmd.extend(["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0"])
    cmd.extend(["-c:v", "copy", "-movflags", "+faststart"])
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)
    return _probe_basic(out_path)


def enhance_video(
    *,
    input_path: Path,
    audio_source: Path,
    out_path: Path,
    upscale_to: tuple[int, int] | None,
    gfpgan_weights: Path,
    yunet_model: Path,
    esrgan_weights: Path,
    fidelity: float,
    stats_out: Path | None,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    gfp_desc = spandrel.ModelLoader().load_from_file(gfpgan_weights)
    gfp_model = gfp_desc.model.eval().to(device="cuda", dtype=torch.float32)

    esr_model = None
    if upscale_to is not None:
        esr_desc = spandrel.ModelLoader().load_from_file(esrgan_weights)
        esr_model = esr_desc.model.eval().to(device="cuda", dtype=torch.float16)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = upscale_to if upscale_to is not None else (src_w, src_h)

    detector = cv2.FaceDetectorYN.create(
        model=str(yunet_model),
        config="",
        input_size=(out_w, out_h),
        score_threshold=0.60,
        nms_threshold=0.30,
        top_k=5000,
    )
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

    tracks: list[dict[str, Any]] = []
    ema_alpha = 0.65
    match_dist_px = 55.0
    max_missed = 3
    faces_detected = 0
    total_face_crops = 0
    frames_done = 0
    first_crop_checked = False
    t0 = time.perf_counter()

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

                _, faces = detector.detect(work_bgr)
                if faces is not None and len(faces) > 0:
                    candidates = [row for row in faces if float(row[14]) >= 0.65]
                    candidates.sort(
                        key=lambda row: float(row[14]) * float(row[2] * row[3]),
                        reverse=True,
                    )
                    # Keep up to 4 largest faces; paste smaller background faces first.
                    selected = list(reversed(candidates[:4]))
                    used_tracks: set[int] = set()
                    frame_landmarks: list[np.ndarray] = []

                    for row in selected:
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
                        best_idx: int | None = None
                        best_dist = match_dist_px
                        for tidx, trk in enumerate(tracks):
                            if tidx in used_tracks:
                                continue
                            dist = float(np.mean(np.linalg.norm(lm5 - trk["landmarks"], axis=1)))
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = tidx

                        if best_idx is not None:
                            smoothed = (
                                ema_alpha * lm5 + (1.0 - ema_alpha) * tracks[best_idx]["landmarks"]
                            )
                            tracks[best_idx] = {"landmarks": smoothed, "missed": 0}
                            used_tracks.add(best_idx)
                            frame_landmarks.append(smoothed)
                        else:
                            tracks.append({"landmarks": lm5, "missed": 0})
                            used_tracks.add(len(tracks) - 1)
                            frame_landmarks.append(lm5)

                    surviving: list[dict[str, Any]] = []
                    for tidx, trk in enumerate(tracks):
                        if tidx in used_tracks:
                            surviving.append(trk)
                        elif trk["missed"] + 1 <= max_missed:
                            surviving.append(
                                {
                                    "landmarks": trk["landmarks"],
                                    "missed": trk["missed"] + 1,
                                }
                            )
                    tracks = surviving

                    frame_had_restore = False
                    for smoothed_lm5 in frame_landmarks:
                        affine, _ = cv2.estimateAffinePartial2D(
                            smoothed_lm5, _FFHQ_TEMPLATE_512, method=cv2.LMEDS
                        )
                        if affine is None:
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
                        restored_bgr = _restore_face_gfpgan(gfp_model, crop_bgr)

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
                            ), f"GFPGAN normalization check failed: MAE={mae:.3f}"
                            first_crop_checked = True

                        blended_crop = np.clip(
                            restored_bgr.astype(np.float32) * fidelity
                            + crop_bgr.astype(np.float32) * (1.0 - fidelity),
                            0.0,
                            255.0,
                        )
                        inv_affine = cv2.invertAffineTransform(affine)
                        warped_face = cv2.warpAffine(
                            blended_crop,
                            inv_affine,
                            (out_w, out_h),
                            flags=cv2.INTER_LINEAR,
                        )
                        warped_mask = cv2.warpAffine(
                            feather_mask,
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
                else:
                    tracks = [
                        {"landmarks": trk["landmarks"], "missed": trk["missed"] + 1}
                        for trk in tracks
                        if trk["missed"] + 1 <= max_missed
                    ]

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
        f"[face_enhance] {out_path.name}: {frames_done}f (faces={faces_detected}/{frames_done}), "
        f"{infer_sec:.2f}s, peak={peak_vram_gib:.2f} GiB",
        flush=True,
    )
    if stats_out is not None:
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GFPGAN v1.4 + YuNet + Real-ESRGAN post-processor."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audio-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upscale-to", type=str, default=None, help="Optional WxH e.g. 1280x736")
    parser.add_argument(
        "--gfpgan-weights", type=Path, default=Path("/workspace/models/face/GFPGANv1.4.pth")
    )
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=Path("/workspace/models/face/face_detection_yunet_2023mar.onnx"),
    )
    parser.add_argument(
        "--esrgan-weights",
        type=Path,
        default=Path("/workspace/models/realesrgan/realesr-general-x4v3.pth"),
    )
    parser.add_argument("--fidelity", type=float, default=0.72)
    parser.add_argument("--stats-out", type=Path, default=None)
    args = parser.parse_args()

    enhance_video(
        input_path=args.input,
        audio_source=args.audio_source,
        out_path=args.out,
        upscale_to=_parse_size(args.upscale_to),
        gfpgan_weights=args.gfpgan_weights,
        yunet_model=args.yunet_model,
        esrgan_weights=args.esrgan_weights,
        fidelity=args.fidelity,
        stats_out=args.stats_out,
    )


if __name__ == "__main__":
    main()

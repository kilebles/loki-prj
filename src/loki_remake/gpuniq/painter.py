"""Keyframe painting via GPUniq Image Jobs (nano-banana-2 / gpt-image-2)."""

from __future__ import annotations

import base64
import io
import os
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import httpx
import numpy as np
from loguru import logger
from PIL import Image, ImageOps

from loki_remake.face_qc import (
    ScaleReport,
    compare_face_scale,
    ensure_local_aux_models,
    normalize_head_scale,
)
from loki_remake.gpuniq.client import GPUniqAPIError, GPUniqClient
from loki_remake.schemas import PaintedRef, RemakeMode


def encode_image_data_url(
    path: Path,
    *,
    max_side: int = 1536,
    mime: str = "image/png",
) -> str:
    """Load an image, resize if larger than max_side, and encode as a base64 data URL."""
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        fmt = "JPEG" if mime == "image/jpeg" else "PNG"
        save_kwargs = {"quality": 90} if fmt == "JPEG" else {}
        img.save(buf, format=fmt, **save_kwargs)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_paint_prompt(
    mode: RemakeMode,
    instruction: str,
    has_user_ref: bool,
    *,
    keep_hair: bool = False,
    scale_feedback: str | None = None,
) -> str:
    if mode == "swap":
        if has_user_ref:
            hair_clause = (
                "Preserve the first image subject's exact hairstyle, hair volume, and outer hair "
                "outline completely unchanged; transfer only the facial features, eyes, nose, "
                "mouth identity, cheekbones, and skin tone from the second reference portrait."
                if keep_hair
                else "Transfer the reference subject's hair style and texture from the second "
                "image while keeping the hair strictly compact inside the exact outer skull "
                "silhouette and head volume of the first image without ballooning."
            )
            prompt = (
                f"Edit the first image so the target subject has the exact facial identity and "
                f"likeness shown in the second reference portrait. Instruction: {instruction}. "
                f"{hair_clause} "
                f"CRITICAL PROPORTIONS AND EXPRESSION: Keep the exact same skull size, head width "
                f"and height, eye-to-eye distance, neck length, head tilt, and head-to-shoulder "
                f"ratio as the first image—never enlarge the head. Preserve the exact facial "
                f"expression, mouth openness or closedness, eye gaze direction, and emotion of the "
                f"first image rather than copying the expression or smile from the reference "
                f"portrait. Naturally match the scene lighting direction, shadows, color "
                f"temperature, and camera grain of the first image while keeping clothing, hands, "
                f"objects, and background unchanged."
            )
            if scale_feedback:
                prompt += f" CRITICAL CORRECTION: {scale_feedback}"
            return prompt
        return (
            f"Edit the first image according to this instruction: {instruction}. "
            f"Keep the exact same camera angle, framing, head proportions, pose, lighting, "
            f"background, and every unchanged person or object identical to the first image."
        )

    if has_user_ref:
        return (
            f"Edit the first image using the second image as the visual reference for the change. "
            f"Instruction: {instruction}. Preserve the exact framing, camera perspective, poses, "
            f"facial expressions, hands, and all untouched elements of the first image."
        )
    return (
        f"Edit the first image seamlessly according to this instruction: {instruction}. "
        f"Preserve the exact camera framing, lighting direction, subject pose, hands, face, "
        f"and every untouched element of the first image."
    )


def _decode_or_download_image(data: dict) -> Image.Image:
    image_block = data.get("image")
    if not isinstance(image_block, dict):
        images_list = data.get("images")
        if isinstance(images_list, list) and images_list:
            image_block = images_list[0]
    if not isinstance(image_block, dict):
        raise RuntimeError(f"Image job completed without image payload: {data}")

    b64_str = image_block.get("b64_json")
    if b64_str:
        raw_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw_bytes))

    url = image_block.get("url")
    if url:
        resp = httpx.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    raise RuntimeError(f"Image payload has neither b64_json nor url: {image_block}")


def _run_single_image_job(
    client: GPUniqClient,
    *,
    model: str,
    prompt: str,
    input_images: list[str],
    aspect_ratio: str,
    poll_interval_sec: float,
    timeout_sec: float,
    sleep_fn: Callable[[float], None],
) -> tuple[dict, str]:
    kickoff = client.create_image_job(
        {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": aspect_ratio,
            "input_images": input_images,
            "response_format": "b64_json",
            "output_format": "png",
        }
    )
    job_id = str(kickoff["job_id"])
    logger.info("Submitted image paint job {} with model {}", job_id, model)

    deadline = time.monotonic() + timeout_sec
    while True:
        poll = client.get_image_job(job_id)
        status = poll.get("status")
        if status == "completed":
            return poll, job_id
        if status == "failed":
            raise GPUniqAPIError(
                str(poll.get("error") or f"Image job {job_id} failed"),
                error_code=poll.get("error_code"),
                request_id=job_id,
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Image job {job_id} timed out after {timeout_sec}s")
        sleep_fn(poll_interval_sec)


def _run_with_fallback(
    client: GPUniqClient,
    *,
    attempts: list[str],
    prompt: str,
    input_images: list[str],
    aspect_ratio: str,
    poll_interval_sec: float,
    timeout_sec: float,
    sleep_fn: Callable[[float], None],
) -> tuple[dict, str, str]:
    last_err: Exception | None = None
    for idx, candidate_model in enumerate(attempts):
        try:
            result_data, job_id = _run_single_image_job(
                client,
                model=candidate_model,
                prompt=prompt,
                input_images=input_images,
                aspect_ratio=aspect_ratio,
                poll_interval_sec=poll_interval_sec,
                timeout_sec=timeout_sec,
                sleep_fn=sleep_fn,
            )
            return result_data, job_id, candidate_model
        except GPUniqAPIError as exc:
            last_err = exc
            is_last = idx == len(attempts) - 1
            if not is_last and exc.error_code in ("content_moderation", "generation_failed", None):
                next_model = attempts[idx + 1]
                logger.warning(
                    "Paint attempt with {} failed ({}: {}), retrying with {}",
                    candidate_model,
                    exc.error_code,
                    exc,
                    next_model,
                )
                continue
            raise

    assert last_err is not None
    raise last_err


def _fit_decoded_image(result_data: dict, target_size: tuple[int, int]) -> Image.Image:
    img = _decode_or_download_image(result_data).convert("RGB")
    if img.size != target_size:
        img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
    return img


def _load_qc_models(aux_models_dir: Path | None) -> tuple[Path, Path] | None:
    cache_dir = aux_models_dir or Path("work/.models")
    if aux_models_dir is None and os.environ.get("PYTEST_CURRENT_TEST"):
        yunet_c = cache_dir / "face_detection_yunet_2023mar.onnx"
        tflite_c = cache_dir / "selfie_multiclass_256x256.tflite"
        if not (yunet_c.exists() and tflite_c.exists()):
            return None
    try:
        return ensure_local_aux_models(cache_dir=cache_dir)
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Skipping keyframe Scale-Lock QC (aux model download failed): {}", exc)
        return None


def paint_keyframe(
    client: GPUniqClient,
    source_frame: Path,
    mode: RemakeMode,
    instruction: str,
    aspect_ratio: str,
    target_size: tuple[int, int],
    out_path: Path,
    *,
    shot_index: int = 1,
    pin_frame: int = 0,
    fps: float = 24.0,
    user_ref: Path | None = None,
    user_refs: list[Path] | None = None,
    custom_prompt: str | None = None,
    keep_hair: bool = False,
    normalize_scale: bool = True,
    scale_tol: float = 0.07,
    max_scale_retries: int = 1,
    aux_models_dir: Path | None = None,
    model: str = "nano-banana-2",
    fallback_model: str | None = "gpt-image-2",
    fallback_models: list[str] | None = None,
    poll_interval_sec: float = 2.5,
    timeout_sec: float = 600.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PaintedRef:
    """Repaint a source keyframe via GPUniq Image Jobs and fit it to the exact target canvas."""
    refs: list[Path] = []
    if user_refs:
        refs.extend(user_refs)
    elif user_ref is not None:
        refs.append(user_ref)

    prompt = (
        custom_prompt
        if custom_prompt is not None
        else _build_paint_prompt(mode, instruction, has_user_ref=bool(refs), keep_hair=keep_hair)
    )
    input_images = [encode_image_data_url(source_frame)]
    for ref_path in refs:
        input_images.append(encode_image_data_url(ref_path))

    candidates = [model]
    if fallback_models:
        candidates.extend(fallback_models)
    elif fallback_model:
        candidates.append(fallback_model)

    attempts: list[str] = []
    for m in candidates:
        if m and m not in attempts:
            attempts.append(m)
    if len(attempts) == 1:
        # Retry transient failures once if no distinct fallback model was configured.
        attempts.append(model)

    result_data, job_id, model_used = _run_with_fallback(
        client,
        attempts=attempts,
        prompt=prompt,
        input_images=input_images,
        aspect_ratio=aspect_ratio,
        poll_interval_sec=poll_interval_sec,
        timeout_sec=timeout_sec,
        sleep_fn=sleep_fn,
    )
    img = _fit_decoded_image(result_data, target_size)
    total_cost = float(result_data.get("cost_usd") or 0.0)

    scale_ratio_raw: float | None = None
    scale_ratio_final: float | None = None
    scale_normalized = False

    if normalize_scale and mode == "swap" and refs and custom_prompt is None:
        aux = _load_qc_models(aux_models_dir)
        if aux is not None:
            yunet_path, tflite_path = aux
            src_bgr = cv2.imread(str(source_frame))
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            report: ScaleReport | None = (
                compare_face_scale(src_bgr, img_bgr, yunet_path, scale_tol=scale_tol)
                if src_bgr is not None
                else None
            )
            if report is not None:
                scale_ratio_raw = report.scale_ratio
                scale_ratio_final = report.scale_ratio
                if not report.within_tolerance and max_scale_retries > 0:
                    delta_pct = abs(report.scale_ratio - 1.0) * 100.0
                    logger.warning(
                        "Keyframe pin_frame={} face scale_ratio={:.3f} (exceeds +/-{:.0%}); "
                        "retrying paint with stricter scale lock",
                        pin_frame,
                        report.scale_ratio,
                        scale_tol,
                    )
                    if report.scale_ratio > 1.0:
                        feedback = (
                            f"The previous output enlarged the head by {delta_pct:.0f}% "
                            "(too large, bobblehead proportion). Shrink the skull and hair volume "
                            "so the face box and eye-to-eye distance match the first image 1:1."
                        )
                    else:
                        feedback = (
                            f"The previous output shrank the head by {delta_pct:.0f}%. Match the "
                            f"exact skull size and eye-to-eye distance of the first image 1:1."
                        )
                    retry_prompt = _build_paint_prompt(
                        mode,
                        instruction,
                        has_user_ref=True,
                        keep_hair=keep_hair,
                        scale_feedback=feedback,
                    )
                    try:
                        r_data, r_job_id, r_model = _run_with_fallback(
                            client,
                            attempts=[model_used] + [m for m in attempts if m != model_used],
                            prompt=retry_prompt,
                            input_images=input_images,
                            aspect_ratio=aspect_ratio,
                            poll_interval_sec=poll_interval_sec,
                            timeout_sec=timeout_sec,
                            sleep_fn=sleep_fn,
                        )
                        total_cost += float(r_data.get("cost_usd") or 0.0)
                        r_img = _fit_decoded_image(r_data, target_size)
                        r_bgr = cv2.cvtColor(np.array(r_img), cv2.COLOR_RGB2BGR)
                        r_report = compare_face_scale(
                            src_bgr, r_bgr, yunet_path, scale_tol=scale_tol
                        )
                        if r_report is not None and abs(r_report.scale_ratio - 1.0) < abs(
                            report.scale_ratio - 1.0
                        ):
                            img = r_img
                            img_bgr = r_bgr
                            report = r_report
                            job_id = r_job_id
                            model_used = r_model
                            scale_ratio_final = r_report.scale_ratio
                            logger.info(
                                "Retry improved pin_frame={} scale_ratio {:.3f} -> {:.3f}",
                                pin_frame,
                                scale_ratio_raw,
                                scale_ratio_final,
                            )
                    except Exception as exc:
                        logger.warning("Scale-lock retry failed, keeping initial paint: {}", exc)

                if not report.within_tolerance:
                    raw_path = out_path.with_name(f"{out_path.stem}_raw.png")
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(raw_path, format="PNG")
                    norm_bgr, post_report = normalize_head_scale(
                        src_bgr,
                        img_bgr,
                        yunet_path=yunet_path,
                        tflite_path=tflite_path,
                        keep_hair=keep_hair,
                        scale_tol=scale_tol,
                    )
                    img = Image.fromarray(cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB))
                    scale_normalized = True
                    if post_report is not None:
                        scale_ratio_final = post_report.scale_ratio
                    logger.info(
                        "Normalized pin_frame={} head scale {:.3f} -> {} (raw saved to {})",
                        pin_frame,
                        scale_ratio_raw,
                        scale_ratio_final,
                        raw_path.name,
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    return PaintedRef(
        shot_index=shot_index,
        pin_frame=pin_frame,
        pin_sec=round(pin_frame / fps, 4),
        source_frame_path=source_frame,
        painted_path=out_path,
        model_used=model_used,
        cost_usd=round(total_cost, 6),
        job_id=job_id,
        scale_ratio_raw=scale_ratio_raw,
        scale_ratio_final=scale_ratio_final,
        scale_normalized=scale_normalized,
    )

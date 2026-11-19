"""Keyframe painting via GPUniq Image Jobs (nano-banana-2 / gpt-image-2)."""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from loguru import logger
from PIL import Image, ImageOps

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


def _build_paint_prompt(mode: RemakeMode, instruction: str, has_user_ref: bool) -> str:
    if mode == "swap":
        if has_user_ref:
            return (
                f"Edit the first image so the target subject has the exact facial identity, "
                f"likeness, eyes, nose, jawline, hair, and facial hair shown in the second "
                f"reference portrait. Instruction: {instruction}. Naturally match the outdoor "
                f"or scene lighting direction, color temperature, shadows, and skin tone of "
                f"the first image. Keep the exact same camera angle, framing, head tilt, body "
                f"pose, clothing, hands, objects, and background from the first image unchanged."
            )
        return (
            f"Edit the first image according to this instruction: {instruction}. "
            f"Keep the exact same camera angle, framing, pose, lighting, background, and every "
            f"unchanged person or object identical to the first image."
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
        else _build_paint_prompt(mode, instruction, has_user_ref=bool(refs))
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

    last_err: Exception | None = None
    result_data: dict | None = None
    job_id = ""
    model_used = model

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
            model_used = candidate_model
            break
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

    if result_data is None:
        assert last_err is not None
        raise last_err

    img = _decode_or_download_image(result_data).convert("RGB")
    if img.size != target_size:
        img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    cost_usd = float(result_data.get("cost_usd") or 0.0)
    return PaintedRef(
        shot_index=shot_index,
        pin_frame=pin_frame,
        pin_sec=round(pin_frame / fps, 4),
        source_frame_path=source_frame,
        painted_path=out_path,
        model_used=model_used,
        cost_usd=round(cost_usd, 6),
        job_id=job_id,
    )

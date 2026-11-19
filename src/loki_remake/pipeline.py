"""End-to-end local job preparation pipeline."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from loguru import logger

from loki_remake.config import Settings
from loki_remake.gpuniq.client import GPUniqClient
from loki_remake.gpuniq.painter import paint_keyframe
from loki_remake.gpuniq.prompter import generate_minimax_prompt
from loki_remake.schemas import PaintedRef, PreparedJob, RemakeMode, RenderTier
from loki_remake.video.normalize import normalize_video
from loki_remake.video.shots import detect_shots_and_keyframes


def prepare_job(
    *,
    video_path: Path,
    mode: RemakeMode,
    instruction: str,
    settings: Settings,
    user_ref_path: Path | None = None,
    tier: RenderTier | None = None,
    max_refs: int = 2,
    out_dir: Path | None = None,
    client: GPUniqClient | None = None,
) -> PreparedJob:
    """Run normalize -> shot detection -> keyframe painting -> 6-section prompt generation."""
    job_id = uuid.uuid4().hex[:10]
    chosen_tier: RenderTier = tier or settings.default_tier
    target_dir = out_dir or (settings.work_dir / f"{mode}_{job_id}")
    target_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    t0 = time.monotonic()
    normalized = normalize_video(video_path, out_dir=target_dir, short_edge=chosen_tier)
    timings["normalize"] = round(time.monotonic() - t0, 3)

    t1 = time.monotonic()
    shots, selected_pins, contact_sheet_path = detect_shots_and_keyframes(
        normalized.video_path,
        num_frames=normalized.num_frames,
        out_dir=target_dir,
        max_refs=max_refs,
        fps=float(normalized.fps),
    )
    timings["shots"] = round(time.monotonic() - t1, 3)

    own_client = client is None
    active_client = client or GPUniqClient(
        api_key=settings.gpuniq_api_key.get_secret_value(),
        base_url=settings.gpuniq_base_url,
    )

    try:
        t2 = time.monotonic()
        painted_dir = target_dir / "painted"
        painted_refs: list[PaintedRef] = []
        for idx, (shot_idx, pin_frame, kf_path) in enumerate(selected_pins, start=1):
            painted_path = painted_dir / f"ref_{idx}.png"
            painted = paint_keyframe(
                active_client,
                source_frame=kf_path,
                mode=mode,
                instruction=instruction,
                aspect_ratio=normalized.aspect_ratio,
                target_size=(normalized.width, normalized.height),
                out_path=painted_path,
                shot_index=shot_idx,
                pin_frame=pin_frame,
                fps=float(normalized.fps),
                user_ref=user_ref_path,
                model=settings.painter_model,
                fallback_model=settings.painter_fallback_model,
            )
            painted_refs.append(painted)
        timings["paint"] = round(time.monotonic() - t2, 3)

        t3 = time.monotonic()
        prompt_text, vlm_cost = generate_minimax_prompt(
            active_client,
            mode=mode,
            instruction=instruction,
            normalized=normalized,
            shots=shots,
            refs=painted_refs,
            contact_sheet=contact_sheet_path,
            vlm_model=settings.vlm_model,
        )
        timings["prompt"] = round(time.monotonic() - t3, 3)
    finally:
        if own_client:
            active_client.close()

    prompt_path = target_dir / "prompt.txt"
    prompt_path.write_text(prompt_text + "\n", encoding="utf-8")

    total_cost = round(sum(r.cost_usd for r in painted_refs) + vlm_cost, 6)
    timings["total_prep"] = round(sum(timings.values()), 3)

    job = PreparedJob(
        job_id=job_id,
        mode=mode,
        tier=chosen_tier,
        instruction=instruction,
        user_ref_path=user_ref_path,
        normalized=normalized,
        shots=shots,
        refs=painted_refs,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
        contact_sheet_path=contact_sheet_path,
        prep_cost_usd=total_cost,
        timings_sec=timings,
    )
    (target_dir / "job.json").write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Prepared job {} in {}s (prep_cost=${:.4f})", job_id, timings["total_prep"], total_cost
    )
    return job

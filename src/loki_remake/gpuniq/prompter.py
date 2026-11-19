"""Six-section MiniMax-H3 prompt generation via VLM and strict validation."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from loki_remake.gpuniq.client import GPUniqClient
from loki_remake.gpuniq.painter import encode_image_data_url
from loki_remake.schemas import (
    NormalizedVideo,
    PaintedRef,
    PromptValidationResult,
    RemakeMode,
    ShotInfo,
)
from loki_remake.video.shots import format_timestamp

SECTION_HEADERS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)

_TIMESTAMP_RE = re.compile(r"\b(\d{2}):(\d{2})\.(\d{3})\b")
_FORBIDDEN_DIM_RE = re.compile(r"\bdim(?:ly\s+lit)?\b", re.IGNORECASE)
_IMPERATIVE_EDIT_RE = re.compile(
    r"\b(?:replace|change|swap|turn\s+into|make\s+the)\b", re.IGNORECASE
)

_ASCII_REPLACEMENTS = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u2014": "-",
    "\u2013": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u202f": " ",
    "\u200b": "",
    "\ufeff": "",
}


def normalize_ascii_punctuation(text: str) -> str:
    """Normalize smart quotes, dashes, ellipsis, and spaces to plain ASCII."""
    out = text
    for src, dst in _ASCII_REPLACEMENTS.items():
        out = out.replace(src, dst)
    # Collapse accidental triple fences if LLM wrapped output in ```...```
    lines = out.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        lines = lines[1:-1]
    return "\n".join(lines).strip()


def _parse_mmss_mmm(mm: str, ss: str, ms: str) -> float:
    return int(mm) * 60.0 + int(ss) + int(ms) / 1000.0


def validate_minimax_prompt(
    prompt: str,
    mode: RemakeMode,
    num_frames: int,
    num_pics: int,
    num_shots: int,
) -> PromptValidationResult:
    """Validate strict MiniMax-H3 / Viggle-Remake six-section prompt rules."""
    errors: list[str] = []
    warnings: list[str] = []

    non_ascii = sorted({ch for ch in prompt if ord(ch) > 127})
    if non_ascii:
        preview = "".join(non_ascii[:8])
        errors.append(f"Prompt contains non-ASCII characters: {preview!r}")

    lines = [line.strip() for line in prompt.splitlines()]
    header_positions: list[tuple[str, int]] = []
    for header in SECTION_HEADERS:
        matches = [i for i, line in enumerate(lines) if line == header]
        if not matches:
            errors.append(f"Missing required section header on its own line: {header}")
        elif len(matches) > 1:
            errors.append(f"Duplicate section header: {header}")
        else:
            header_positions.append((header, matches[0]))

    if len(header_positions) == len(SECTION_HEADERS):
        indices = [pos for _, pos in header_positions]
        if indices != sorted(indices):
            errors.append(
                "Section headers are out of order; expected: " + ", ".join(SECTION_HEADERS)
            )

    if "<Video 1>" not in prompt:
        errors.append("Prompt must reference <Video 1>")

    for pic_idx in range(1, num_pics + 1):
        tag = f"<Picture {pic_idx}>"
        if tag not in prompt:
            errors.append(f"Prompt must reference {tag}")

    if "<Subject 1>" not in prompt:
        errors.append("Prompt must define and reference <Subject 1>")

    for shot_idx in range(1, num_shots + 1):
        shot_tag = f"[Shot {shot_idx}]"
        if shot_tag not in prompt:
            errors.append(f"detailed_description must include {shot_tag}")

    if _FORBIDDEN_DIM_RE.search(prompt):
        errors.append("Forbidden word 'dim' / 'dimly lit' found (darkens MiniMax-H3 renders)")

    max_sec = (num_frames - 1) / 24.0
    for mm, ss, ms in _TIMESTAMP_RE.findall(prompt):
        ts_sec = _parse_mmss_mmm(mm, ss, ms)
        if ts_sec > max_sec + 0.05:
            errors.append(
                f"Timestamp {mm}:{ss}.{ms} ({ts_sec:.3f}s) exceeds clip end "
                f"{format_timestamp(num_frames - 1)} ({max_sec:.3f}s)"
            )

    # Check detailed_description for imperative editing phrasing.
    dd_idx = next((i for i, line in enumerate(lines) if line == "detailed_description:"), None)
    os_idx = next((i for i, line in enumerate(lines) if line == "overall_soundscape:"), None)
    if dd_idx is not None and os_idx is not None and dd_idx < os_idx:
        dd_body = "\n".join(lines[dd_idx + 1 : os_idx])
        if _IMPERATIVE_EDIT_RE.search(dd_body):
            warnings.append(
                "detailed_description uses edit instruction words; "
                "describe the target video directly"
            )

    if mode == "swap" and "character replacement" not in prompt.lower():
        warnings.append(
            "swap summary typically starts with [video editing + character replacement]"
        )

    return PromptValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _system_instructions(mode: RemakeMode) -> str:
    mode_guidance = (
        "Mode is SWAP (character/person replacement): summary starts with "
        "`[video editing + character replacement]`. Bind `<Subject 1>` in `subject_definitions` "
        "to `<Picture N>` and state who they replace in `<Video 1>`. "
        "In `retention_analysis`, mark `<Video 1>` as `fully_preserved -`, `<Picture N>` as "
        "`reference -`, and `<Subject N>` as `fully_preserved -`."
        if mode == "swap"
        else "Mode is EDIT (outfit, object, or background/interior edit): summary starts with "
        "`[video editing]`. Define both what changed (`<Subject 1>`) and the preserved subject or "
        "setting (`<Subject 2>`). In `retention_analysis`, mark `<Video 1>` and `<Picture N>` as "
        "`fully_preserved -`, and state what is `replaced -` / `removed -` / `new -` vs "
        "`fully_preserved -`."
    )
    return (
        "You write prompts for MiniMax-H3 / Viggle-Remake.\n"
        "Output ONLY the six-section prompt in plain English, plain ASCII, present tense.\n"
        "The six section headers MUST appear in this exact order, each alone on its own line:\n"
        "subject_definitions:\n"
        "summary:\n"
        "retention_analysis:\n"
        "detailed_description:\n"
        "overall_soundscape:\n"
        "non_diegetic_music:\n\n"
        "Strict rules:\n"
        "1. In `subject_definitions:`, start with one line for `<Video 1>`, one line placing each "
        "`<Picture N>`, then define `<Subject 1>`, `<Subject 2>`, ...\n"
        "2. Describe the TARGET video as it already plays; NEVER write imperative edit commands "
        "('change', 'make', 'swap') inside detailed_description.\n"
        "3. Never use the words 'dim' or 'dimly lit'.\n"
        "4. Refer to the source clip as `<Video 1>`, each repainted keyframe as `<Picture 1>`, "
        "`<Picture 2>`, ..., and subjects as `<Subject 1>`, `<Subject 2>`, ...\n"
        "5. Start detailed_description with one sentence on visual look, lighting, and camera "
        "style, then cover every `[Shot N]` in order from `[Shot 1]` through the last shot.\n"
        "6. All `MM:SS.mmm` timestamps MUST stay within `00:00.000` .. clip end timestamp.\n"
        "7. If anyone speaks or sings, write `<Subject 1> (S1) speaks <d>[unclear]</d>` "
        "without guessing words.\n"
        "8. Keep `overall_soundscape:` and `non_diegetic_music:` to one line each (or `N/A`).\n"
        f"\n{mode_guidance}"
    )


def _build_user_context(
    mode: RemakeMode,
    instruction: str,
    normalized: NormalizedVideo,
    shots: list[ShotInfo],
    refs: list[PaintedRef],
) -> str:
    end_ts = format_timestamp(normalized.num_frames - 1, float(normalized.fps))
    shot_lines = [
        f"- [Shot {s.index}]: frames {s.start_frame}..{s.end_frame} "
        f"({format_timestamp(s.start_frame)} to {format_timestamp(s.end_frame)})"
        for s in shots
    ]
    pic_lines = [
        f"- <Picture {i}>: pinned to frame {r.pin_frame} ({format_timestamp(r.pin_frame)}) "
        f"in [Shot {r.shot_index}]"
        for i, r in enumerate(refs, start=1)
    ]
    return (
        f"Mode: {mode}\n"
        f"User target instruction: {instruction}\n"
        f"Clip specs: {normalized.width}x{normalized.height}, {normalized.fps} fps, "
        f"{normalized.num_frames} frames, max valid timestamp = {end_ts}\n"
        f"Shots ({len(shots)}):\n" + "\n".join(shot_lines) + "\n"
        f"Painted reference frames ({len(refs)}):\n" + "\n".join(pic_lines) + "\n\n"
        "Attached images in order:\n"
        "1. Annotated contact sheet of <Video 1> showing framing, cuts, and timestamps.\n"
        "2+. Each painted target keyframe (<Picture 1>, ...).\n"
        "Write the exact 6-section MiniMax-H3 prompt now."
    )


def _extract_cost(resp: dict) -> float:
    usage = resp.get("usage")
    if isinstance(usage, dict) and usage.get("cost") is not None:
        return float(usage["cost"])
    if resp.get("cost_usd") is not None:
        return float(resp["cost_usd"])
    return 0.0


def generate_minimax_prompt(
    client: GPUniqClient,
    mode: RemakeMode,
    instruction: str,
    normalized: NormalizedVideo,
    shots: list[ShotInfo],
    refs: list[PaintedRef],
    contact_sheet: Path,
    *,
    vlm_model: str = "gemini-3-flash",
) -> tuple[str, float]:
    """Generate and validate a six-section MiniMax-H3 prompt using GPUniq VLM."""
    content_parts: list[dict] = [
        {
            "type": "text",
            "text": _build_user_context(mode, instruction, normalized, shots, refs),
        },
        {
            "type": "image_url",
            "image_url": {"url": encode_image_data_url(contact_sheet, mime="image/jpeg")},
        },
    ]
    for ref in refs:
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_url(ref.painted_path, mime="image/png")},
            }
        )

    messages: list[dict] = [
        {"role": "system", "content": _system_instructions(mode)},
        {"role": "user", "content": content_parts},
    ]

    total_cost = 0.0
    resp = client.chat_completions(
        {
            "model": vlm_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,
        }
    )
    total_cost += _extract_cost(resp)
    finish_reason = resp["choices"][0].get("finish_reason")
    if finish_reason and finish_reason != "stop":
        logger.warning("VLM completion ended with finish_reason={}", finish_reason)
    draft = normalize_ascii_punctuation(resp["choices"][0]["message"]["content"])

    validation = validate_minimax_prompt(
        draft,
        mode=mode,
        num_frames=normalized.num_frames,
        num_pics=len(refs),
        num_shots=len(shots),
    )
    if validation.valid:
        return draft, round(total_cost, 6)

    logger.warning("Initial prompt failed validation ({}), running repair turn", validation.errors)
    repair_prompt = (
        "Your prompt failed strict validation. Fix ALL errors below and return ONLY the corrected "
        "six-section prompt in plain ASCII:\n- " + "\n- ".join(validation.errors)
    )
    messages.append({"role": "assistant", "content": draft})
    messages.append({"role": "user", "content": repair_prompt})

    resp_fix = client.chat_completions(
        {
            "model": vlm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
        }
    )
    total_cost += _extract_cost(resp_fix)
    fixed = normalize_ascii_punctuation(resp_fix["choices"][0]["message"]["content"])

    recheck = validate_minimax_prompt(
        fixed,
        mode=mode,
        num_frames=normalized.num_frames,
        num_pics=len(refs),
        num_shots=len(shots),
    )
    if not recheck.valid:
        raise ValueError(
            "Prompt remained invalid after repair attempt: " + "; ".join(recheck.errors)
        )

    return fixed, round(total_cost, 6)

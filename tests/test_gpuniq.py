"""HTTP-mocked unit tests for GPUniq client, keyframe painter, and VLM prompter."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image
from pytest_httpx import HTTPXMock

from loki_remake.gpuniq.client import GPUniqClient
from loki_remake.gpuniq.painter import paint_keyframe
from loki_remake.gpuniq.prompter import generate_minimax_prompt
from loki_remake.schemas import NormalizedVideo, PaintedRef, ShotInfo

_SAMPLE_EDIT_PROMPT = """subject_definitions:
<Video 1> is the source video for the target video edit.
<Picture 1> is a frame of the target video, exactly as the target video shows it at that moment.
<Subject 1> is the red leather jacket worn by the man in <Picture 1>.
<Subject 2> is the park setting and pizza dough from <Video 1>.

summary:
[video editing] The target video is an edited version of <Video 1> in which the man
wears <Subject 1> instead of a grey hoodie while tossing dough in <Subject 2>.

retention_analysis:
<Video 1> (source video editing): fully_preserved - camera motion, framing, and timing
are copied frame for frame.
<Picture 1> (target keyframe): fully_preserved - target passes through this frame.
<Subject 1> (appears in [Shot 1]): new - red leather jacket appears in every frame.
<Subject 2> (appears in [Shot 1]): fully_preserved - park background remains unchanged.

detailed_description:
A realistic sunny outdoor handheld look.
[Shot 1] From 00:00.000 to 00:09.375, the man wears <Subject 1> as he tosses dough
in <Subject 2>, matching <Picture 1> at 00:00.500.

overall_soundscape:
The soundtrack of <Video 1>, unchanged.

non_diegetic_music:
N/A"""


def _make_png(
    path: Path,
    size: tuple[int, int] = (240, 416),
    color: tuple[int, int, int] = (40, 90, 200),
) -> Path:
    img = Image.new("RGB", size, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def _png_b64(size: tuple[int, int] = (480, 832)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (220, 40, 40)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_client_balances(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/workspaces/balances",
        json={
            "exception": 0,
            "data": {"account": {"balance_usd": 112.14, "free_usd": 112.14, "reserved_usd": 0.0}},
        },
    )
    with GPUniqClient(api_key="gpuniq_test") as client:
        data = client.get_balances()
    assert data["account"]["balance_usd"] == 112.14


def test_painter_moderation_fallback(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    src_frame = _make_png(tmp_path / "src.png")
    out_path = tmp_path / "painted.png"

    # Attempt 1: nano-banana-2 fails content_moderation
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/images/jobs",
        json={"exception": 0, "data": {"job_id": "job_1"}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/images/jobs/job_1",
        json={
            "exception": 0,
            "data": {
                "status": "failed",
                "error_code": "content_moderation",
                "error": "Blocked by filter",
            },
        },
    )
    # Attempt 2: gpt-image-2 succeeds after 1 pending poll
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/images/jobs",
        json={"exception": 0, "data": {"job_id": "job_2"}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/images/jobs/job_2",
        json={"exception": 0, "data": {"status": "processing"}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/images/jobs/job_2",
        json={
            "exception": 0,
            "data": {
                "status": "completed",
                "cost_usd": 0.0464,
                "image": {"b64_json": _png_b64((512, 896))},
            },
        },
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        ref = paint_keyframe(
            client,
            source_frame=src_frame,
            mode="edit",
            instruction="Red jacket",
            aspect_ratio="9:16",
            target_size=(480, 832),
            out_path=out_path,
            model="nano-banana-2",
            fallback_model="gpt-image-2",
            poll_interval_sec=0.0,
            sleep_fn=lambda _: None,
        )

    assert ref.model_used == "gpt-image-2"
    assert ref.cost_usd == 0.0464
    assert out_path.exists()
    with Image.open(out_path) as out_img:
        assert out_img.size == (480, 832)


def test_prompter_repair_turn(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    sheet = _make_png(tmp_path / "sheet.jpg", size=(320, 320))
    painted = _make_png(tmp_path / "ref1.png", size=(480, 832))

    # First VLM response has invalid timestamp past 00:09.375
    bad_draft = _SAMPLE_EDIT_PROMPT.replace("00:09.375", "00:15.000")
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/openai/chat/completions",
        json={
            "choices": [{"message": {"content": bad_draft}}],
            "usage": {"cost": 0.0012},
        },
    )
    # Repair turn fixes it
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/openai/chat/completions",
        json={
            "choices": [{"message": {"content": _SAMPLE_EDIT_PROMPT}}],
            "usage": {"cost": 0.0008},
        },
    )

    norm = NormalizedVideo(
        source_path=tmp_path / "in.mp4",
        video_path=tmp_path / "cond.mp4",
        width=480,
        height=832,
        fps=24,
        num_frames=226,
        duration_sec=9.4167,
        aspect_ratio="9:16",
    )
    shots = [
        ShotInfo(
            index=1,
            start_frame=0,
            end_frame=225,
            start_sec=0.0,
            end_sec=9.375,
            keyframe_index=12,
            keyframe_sec=0.5,
            sharpness_score=100.0,
            keyframe_path=painted,
        )
    ]
    refs = [
        PaintedRef(
            shot_index=1,
            pin_frame=12,
            pin_sec=0.5,
            source_frame_path=painted,
            painted_path=painted,
            model_used="nano-banana-2",
            cost_usd=0.05,
        )
    ]

    with GPUniqClient(api_key="gpuniq_test") as client:
        prompt_text, cost = generate_minimax_prompt(
            client,
            mode="edit",
            instruction="Red leather jacket",
            normalized=norm,
            shots=shots,
            refs=refs,
            contact_sheet=sheet,
        )

    assert "00:09.375" in prompt_text
    assert cost == 0.002

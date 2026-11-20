"""HTTP-mocked unit tests for GPUniq client, keyframe painter, and VLM prompter."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image
from pytest_httpx import HTTPXMock

from loki_remake.gpuniq.client import GPUniqClient
from loki_remake.gpuniq.ops import extract_ssh_connection, pick_best_h200, rank_pro6000_gpus
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


def test_pick_best_h200_filters_and_ranks(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/marketplace/list?sort_by=price-low&page=1&page_size=30&gpu_model=H200&gpu_model=H200+NVL&min_vram_gb=138.0&min_ram_gb=150.0&min_disk_gb=280.0&min_inet_speed_mbps=800.0&max_price_per_hour=6.0&max_gpu_count=1&provider=vast",
        json={
            "exception": 0,
            "data": {
                "agents": [
                    {
                        "id": 1,
                        "gpu_model": "H200",
                        "gpu_count": 2,
                        "status": "online",
                        "instances_available": 1,
                        "vram_gb": 280,
                        "ram_gb": 300,
                        "storage_gb": 500,
                        "down_mbps": 5000,
                        "reliability": 0.99,
                        "price_per_hour": 4.0,
                    },
                    {
                        "id": 2,
                        "gpu_model": "H200 NVL",
                        "gpu_count": 1,
                        "status": "online",
                        "instances_available": 1,
                        "vram_gb": 140,
                        "ram_gb": 251,
                        "storage_gb": 900,
                        "down_mbps": 877,
                        "reliability": 0.995,
                        "cuda_max_good": 13.2,
                        "price_per_hour": 4.88,
                    },
                    {
                        "id": 3,
                        "gpu_model": "H200 NVL",
                        "gpu_count": 1,
                        "status": "online",
                        "instances_available": 1,
                        "vram_gb": 140,
                        "ram_gb": 188,
                        "storage_gb": 1500,
                        "down_mbps": 2336,
                        "reliability": 0.95,
                        "cuda_max_good": 13.5,
                        "price_per_hour": 4.52,
                    },
                ]
            },
        },
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        best = pick_best_h200(client)
    assert best["id"] == 3


def test_extract_ssh_connection_variants() -> None:
    assert extract_ssh_connection(
        {"ssh_host": "1.2.3.4", "ssh_port": 22022, "ssh_user": "root"}
    ) == ("1.2.3.4", 22022, "root")

    assert extract_ssh_connection(
        {"connection": {"host": "5.6.7.8", "port": "31337"}, "username": "ubuntu"}
    ) == ("5.6.7.8", 31337, "ubuntu")

    assert extract_ssh_connection(
        {"ssh_command": "ssh -p 44123 vast@9.8.7.6"}
    ) == ("9.8.7.6", 44123, "vast")

    assert extract_ssh_connection(
        {
            "server_connection": {
                "ssh_host": "ssh2.gpuniq.com",
                "ssh_port": 20042,
                "ssh_username": "root",
                "direct_ssh_host": "207.148.106.151",
                "direct_ssh_port": 43499,
            }
        }
    ) == ("207.148.106.151", 43499, "root")

    assert extract_ssh_connection(
        {
            "container": {
                "ssh_host": "ssh2.gpuniq.com",
                "ssh_port": 20042,
                "ssh_username": "root",
                "direct_ssh_host": "207.148.106.151",
                "direct_ssh_port": 43499,
            }
        }
    ) == ("207.148.106.151", 43499, "root")


def test_painter_multi_model_cascade(httpx_mock: HTTPXMock, tmp_path: Path) -> None:
    src_frame = _make_png(tmp_path / "src.png")
    user_ref = _make_png(tmp_path / "celeb.png", color=(20, 200, 20))
    out_path = tmp_path / "painted.png"

    # Attempt 1: nano-banana-2 rejected by content_moderation
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/images/jobs",
        json={"exception": 0, "data": {"job_id": "job_nb"}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/images/jobs/job_nb",
        json={
            "exception": 0,
            "data": {"status": "failed", "error_code": "content_moderation", "error": "Celebrity"},
        },
    )
    # Attempt 2: seedream-4-5 succeeds
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/images/jobs",
        json={"exception": 0, "data": {"job_id": "job_sd"}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/images/jobs/job_sd",
        json={
            "exception": 0,
            "data": {
                "status": "completed",
                "cost_usd": 0.04,
                "image": {"b64_json": _png_b64((512, 896))},
            },
        },
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        ref = paint_keyframe(
            client,
            source_frame=src_frame,
            mode="swap",
            instruction="Swap face",
            aspect_ratio="9:16",
            target_size=(480, 832),
            out_path=out_path,
            user_ref=user_ref,
            model="nano-banana-2",
            fallback_models=["seedream-4-5", "flux-2-pro", "gpt-image-2"],
            poll_interval_sec=0.0,
            sleep_fn=lambda _: None,
        )

    assert ref.model_used == "seedream-4-5"
    assert ref.cost_usd == 0.04
    assert out_path.exists()


def test_rank_pro6000_gpus(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/marketplace/list?sort_by=price-low&page=1&page_size=25&gpu_model=RTX+PRO+6000&gpu_model=RTX+PRO+6000+S&gpu_model=RTX+PRO+6000+WS&gpu_model=RTX+PRO+6000+Max-Q&gpu_model=RTX+PRO+6000+Blackwell&min_vram_gb=90.0&min_ram_gb=150.0&min_disk_gb=280.0&min_inet_speed_mbps=500.0&max_price_per_hour=3.0&max_gpu_count=1&provider=vast",
        json={
            "exception": 0,
            "data": {
                "agents": [
                    {
                        "id": 100,
                        "gpu_model": "RTX PRO 6000 Max-Q",
                        "gpu_count": 1,
                        "status": "online",
                        "instances_available": 1,
                        "vram_gb": 95,
                        "ram_gb": 188,
                        "storage_gb": 500,
                        "down_mbps": 900,
                        "reliability": 0.99,
                        "cuda_max_good": 13.0,
                        "price_per_hour": 2.0,
                    },
                    {
                        "id": 101,
                        "gpu_model": "RTX PRO 6000 S",
                        "gpu_count": 1,
                        "status": "online",
                        "instances_available": 1,
                        "vram_gb": 95,
                        "ram_gb": 251,
                        "storage_gb": 900,
                        "down_mbps": 1850,
                        "reliability": 0.993,
                        "cuda_max_good": 13.2,
                        "price_per_hour": 2.26,
                    },
                ]
            },
        },
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        ranked = rank_pro6000_gpus(client)
    assert [a["id"] for a in ranked] == [101, 100]


def test_painter_multi_user_refs_and_custom_prompt(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    src_frame = tmp_path / "src.png"
    ref_a = tmp_path / "ref_a.png"
    ref_b = tmp_path / "ref_b.png"
    out_path = tmp_path / "painted.png"
    for idx, p in enumerate((src_frame, ref_a, ref_b)):
        Image.new("RGB", (64, 64), (30 * idx + 10, 50, 90)).save(p)

    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/images/jobs",
        json={"exception": 0, "data": {"job_id": "job_multi_1"}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/images/jobs/job_multi_1",
        json={
            "exception": 0,
            "data": {
                "status": "completed",
                "cost_usd": 0.05,
                "image": {"b64_json": _png_b64((512, 288))},
            },
        },
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        painted = paint_keyframe(
            client,
            source_frame=src_frame,
            mode="swap",
            instruction="Swap both",
            aspect_ratio="16:9",
            target_size=(1280, 736),
            out_path=out_path,
            user_refs=[ref_a, ref_b],
            custom_prompt="Explicit Image 1/2/3 prompt",
            poll_interval_sec=0.0,
            sleep_fn=lambda _: None,
        )

    post_req = httpx_mock.get_requests()[0]
    payload = json.loads(post_req.content.decode("utf-8"))
    assert payload["prompt"] == "Explicit Image 1/2/3 prompt"
    assert len(payload["input_images"]) == 3
    assert painted.cost_usd == 0.05
    assert Image.open(out_path).size == (1280, 736)


def test_run_ssh_idle_timeout(monkeypatch, tmp_path: Path) -> None:
    import sys

    import pytest

    from loki_remake.gpuniq import ops

    monkeypatch.setattr(
        ops,
        "_ssh_base_args",
        lambda _k, _p: [sys.executable, "-c", "import sys; exec(sys.argv[2])"],
    )
    with pytest.raises(TimeoutError, match="no output"):
        ops.run_ssh(
            "ignored",
            22,
            "root",
            tmp_path / "id_ed25519",
            "import time; time.sleep(5)",
            stream=True,
            idle_timeout=0.3,
        )


def test_create_marketplace_order_never_retries_on_timeout(httpx_mock: HTTPXMock) -> None:
    import httpx
    import pytest

    from loki_remake.gpuniq.client import GPUniqAPIError

    httpx_mock.add_exception(
        httpx.ReadTimeout("timed out"),
        method="POST",
        url="https://api.gpuniq.com/v1/marketplace/order",
    )
    with GPUniqClient(api_key="gpuniq_test") as client:
        with pytest.raises(GPUniqAPIError, match="Transport error"):
            client.create_marketplace_order(agent_id=123)

    assert len(httpx_mock.get_requests()) == 1

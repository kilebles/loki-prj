"""Unit tests for Topaz API, upscaler GPU selection, and visual comparison grids."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image
from pytest_httpx import HTTPXMock

from loki_remake.gpuniq.client import GPUniqClient
from loki_remake.gpuniq.ops import pick_best_upscaler_gpu
from loki_remake.video.upscale import build_comparison_grids, upscale_via_topaz


def _make_tiny_mp4(
    path: Path,
    *,
    width: int = 320,
    height: int = 192,
    duration_sec: float = 0.5,
    with_audio: bool = True,
    color: str = "blue",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r=24:d={duration_sec}",
    ]
    if with_audio:
        cmd.extend(["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration_sec}"])
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if with_audio:
        cmd.extend(["-c:a", "aac", "-shortest"])
    cmd.append(str(path))
    subprocess.run(cmd, check=True)
    return path


def test_topaz_client_endpoints(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/topaz/models",
        json={
            "exception": 0,
            "data": {
                "usd_per_credit": 0.14,
                "video_models": [{"slug": "topaz-video-proteus"}],
            },
        },
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs",
        json={"exception": 0, "data": {"job_id": "tv_1", "estimated_cost_usd": 0.28}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs/tv_1",
        json={"exception": 0, "data": {"status": "completed", "credits": 2, "cost_usd": 0.28}},
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        models = client.list_topaz_models()
        job = client.create_topaz_video_job({"model": "topaz-video-proteus"})
        done = client.get_topaz_video_job(job["job_id"])

    assert models["usd_per_credit"] == 0.14
    assert job["job_id"] == "tv_1"
    assert done["cost_usd"] == 0.28


def test_upscale_via_topaz_remuxes_audio_and_polls(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    src = _make_tiny_mp4(tmp_path / "src_480.mp4", width=320, height=192, with_audio=True)
    upstream_no_audio = _make_tiny_mp4(
        tmp_path / "upstream_736.mp4", width=640, height=384, with_audio=False, color="red"
    )
    out = tmp_path / "out_topaz.mp4"

    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs",
        json={"exception": 0, "data": {"job_id": "tv_2", "estimated_cost_usd": 0.42}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs/tv_2",
        json={"exception": 0, "data": {"status": "processing"}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs/tv_2",
        json={
            "exception": 0,
            "data": {
                "status": "completed",
                "credits": 3.0,
                "cost_usd": 0.42,
                "video": {"url": "https://cdn.example.com/result.mp4"},
            },
        },
    )
    httpx_mock.add_response(
        method="GET",
        url="https://cdn.example.com/result.mp4",
        content=upstream_no_audio.read_bytes(),
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        res = upscale_via_topaz(
            client,
            src,
            out,
            model="topaz-video-proteus",
            output_width=640,
            output_height=384,
            poll_interval_sec=0.0,
            sleep_fn=lambda _: None,
        )

    assert out.exists()
    assert res.width == 640
    assert res.height == 384
    assert res.has_audio is True
    assert res.credits == 3.0
    assert res.upscale_cost_usd == 0.42


def test_upscale_via_topaz_temp_upload_fallback(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch
) -> None:
    src = _make_tiny_mp4(tmp_path / "src.mp4", width=320, height=192, with_audio=True)
    upstream_ok = _make_tiny_mp4(tmp_path / "up.mp4", width=640, height=384, with_audio=True)
    out = tmp_path / "out.mp4"

    monkeypatch.setattr(
        "loki_remake.video.upscale.upload_temp_https",
        lambda path, http_client=None: "https://litterbox.catbox.moe/abc.mp4",
    )

    # First call (data URI) rejected with 400
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs",
        status_code=400,
        json={"error": {"code": "invalid_request_error", "message": "https url required"}},
    )
    # Second call with temp https url succeeds
    httpx_mock.add_response(
        method="POST",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs",
        json={"exception": 0, "data": {"job_id": "tv_3", "estimated_cost_usd": 0.28}},
    )
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/llm/topaz/video/jobs/tv_3",
        json={
            "exception": 0,
            "data": {
                "status": "completed",
                "credits": 2.0,
                "cost_usd": 0.28,
                "video": {"url": "https://cdn.example.com/ok.mp4"},
            },
        },
    )
    httpx_mock.add_response(
        method="GET",
        url="https://cdn.example.com/ok.mp4",
        content=upstream_ok.read_bytes(),
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        res = upscale_via_topaz(
            client,
            src,
            out,
            model="topaz-video-iris-2",
            output_width=640,
            output_height=384,
            allow_temp_upload=True,
            poll_interval_sec=0.0,
            sleep_fn=lambda _: None,
        )

    assert res.upscale_cost_usd == 0.28
    assert out.exists()


def test_pick_best_upscaler_gpu_fallback(httpx_mock: HTTPXMock) -> None:
    # Tier 1 returns no valid single-GPU online agents
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/marketplace/list?sort_by=price-low&page=1&page_size=20&gpu_model=A100+SXM4&gpu_model=L40S&gpu_model=A100+PCIe&min_vram_gb=39.0&min_ram_gb=40.0&min_disk_gb=80.0&min_inet_speed_mbps=500.0&max_price_per_hour=1.3&max_gpu_count=1&provider=vast",
        json={"exception": 0, "data": {"agents": [{"id": 1, "gpu_count": 2, "status": "online"}]}},
    )
    # Tier 2 returns RTX 4090 and RTX 5090
    httpx_mock.add_response(
        method="GET",
        url="https://api.gpuniq.com/v1/marketplace/list?sort_by=price-low&page=1&page_size=20&gpu_model=RTX+5090&gpu_model=RTX+4090&min_vram_gb=23.0&min_ram_gb=30.0&min_disk_gb=80.0&min_inet_speed_mbps=500.0&max_price_per_hour=1.0&max_gpu_count=1&provider=vast",
        json={
            "exception": 0,
            "data": {
                "agents": [
                    {
                        "id": 10,
                        "gpu_model": "RTX 5090",
                        "gpu_count": 1,
                        "status": "online",
                        "instances_available": 1,
                        "vram_gb": 31,
                        "ram_gb": 32,
                        "storage_gb": 120,
                        "down_mbps": 800,
                        "reliability": 0.96,
                        "cuda_max_good": 12.8,
                        "price_per_hour": 0.59,
                    },
                    {
                        "id": 11,
                        "gpu_model": "RTX 4090",
                        "gpu_count": 1,
                        "status": "online",
                        "instances_available": 1,
                        "vram_gb": 24,
                        "ram_gb": 32,
                        "storage_gb": 120,
                        "down_mbps": 900,
                        "reliability": 0.98,
                        "cuda_max_good": 12.8,
                        "price_per_hour": 0.49,
                    },
                ]
            },
        },
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        best = pick_best_upscaler_gpu(client)
    assert best["id"] == 11


def test_build_comparison_grids(tmp_path: Path) -> None:
    v1 = _make_tiny_mp4(tmp_path / "v480.mp4", width=416, height=240, color="green")
    v2 = _make_tiny_mp4(tmp_path / "v736.mp4", width=640, height=368, color="yellow")
    crops = tmp_path / "crops.jpg"
    full = tmp_path / "full.jpg"

    build_comparison_grids(
        [
            {"path": v1, "title": "Raw 480p", "subtitle": "95.7s | $0.148"},
            {"path": v2, "frame_index": 2, "title": "Native 736p", "subtitle": "368.0s | $0.570"},
        ],
        frame_index=3,
        target_size=(640, 368),
        crop_box=(100, 50, 300, 250),
        crops_out_path=crops,
        full_out_path=full,
        full_subset_indices=[0, 1],
    )

    assert crops.exists() and crops.stat().st_size > 1000
    assert full.exists() and full.stat().st_size > 1000
    with Image.open(crops) as im:
        assert im.size[0] > 400

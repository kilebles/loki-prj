"""Unit tests for video probing and normalization."""

from __future__ import annotations

import subprocess
from pathlib import Path

from loki_remake.video.normalize import (
    compute_canvas,
    normalize_video,
    probe_video,
    snap_frame_count,
)


def test_snap_frame_count() -> None:
    assert snap_frame_count(1) == 90
    assert snap_frame_count(90) == 90
    assert snap_frame_count(91) == 107
    assert snap_frame_count(120) == 124
    assert snap_frame_count(226) == 226
    assert snap_frame_count(345) == 345
    assert snap_frame_count(500) == 345
    for raw in range(80, 360, 7):
        snapped = snap_frame_count(raw)
        assert 90 <= snapped <= 345
        assert (snapped - 5) % 17 == 0


def test_compute_canvas() -> None:
    assert compute_canvas(1080, 1920, 480) == (480, 832, "9:16")
    assert compute_canvas(1080, 1920, 736) == (736, 1280, "9:16")
    assert compute_canvas(1920, 1080, 480) == (832, 480, "16:9")
    assert compute_canvas(1920, 1080, 736) == (1280, 736, "16:9")

    w, h, slug = compute_canvas(200, 2000, 480)
    assert w == 480
    assert h % 32 == 0
    assert h <= 1152
    assert slug == "9:16"


def test_normalize_synthetic_vertical_video(tmp_path: Path) -> None:
    src = tmp_path / "src_4s.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=720x1280:rate=30:duration=4.0",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=4.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(src),
    ]
    subprocess.run(cmd, check=True)

    probe = probe_video(src)
    assert probe.width == 720
    assert probe.height == 1280
    assert probe.has_audio is True

    norm = normalize_video(src, out_dir=tmp_path / "out", short_edge=480)
    assert norm.width == 480
    assert norm.height == 832
    assert norm.fps == 24
    # 4.0s at 24fps = 96 frames -> snapped to 107 frames
    assert norm.num_frames == 107
    assert (norm.num_frames - 5) % 17 == 0
    assert norm.padded_frames > 0
    assert norm.aspect_ratio == "9:16"
    assert norm.video_path.exists()
    assert norm.audio_path is not None and norm.audio_path.exists()

    out_probe = probe_video(norm.video_path)
    assert out_probe.width == 480
    assert out_probe.height == 832
    assert out_probe.frame_count == 107
    assert abs(out_probe.fps - 24.0) < 0.05

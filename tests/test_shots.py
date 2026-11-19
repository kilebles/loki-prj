"""Unit tests for shot detection, keyframe extraction, and contact sheet generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from loki_remake.video.shots import detect_shots_and_keyframes, format_timestamp


def _write_two_shot_video(path: Path, shot1_frames: int = 45, shot2_frames: int = 45) -> int:
    w, h = 240, 416
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    assert writer.isOpened()

    # Shot 1: blue/cyan frame with sharp white checkerboard text/lines
    base1 = np.zeros((h, w, 3), dtype=np.uint8)
    base1[:, :] = (200, 120, 20)
    cv2.rectangle(base1, (30, 60), (210, 300), (255, 255, 255), 4)
    cv2.line(base1, (30, 60), (210, 300), (255, 255, 255), 3)

    # Shot 2: bright red/orange frame with dark circle
    base2 = np.zeros((h, w, 3), dtype=np.uint8)
    base2[:, :] = (20, 40, 230)
    cv2.circle(base2, (120, 208), 70, (20, 240, 240), -1)

    for i in range(shot1_frames):
        frame = base1.copy()
        if i == 15:
            # Make frame 15 extra sharp with fine detail
            for y in range(80, 280, 12):
                cv2.line(frame, (40, y), (200, y), (255, 255, 255), 1)
        writer.write(frame)

    for _ in range(shot2_frames):
        writer.write(base2.copy())

    writer.release()
    return shot1_frames + shot2_frames


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00.000"
    assert format_timestamp(24) == "00:01.000"
    assert format_timestamp(225) == "00:09.375"


def test_detect_two_shots_and_contact_sheet(tmp_path: Path) -> None:
    video_path = tmp_path / "two_shots.mp4"
    total_frames = _write_two_shot_video(video_path)

    shots, pins, sheet_path = detect_shots_and_keyframes(
        video_path=video_path,
        num_frames=total_frames,
        out_dir=tmp_path / "out",
        max_refs=2,
    )

    assert len(shots) == 2
    assert shots[0].start_frame == 0
    assert shots[1].start_frame == 45
    assert shots[0].keyframe_path.exists()
    assert shots[1].keyframe_path.exists()
    assert len(pins) == 2
    assert pins[0][0] == 1 and pins[1][0] == 2
    assert sheet_path.exists() and sheet_path.stat().st_size > 1000

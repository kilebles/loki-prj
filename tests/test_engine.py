"""Unit tests for MiniMax-H3 / Viggle-Remake bf16 command builder and parser."""

from __future__ import annotations

from pathlib import Path

from loki_remake.engine.viggle_bf16 import build_sample_command, parse_sample_output
from loki_remake.schemas import NormalizedVideo, PaintedRef, PreparedJob, ShotInfo


def _sample_job(tmp_path: Path) -> PreparedJob:
    norm = NormalizedVideo(
        source_path=tmp_path / "src.mp4",
        video_path=tmp_path / "cond.mp4",
        audio_path=tmp_path / "audio.m4a",
        width=832,
        height=480,
        fps=24,
        num_frames=226,
        duration_sec=9.4167,
        aspect_ratio="16:9",
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
            sharpness_score=200.0,
            keyframe_path=tmp_path / "kf1.png",
        )
    ]
    refs = [
        PaintedRef(
            shot_index=1,
            pin_frame=12,
            pin_sec=0.5,
            source_frame_path=tmp_path / "kf1.png",
            painted_path=tmp_path / "painted" / "ref_1.png",
            model_used="nano-banana-2",
        ),
        PaintedRef(
            shot_index=1,
            pin_frame=160,
            pin_sec=6.6667,
            source_frame_path=tmp_path / "kf2.png",
            painted_path=tmp_path / "painted" / "ref_2.png",
            model_used="nano-banana-2",
        ),
    ]
    return PreparedJob(
        job_id="abc123",
        mode="swap",
        tier=480,
        instruction="Replace man with pineapple mascot",
        normalized=norm,
        shots=shots,
        refs=refs,
        prompt_path=tmp_path / "prompt.txt",
        prompt_text="prompt",
        contact_sheet_path=tmp_path / "sheet.jpg",
    )


def test_build_sample_command_with_pins_and_override(tmp_path: Path) -> None:
    job = _sample_job(tmp_path)
    cmd = build_sample_command(
        job,
        base_dir=Path("/workspace/models/MiniMax-H3"),
        lora_repo=Path("/workspace/models/Viggle-Remake"),
        out_video=Path("/workspace/jobs/abc123/out_736.mp4"),
        video_path=Path("/workspace/jobs/abc123/cond.mp4"),
        audio_path=Path("/workspace/jobs/abc123/audio.m4a"),
        prompt_path=Path("/workspace/jobs/abc123/prompt.txt"),
        ref_paths=[
            Path("/workspace/jobs/abc123/painted/ref_1.png"),
            Path("/workspace/jobs/abc123/painted/ref_2.png"),
        ],
        short_edge=736,
        seed=42,
    )

    assert cmd[0] == "/workspace/venv/bin/python"
    assert cmd[1] == "/workspace/models/Viggle-Remake/inference/sample.py"
    assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "swap"
    assert "--short-edge" in cmd and cmd[cmd.index("--short-edge") + 1] == "736"
    assert "--frames" in cmd and cmd[cmd.index("--frames") + 1] == "226"
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "42"
    assert "--audio" in cmd and cmd[cmd.index("--audio") + 1] == "/workspace/jobs/abc123/audio.m4a"
    assert "--attn-backend" in cmd and cmd[cmd.index("--attn-backend") + 1] == "_native_cudnn"

    ref_indices = [i for i, tok in enumerate(cmd) if tok == "--ref"]
    pin_indices = [i for i, tok in enumerate(cmd) if tok == "--pin"]
    assert len(ref_indices) == 2 and len(pin_indices) == 2
    assert cmd[pin_indices[0] + 1] == "12"
    assert cmd[pin_indices[1] + 1] == "160"


def test_parse_sample_output() -> None:
    log = (
        "Loading checkpoint shards: 100%|██| 12/12\n"
        "render 58.4s at 832x480x226, peak 87.2 GiB allocated / 91.0 GiB reserved "
        "-> /workspace/jobs/abc123/out_480.mp4\n"
    )
    stats = parse_sample_output(log)
    assert stats["render_sec"] == 58.4
    assert stats["width"] == 832.0
    assert stats["height"] == 480.0
    assert stats["num_frames"] == 226.0
    assert stats["peak_alloc_gib"] == 87.2
    assert stats["peak_reserved_gib"] == 91.0

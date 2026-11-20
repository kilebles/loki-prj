"""Unit tests for face scale QC, painter retry/fallback, and face enhancement helpers."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pytest_httpx import HTTPXMock
from typer.testing import CliRunner

from loki_remake import face_qc
from loki_remake.cli import app
from loki_remake.gpuniq import painter
from loki_remake.gpuniq.client import GPUniqClient


def _b64_png(
    size: tuple[int, int] = (480, 832),
    color: tuple[int, int, int] = (90, 120, 150),
) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_face(cx: float, cy: float, w: float, h: float, iod: float) -> face_qc.FaceMeasurement:
    lm5 = np.array(
        [
            [cx - iod * 0.5, cy],
            [cx + iod * 0.5, cy],
            [cx, cy + h * 0.18],
            [cx - w * 0.18, cy + h * 0.35],
            [cx + w * 0.18, cy + h * 0.35],
        ],
        dtype=np.float32,
    )
    return face_qc.FaceMeasurement(
        bbox=(cx - w * 0.5, cy - h * 0.4, w, h),
        lm5=lm5,
        iod=iod,
        eye_center=(cx, cy),
        score=0.98,
    )


def test_compare_face_scale_and_letterbox_safe_remap(monkeypatch, tmp_path: Path) -> None:
    h, w = 240, 320
    src_bgr = np.full((h, w, 3), 180, dtype=np.uint8)
    src_bgr[:30, :] = 0
    src_bgr[210:, :] = 0

    painted_bgr = np.full((h, w, 3), 160, dtype=np.uint8)
    painted_bgr[:30, :] = 0
    painted_bgr[210:, :] = 0

    src_face = _make_face(160.0, 110.0, 60.0, 80.0, 28.0)
    big_face = _make_face(160.0, 112.0, 75.0, 100.0, 35.0)
    fixed_face = _make_face(160.0, 110.0, 60.5, 80.5, 28.2)

    call_count = {"n": 0}

    def fake_detect(img: np.ndarray, _path: Path, score_threshold: float = 0.5):
        del score_threshold
        call_count["n"] += 1
        if img is src_bgr or np.all(img[100, 100] == 180):
            return [src_face]
        if call_count["n"] <= 2:
            return [big_face]
        return [fixed_face]

    probs = np.zeros((h, w, 6), dtype=np.float32)
    probs[:, :, 0] = 1.0
    probs[175:, :, 4] = 0.9

    monkeypatch.setattr(face_qc, "detect_faces_yunet", fake_detect)
    monkeypatch.setattr(face_qc, "segment_multiclass", lambda _img, _p: probs)

    out_bgr, report = face_qc.normalize_head_scale(
        src_bgr,
        painted_bgr,
        yunet_path=tmp_path / "yunet.onnx",
        tflite_path=tmp_path / "seg.tflite",
        scale_tol=0.07,
    )

    assert report is not None
    assert report.normalized_applied is True
    assert out_bgr.shape == painted_bgr.shape
    assert np.array_equal(out_bgr[:30], painted_bgr[:30])
    assert np.array_equal(out_bgr[210:], painted_bgr[210:])


def test_painter_scale_lock_retry_and_fallback(
    httpx_mock: HTTPXMock, monkeypatch, tmp_path: Path
) -> None:
    src_frame = tmp_path / "src.png"
    user_ref = tmp_path / "ref.png"
    out_path = tmp_path / "painted.png"
    Image.new("RGB", (64, 64), (100, 110, 120)).save(src_frame)
    Image.new("RGB", (64, 64), (20, 180, 90)).save(user_ref)

    b64_img = _b64_png((480, 832), (80, 90, 100))

    for jid in ("job_1", "job_2"):
        httpx_mock.add_response(
            method="POST",
            url="https://api.gpuniq.com/v1/llm/images/jobs",
            json={"exception": 0, "data": {"job_id": jid, "status": "queued"}},
        )
        httpx_mock.add_response(
            method="GET",
            url=f"https://api.gpuniq.com/v1/llm/images/jobs/{jid}",
            json={
                "exception": 0,
                "data": {
                    "job_id": jid,
                    "status": "completed",
                    "cost_usd": 0.05,
                    "image": {"b64_json": b64_img},
                },
            },
        )

    sf = _make_face(240.0, 300.0, 120.0, 160.0, 56.0)
    rep_1 = face_qc.ScaleReport(sf, sf, 1.22, 1.22, 1.22, 1.22, within_tolerance=False)
    rep_2 = face_qc.ScaleReport(sf, sf, 1.12, 1.12, 1.12, 1.12, within_tolerance=False)
    rep_3 = face_qc.ScaleReport(
        sf, sf, 1.01, 1.01, 1.01, 1.01, within_tolerance=True, normalized_applied=True
    )

    calls = {"cmp": 0}

    def fake_cmp(*_args, **_kwargs):
        calls["cmp"] += 1
        return rep_1 if calls["cmp"] == 1 else rep_2

    qc_paths = (tmp_path / "y.onnx", tmp_path / "s.tflite")
    monkeypatch.setattr(painter, "_load_qc_models", lambda _d: qc_paths)
    monkeypatch.setattr(painter, "compare_face_scale", fake_cmp)
    monkeypatch.setattr(
        painter,
        "normalize_head_scale",
        lambda _s, p_bgr, **_kw: (p_bgr, rep_3),
    )

    with GPUniqClient(api_key="gpuniq_test") as client:
        painted = painter.paint_keyframe(
            client,
            source_frame=src_frame,
            mode="swap",
            instruction="Swap face naturally",
            aspect_ratio="9:16",
            target_size=(480, 832),
            out_path=out_path,
            user_ref=user_ref,
            normalize_scale=True,
            max_scale_retries=1,
            sleep_fn=lambda _: None,
        )

    reqs = [r for r in httpx_mock.get_requests() if r.method == "POST"]
    assert len(reqs) == 2
    retry_payload = json.loads(reqs[1].content.decode("utf-8"))
    assert "CRITICAL CORRECTION" in retry_payload["prompt"]
    assert "CRITICAL PROPORTIONS AND EXPRESSION" in retry_payload["prompt"]
    assert painted.cost_usd == 0.1
    assert painted.scale_ratio_raw == 1.22
    assert painted.scale_ratio_final == 1.01
    assert painted.scale_normalized is True
    assert (tmp_path / "painted_raw.png").exists()
    assert out_path.exists()


def test_face_enhance_lab_and_grain_helpers() -> None:
    infra_dir = str(Path(__file__).resolve().parents[1] / "infra")
    if infra_dir not in sys.path:
        sys.path.insert(0, infra_dir)
    from face_enhance import _lab_color_match, _make_feathered_mask, _reinject_luminance_grain

    mask = _make_feathered_mask(128)[:, :, 0]
    src = np.full((128, 128, 3), 120, dtype=np.uint8)
    src[::2, ::2] = 135
    res = np.full((128, 128, 3), 210, dtype=np.uint8)

    matched = _lab_color_match(src, res, mask)
    assert abs(float(matched.mean()) - float(src.mean())) < 15.0

    grained = _reinject_luminance_grain(src, np.full((128, 128, 3), 128, dtype=np.uint8), 0.5)
    assert grained.dtype == np.uint8
    assert float(grained.std()) > 0.5


def test_cli_help_includes_new_flags() -> None:
    runner = CliRunner()
    env = {"COLUMNS": "140"}
    res_up = runner.invoke(app, ["gpu-up", "--help"], env=env)
    assert res_up.exit_code == 0
    assert "--image" in res_up.output

    res_prep = runner.invoke(app, ["prepare", "--help"], env=env)
    assert res_prep.exit_code == 0
    assert "--keep-hair" in res_prep.output
    assert "--normalize-scale" in res_prep.output

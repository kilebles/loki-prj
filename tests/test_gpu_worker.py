"""Unit tests for the resident GPU worker HTTP daemon and CLI subcommands."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from loki_remake.cli import app


def _load_gpu_worker_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "infra" / "gpu_worker.py"
    spec = importlib.util.spec_from_file_location("gpu_worker_mod", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeEngine:
    def __init__(self) -> None:
        self.jobs_completed = 0
        self.render_calls: list[dict[str, Any]] = []
        self.enhance_calls: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "loaded_mode": "swap",
            "quant": "fp8",
            "gpu_name": "Fake H200",
            "vram_alloc_gib": 41.3,
            "vram_total_gib": 140.0,
            "jobs_completed": self.jobs_completed,
        }

    def render(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "video" not in payload or "out" not in payload:
            raise ValueError("render payload requires 'video' and 'out'")
        self.render_calls.append(payload)
        self.jobs_completed += 1
        return {
            "ok": True,
            "mode": payload.get("mode", "swap"),
            "width": 1280,
            "height": 736,
            "num_frames": 361,
            "text_encode_sec": 2.4,
            "render_sec": 490.0,
            "enhance_sec": 0.0,
            "total_sec": 492.4,
            "peak_alloc_gib": 41.3,
            "peak_reserved_gib": 68.0,
            "out": payload["out"],
            "enhanced_out": None,
            "jobs_completed": self.jobs_completed,
        }

    def enhance(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "input" not in payload or "out" not in payload:
            raise ValueError("enhance payload requires 'input' and 'out'")
        self.enhance_calls.append(payload)
        self.jobs_completed += 1
        return {
            "ok": True,
            "enhance_sec": 18.5,
            "out": payload["out"],
            "jobs_completed": self.jobs_completed,
        }


def _http_json(
    url: str, *, method: str = "GET", body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def test_worker_http_routes_and_shutdown() -> None:
    mod = _load_gpu_worker_module()
    engine = _FakeEngine()
    server = mod.create_worker_server(engine, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()

    try:
        base = f"http://127.0.0.1:{port}"

        status, health = _http_json(f"{base}/health")
        assert status == 200
        assert health["status"] == "ready"
        assert health["loaded_mode"] == "swap"

        # Invalid render payload -> 400
        status, err = _http_json(f"{base}/render", method="POST", body={"mode": "swap"})
        assert status == 400
        assert err["ok"] is False

        # Valid render payload -> 200
        status, res = _http_json(
            f"{base}/render",
            method="POST",
            body={"mode": "swap", "video": "/tmp/in.mp4", "out": "/tmp/out.mp4"},
        )
        assert status == 200
        assert res["ok"] is True
        assert res["text_encode_sec"] == 2.4
        assert res["jobs_completed"] == 1

        # Valid enhance payload -> 200
        status, enh = _http_json(
            f"{base}/enhance",
            method="POST",
            body={"input": "/tmp/out.mp4", "out": "/tmp/out_rf.mp4", "restorer": "restoreformer"},
        )
        assert status == 200
        assert enh["ok"] is True
        assert enh["jobs_completed"] == 2

        # Unknown path -> 404
        status, not_found = _http_json(f"{base}/nope")
        assert status == 404
        assert not_found["ok"] is False

        # Graceful shutdown
        status, shut = _http_json(f"{base}/shutdown", method="POST", body={})
        assert status == 200
        assert shut["ok"] is True
        thread.join(timeout=3.0)
        assert not thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()


def test_worker_cli_commands_help() -> None:
    runner = CliRunner()
    for cmd in ("gpu-worker-start", "gpu-worker-status", "gpu-worker-render", "gpu-worker-stop"):
        res = runner.invoke(app, [cmd, "--help"])
        assert res.exit_code == 0, f"{cmd} --help failed: {res.output}"

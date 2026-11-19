"""GPUniq GPU provisioning, SSH readiness, remote execution, and state persistence."""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from loki_remake.gpuniq.client import GPUniqClient
from loki_remake.schemas import GPUInstanceState

_SSH_CMD_HOST_RE = re.compile(
    r"(?:(?P<user>[A-Za-z0-9_.-]+)@)?(?P<host>\d+\.\d+\.\d+\.\d+|[A-Za-z0-9.-]+)"
)
_SSH_CMD_PORT_RE = re.compile(r"-p\s+(?P<port>\d+)")

DEFAULT_GPU_STATE_PATH = Path("work/active_gpu.json")


def _ssh_base_args(ssh_key_path: Path, port: int) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        "-i",
        str(ssh_key_path),
        "-p",
        str(port),
    ]


def ensure_ssh_key(
    client: GPUniqClient,
    *,
    pubkey_path: Path | None = None,
    key_name: str = "elisey-mac",
) -> int:
    """Return existing GPUniq SSH key id matching local pubkey, or register it."""
    path = pubkey_path or (Path.home() / ".ssh" / "id_ed25519.pub")
    if not path.exists():
        raise FileNotFoundError(f"SSH public key not found: {path}")

    pub_text = path.read_text(encoding="utf-8").strip()
    parts = pub_text.split()
    fingerprint_core = " ".join(parts[:2]) if len(parts) >= 2 else pub_text

    listed = client.list_ssh_keys()
    for item in listed.get("ssh_keys", []):
        existing = str(item.get("public_key") or "").strip()
        existing_parts = existing.split()
        existing_core = " ".join(existing_parts[:2]) if len(existing_parts) >= 2 else existing
        if existing_core == fingerprint_core and item.get("is_active", True):
            return int(item["id"])

    created = client.create_ssh_key(key_name=key_name, public_key=pub_text)
    key_id = int(created["id"])
    logger.info("Registered SSH key '{}' on GPUniq with id={}", key_name, key_id)
    return key_id


def _filter_gpu_candidates(
    agents: list[dict[str, Any]],
    *,
    min_reliability: float,
    min_vram_gb: float,
    min_ram_gb: float,
    min_disk_gb: float,
    min_cuda: float = 12.8,
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for a in agents:
        if int(a.get("gpu_count") or 0) != 1:
            continue
        if str(a.get("status") or "").lower() != "online":
            continue
        if int(a.get("instances_available") or 0) < 1:
            continue
        if float(a.get("vram_gb") or 0) < min_vram_gb:
            continue
        if float(a.get("ram_gb") or 0) < min_ram_gb:
            continue
        if float(a.get("storage_gb") or 0) < min_disk_gb:
            continue
        if float(a.get("reliability") or 0) < min_reliability:
            continue
        cuda_max = a.get("cuda_max_good")
        if cuda_max is not None and float(cuda_max) < min_cuda:
            continue
        valid.append(a)
    return valid


def pick_best_h200(
    client: GPUniqClient,
    *,
    min_vram_gb: float = 138.0,
    min_ram_gb: float = 150.0,
    min_disk_gb: float = 280.0,
    min_down_mbps: float = 800.0,
    min_reliability: float = 0.90,
    max_price: float = 6.0,
) -> dict[str, Any]:
    """Select the fastest/cheapest reliable 1x H200 / H200 NVL offer on GPUniq Marketplace."""
    for threshold in (min_down_mbps, 400.0):
        resp = client.list_marketplace(
            gpu_model=["H200", "H200 NVL"],
            max_gpu_count=1,
            min_vram_gb=min_vram_gb,
            min_ram_gb=min_ram_gb,
            min_disk_gb=min_disk_gb,
            min_inet_speed_mbps=threshold,
            max_price_per_hour=max_price,
            provider="vast",
            sort_by="price-low",
            page_size=30,
        )
        candidates = _filter_gpu_candidates(
            resp.get("agents", []),
            min_reliability=min_reliability,
            min_vram_gb=min_vram_gb,
            min_ram_gb=min_ram_gb,
            min_disk_gb=min_disk_gb,
            min_cuda=12.8,
        )
        if candidates:
            return min(
                candidates,
                key=lambda a: (
                    0 if float(a.get("down_mbps") or 0) >= 1500.0 else 1,
                    float(a["price_per_hour"])
                    / min(max(float(a.get("down_mbps") or 400.0), 400.0), 4000.0),
                ),
            )

    raise RuntimeError(
        "No matching 1x H200 offer found on GPUniq Marketplace "
        f"(RAM>={min_ram_gb}GB, disk>={min_disk_gb}GB, max_price=${max_price}/h)"
    )


def rank_upscaler_gpus(
    client: GPUniqClient,
    *,
    min_disk_gb: float = 80.0,
    min_down_mbps: float = 500.0,
    min_reliability: float = 0.96,
) -> list[dict[str, Any]]:
    """Return ranked affordable reliable 1x GPUs (40GB+ A100/L40S first, then 5090/4090)."""
    tiers: list[tuple[list[str], float, float, float]] = [
        (["A100 SXM4", "L40S", "A100 PCIe"], 39.0, 40.0, 1.30),
        (["RTX 5090", "RTX 4090"], 23.0, 30.0, 1.00),
    ]
    ranked: list[dict[str, Any]] = []
    for gpu_models, min_vram, min_ram, max_price in tiers:
        resp = client.list_marketplace(
            gpu_model=gpu_models,
            max_gpu_count=1,
            min_vram_gb=min_vram,
            min_ram_gb=min_ram,
            min_disk_gb=min_disk_gb,
            min_inet_speed_mbps=min_down_mbps,
            max_price_per_hour=max_price,
            provider="vast",
            sort_by="price-low",
            page_size=20,
        )
        candidates = _filter_gpu_candidates(
            resp.get("agents", []),
            min_reliability=min_reliability,
            min_vram_gb=min_vram,
            min_ram_gb=min_ram,
            min_disk_gb=min_disk_gb,
            min_cuda=12.1,
        )
        ranked.extend(
            sorted(
                candidates,
                key=lambda a: (
                    0 if float(a.get("down_mbps") or 0) >= 1500.0 else 1,
                    float(a.get("price_per_hour") or 999.0),
                    -float(a.get("down_mbps") or 0.0),
                ),
            )
        )

    if ranked:
        return ranked

    raise RuntimeError(
        "No matching 1x upscaler GPU found on GPUniq Marketplace "
        "(A100/L40S <= $1.30/h or RTX 5090/4090 <= $1.00/h)"
    )


def pick_best_upscaler_gpu(
    client: GPUniqClient,
    *,
    min_disk_gb: float = 80.0,
    min_down_mbps: float = 500.0,
    min_reliability: float = 0.96,
) -> dict[str, Any]:
    """Select an affordable reliable 1x GPU (preferring 40GB+ A100/L40S, fallback 5090/4090)."""
    return rank_upscaler_gpus(
        client,
        min_disk_gb=min_disk_gb,
        min_down_mbps=min_down_mbps,
        min_reliability=min_reliability,
    )[0]


def rank_pro6000_gpus(
    client: GPUniqClient,
    *,
    min_vram_gb: float = 90.0,
    min_ram_gb: float = 150.0,
    min_disk_gb: float = 280.0,
    min_down_mbps: float = 500.0,
    min_reliability: float = 0.95,
    max_price: float = 3.0,
) -> list[dict[str, Any]]:
    """Select ranked reliable 1x RTX PRO 6000 96GB offers with >=150GB RAM for FP8 rendering."""
    resp = client.list_marketplace(
        gpu_model=[
            "RTX PRO 6000",
            "RTX PRO 6000 S",
            "RTX PRO 6000 WS",
            "RTX PRO 6000 Max-Q",
            "RTX PRO 6000 Blackwell",
        ],
        max_gpu_count=1,
        min_vram_gb=min_vram_gb,
        min_ram_gb=min_ram_gb,
        min_disk_gb=min_disk_gb,
        min_inet_speed_mbps=min_down_mbps,
        max_price_per_hour=max_price,
        provider="vast",
        sort_by="price-low",
        page_size=25,
    )
    candidates = _filter_gpu_candidates(
        resp.get("agents", []),
        min_reliability=min_reliability,
        min_vram_gb=min_vram_gb,
        min_ram_gb=min_ram_gb,
        min_disk_gb=min_disk_gb,
        min_cuda=12.8,
    )
    if candidates:
        return sorted(
            candidates,
            key=lambda a: (
                0 if float(a.get("down_mbps") or 0) >= 1500.0 else 1,
                float(a.get("price_per_hour") or 999.0),
                -float(a.get("down_mbps") or 0.0),
            ),
        )

    raise RuntimeError(
        "No matching 1x RTX PRO 6000 96GB offer found on GPUniq Marketplace "
        f"(RAM>={min_ram_gb}GB, disk>={min_disk_gb}GB, max_price=${max_price}/h)"
    )


def extract_ssh_connection(instance_data: dict[str, Any]) -> tuple[str | None, int | None, str]:
    """Extract (host, port, user) from GPUniq instance metadata."""
    containers: list[dict[str, Any]] = [instance_data]
    for nested_key in (
        "server_connection",
        "container",
        "ssh",
        "connection",
        "credentials",
        "network",
    ):
        nested = instance_data.get(nested_key)
        if isinstance(nested, dict):
            containers.append(nested)

    host: str | None = None
    port: int | None = None
    user = "root"

    for box in containers:
        if host is None and port is None:
            d_host = box.get("direct_ssh_host")
            d_port = box.get("direct_ssh_port")
            if isinstance(d_host, str) and d_host.strip() and d_port and str(d_port).isdigit():
                host = d_host.strip()
                port = int(d_port)
        if host is None:
            for k in ("ssh_host", "host", "public_ip", "public_ipaddr", "ip"):
                val = box.get(k)
                if isinstance(val, str) and val.strip():
                    host = val.strip()
                    break
        if port is None:
            for k in ("ssh_port", "port", "direct_port"):
                val = box.get(k)
                if val is not None and str(val).isdigit():
                    port = int(val)
                    break
        for k in ("ssh_username", "ssh_user", "username", "user"):
            val = box.get(k)
            if isinstance(val, str) and val.strip():
                user = val.strip()
                break

    if host is None or port is None:
        for box in containers:
            for k in ("ssh_command", "ssh_cmd", "command"):
                cmd_str = box.get(k)
                if not isinstance(cmd_str, str) or "@" not in cmd_str:
                    continue
                port_match = _SSH_CMD_PORT_RE.search(cmd_str)
                if port is None and port_match:
                    port = int(port_match.group("port"))
                for token in cmd_str.split():
                    if "@" in token:
                        m = _SSH_CMD_HOST_RE.fullmatch(token.strip())
                        if m:
                            if m.group("user"):
                                user = m.group("user")
                            host = host or m.group("host")
                        break

    return host, port, user


def wait_for_ssh_ready(
    client: GPUniqClient,
    task_id: int,
    ssh_key_path: Path,
    *,
    timeout_sec: float = 420.0,
    poll_interval_sec: float = 5.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[str, int, str]:
    """Poll instance status until SSH host/port are assigned and reachable."""
    deadline = time.monotonic() + timeout_sec
    last_host: str | None = None
    last_port: int | None = None
    last_user = "root"

    while True:
        details = client.get_instance(task_id)
        host, port, user = extract_ssh_connection(details)
        if host and port:
            last_host, last_port, last_user = host, port, user
            probe = subprocess.run(
                [*_ssh_base_args(ssh_key_path, port), f"{user}@{host}", "echo ready"],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0 and "ready" in probe.stdout:
                logger.info("SSH ready on {}@{}:{} (task_id={})", user, host, port, task_id)
                return host, port, user

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"SSH for task_id={task_id} not ready after {timeout_sec}s "
                f"(last={last_user}@{last_host}:{last_port})"
            )
        sleep_fn(poll_interval_sec)


def run_ssh(
    host: str,
    port: int,
    user: str,
    key_path: Path,
    command: str,
    *,
    stream: bool = True,
    timeout: float | None = None,
    idle_timeout: float | None = None,
) -> tuple[int, str]:
    """Execute a shell command over SSH with optional wall-clock and idle watchdogs."""
    cmd = [*_ssh_base_args(key_path, port), f"{user}@{host}", command]
    if not stream:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    proc_p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc_p.stdout is not None
    fd = proc_p.stdout.fileno()
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    chunks: list[str] = []
    start = time.monotonic()
    last_activity = start

    try:
        while True:
            now = time.monotonic()
            remaining: list[float] = []
            if timeout is not None:
                remaining.append(max(0.0, timeout - (now - start)))
            if idle_timeout is not None:
                remaining.append(max(0.0, idle_timeout - (now - last_activity)))
            wait_sec = min(1.0, *remaining) if remaining else 1.0

            events = sel.select(timeout=wait_sec)
            now = time.monotonic()
            if events:
                data = os.read(fd, 4096)
                if not data:
                    break
                last_activity = now
                text = data.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                chunks.append(text)
                continue

            if timeout is not None and (now - start) > timeout:
                proc_p.kill()
                proc_p.wait()
                raise TimeoutError(f"SSH command exceeded wall timeout ({timeout}s)")
            if idle_timeout is not None and (now - last_activity) > idle_timeout:
                proc_p.kill()
                proc_p.wait()
                raise TimeoutError(
                    f"SSH command produced no output for {idle_timeout}s"
                )
            if proc_p.poll() is not None:
                # Drain any buffered bytes after child exits.
                while True:
                    tail = os.read(fd, 4096)
                    if not tail:
                        break
                    text = tail.decode("utf-8", errors="replace")
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    chunks.append(text)
                break
    finally:
        sel.close()
        proc_p.stdout.close()

    wait_left = None if timeout is None else max(1.0, timeout - (time.monotonic() - start))
    rc = proc_p.wait(timeout=wait_left)
    return rc, "".join(chunks)


def rsync_to_remote(
    host: str,
    port: int,
    user: str,
    key_path: Path,
    local_path: Path,
    remote_path: str,
) -> None:
    """Upload a file or directory to remote GPU over rsync+ssh."""
    ssh_flag = (
        f"ssh -i {key_path} -p {port} -o BatchMode=yes "
        "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        "-o ServerAliveInterval=30 -o ServerAliveCountMax=6"
    )
    src = f"{local_path}/" if local_path.is_dir() else str(local_path)
    cmd = ["rsync", "-az", "--progress", "-e", ssh_flag, src, f"{user}@{host}:{remote_path}"]
    subprocess.run(cmd, check=True)


def rsync_from_remote(
    host: str,
    port: int,
    user: str,
    key_path: Path,
    remote_path: str,
    local_path: Path,
) -> None:
    """Download a file or directory from remote GPU over rsync+ssh."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    ssh_flag = (
        f"ssh -i {key_path} -p {port} -o BatchMode=yes "
        "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        "-o ServerAliveInterval=30 -o ServerAliveCountMax=6"
    )
    cmd = [
        "rsync",
        "-az",
        "--progress",
        "-e",
        ssh_flag,
        f"{user}@{host}:{remote_path}",
        str(local_path),
    ]
    subprocess.run(cmd, check=True)


def save_active_gpu(state: GPUInstanceState, path: Path = DEFAULT_GPU_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_active_gpu(path: Path = DEFAULT_GPU_STATE_PATH) -> GPUInstanceState | None:
    if not path.exists():
        return None
    return GPUInstanceState.model_validate_json(path.read_text(encoding="utf-8"))


def clear_active_gpu(path: Path = DEFAULT_GPU_STATE_PATH) -> None:
    if path.exists():
        path.unlink()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()

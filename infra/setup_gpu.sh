#!/usr/bin/env bash
set -euo pipefail

echo "=== Host GPU / RAM / Disk diagnostics ==="
nvidia-smi
free -h
df -h /workspace / || df -h /

BASE_DIR="/workspace"
mkdir -p "${BASE_DIR}/models/face" "${BASE_DIR}/models/realesrgan" "${BASE_DIR}/jobs"

AVAIL_GB=$(df -BG --output=avail "${BASE_DIR}" | tail -n 1 | tr -dc '0-9')
echo "Available space in ${BASE_DIR}: ${AVAIL_GB} GB"
if [ "${AVAIL_GB:-0}" -lt 200 ]; then
  echo "ERROR: Need at least 200 GB free in ${BASE_DIR}, found ${AVAIL_GB} GB" >&2
  exit 1
fi

echo "=== Installing base system tools ==="
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq ffmpeg git rsync curl
fi

echo "=== Setting up Python 3.12 venv at ${BASE_DIR}/venv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -x "${BASE_DIR}/venv/bin/python" ]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "${BASE_DIR}/venv"
  else
    uv venv --python 3.12 "${BASE_DIR}/venv"
  fi
fi

VENV_PY="${BASE_DIR}/venv/bin/python"
export VIRTUAL_ENV="${BASE_DIR}/venv"
export PATH="${BASE_DIR}/venv/bin:${HOME}/.local/bin:${PATH}"

echo "=== Installing validated Viggle-Remake / MiniMax-H3 + FP8 + Face Restoration deps ==="
uv pip install --python "${VENV_PY}" --upgrade pip
uv pip install --python "${VENV_PY}" \
  torch==2.9.1 torchvision==0.24.1 torchao==0.15.0 \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python "${VENV_PY}" \
  "git+https://github.com/huggingface/diffusers@d6726f3" \
  transformers==5.14.1 \
  huggingface_hub==1.27.0 \
  hf_xet \
  safetensors==0.8.0 \
  peft==0.18.0 \
  accelerate==1.12.0 \
  tokenizers==0.22.1 \
  av==16.1.0 \
  numpy==2.3.5 \
  Pillow==12.2.0 \
  spandrel \
  spandrel-extra-arches \
  opencv-python-headless

echo "=== Fast pre-download check: CUDA, diffusers, torchao==0.15.0, spandrel, YuNet ==="
"${VENV_PY}" - <<'PY'
import cv2
import spandrel
import torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel
from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

assert torch.cuda.is_available(), "CUDA is not available"
props = torch.cuda.get_device_properties(0)
print(
    f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB | "
    f"sm={props.major}.{props.minor}"
)
print(f"cuDNN SDPA enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")
assert hasattr(cv2, "FaceDetectorYN"), "OpenCV FaceDetectorYN missing"

ok_quant: list[str] = []
x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
for name, cfg in [("fp8", Float8WeightOnlyConfig()), ("int8", Int8WeightOnlyConfig())]:
    try:
        m = nn.Linear(128, 128).to("cuda", torch.bfloat16)
        quantize_(m, cfg)
        y = m(x)
        torch.cuda.synchronize()
        assert y.shape == (4, 128)
        ok_quant.append(name)
    except Exception as exc:
        print(f"Probe {name} unavailable: {exc}")

assert ok_quant, "Neither Float8WeightOnlyConfig nor Int8WeightOnlyConfig works on CUDA"
print(f"Verified {MiniMaxH3Transformer3DModel.__name__}, {spandrel.__name__}, torchao CUDA kernels={ok_quant}")
PY

HF_BIN="${BASE_DIR}/venv/bin/hf"
export HF_XET_HIGH_PERFORMANCE=1

hf_download_resilient() {
  local pass1_sec="$1"
  local pass2_sec="$2"
  shift 2
  if timeout --kill-after=15 "${pass1_sec}" "${HF_BIN}" download "$@"; then
    return 0
  fi
  echo "=== HF Xet pass did not finish in ${pass1_sec}s; resuming via plain HTTP (HF_HUB_DISABLE_XET=1) ==="
  HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
    timeout --kill-after=15 "${pass2_sec}" "${HF_BIN}" download "$@" --max-workers 6
}

echo "=== Downloading MiniMaxAI/MiniMax-H3 (~144 GB required weights) ==="
hf_download_resilient 300 600 MiniMaxAI/MiniMax-H3 \
  --local-dir "${BASE_DIR}/models/MiniMax-H3" \
  --include "modular_model_index.json" \
  --include "transformer/*" \
  --include "text_encoder/*" \
  --include "tokenizer/*" \
  --include "processor/*" \
  --include "vae/*" \
  --include "audio_vae/*" \
  --include "scheduler/*" \
  --include "audio_scheduler/*"

echo "=== Downloading Viggle/Viggle-Remake LoRAs and sample.py ==="
hf_download_resilient 120 180 Viggle/Viggle-Remake \
  --local-dir "${BASE_DIR}/models/Viggle-Remake" \
  --include "lora/*" \
  --include "inference/*" \
  --include "requirements.txt"

echo "=== Downloading commercial-safe upscale & face restoration weights ==="
if [ ! -f "${BASE_DIR}/models/face/GFPGANv1.4.pth" ]; then
  curl --connect-timeout 20 --max-time 120 -fsSL -L -o "${BASE_DIR}/models/face/GFPGANv1.4.pth" \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
fi
if [ ! -f "${BASE_DIR}/models/face/face_detection_yunet_2023mar.onnx" ]; then
  curl --connect-timeout 20 --max-time 120 -fsSL -L -o "${BASE_DIR}/models/face/face_detection_yunet_2023mar.onnx" \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
fi
if [ ! -f "${BASE_DIR}/models/realesrgan/realesr-general-x4v3.pth" ]; then
  curl --connect-timeout 20 --max-time 120 -fsSL -L -o "${BASE_DIR}/models/realesrgan/realesr-general-x4v3.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
fi

echo "=== Verifying CUDA, cuDNN SDPA, diffusers, torchao, spandrel, YuNet, and weights ==="
"${VENV_PY}" - <<'PY'
from pathlib import Path
import cv2
import spandrel
import torch
from diffusers import MiniMaxH3Transformer3DModel
from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

assert torch.cuda.is_available(), "CUDA is not available"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB | sm={props.major}.{props.minor}")
print(f"cuDNN SDPA enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")
print(
    f"Verified libs: {MiniMaxH3Transformer3DModel.__name__}, {spandrel.__name__}, "
    f"torchao ({Float8WeightOnlyConfig.__name__}, {Int8WeightOnlyConfig.__name__}, {quantize_.__name__})"
)
assert hasattr(cv2, "FaceDetectorYN"), "OpenCV FaceDetectorYN missing"

base = Path("/workspace/models/MiniMax-H3")
lora = Path("/workspace/models/Viggle-Remake")
for req in [
    base / "modular_model_index.json",
    lora / "inference/sample.py",
    lora / "lora/swap/pytorch_lora_weights.safetensors",
    lora / "lora/edit/pytorch_lora_weights.safetensors",
    Path("/workspace/models/face/GFPGANv1.4.pth"),
    Path("/workspace/models/face/face_detection_yunet_2023mar.onnx"),
    Path("/workspace/models/realesrgan/realesr-general-x4v3.pth"),
]:
    assert req.exists(), f"Missing required path: {req}"
    print(f"Verified: {req}")
PY

echo "=== GPU setup complete ==="

#!/usr/bin/env bash
set -euo pipefail

echo "=== Host GPU / RAM / Disk diagnostics ==="
nvidia-smi
free -h
df -h /workspace / || df -h /

BASE_DIR="/workspace"
mkdir -p "${BASE_DIR}/models" "${BASE_DIR}/jobs"

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

echo "=== Installing validated Viggle-Remake / MiniMax-H3 dependencies ==="
uv pip install --python "${VENV_PY}" --upgrade pip
uv pip install --python "${VENV_PY}" \
  torch==2.9.1 torchvision==0.24.1 \
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
  Pillow==12.2.0

HF_BIN="${BASE_DIR}/venv/bin/hf"
export HF_XET_HIGH_PERFORMANCE=1

echo "=== Downloading MiniMaxAI/MiniMax-H3 (~144 GB required weights) ==="
"${HF_BIN}" download MiniMaxAI/MiniMax-H3 \
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
"${HF_BIN}" download Viggle/Viggle-Remake \
  --local-dir "${BASE_DIR}/models/Viggle-Remake" \
  --include "lora/*" \
  --include "inference/*" \
  --include "requirements.txt"

echo "=== Verifying CUDA, cuDNN SDPA, diffusers, and model files ==="
"${VENV_PY}" - <<'PY'
from pathlib import Path
import torch
from diffusers import MiniMaxH3Transformer3DModel

assert torch.cuda.is_available(), "CUDA is not available"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB")
print(f"cuDNN SDPA enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")
print(f"MiniMaxH3Transformer3DModel imported: {MiniMaxH3Transformer3DModel.__name__}")

base = Path("/workspace/models/MiniMax-H3")
lora = Path("/workspace/models/Viggle-Remake")
for req in [
    base / "modular_model_index.json",
    lora / "inference/sample.py",
    lora / "lora/swap/pytorch_lora_weights.safetensors",
    lora / "lora/edit/pytorch_lora_weights.safetensors",
]:
    assert req.exists(), f"Missing required path: {req}"
    print(f"Verified: {req}")
PY

echo "=== GPU setup complete ==="

#!/usr/bin/env bash
set -euo pipefail

echo "=== Host GPU / RAM / Disk diagnostics ==="
nvidia-smi
free -h
df -h /workspace / || df -h /

BASE_DIR="/workspace"
mkdir -p "${BASE_DIR}/models/realesrgan" "${BASE_DIR}/models/SEEDVR2" "${BASE_DIR}/in" "${BASE_DIR}/out"

echo "=== Ensuring system tools (ffmpeg, git, curl, rsync) ==="
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq ffmpeg git curl rsync
fi

echo "=== Locating working CUDA PyTorch interpreter ==="
PY_BIN=""
for candidate in \
  /venv/main/bin/python \
  /opt/conda/bin/python \
  "$(command -v python3 || true)" \
  "$(command -v python || true)"
do
  [ -n "${candidate}" ] && [ -x "${candidate}" ] || continue
  if "${candidate}" -c "import torch; assert torch.cuda.is_available(), 'no cuda'" >/dev/null 2>&1; then
    PY_BIN="${candidate}"
    break
  fi
done

if [ -z "${PY_BIN}" ]; then
  echo "ERROR: Could not find a Python interpreter with CUDA-enabled PyTorch" >&2
  exit 1
fi

echo "Using CUDA Python: ${PY_BIN}"
printf '%s\n' "${PY_BIN}" > "${BASE_DIR}/upscaler_python.txt"

echo "=== Cloning ComfyUI-SeedVR2_VideoUpscaler (standalone CLI) ==="
if [ ! -d "${BASE_DIR}/SeedVR2/.git" ]; then
  git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "${BASE_DIR}/SeedVR2"
else
  git -C "${BASE_DIR}/SeedVR2" pull --ff-only || true
fi

echo "=== Installing non-torch dependencies (preserving preinstalled CUDA torch) ==="
"${PY_BIN}" -m pip install -q --upgrade-strategy only-if-needed \
  spandrel \
  safetensors \
  einops \
  "omegaconf>=2.3.0" \
  "diffusers>=0.33.1" \
  "peft>=0.17.0" \
  "rotary_embedding_torch>=0.5.3" \
  opencv-python-headless \
  gguf \
  matplotlib \
  hf_xet \
  tqdm \
  psutil

echo "=== Downloading RealESRGAN_x2plus.pth ==="
REALESRGAN_PATH="${BASE_DIR}/models/realesrgan/RealESRGAN_x2plus.pth"
if [ ! -f "${REALESRGAN_PATH}" ]; then
  curl -fsSL -L -o "${REALESRGAN_PATH}" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
fi

echo "=== Verifying CUDA, spandrel, diffusers, and SeedVR2 CLI ==="
"${PY_BIN}" - <<'PY'
from pathlib import Path
import torch
import spandrel
import diffusers

assert torch.cuda.is_available(), "CUDA lost after pip install!"
props = torch.cuda.get_device_properties(0)
print(
    f"Verified GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GiB | "
    f"torch={torch.__version__} | diffusers={diffusers.__version__} | spandrel={spandrel.__name__}"
)
assert Path("/workspace/models/realesrgan/RealESRGAN_x2plus.pth").stat().st_size > 1_000_000
assert Path("/workspace/SeedVR2/inference_cli.py").exists()
PY

echo "=== Upscaler GPU setup complete ==="

#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${LOKI_BASE_DIR:-/workspace}"
VENV_DIR="${VIRTUAL_ENV:-${BASE_DIR}/venv}"
HF_BIN="${VENV_DIR}/bin/hf"

mkdir -p "${BASE_DIR}/models/face" "${BASE_DIR}/models/realesrgan" "${BASE_DIR}/jobs"

# Link pre-baked auxiliary weights if present in /opt/loki-models, otherwise download.
if [ -d /opt/loki-models ]; then
  for sub in face realesrgan; do
    if [ -d "/opt/loki-models/${sub}" ]; then
      for f in "/opt/loki-models/${sub}/"*; do
        [ -e "${f}" ] || continue
        target="${BASE_DIR}/models/${sub}/$(basename "${f}")"
        [ -e "${target}" ] || ln -s "${f}" "${target}"
      done
    fi
  done
fi

download_if_missing() {
  local dest="$1"
  local url="$2"
  if [ ! -f "${dest}" ]; then
    curl --connect-timeout 20 --max-time 180 -fsSL -L -o "${dest}" "${url}"
  fi
}

download_if_missing "${BASE_DIR}/models/face/GFPGANv1.4.pth" \
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
download_if_missing "${BASE_DIR}/models/face/RestoreFormer++.ckpt" \
  "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt"
download_if_missing "${BASE_DIR}/models/face/face_detection_yunet_2023mar.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
download_if_missing "${BASE_DIR}/models/face/selfie_multiclass_256x256.tflite" \
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
download_if_missing "${BASE_DIR}/models/realesrgan/realesr-general-x4v3.pth" \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"

export HF_XET_HIGH_PERFORMANCE=1

hf_download_resilient() {
  local pass1_sec="$1"
  local pass2_sec="$2"
  local pass3_sec="$3"
  shift 3
  if timeout --kill-after=15 "${pass1_sec}" "${HF_BIN}" download "$@"; then
    return 0
  fi
  echo "=== HF Xet pass did not finish in ${pass1_sec}s; resuming via Rust hf_transfer ==="
  if HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
      timeout --kill-after=15 "${pass2_sec}" "${HF_BIN}" download "$@"; then
    return 0
  fi
  echo "=== HF hf_transfer did not finish in ${pass2_sec}s; resuming via multi-worker HTTP ==="
  HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60 \
    timeout --kill-after=15 "${pass3_sec}" "${HF_BIN}" download "$@" --max-workers 6
}

echo "=== Ensuring MiniMaxAI/MiniMax-H3 (~144 GB required weights) ==="
hf_download_resilient 300 300 600 MiniMaxAI/MiniMax-H3 \
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

echo "=== Ensuring Viggle/Viggle-Remake LoRAs and sample.py ==="
hf_download_resilient 120 120 180 Viggle/Viggle-Remake \
  --local-dir "${BASE_DIR}/models/Viggle-Remake" \
  --include "lora/*" \
  --include "inference/*" \
  --include "requirements.txt"

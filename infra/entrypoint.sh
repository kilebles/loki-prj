#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/models/face /workspace/models/realesrgan /workspace/jobs /root/.ssh
chmod 700 /root/.ssh

if [ -d /opt/loki-venv ] && [ ! -e /workspace/venv ]; then
  ln -s /opt/loki-venv /workspace/venv
fi

if [ -d /opt/loki-models ]; then
  for sub in face realesrgan; do
    if [ -d "/opt/loki-models/${sub}" ]; then
      for f in "/opt/loki-models/${sub}/"*; do
        [ -e "${f}" ] || continue
        target="/workspace/models/${sub}/$(basename "${f}")"
        if [ ! -e "${target}" ]; then
          ln -s "${f}" "${target}"
        fi
      done
    fi
  done
fi

PUB_KEY="${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-}}"
if [ -n "${PUB_KEY}" ]; then
  if ! grep -qxF "${PUB_KEY}" /root/.ssh/authorized_keys 2>/dev/null; then
    printf '%s\n' "${PUB_KEY}" >> /root/.ssh/authorized_keys
  fi
  chmod 600 /root/.ssh/authorized_keys
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
  ssh-keygen -A
fi

# Production resident worker mode: bootstrap base models if absent, keep sshd optional in background,
# and exec the resident GPU HTTP worker so models stay warm in VRAM across requests.
if [ "${LOKI_WORKER_MODE:-0}" = "1" ]; then
  if [ -n "${PUB_KEY}" ]; then
    /usr/sbin/sshd -e &
  fi
  if [ ! -f "${LOKI_MODELS_DIR:-/workspace/models}/MiniMax-H3/modular_model_index.json" ]; then
    /opt/loki-app/bootstrap_models.sh
  fi
  exec /opt/loki-venv/bin/python /opt/loki-app/gpu_worker.py \
    --host "${LOKI_WORKER_HOST:-0.0.0.0}" \
    --port "${LOKI_WORKER_PORT:-8000}" \
    --preload-mode "${LOKI_PRELOAD_MODE:-swap}" \
    --quant "${LOKI_QUANT:-fp8}"
fi

exec /usr/sbin/sshd -D -e

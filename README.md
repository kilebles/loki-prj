# LOKI Remake

Video-to-video remake pipeline (`swap` / `edit`) built around **MiniMax-H3 / Viggle-Remake** and **GPUniq** cloud GPUs + media/VLM APIs.

## Architecture Overview

1. **Video Normalization (`loki_remake.video.normalize`)**:
   - Probes input video (`ffprobe`), normalizes to 24 fps CFR, enforces 9:16 / aspect-clamped multiple-of-32 dimensions (`480x832` or `736x1280`), snaps frame count to the MiniMax-H3 `17n + 5` grid (`90..345` frames, up to 14.4 s), and preserves clean audio (`audio.m4a`).
2. **Shot & Keyframe Selection (`loki_remake.video.shots`)**:
   - Detects hard cuts, selects sharpest non-transition frames per shot (plus split keyframes on long continuous takes), and renders an annotated contact sheet for prompt generation.
3. **Keyframe Painting & Prompting (`loki_remake.gpuniq`)**:
   - Repaints keyframe(s) via GPUniq Image Jobs (`nano-banana-2` with `gpt-image-2` fallback) aligned to the target canvas.
   - Generates and strictly validates the six-section MiniMax-H3 prompt via GPUniq Chat Completions (`gemini-3-flash`), with automatic repair on validation failures.
4. **GPU Render & Production Orchestration** *(next stages)*:
   - Reference `bf16` execution on `H200` / `RTX PRO 6000` (`Q8_0` ComfyUI worker), `480p -> 720p` upscaling comparison, HTTP API + Redis queue + GPUniq Burst autoscaler.

## Setup

Requires Python >= 3.11, `uv`, and `ffmpeg` / `ffprobe` on `PATH`.

```bash
uv sync
cp .env.example .env
# Edit .env and set GPUNIQ_API_KEY
```

## CLI Usage

```bash
# Check GPUniq workspace balance
uv run loki-remake balance

# List available 80GB+ cloud GPUs on GPUniq
uv run loki-remake gpus --min-vram 80

# Prepare a video remake job (normalize + shots + paint keyframes + 6-section prompt)
uv run loki-remake prepare \
  --video samples/source.mp4 \
  --mode edit \
  --instruction "Change the grey hoodie into a bright red leather jacket" \
  --tier 480 \
  --max-refs 1 \
  --out work/demo

# Validate an existing prompt file
uv run loki-remake validate-prompt \
  --prompt work/demo/prompt.txt \
  --frames 226 \
  --pics 1 \
  --shots 1 \
  --mode edit
```

## Tests & Linting

```bash
uv run ruff check src tests
uv run pytest -v
```

#!/usr/bin/env python3
"""Resident GPU worker daemon for MiniMax-H3 + Viggle-Remake + Face Enhancement.

Keeps the quantized MiniMax-H3 + fused Viggle-Remake LoRA pipeline resident in VRAM
(~41 GiB in FP8), Qwen3-VL text encoder resident in CPU RAM (~63 GiB, swapped to CUDA
for ~2.5s per prompt on 140-180 GB GPUs or with automatic transformer offload on 96 GB GPUs),
and face restoration models (GFPGAN v1.4, RestoreFormer++, YuNet, MediaPipe) warm in memory.

Eliminates the ~315s cold-load + FP8 quantization tax on every job, reducing inter-job
startup overhead to ~3-4s for both iterative testing and production deployments.

Zero extra Python dependencies beyond the base worker venv (uses stdlib ThreadingHTTPServer).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Ensure sibling infra scripts (sample_fp8, face_enhance, restoreformer_arch) are importable.
_INFRA_DIR = str(Path(__file__).resolve().parent)
if _INFRA_DIR not in sys.path:
    sys.path.insert(0, _INFRA_DIR)

SIGMAS: dict[str, list[float]] = {
    "swap": [1.0, 0.985, 0.96, 0.90, 0.78, 0.56, 0.0],
    "edit": [1.0, 0.98, 0.94, 0.86, 0.72, 0.50, 0.0],
}
LORA_FILES: dict[str, str] = {
    "swap": "character_swap/lora.safetensors",
    "edit": "video_edit/lora.safetensors",
}


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


def _resolve_model_file(filename: str, subdir: str = "face") -> Path:
    """Resolve auxiliary model path checking /workspace/models first, then /opt/loki-models."""
    models_root = _env_path("LOKI_MODELS_DIR", "/workspace/models")
    baked_root = _env_path("LOKI_BAKED_MODELS_DIR", "/opt/loki-models")
    primary = models_root / subdir / filename
    if primary.exists():
        return primary
    fallback = baked_root / subdir / filename
    if fallback.exists():
        return fallback
    return primary


class ResidentRemakeEngine:
    """Thread-safe resident GPU inference engine for MiniMax-H3 + face enhancement."""

    def __init__(
        self,
        *,
        base_dir: Path,
        lora_repo: Path,
        quant: str = "fp8",
        attn_backend: str = "_native_cudnn",
        preload_mode: str | None = "swap",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.lora_repo = Path(lora_repo)
        self.quant = quant
        self.attn_backend = attn_backend

        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._status = "initializing"
        self._last_error: str | None = None
        self._jobs_completed = 0

        # Resident MiniMax-H3 + Qwen3-VL state
        self._loaded_mode: str | None = None
        self._pipe: Any = None
        self._orig_vid_set_timesteps: Any = None
        self._orig_aud_set_timesteps: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._text_encoder: Any = None
        self._hooks_installed = False
        self._job_ctx: dict[str, Any] = {
            "prompt_embeds": None,
            "text_token_tags": None,
            "kf_images": [],
            "kf_anchors": [],
            "nkf": 0,
            "keyframe_until": 0.4,
        }

        # Lazily cached face-enhancement models (~1 GiB total when warm)
        self._face_models: dict[str, Any] = {}
        self._esr_model: Any = None
        self._seg_interp: Any = None
        self._detector: Any = None

        if preload_mode is not None:
            self.warmup(preload_mode)
        else:
            self._status = "ready"

    def health(self) -> dict[str, Any]:
        gpu_name = "unknown"
        vram_alloc = 0.0
        vram_reserved = 0.0
        vram_total = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_alloc = round(torch.cuda.memory_allocated() / 2**30, 2)
                vram_reserved = round(torch.cuda.memory_reserved() / 2**30, 2)
                vram_total = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        except Exception:
            pass

        return {
            "status": self._status,
            "loaded_mode": self._loaded_mode,
            "quant": self.quant,
            "attn_backend": self.attn_backend,
            "gpu_name": gpu_name,
            "vram_alloc_gib": vram_alloc,
            "vram_reserved_gib": vram_reserved,
            "vram_total_gib": vram_total,
            "text_encoder_resident_ram": self._text_encoder is not None,
            "cached_enhancers": sorted(self._face_models.keys()),
            "cached_esrgan": self._esr_model is not None,
            "cached_mediapipe": self._seg_interp is not None,
            "jobs_completed": self._jobs_completed,
            "uptime_sec": round(time.monotonic() - self._started_at, 1),
            "last_error": self._last_error,
        }

    def _ensure_text_encoder_in_ram(self) -> None:
        """Load Qwen3-VL text encoder once into CPU RAM (reused across all jobs)."""
        if self._text_encoder is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = time.perf_counter()
        print("[worker] loading Qwen3-VL text encoder into CPU RAM...", flush=True)
        self._processor = AutoProcessor.from_pretrained(self.base_dir / "processor")
        self._tokenizer = self._processor.tokenizer
        self._text_encoder = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                self.base_dir / "text_encoder",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to("cpu")
        )
        print(
            f"[worker] Qwen3-VL resident in CPU RAM ({time.perf_counter() - t0:.1f}s)", flush=True
        )

    def _install_minimax_hooks_once(self) -> None:
        """Install MiniMax-H3 hooks once, reading per-job mutable state from _job_ctx."""
        if self._hooks_installed:
            return
        import torch
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3Ref2VAPrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3.denoise import (
            MiniMaxH3LoopSchedulerStep,
            MiniMaxH3Ref2VALoopDenoiser,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3Ref2VAReferenceEncoderStep,
            MiniMaxH3Ref2VATextEncoderStep,
        )

        ctx = self._job_ctx

        # 1. Precomputed text embeddings injection
        def _text_step_call(_self: Any, _components: Any, state: Any) -> tuple[Any, Any]:
            state.set("prompt_embeds", ctx["prompt_embeds"])
            state.set("text_token_tags", ctx["text_token_tags"])
            return _components, state

        MiniMaxH3Ref2VATextEncoderStep.__call__ = _text_step_call

        # 2. Row timesteps override (pass 1.0 at index 6)
        orig_row = MiniMaxH3SetTimestepsStep.build_row_timesteps

        def _row_ts(self_step: Any, *args: Any, **kwargs: Any) -> Any:
            a = list(args)
            a[6] = 1.0
            return orig_row(self_step, *a, **kwargs)

        MiniMaxH3SetTimestepsStep.build_row_timesteps = _row_ts

        # 3. Video-only scheduler step
        def _video_only_step(
            _self: Any, components: Any, block_state: Any, i: int, t: Any
        ) -> tuple[Any, Any]:
            if components.scheduler.step_index is None:
                components.scheduler._init_step_index(t)
            step = components.scheduler.step_index
            dt = components.scheduler.sigmas[step + 1] - components.scheduler.sigmas[step]
            block_state.latents = (
                block_state.latents.to(torch.float32)
                + dt * block_state.noise_pred.to(torch.float32)
            ).to(block_state.latents.dtype)
            components.scheduler._step_index += 1
            return components, block_state

        MiniMaxH3LoopSchedulerStep.__call__ = _video_only_step

        # 4. Reference keyframe encoder hook
        orig_ref_enc = MiniMaxH3Ref2VAReferenceEncoderStep.__call__

        def _ref_enc(self_step: Any, components: Any, state: Any) -> tuple[Any, Any]:
            components, state = orig_ref_enc(self_step, components, state)
            kf_images = ctx["kf_images"]
            if kf_images:
                ref_latents = list(state.get("ref_image_latents") or [])
                h, w = state.get("height"), state.get("width")
                vae_dev = next(components.vae.parameters()).device
                vae_dt = next(components.vae.parameters()).dtype
                for img in kf_images:
                    t = (
                        components.video_processor.preprocess(img, height=h, width=w)
                        .unsqueeze(2)
                        .to(vae_dev, vae_dt)
                    )
                    ref_latents.append(self_step.encode_video_latent(components, t))
                state.set("ref_image_latents", ref_latents)
            return components, state

        MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = _ref_enc

        # 5. Layout builder with per-job keyframe anchors
        orig_layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence

        def _layout_with_keyframes(
            self_step: Any,
            *,
            latents: Any,
            ref_image_latents: Any,
            ref_video_latents: Any,
            audio_latents: Any,
            ref_audio_latents: Any,
        ) -> Any:
            anchors = ctx["kf_anchors"]
            if not anchors or not ref_image_latents:
                return orig_layout(
                    self_step,
                    latents=latents,
                    ref_image_latents=ref_image_latents,
                    ref_video_latents=ref_video_latents,
                    audio_latents=audio_latents,
                    ref_audio_latents=ref_audio_latents,
                )
            n = len(anchors)
            orig_refs, kf_refs = ref_image_latents[:-n], ref_image_latents[-n:]
            packed, spans, ref_shapes = orig_layout(
                self_step,
                latents=latents,
                ref_image_latents=orig_refs,
                ref_video_latents=ref_video_latents,
                audio_latents=audio_latents,
                ref_audio_latents=ref_audio_latents,
            )
            kf_chunks = [self_step.pack_latent_video(r) for r in kf_refs]
            packed = torch.cat([packed, *kf_chunks], dim=1)
            spans = dict(spans)
            spans["keyframes"] = list(zip([c.shape[1] for c in kf_chunks], anchors, strict=True))
            return packed, spans, ref_shapes

        MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = _layout_with_keyframes

        # 6. Loop denoiser with keyframe_until gating
        orig_denoise = MiniMaxH3Ref2VALoopDenoiser.__call__

        def _denoise_with_keyframes(
            self_step: Any, components: Any, block_state: Any, i: int, t: Any
        ) -> tuple[Any, Any]:
            nkf = ctx["nkf"]
            if not nkf:
                return orig_denoise(self_step, components, block_state, i, t)
            kfs = block_state.token_spans.get("keyframes") or []
            kf_len = sum(length for length, _ in kfs)
            if not kf_len:
                return orig_denoise(self_step, components, block_state, i, t)
            sigma = float(components.scheduler.sigmas[i])
            if sigma <= ctx["keyframe_until"]:
                saved_seq = block_state.packed_sequence
                block_state.packed_sequence = saved_seq[:, :-kf_len]
                saved_kfs = block_state.token_spans.pop("keyframes")
                try:
                    return orig_denoise(self_step, components, block_state, i, t)
                finally:
                    block_state.packed_sequence = saved_seq
                    block_state.token_spans["keyframes"] = saved_kfs
            bsz = block_state.latents.shape[0]
            dev = block_state.latents.device
            h_lat, w_lat = block_state.latents.shape[-2], block_state.latents.shape[-1]
            kf_t = [
                torch.full((bsz, length), float(anchor), device=dev, dtype=torch.float32)
                for length, anchor in kfs
            ]
            kf_shapes = [(1, h_lat, w_lat)] * len(kfs)
            saved_ts = block_state.row_timesteps
            saved_rs = block_state.ref_image_latent_shapes
            block_state.row_timesteps = torch.cat([saved_ts, *kf_t], dim=1)
            block_state.ref_image_latent_shapes = [*saved_rs, *kf_shapes]
            try:
                return orig_denoise(self_step, components, block_state, i, t)
            finally:
                block_state.row_timesteps = saved_ts
                block_state.ref_image_latent_shapes = saved_rs

        MiniMaxH3Ref2VALoopDenoiser.__call__ = _denoise_with_keyframes
        self._hooks_installed = True
        print("[worker] installed MiniMax-H3 reusable hooks once", flush=True)

    def _load_pipeline_for_mode(self, mode: str) -> None:
        """Load MiniMax-H3 + fuse Viggle-Remake LoRA + quantize once onto CUDA."""
        if self._pipe is not None and self._loaded_mode == mode:
            return
        if mode not in LORA_FILES:
            raise ValueError(f"Unsupported mode: {mode!r} (expected one of {sorted(LORA_FILES)})")

        import torch
        from diffusers import ModularPipeline
        from peft import LoraConfig
        from safetensors.torch import load_file
        from sample_fp8 import quantize_transformer

        self._status = "loading"
        if self._pipe is not None:
            print(f"[worker] switching LoRA mode {self._loaded_mode} -> {mode}", flush=True)
            del self._pipe
            self._pipe = None
            self._loaded_mode = None
            gc.collect()
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        self._ensure_text_encoder_in_ram()
        self._install_minimax_hooks_once()

        print(
            f"[worker] loading MiniMax-H3 pipeline (mode={mode}, quant={self.quant})...", flush=True
        )
        pipe = ModularPipeline.from_pretrained(str(self.base_dir), trust_remote_code=True)
        pipe.load_components(
            names=[
                "transformer_ref",
                "vae",
                "audio_vae",
                "scheduler",
                "audio_scheduler",
                "video_processor",
            ],
            torch_dtype=torch.bfloat16,
            device_map=None,
        )

        lora_path = self.lora_repo / LORA_FILES[mode]
        raw = load_file(str(lora_path))
        rank = next(v.shape[0] for k, v in raw.items() if k.endswith(".lora_A.weight"))
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
            init_lora_weights=False,
        )
        pipe.transformer_ref.add_adapter(lora_cfg, adapter_name="default")
        state = {
            k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            ): v
            for k, v in raw.items()
        }
        missing, unexpected = pipe.transformer_ref.load_state_dict(state, strict=False)
        assert not unexpected and not [m for m in missing if "lora_" in m], (
            f"LoRA mismatch: {missing[:3]} {unexpected[:3]}"
        )
        pipe.transformer_ref.fuse_lora()
        pipe.transformer_ref.delete_adapters("default")
        del raw, state
        gc.collect()

        pipe.transformer_ref.set_attention_backend(self.attn_backend)
        quantize_transformer(pipe.transformer_ref, self.quant)

        pipe.to("cuda")
        torch.cuda.empty_cache()

        self._orig_vid_set_timesteps = pipe.scheduler.set_timesteps
        self._orig_aud_set_timesteps = pipe.audio_scheduler.set_timesteps
        self._pipe = pipe
        self._loaded_mode = mode
        print(
            f"[worker] pipeline resident on CUDA ({mode}, {self.quant}) in "
            f"{time.perf_counter() - t0:.1f}s | VRAM alloc="
            f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB",
            flush=True,
        )

    def warmup(self, mode: str = "swap") -> dict[str, Any]:
        with self._lock:
            try:
                self._load_pipeline_for_mode(mode)
                self._status = "ready"
                self._last_error = None
            except Exception as exc:
                self._status = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise
        return self.health()

    def _encode_prompt(self, prompt_text: str) -> tuple[Any, Any, float]:
        """Encode prompt using RAM-resident Qwen3-VL, swapping to CUDA briefly (~2.5s)."""
        import torch
        from sample_fp8 import encode_prompt

        t0 = time.perf_counter()
        self._ensure_text_encoder_in_ram()

        free_vram_gib = torch.cuda.mem_get_info()[0] / 2**30
        offloaded_transformer = False
        if free_vram_gib < 68.0 and self._pipe is not None:
            print(
                f"[worker] free VRAM {free_vram_gib:.1f} GiB < 68 GiB; "
                "temporarily offloading transformer to CPU for Qwen3-VL",
                flush=True,
            )
            self._pipe.transformer_ref.to("cpu")
            torch.cuda.empty_cache()
            offloaded_transformer = True

        try:
            self._text_encoder.to("cuda")
            with torch.inference_mode():
                prompt_embeds, text_token_tags = encode_prompt(
                    self._processor, self._tokenizer, self._text_encoder, prompt_text
                )
        finally:
            self._text_encoder.to("cpu")
            torch.cuda.empty_cache()
            if offloaded_transformer and self._pipe is not None:
                self._pipe.transformer_ref.to("cuda")
                torch.cuda.empty_cache()

        return prompt_embeds, text_token_tags, time.perf_counter() - t0

    def _get_enhancer_models(
        self,
        *,
        restorer: str,
        use_semantic_mask: bool,
        upscale_to: tuple[int, int] | None,
        gfpgan_weights: Path,
        restoreformer_weights: Path,
        yunet_model: Path,
        seg_model: Path | None,
        esrgan_weights: Path,
    ) -> tuple[Any, Any, Any, Any]:
        """Lazily load and cache face restoration + segmentation + detector models."""
        import cv2
        import spandrel
        import torch

        if restorer not in self._face_models:
            t0 = time.perf_counter()
            if restorer == "restoreformer":
                from restoreformer_arch import load_restoreformer_pp

                self._face_models[restorer] = load_restoreformer_pp(
                    restoreformer_weights, device="cuda"
                )
            else:
                gfp_desc = spandrel.ModelLoader().load_from_file(gfpgan_weights)
                self._face_models[restorer] = gfp_desc.model.eval().to(
                    device="cuda", dtype=torch.float32
                )
            print(
                f"[worker] cached face restorer '{restorer}' in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

        if use_semantic_mask and self._seg_interp is None and seg_model and seg_model.exists():
            from ai_edge_litert.interpreter import Interpreter

            interp = Interpreter(model_path=str(seg_model), num_threads=4)
            interp.allocate_tensors()
            self._seg_interp = interp
            print("[worker] cached MediaPipe selfie_multiclass segmenter (4 threads)", flush=True)

        if upscale_to is not None and self._esr_model is None:
            esr_desc = spandrel.ModelLoader().load_from_file(esrgan_weights)
            self._esr_model = esr_desc.model.eval().to(device="cuda", dtype=torch.float16)
            print("[worker] cached Real-ESRGAN upscaler", flush=True)

        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(yunet_model),
                config="",
                input_size=(640, 640),
                score_threshold=0.58,
                nms_threshold=0.30,
                top_k=5000,
            )
            print("[worker] cached YuNet face detector", flush=True)

        return (
            self._face_models[restorer],
            self._esr_model if upscale_to is not None else None,
            self._seg_interp if use_semantic_mask else None,
            self._detector,
        )

    def _run_enhance_locked(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Run face enhancement with warm cached models (caller must hold self._lock)."""
        from face_enhance import _parse_size, enhance_video

        input_path = Path(cfg["input"])
        audio_source = Path(cfg.get("audio_source") or input_path)
        out_path = Path(cfg["out"])
        restorer = str(cfg.get("restorer", "restoreformer"))
        if restorer not in ("gfpgan", "restoreformer"):
            raise ValueError(f"Unsupported restorer: {restorer!r}")

        upscale_to = _parse_size(cfg.get("upscale_to"))
        use_semantic_mask = bool(cfg.get("semantic_mask", True))
        color_match = bool(cfg.get("color_match", True))
        grain_strength = float(cfg.get("grain_strength", 0.35))
        fidelity = float(cfg.get("fidelity", 0.65))
        stats_out = Path(cfg["stats_out"]) if cfg.get("stats_out") else None

        gfpgan_weights = Path(
            cfg.get("gfpgan_weights") or _resolve_model_file("GFPGANv1.4.pth", "face")
        )
        restoreformer_weights = Path(
            cfg.get("restoreformer_weights") or _resolve_model_file("RestoreFormer++.ckpt", "face")
        )
        yunet_model = Path(
            cfg.get("yunet_model")
            or _resolve_model_file("face_detection_yunet_2023mar.onnx", "face")
        )
        seg_model = Path(
            cfg.get("seg_model") or _resolve_model_file("selfie_multiclass_256x256.tflite", "face")
        )
        esrgan_weights = Path(
            cfg.get("esrgan_weights")
            or _resolve_model_file("realesr-general-x4v3.pth", "realesrgan")
        )

        face_model, esr_model, seg_interp, detector = self._get_enhancer_models(
            restorer=restorer,
            use_semantic_mask=use_semantic_mask,
            upscale_to=upscale_to,
            gfpgan_weights=gfpgan_weights,
            restoreformer_weights=restoreformer_weights,
            yunet_model=yunet_model,
            seg_model=seg_model,
            esrgan_weights=esrgan_weights,
        )

        return enhance_video(
            input_path=input_path,
            audio_source=audio_source,
            out_path=out_path,
            upscale_to=upscale_to,
            restorer=restorer,
            gfpgan_weights=gfpgan_weights,
            restoreformer_weights=restoreformer_weights,
            yunet_model=yunet_model,
            seg_model=seg_model,
            use_semantic_mask=use_semantic_mask,
            color_match=color_match,
            grain_strength=grain_strength,
            esrgan_weights=esrgan_weights,
            fidelity=fidelity,
            stats_out=stats_out,
            preloaded_face_model=face_model,
            preloaded_esr_model=esr_model,
            preloaded_seg_interp=seg_interp,
            preloaded_detector=detector,
        )

    def enhance(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run standalone face enhancement using cached warm models."""
        if not payload.get("input") or not payload.get("out"):
            raise ValueError("enhance payload requires 'input' and 'out' paths")
        with self._lock:
            self._status = "busy"
            t0 = time.perf_counter()
            try:
                stats = self._run_enhance_locked(payload)
                self._jobs_completed += 1
                self._status = "ready"
                self._last_error = None
                return {
                    "ok": True,
                    "enhance_sec": round(time.perf_counter() - t0, 2),
                    "out": str(payload["out"]),
                    "stats": stats,
                    "jobs_completed": self._jobs_completed,
                }
            except Exception as exc:
                self._status = "ready"
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

    def render(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run MiniMax-H3 render (and optional inline face enhancement) with zero cold start."""
        import numpy as np
        import torch
        from diffusers.utils import export_to_video, load_image, load_video
        from PIL import Image

        mode = str(payload.get("mode", "swap"))
        if mode not in SIGMAS:
            raise ValueError(f"Unsupported mode: {mode!r}")

        video_path = payload.get("video")
        out_path_raw = payload.get("out")
        if not video_path or not out_path_raw:
            raise ValueError("render payload requires 'video' and 'out'")

        prompt_raw = payload.get("prompt")
        if not prompt_raw:
            raise ValueError("render payload requires 'prompt' (file path or text)")
        prompt_candidate = Path(str(prompt_raw))
        prompt_text = (
            prompt_candidate.read_text(encoding="utf-8").strip()
            if prompt_candidate.is_file()
            else str(prompt_raw).strip()
        )

        # Accept either refs=[{"path": ..., "pin": 0}, ...] or ref=[...] + pin=[...]
        refs_in = payload.get("refs")
        if refs_in is not None:
            ref_paths = [str(item["path"]) for item in refs_in]
            pins = [int(item.get("pin", 0)) for item in refs_in]
        else:
            ref_paths = [str(p) for p in (payload.get("ref") or [])]
            pins = [int(p) for p in (payload.get("pin") or [])]
            if not pins and ref_paths:
                pins = [0] * len(ref_paths)
        if not ref_paths or len(ref_paths) != len(pins):
            raise ValueError("render payload requires at least one ref with matching pin")

        audio_path = payload.get("audio")
        frames_req = int(payload.get("frames", 0))
        short_edge = int(payload.get("short_edge", 736))
        seed = int(payload.get("seed", 0))
        keyframe_until = float(payload.get("keyframe_until", 0.4))
        out_path = Path(str(out_path_raw))

        with self._lock:
            self._status = "busy"
            t_total = time.perf_counter()
            try:
                self._load_pipeline_for_mode(mode)
                pipe = self._pipe
                assert pipe is not None

                # 1. Encode prompt with RAM-resident Qwen3-VL (~2.5s on H200/B200)
                prompt_embeds, text_token_tags, text_encode_sec = self._encode_prompt(prompt_text)

                # 2. Load conditioning video & compute canvas geometry
                cond = load_video(str(video_path))
                if frames_req > 0:
                    cond = cond[:frames_req]
                w0, h0 = cond[0].size
                scale = short_edge / min(h0, w0)
                h = max(32, int(round(h0 * scale / 32)) * 32)
                w = max(32, int(round(w0 * scale / 32)) * 32)
                cond = [f.resize((w, h), Image.Resampling.LANCZOS) for f in cond]
                n = len(cond)
                nf = ((n - 1) // 4) * 4 + 1
                cond = cond[:nf]
                if len(cond) < nf:
                    cond += [cond[-1]] * (nf - len(cond))

                # 3. Configure schedulers cleanly for this job's mode
                sigmas = SIGMAS[mode]
                orig_vid_st = self._orig_vid_set_timesteps
                orig_aud_st = self._orig_aud_set_timesteps
                pipe.scheduler.set_timesteps = lambda num_inference_steps=None, device=None, **kw: (
                    orig_vid_st(sigmas=sigmas, device=device)
                )
                pipe.audio_scheduler.set_timesteps = (
                    lambda num_inference_steps=None, device=None, **kw: orig_aud_st(
                        sigmas=sigmas, device=device
                    )
                )

                se = min(h, w)
                pipe.register_to_config(
                    canvas_short_edge=se,
                    canvas_max_pixels=se * max(h, w),
                    reference_image_short_edge=se,
                )

                # 4. Populate per-job context for MiniMax-H3 hooks
                kf_images = [
                    load_image(p).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
                    for p in ref_paths
                ]
                nlat = (nf - 1) // 4 + 1
                kf_anchors = [min(nlat - 1, max(0, int(round(p / 4.0)))) for p in pins]
                self._job_ctx.update(
                    {
                        "prompt_embeds": prompt_embeds,
                        "text_token_tags": text_token_tags,
                        "kf_images": kf_images,
                        "kf_anchors": kf_anchors,
                        "nkf": len(kf_images),
                        "keyframe_until": keyframe_until,
                    }
                )

                torch.cuda.reset_peak_memory_stats()
                t_render = time.perf_counter()
                with torch.inference_mode():
                    out = pipe(
                        prompt="x",
                        reference_videos=[cond],
                        num_frames=nf,
                        height=h,
                        width=w,
                        fps=24,
                        num_inference_steps=len(sigmas) - 1,
                        guidance_scale=1.0,
                        generator=torch.Generator("cuda").manual_seed(seed),
                    )
                render_sec = time.perf_counter() - t_render
                peak_alloc = torch.cuda.max_memory_allocated() / 2**30
                peak_res = torch.cuda.max_memory_reserved() / 2**30

                # Clear per-job tensors promptly
                self._job_ctx["prompt_embeds"] = None
                self._job_ctx["text_token_tags"] = None
                self._job_ctx["kf_images"] = []
                self._job_ctx["kf_anchors"] = []
                self._job_ctx["nkf"] = 0
                del prompt_embeds, text_token_tags, kf_images, cond

                video = out.values["videos"][0]
                frames_np = [np.asarray(f) for f in video]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as tmp:
                    raw_mp4 = Path(tmp) / "v.mp4"
                    export_to_video(frames_np, str(raw_mp4), fps=24)
                    if audio_path and Path(str(audio_path)).exists():
                        dur = len(frames_np) / 24.0
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-nostdin",
                                "-y",
                                "-loglevel",
                                "error",
                                "-i",
                                str(raw_mp4),
                                "-i",
                                str(audio_path),
                                "-t",
                                f"{dur:.3f}",
                                "-c:v",
                                "copy",
                                "-c:a",
                                "aac",
                                "-b:a",
                                "192k",
                                "-shortest",
                                str(out_path),
                            ],
                            stdin=subprocess.DEVNULL,
                            check=True,
                        )
                    else:
                        shutil.copy(raw_mp4, out_path)

                del out, video, frames_np
                torch.cuda.empty_cache()

                # 5. Optional inline face enhancement (reuses warm enhancers!)
                enhance_sec = 0.0
                enhanced_out: str | None = None
                enhance_stats: dict[str, Any] | None = None
                enhance_cfg = payload.get("enhance")
                if isinstance(enhance_cfg, dict) and enhance_cfg.get("restorer", "none") != "none":
                    t_enh = time.perf_counter()
                    enh_payload = dict(enhance_cfg)
                    enh_payload.setdefault("input", str(out_path))
                    enh_payload.setdefault("audio_source", str(audio_path or out_path))
                    restorer_name = str(enh_payload.get("restorer", "restoreformer"))
                    enh_payload.setdefault(
                        "out",
                        str(out_path.with_name(f"{out_path.stem}_{restorer_name}.mp4")),
                    )
                    enhance_stats = self._run_enhance_locked(enh_payload)
                    enhance_sec = time.perf_counter() - t_enh
                    enhanced_out = str(enh_payload["out"])

                self._jobs_completed += 1
                self._status = "ready"
                self._last_error = None
                total_sec = time.perf_counter() - t_total

                print(
                    f"[worker] job #{self._jobs_completed} done: "
                    f"encode={text_encode_sec:.1f}s render={render_sec:.1f}s "
                    f"enhance={enhance_sec:.1f}s total={total_sec:.1f}s "
                    f"({w}x{h}x{nf}, peak={peak_alloc:.1f}/{peak_res:.1f} GiB)",
                    flush=True,
                )
                return {
                    "ok": True,
                    "mode": mode,
                    "quant": self.quant,
                    "width": w,
                    "height": h,
                    "num_frames": nf,
                    "text_encode_sec": round(text_encode_sec, 2),
                    "render_sec": round(render_sec, 2),
                    "enhance_sec": round(enhance_sec, 2),
                    "total_sec": round(total_sec, 2),
                    "peak_alloc_gib": round(peak_alloc, 2),
                    "peak_reserved_gib": round(peak_res, 2),
                    "out": str(out_path),
                    "enhanced_out": enhanced_out,
                    "enhance_stats": enhance_stats,
                    "jobs_completed": self._jobs_completed,
                }
            except Exception as exc:
                self._status = "ready"
                self._last_error = f"{type(exc).__name__}: {exc}"
                torch.cuda.empty_cache()
                raise


def create_worker_server(
    engine: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer bound to (host, port) serving the given engine."""

    class WorkerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[http] {self.address_string()} - {format % args}", flush=True)

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            data = self.rfile.read(length)
            parsed = json.loads(data.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("JSON body must be an object")
            return parsed

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, engine.health())
                return
            self._send_json(
                HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path {self.path}"}
            )

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
            except Exception as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Invalid JSON: {exc}"}
                )
                return

            if self.path == "/render":
                try:
                    result = engine.render(payload)
                    self._send_json(HTTPStatus.OK, result)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    tb = traceback.format_exc()
                    print(f"[worker] /render failed:\n{tb}", flush=True)
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": tb[-1500:],
                        },
                    )
                return

            if self.path == "/enhance":
                try:
                    result = engine.enhance(payload)
                    self._send_json(HTTPStatus.OK, result)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    tb = traceback.format_exc()
                    print(f"[worker] /enhance failed:\n{tb}", flush=True)
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": tb[-1500:],
                        },
                    )
                return

            if self.path == "/shutdown":
                self._send_json(HTTPStatus.OK, {"ok": True, "message": "shutting down"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            self._send_json(
                HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path {self.path}"}
            )

    server = ThreadingHTTPServer((host, port), WorkerHandler)
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="Resident LOKI Remake GPU Worker HTTP Daemon")
    ap.add_argument(
        "--host", default="127.0.0.1", help="Bind address (127.0.0.1 for SSH, 0.0.0.0 for prod)"
    )
    ap.add_argument("--port", type=int, default=8000, help="HTTP listen port")
    ap.add_argument(
        "--base",
        type=Path,
        default=_env_path("LOKI_MODELS_DIR", "/workspace/models") / "MiniMax-H3",
    )
    ap.add_argument(
        "--lora-repo",
        type=Path,
        default=_env_path("LOKI_MODELS_DIR", "/workspace/models") / "Viggle-Remake",
    )
    ap.add_argument("--quant", choices=("bf16", "fp8", "int8"), default="fp8")
    ap.add_argument("--attn-backend", default="_native_cudnn")
    ap.add_argument(
        "--preload-mode",
        choices=("swap", "edit", "none"),
        default="swap",
        help="LoRA mode to preload at startup (default: swap; 'none' loads lazily on first job)",
    )
    args = ap.parse_args()

    preload = None if args.preload_mode == "none" else args.preload_mode
    engine = ResidentRemakeEngine(
        base_dir=args.base,
        lora_repo=args.lora_repo,
        quant=args.quant,
        attn_backend=args.attn_backend,
        preload_mode=preload,
    )
    server = create_worker_server(engine, host=args.host, port=args.port)

    def _handle_sig(_signum: int, _frame: Any) -> None:
        print("[worker] received shutdown signal, stopping server...", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    actual_port = server.server_address[1]
    print(
        f"[worker] listening on http://{args.host}:{actual_port} "
        f"(mode={engine._loaded_mode}, quant={args.quant})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        print("[worker] stopped cleanly", flush=True)


if __name__ == "__main__":
    main()

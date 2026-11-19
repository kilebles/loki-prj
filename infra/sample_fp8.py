#!/usr/bin/env python3
"""Quantized FP8/INT8 Viggle-Remake + MiniMax-H3 inference runner for 96 GB GPUs (RTX PRO 6000)."""

from __future__ import annotations

import argparse
import gc
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, attention_backend
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    _ROPE_FRAME_RESCALE,
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    _frame_position_grid,
)
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3LoopSchedulerStep,
    MiniMaxH3Ref2VALoopDenoiser,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    encode_vae_condition,
    get_qwen3vl_prompt_embeds,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames
from diffusers.utils.export_utils import encode_video
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

# 6-step grids from Viggle-Remake sample.py, in the video stream's sigma.
SIGMAS = {
    "swap": [1, 0.990476, 0.965243, 0.923077, 0.8, 0.631579, 0],
    "edit": [1, 0.983607, 0.96, 0.923077, 0.857143, 0.705882, 0],
}
TEXT_TAG = 1


def _quantize_transformer(transformer: MiniMaxH3Transformer3DModel, quant: str) -> str:
    """Apply torchao weight-only quantization after fuse_lora() on CPU."""
    if quant == "bf16":
        return "bf16"

    from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

    if quant == "fp8":
        try:
            # Probe Float8WeightOnlyConfig on this exact GPU/CUDA build before quantizing 30GB.
            probe = torch.nn.Linear(128, 128, bias=False, device="cuda", dtype=torch.bfloat16)
            quantize_(probe, Float8WeightOnlyConfig())
            _ = probe(torch.randn(1, 128, device="cuda", dtype=torch.bfloat16))
            torch.cuda.synchronize()
            del probe
            quantize_(transformer, Float8WeightOnlyConfig())
            return "fp8"
        except Exception as exc:
            print(f"WARN: Float8WeightOnlyConfig unavailable ({exc}); using int8", flush=True)
            quantize_(transformer, Int8WeightOnlyConfig())
            return "int8"

    quantize_(transformer, Int8WeightOnlyConfig())
    return "int8"


def main() -> None:
    parser = argparse.ArgumentParser(description="FP8/INT8 Viggle-Remake inference runner.")
    parser.add_argument("--mode", required=True, choices=["swap", "edit"])
    parser.add_argument("--video", required=True, help="Source clip")
    parser.add_argument("--audio", default=None, help="Soundtrack pinned into output")
    parser.add_argument("--prompt", required=True, help="Six-section prompt file")
    parser.add_argument("--ref", action="append", required=True, help="Reference image path")
    parser.add_argument("--pin", action="append", type=int, default=[], help="Pin frame per ref")
    parser.add_argument("--frames", type=int, default=124, help="17n+5 in 90..345 at 24 fps")
    parser.add_argument("--short-edge", type=int, default=480, help="480 or 736 (multiple of 32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base", required=True, help="Local MiniMaxAI/MiniMax-H3 path")
    parser.add_argument("--lora-repo", required=True, help="Local Viggle/Viggle-Remake path")
    parser.add_argument("--rank", type=int, default=256, choices=[256, 128, 64])
    parser.add_argument("--keyframe-until", type=float, default=0.9)
    parser.add_argument("--attn-backend", default="_native_cudnn")
    parser.add_argument("--quant", default="fp8", choices=["fp8", "int8", "bf16"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    nf, se, refs, pin = args.frames, args.short_edge, args.ref, args.pin
    assert 90 <= nf <= 345 and (nf - 5) % 17 == 0, f"--frames must be 17n+5 in 90..345, got {nf}"
    assert se % 32 == 0, f"--short-edge must be multiple of 32, got {se}"
    assert len(pin) in (0, len(refs)), "--pin must be omitted or given once per --ref"

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    sigmas = SIGMAS[args.mode]

    video = MiniMaxH3VideoReference.from_file(args.video)
    video.audio, video.sample_rate = None, None
    height, width = video.frames.shape[1], video.frames.shape[2]

    # Step 1: Load 64-layer Qwen3-VL conditioner on GPU, encode prompt, and free completely.
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.base, subfolder="processor")
    tokenizer = AutoTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, subfolder="text_encoder", dtype=torch.bfloat16
    ).to("cuda")
    t1 = time.time()
    _, stamps = MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
        np.zeros((nf, 1, 1, 3), dtype=np.uint8),
        24.0,
        2.0,
        processor.video_processor.temporal_patch_size,
    )
    segments = [
        "<Video 1>: ",
        *[f"<{s:.1f} seconds>" for s in stamps],
        *[f"<Picture {i + 1}>: " for i in range(len(refs))],
        prompt,
    ]
    token_ids = [i for s in segments for i in tokenizer(s, add_special_tokens=False)["input_ids"]]
    with torch.no_grad():
        prompt_embeds = get_qwen3vl_prompt_embeds(
            text_encoder,
            processor,
            token_ids,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        ).cpu()
    text_token_tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)
    t2 = time.time()
    print(
        f"text encoder: load {t1 - t0:.0f}s, encode {t2 - t1:.1f}s, {len(token_ids)} tokens, "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Load pipeline + transformer on CPU, fuse LoRA BEFORE quantizing, then move to CUDA.
    pipe = ModularPipeline.from_pretrained(args.base, workflow="ref2va")
    type(pipe).min_duration = property(lambda self: 90 / 24)
    pipe.load_components(
        names=["vae", "audio_vae", "scheduler", "audio_scheduler"],
        pretrained_model_name_or_path=args.base,
        dtype=torch.bfloat16,
    )
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.base, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    lora_sub = (
        f"lora/{args.mode}" if args.rank == 256 else f"lora_lowrank/{args.mode}-r{args.rank}"
    )
    lora = Path(args.lora_repo) / lora_sub
    transformer.load_lora_adapter(
        str(lora), weight_name="pytorch_lora_weights.safetensors", prefix=None
    )
    # Fuse and unload LoRA in bf16 before torchao wraps nn.Linear layers.
    transformer.fuse_lora()
    transformer.unload_lora()

    t_q0 = time.time()
    quant_used = _quantize_transformer(transformer, args.quant)
    t_q1 = time.time()
    print(f"transformer quant={quant_used} in {t_q1 - t_q0:.1f}s", flush=True)

    pipe.transformer_ref = transformer
    pipe.transformer_ref.set_attention_backend(args.attn_backend)
    pipe.to("cuda")
    pipe.register_to_config(
        canvas_short_edge=se,
        canvas_max_pixels=se * max(height, width),
        reference_image_short_edge=se,
    )
    for sched, s_val in ((pipe.scheduler, 12.0), (pipe.audio_scheduler, 3.0)):
        grid = [
            s_val * (x / (12.0 - 11.0 * x)) / (1.0 + (s_val - 1.0) * (x / (12.0 - 11.0 * x)))
            for x in sigmas
        ]
        sched.set_timesteps = (
            lambda f, g: lambda *a, **k: f(sigmas=g, device=k.get("device"))
        )(sched.set_timesteps, grid)
    t3 = time.time()
    print(
        f"pipeline on cuda ({quant_used}) in {t3 - t2:.0f}s "
        f"(alloc={torch.cuda.memory_allocated() / 2**30:.1f} GiB)",
        flush=True,
    )

    def use_fixed_embeds(self, components, state):
        block_state = self.get_block_state(state)
        block_state.prompt_embeds = prompt_embeds.to(components._execution_device, torch.bfloat16)
        block_state.text_token_tags = text_token_tags
        self.set_block_state(state, block_state)
        return components, state

    MiniMaxH3Ref2VATextEncoderStep.__call__ = use_fixed_embeds

    _build_row_timesteps = MiniMaxH3SetTimestepsStep.build_row_timesteps
    MiniMaxH3SetTimestepsStep.build_row_timesteps = staticmethod(
        lambda *a: _build_row_timesteps(*a[:6], 1.0, *a[7:])
    )

    @torch.no_grad()
    def video_only_step(self, components, block_state, i, t):
        n = block_state.num_condition_video_rows
        block_state.latents[n:] = components.scheduler.step(
            block_state.noise_pred[0, n:].float(),
            t,
            block_state.latents[n:],
            return_dict=False,
        )[0]
        return components, block_state

    MiniMaxH3LoopSchedulerStep.__call__ = video_only_step

    kf_state: dict[str, list] = {"images": [], "anchors": list(pin)}
    _encode_references = MiniMaxH3Ref2VAReferenceEncoderStep.__call__

    def encode_references_then_keyframes(self, components, state):
        components, state = _encode_references(self, components, state)
        with torch.no_grad():
            kf = [
                encode_vae_condition(
                    components.vae,
                    torch.from_numpy(np.array(im))
                    .to(components._execution_device)
                    .permute(2, 0, 1)[None, :, None],
                    components.pixel_mean,
                    components.pixel_std,
                    components.keyframe_encode_seed,
                )
                for im in kf_state["images"]
            ]
        state.set("condition_latents", list(state.get("condition_latents") or []) + kf)
        return components, state

    MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = encode_references_then_keyframes
    _layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence

    def layout_with_keyframes(
        text_token_tags,
        references,
        condition_latents,
        audio_condition_latents,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
        audio_channels,
        audio_tag,
        video_tag,
    ):
        k = len(kf_state["anchors"])
        pos, tags, vi, ai, ti, ncv, nca = _layout(
            text_token_tags,
            references,
            condition_latents[: len(condition_latents) - k],
            audio_condition_latents,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            audio_channels,
            audio_tag,
            video_tag,
        )
        if not k:
            return pos, tags, vi, ai, ti, ncv, nca
        s = text_token_tags.shape[0] + ncv + nca
        origin = float(pos[s, 0])
        blocks = []
        for anchor, lat in zip(kf_state["anchors"], condition_latents[-k:], strict=True):
            grid, _ = _frame_position_grid(lat.shape[3], lat.shape[4], patch_size[1], patch_size[2])
            p = torch.zeros(grid.shape[0], 3, dtype=torch.float64)
            p[:, 0] = origin + _ROPE_FRAME_RESCALE * float(anchor)
            p[:, 1:] = grid
            blocks.append(p)
        n = sum(b.shape[0] for b in blocks)
        pos = torch.cat([pos[:s], *blocks, pos[s:]])
        tags = torch.cat([tags[:s], torch.full((n,), video_tag, dtype=tags.dtype), tags[s:]])
        vi = torch.cat([vi[vi < s], torch.arange(s, s + n), vi[vi >= s] + n])
        ai = torch.where(ai >= s, ai + n, ai)
        return pos, tags, vi, ai, ti, ncv + n, nca

    MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = staticmethod(
        layout_with_keyframes
    )

    r_comp = pipe.vae_spatial_compression_ratio
    _, ph, pw = pipe.patch_size
    nkf = len(pin) * (height // r_comp // ph) * (width // r_comp // pw)
    _denoise = MiniMaxH3Ref2VALoopDenoiser.__call__

    def kf_release(self, components, block_state, i, t):
        if not nkf or 1 - float(t) >= args.keyframe_until:
            return _denoise(self, components, block_state, i, t)
        ncv = block_state.num_condition_video_rows
        f = block_state.denoiser_input_fields
        vi, ai = f["video_indices"], f["audio_indices"]
        lo, hi = int(vi[ncv - nkf]), int(vi[ncv - 1]) + 1
        assert hi - lo == nkf and hi == int(ai[block_state.num_condition_audio_rows]), (lo, hi, nkf)
        keep = torch.ones(f["position_ids"].shape[0], dtype=torch.bool, device=vi.device)
        keep[lo:hi] = False
        plan = list(block_state.row_timestep_plan)
        plan[i] = (plan[i][0], plan[i][1][keep])
        full = block_state.latents, f, block_state.row_timestep_plan
        block_state.latents = torch.cat(
            [block_state.latents[: ncv - nkf], block_state.latents[ncv:]]
        )
        block_state.denoiser_input_fields = dict(
            f,
            position_ids=f["position_ids"][keep],
            token_tags=f["token_tags"][keep],
            video_indices=torch.cat([vi[: ncv - nkf], vi[ncv:] - nkf]),
            audio_indices=ai - nkf * (ai >= hi),
        )
        block_state.row_timestep_plan = plan
        components, block_state = _denoise(self, components, block_state, i, t)
        block_state.latents, block_state.denoiser_input_fields, block_state.row_timestep_plan = full
        p = block_state.noise_pred
        block_state.noise_pred = torch.cat(
            [p[:, : ncv - nkf], p.new_zeros(1, nkf, p.shape[2]), p[:, ncv - nkf :]], 1
        )
        return components, block_state

    MiniMaxH3Ref2VALoopDenoiser.__call__ = kf_release

    if pin:
        for i, im in enumerate(Image.open(x).convert("RGB") for x in refs):
            if im.size != (width, height) and i == 0:
                im = pipe.image_processor.resize(im, height=height, width=width)
            elif im.size != (width, height):
                scale = max(width / im.size[0], height / im.size[1])
                size = (
                    max(width, round(im.size[0] * scale)),
                    max(height, round(im.size[1] * scale)),
                )
                left, top = max(0, (size[0] - width) // 2), max(0, (size[1] - height) // 2)
                im = im.resize(size, Image.Resampling.LANCZOS).crop(
                    (left, top, left + width, top + height)
                )
            kf_state["images"].append(im)

    asr = pipe.audio_vae.config.sampling_rate
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            args.audio or args.video,
            "-t",
            f"{nf / 24:.3f}",
            "-vn",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(asr),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    wav = torch.frombuffer(bytearray(pcm), dtype=torch.float32).view(-1, 2).T
    n_samp = round((nf + 1) / 24 * asr)
    wav = torch.nn.functional.pad(wav, (0, max(0, n_samp - wav.shape[1])))[:, :n_samp]
    with attention_backend(), torch.no_grad():
        posterior = pipe.audio_vae.encode(
            wav[:, None].to("cuda", torch.bfloat16), return_dict=False
        )[0]
    mean = torch.tensor(pipe.audio_vae.config.latents_mean).view(1, 1, -1)
    std = torch.tensor(pipe.audio_vae.config.latents_std).view(1, 1, -1)
    lat_slice = posterior.mode().float().cpu().transpose(1, 2)[:, : audio_latent_num_frames(nf)]
    rows = (lat_slice - mean) / std
    audio_latents = rows.permute(0, 2, 1).contiguous()

    torch.cuda.reset_peak_memory_stats()
    t4 = time.time()
    result = pipe(
        prompt="".join(segments),
        references=[video, *[MiniMaxH3ImageReference.from_file(x) for x in refs]],
        num_frames=nf,
        height=height,
        width=width,
        num_inference_steps=len(sigmas) - 1,
        generator=torch.Generator().manual_seed(args.seed),
        audio_latents=audio_latents,
        output=["videos", "audio", "sampling_rate"],
    )
    t5 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        result["videos"][0],
        fps=24,
        output_path=args.out,
        audio=result["audio"][0],
        audio_sample_rate=result["sampling_rate"],
    )
    print(
        f"render {t5 - t4:.1f}s at {width}x{height}x{nf}, "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated / "
        f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

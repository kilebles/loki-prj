"""Typed data contracts for LOKI Remake preparation and rendering."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

RemakeMode = Literal["swap", "edit"]
RenderTier = Literal[480, 736]


class VideoProbe(BaseModel):
    """Metadata extracted from source media via ffprobe."""

    width: int
    height: int
    fps: float
    duration_sec: float
    frame_count: int
    has_audio: bool


class NormalizedVideo(BaseModel):
    """Conditioning video and audio normalized for MiniMax-H3 / Viggle-Remake."""

    source_path: Path
    video_path: Path
    audio_path: Path | None = None
    width: int
    height: int
    fps: int = 24
    num_frames: int
    duration_sec: float
    padded_frames: int = 0
    aspect_ratio: str


class ShotInfo(BaseModel):
    """Detected camera shot boundaries and best sharp keyframe."""

    index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    keyframe_index: int
    keyframe_sec: float
    sharpness_score: float
    keyframe_path: Path


class PaintedRef(BaseModel):
    """Repainted reference keyframe pinned to a source frame."""

    shot_index: int
    pin_frame: int
    pin_sec: float
    source_frame_path: Path
    painted_path: Path
    model_used: str
    cost_usd: float = 0.0
    job_id: str = ""


class PromptValidationResult(BaseModel):
    """Result of validating a MiniMax-H3 six-section prompt."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PreparedJob(BaseModel):
    """Complete output of the local preparation stage ready for GPU rendering."""

    job_id: str
    mode: RemakeMode
    tier: RenderTier
    instruction: str
    user_ref_path: Path | None = None
    normalized: NormalizedVideo
    shots: list[ShotInfo]
    refs: list[PaintedRef]
    prompt_path: Path
    prompt_text: str
    contact_sheet_path: Path
    prep_cost_usd: float = 0.0
    timings_sec: dict[str, float] = Field(default_factory=dict)


class GPUInstanceState(BaseModel):
    """Persisted metadata for an active rented GPUniq instance."""

    task_id: int
    agent_id: int
    gpu_model: str
    price_per_hour: float
    ssh_host: str
    ssh_port: int
    ssh_user: str = "root"
    ssh_key_path: Path
    status: str = "running"
    created_at: str


class RenderResult(BaseModel):
    """Measured output and cost telemetry for one remote GPU render."""

    job_id: str
    mode: RemakeMode
    tier: RenderTier
    width: int
    height: int
    num_frames: int
    seed: int = 0
    attn_backend: str = "_native_cudnn"
    wall_sec: float
    render_sec: float
    peak_alloc_gib: float = 0.0
    peak_reserved_gib: float = 0.0
    gpu_model: str
    gpu_price_per_hour: float
    render_cost_usd: float
    inference_only_cost_usd: float = 0.0
    out_video_path: Path
    has_audio: bool = False

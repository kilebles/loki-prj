"""Application settings loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for LOKI Remake."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gpuniq_api_key: SecretStr
    gpuniq_base_url: str = "https://api.gpuniq.com"
    painter_model: str = "nano-banana-2"
    painter_fallback_model: str | None = "gpt-image-2"
    painter_fallback_models: list[str] = Field(
        default_factory=lambda: ["seedream-4-5", "flux-2-pro", "gpt-image-2"]
    )
    vlm_model: str = "gemini-3-flash"
    default_tier: Literal[480, 736] = 480
    work_dir: Path = Path("work")
    log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"

    @field_validator("default_tier", mode="before")
    @classmethod
    def _coerce_tier(cls, v: Any) -> Any:
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

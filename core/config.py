"""Application configuration management."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # Optional for local flux-kontext runs; required only when IMAGE_PROVIDER=gemini.
    GOOGLE_API_KEY: str = ""

    # Image generation backend configuration
    # - "flux-kontext": FREE, fully local. Open-weights FLUX.1 Kontext [dev] model
    #   that keeps characters/objects consistent across scenes via reference images.
    # - "gemini": cloud API (requires GOOGLE_API_KEY).
    IMAGE_PROVIDER: Literal["flux-kontext", "gemini"] = "flux-kontext"

    # FLUX Kontext settings (for flux-kontext backend)
    # Open-weights image model with native character/object consistency.
    FLUX_KONTEXT_MODEL: str = "black-forest-labs/FLUX.1-Kontext-dev"
    # Embedded guidance scale for Kontext (BFL recommends ~2.5).
    FLUX_KONTEXT_GUIDANCE_SCALE: float = 2.5
    # Offload model weights to CPU layer-by-layer to fit in less VRAM/RAM.
    # Slower but lets the 12B model run on machines with limited memory.
    # Recommended True on Macs / GPUs with < 24GB.
    IMAGE_ENABLE_CPU_OFFLOAD: bool = True

    # Device configuration (auto, cuda, mps, cpu)
    IMAGE_DEVICE: str = "auto"  # auto will choose mps for Mac, cuda for NVIDIA

    # Generation parameters
    IMAGE_NUM_INFERENCE_STEPS: int = 30  # Lower = faster, higher = better quality

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

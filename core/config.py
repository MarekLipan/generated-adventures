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

    # GGUF pre-quantized models (recommended over runtime quantization).
    # Format: "repo_id:filename" — files are resolved from the HuggingFace cache.
    # When both are set, GGUF loading is used and IMAGE_QUANTIZATION is ignored.
    # Transformer Q8_0 (~12.7 GB) + T5 Q5_K_M (~3.4 GB) = ~20 GB total,
    # fits in 32 GB unified memory with no CPU offload.
    FLUX_KONTEXT_GGUF_TRANSFORMER: str = ""
    FLUX_KONTEXT_GGUF_T5: str = ""

    # Runtime quantization via optimum-quanto (fallback when GGUF is not set).
    # - "none":  bfloat16 full precision (~24 GB) — requires sequential CPU offload on 32 GB Mac
    # - "int8":  8-bit weights (~12 GB)  — fits in 32 GB without any CPU offload
    # - "int4":  4-bit weights (~6 GB)   — fits easily, some quality loss
    # Requires: pip install optimum-quanto
    IMAGE_QUANTIZATION: Literal["none", "int8", "int4"] = "none"

    # Generation parameters
    IMAGE_NUM_INFERENCE_STEPS: int = 30  # Lower = faster, higher = better quality

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

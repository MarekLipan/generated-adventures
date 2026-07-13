"""Application configuration management."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # Hugging Face token for downloading gated models (e.g. FLUX.1-Kontext-dev).
    HUGGING_FACE_HUB_TOKEN: str = ""

    # Optional for local flux-kontext runs; required only when IMAGE_PROVIDER=gemini.
    GOOGLE_API_KEY: str = ""

    # Image generation backend configuration
    # - "flux-kontext":    FREE, fully local. FLUX.1 Kontext [dev] 12B. Single stitched
    #                      reference image for multi-character scenes. 30 steps.
    # - "flux-klein":      FREE, fully local. FLUX.2 Klein [9B or 4B]. Native list of
    #                      reference images — no stitching. 9B distilled runs in 4 steps.
    # - "gemini":          Cloud API (requires GOOGLE_API_KEY).
    IMAGE_PROVIDER: Literal["flux-kontext", "flux-klein", "gemini"] = "flux-kontext"

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

    # FLUX.2 Klein settings (for flux-klein backend)
    # Distilled successor to FLUX.1 Kontext with native multi-image reference support.
    # 9B distilled: 4 steps, guidance ~3.5. 4B base: ~50 steps, guidance ~3.5.
    FLUX_KLEIN_MODEL: str = "black-forest-labs/FLUX.2-klein-9B"
    FLUX_KLEIN_GUIDANCE_SCALE: float = 3.5
    # 4 for 9B distilled model; set to 50 if using 4B base (non-distilled).
    FLUX_KLEIN_NUM_INFERENCE_STEPS: int = 4
    # Edge length each reference image is resized to before being fed to Klein.
    # Attention cost grows ~quadratically with this, and multiple references add
    # up fast. CUDA has flash-attention so it stays bounded at 1024.
    FLUX_KLEIN_REFERENCE_SIZE: int = 1024
    # MPS has no flash-attention kernel, so the full attention matrix is
    # materialized in one allocation. With a multi-character party, 1024px
    # references OOM a 32 GB Mac (a single ~30 GB MTLBuffer); 512px fits with
    # headroom. This smaller size is used automatically when running on MPS.
    FLUX_KLEIN_MPS_REFERENCE_SIZE: int = 512

    # Generation parameters
    IMAGE_NUM_INFERENCE_STEPS: int = 30  # Used by flux-kontext backend

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

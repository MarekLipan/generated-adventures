"""Application configuration management."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # Hugging Face token for downloading gated models (e.g. FLUX.1-Kontext-dev).
    HUGGING_FACE_HUB_TOKEN: str = ""

    # Optional for local flux-kontext runs; required only when IMAGE_PROVIDER=gemini
    # or LLM_PROVIDER=gemini.
    GOOGLE_API_KEY: str = ""

    # ── Text LLM (story / scene generation) ─────────────────────────────────────
    # - "gemini":            Google Gemini API (requires GOOGLE_API_KEY). Default
    #                        model is the free-tier-friendly Gemini 3.1 Flash-Lite.
    # - "ollama" /
    #   "openai-compatible": any OpenAI-compatible endpoint (Ollama, llama.cpp
    #                        server, LM Studio) reached via LLM_BASE_URL.
    LLM_PROVIDER: Literal["gemini", "ollama", "openai-compatible"] = "gemini"
    # Model id for the chosen provider. For gemini, confirm the exact id in Google
    # AI Studio (it may carry a "-preview" suffix); for local, e.g.
    # "qwen2.5:14b-instruct" or "mistral-nemo:12b-instruct".
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    # OpenAI-compatible endpoint settings (used for ollama / openai-compatible only).
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = "ollama"

    # Image generation backend configuration
    # - "flux-kontext":    FREE, fully local. FLUX.1 Kontext [dev] 12B. Single stitched
    #                      reference image for multi-character scenes. 30 steps.
    # - "flux-klein":      FREE, fully local. FLUX.2 Klein [9B or 4B]. Native list of
    #                      reference images — no stitching. 9B distilled runs in 4 steps.
    # - "gemini":          Cloud API (requires GOOGLE_API_KEY).
    # - "http":            Remote inference service (this app's image_server.py) that
    #                      keeps a model warm and runs generation over HTTP. The game
    #                      becomes a thin client — see IMAGE_SERVER_URL below.
    IMAGE_PROVIDER: Literal["flux-kontext", "flux-klein", "gemini", "http"] = "flux-kontext"

    # Privacy guardrail: never upload reference images (including players' profile
    # photos) to a CLOUD image backend unless explicitly enabled. Local backends
    # (flux-kontext / flux-klein) run on-device and are unaffected. When False
    # (default), the Gemini image backend generates text-to-image only and drops
    # any reference photos, so player photos never leave the machine by accident.
    ALLOW_CLOUD_IMAGE_UPLOAD: bool = False

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

    # bitsandbytes quantization for the Klein backend (CUDA only).
    # Klein loads a 9B transformer (~18 GB bf16) AND a Qwen3-8B text encoder
    # (~16 GB bf16). On a <24 GB card neither module fits, so the bf16 pipeline
    # falls back to slow layer-by-layer sequential CPU offload (~30-66 s/image).
    # Quantizing BOTH modules with bitsandbytes lets model-level offload keep the
    # active module resident, cutting generation to ~12-20 s/image at the same
    # visual quality. NF4 is near-lossless here and fits with room to spare.
    # - "nf4":  4-bit NormalFloat (~5.3 GB transformer, ~4.5 GB encoder) — recommended.
    # - "int8": 8-bit (higher fidelity, ~9.5 GB transformer — tighter on 10 GB cards).
    # - "none": full bfloat16 (uses sequential CPU offload on small cards).
    # NOTE: on-the-fly GGUF and optimum-quanto are NOT used here — both are far
    # slower or crash on the current torch/CUDA build; bitsandbytes has fast kernels.
    FLUX_KLEIN_QUANTIZATION: Literal["none", "nf4", "int8"] = "nf4"

    # Edge length each reference image is resized to before being fed to Klein.
    # Attention cost grows ~quadratically with this, and multiple references add
    # up fast. On a 10 GB card, references >~640 px push peak VRAM over the limit
    # and trigger driver memory paging (3 refs @768 px = ~90 s vs @512 px = ~15 s),
    # so 512 keeps multi-character scenes fast. Raise on cards with more VRAM.
    FLUX_KLEIN_REFERENCE_SIZE: int = 512
    # MPS has no flash-attention kernel, so the full attention matrix is
    # materialized in one allocation. With a multi-character party, 1024px
    # references OOM a 32 GB Mac (a single ~30 GB MTLBuffer); 512px fits with
    # headroom. This smaller size is used automatically when running on MPS.
    FLUX_KLEIN_MPS_REFERENCE_SIZE: int = 512

    # Generation parameters
    IMAGE_NUM_INFERENCE_STEPS: int = 30  # Used by flux-kontext backend

    # ── Remote inference service (image_server.py) ──────────────────────────────
    # The server loads a local backend (flux-klein/flux-kontext) once, keeps it warm,
    # and serves generation over HTTP. Run it with:
    #     uv run python image_server.py
    # The server reads IMAGE_PROVIDER to decide which local backend to host, so set
    # that to e.g. "flux-klein" IN THE SERVER's environment, and "http" in the CLIENT
    # (game) environment.
    #
    # Bind address for the server. 0.0.0.0 = listen on all interfaces so other
    # devices on the same wifi/LAN can reach it; 127.0.0.1 = this machine only.
    IMAGE_SERVER_HOST: str = "0.0.0.0"
    IMAGE_SERVER_PORT: int = 8000
    # Optional shared secret. If set, the server requires header "X-API-Key: <value>"
    # and the client sends it. Leave empty to disable auth (fine on a trusted LAN).
    IMAGE_SERVER_API_KEY: str = ""

    # Client (IMAGE_PROVIDER="http") — base URL of the running image server.
    # Same machine: http://127.0.0.1:8000. From a laptop on the same wifi, point this
    # at the server machine, e.g. http://192.168.1.42:8000 or http://my-desktop.local:8000
    IMAGE_SERVER_URL: str = "http://127.0.0.1:8000"
    # Per-request timeout (seconds). Generous: covers model warmup + queued requests.
    IMAGE_SERVER_TIMEOUT: float = 300.0

    # ── Text-to-speech (scene narration) ────────────────────────────────────────
    # - "kokoro": FREE, fully local. Kokoro-82M via onnxruntime on CPU (~5x realtime).
    #             Runs off the GPU so it doesn't contend with local image generation.
    # - "gemini": Cloud API (requires GOOGLE_API_KEY). Expressive, but paid/online.
    # - "none":   Disable narration entirely.
    TTS_PROVIDER: Literal["kokoro", "gemini", "none"] = "kokoro"

    # Kokoro (local) settings. Model files are downloaded once into models/kokoro/
    # (see README). Paths are relative to the project root by default.
    KOKORO_MODEL_PATH: str = "models/kokoro/kokoro-v1.0.onnx"
    KOKORO_VOICES_PATH: str = "models/kokoro/voices-v1.0.bin"
    # Narrator voice. British male "bm_lewis" is the game's default DM voice.
    # Other options include bm_george, am_michael, am_adam, af_heart, bf_emma.
    KOKORO_VOICE: str = "bm_lewis"
    # Language code passed to the phonemizer; match the voice's region
    # (en-gb for British voices, en-us for American).
    KOKORO_LANG: str = "en-gb"
    # Playback speed multiplier (1.0 = natural). Slightly <1 reads more
    # deliberately — 0.92 gives a measured, engaged storyteller pace for the DM.
    KOKORO_SPEED: float = 0.92

    # Gemini TTS voice name (used only when TTS_PROVIDER="gemini").
    GEMINI_TTS_VOICE: str = "Algieba"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

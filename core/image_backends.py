"""Image generation backends for different providers."""

import io
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)


class ImageGenerator(ABC):
    """Abstract base class for image generation backends."""

    @abstractmethod
    def generate_character_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
    ) -> Optional[pathlib.Path]:
        """Generate a character portrait image.

        Args:
            prompt: Detailed text prompt for image generation
            output_path: Where to save the generated image

        Returns:
            Path to saved image or None if generation failed
        """
        pass

    @abstractmethod
    def generate_scene_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
    ) -> Optional[pathlib.Path]:
        """Generate a scene image with optional character references.

        Args:
            prompt: Detailed text prompt for scene composition
            output_path: Where to save the generated image
            reference_images: Optional list of character/asset images for consistency

        Returns:
            Path to saved image or None if generation failed
        """
        pass


class GeminiImageGenerator(ImageGenerator):
    """Image generator using Google Gemini API."""

    def __init__(self, api_key: str):
        from google import genai as genai_client  # type: ignore

        self.client = genai_client.Client(api_key=api_key)

    def generate_character_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
    ) -> Optional[pathlib.Path]:
        """Generate character image using Gemini."""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[prompt],
            )

            # Extract image from response
            for part in response.candidates[0].content.parts:  # type: ignore[index]
                if getattr(part, "inline_data", None) and getattr(
                    part.inline_data, "data", None
                ):
                    img = Image.open(io.BytesIO(part.inline_data.data))
                    img.save(output_path)
                    logger.info(f"✓ Gemini: Saved image to {output_path}")
                    return output_path

            logger.warning("Gemini returned no image data")
            return None

        except Exception as e:
            logger.error(f"Gemini image generation failed: {e}")
            return None

    def generate_scene_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
    ) -> Optional[pathlib.Path]:
        """Generate scene image with Gemini (supports reference images)."""
        from google.genai import types as genai_config  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        try:
            content_parts = [prompt]

            # Add reference images if provided (Gemini's strength)
            if reference_images:
                for ref_path in reference_images:
                    if ref_path.exists():
                        with open(ref_path, "rb") as f:
                            image_data = f.read()
                        content_parts.append(
                            genai_types.Part.from_bytes(
                                data=image_data, mime_type="image/png"
                            )
                        )
                logger.info(f"✓ Gemini: Using {len(reference_images)} reference images")

            config = genai_config.GenerateContentConfig(
                response_modalities=["image"],
            )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=content_parts,
                config=config,
            )

            # Extract and save image
            for part in response.candidates[0].content.parts:  # type: ignore[index]
                if getattr(part, "inline_data", None) and getattr(
                    part.inline_data, "data", None
                ):
                    img = Image.open(io.BytesIO(part.inline_data.data))
                    img.save(output_path)
                    logger.info(f"✓ Gemini: Saved scene image to {output_path}")
                    return output_path

            logger.warning("Gemini returned no scene image")
            return None

        except Exception as e:
            logger.error(f"Gemini scene generation failed: {e}")
            return None


class FluxKontextImageGenerator(ImageGenerator):
    """Image generator using FLUX.1 Kontext [dev] - free, local, character-consistent.

    FLUX.1 Kontext [dev] is an open-weights 12B rectified-flow transformer that
    supports *in-context* generation: you give it one or more reference images
    plus a text instruction, and it keeps characters, styles and objects
    consistent across new scenes WITHOUT any finetuning. This is the same
    capability the Gemini backend relies on, but it runs entirely on your own
    machine for free.

    - Character portraits: pure text-to-image (no reference needed).
    - Scenes: pass the party's character portraits as references so the same
      characters reappear consistently. Multiple references are stitched into a
      single conditioning image (Kontext's recommended multi-reference approach).
    """

    # Shared art-style anchor appended to every prompt so all images —
    # portraits and scenes alike — share the same visual language.
    # Named style anchors ("Hearthstone", "World of Warcraft") are far more
    # effective than adjective lists because FLUX has seen millions of labeled
    # examples and can reproduce the exact aesthetic on demand.
    GAME_ART_STYLE = (
        "Hearthstone card art, World of Warcraft Blizzard Entertainment concept art, "
        "vibrant saturated colors, hand-painted digital illustration, visible painterly "
        "brushstrokes, exaggerated heroic proportions, chunky oversized armor and "
        "weapons, bold clear silhouette with strong outline, warm inviting lighting, "
        "bright warm face illumination, strong rim lighting separating figure from "
        "background, whimsical light-hearted fantasy tone, no photorealism, no text, "
        "no watermarks"
    )

    def __init__(self):
        try:
            import torch
            from diffusers import FluxKontextPipeline

            logger.info(
                f"Initializing FLUX Kontext model: {settings.FLUX_KONTEXT_MODEL}"
            )

            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")

            # Use device-appropriate dtype to balance quality and memory.
            if self.device == "cpu":
                self.torch_dtype = torch.float32
            elif self.device == "mps":
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.bfloat16

            use_gguf = bool(
                settings.FLUX_KONTEXT_GGUF_TRANSFORMER
                and settings.FLUX_KONTEXT_GGUF_T5
            )

            hf_token = settings.HUGGING_FACE_HUB_TOKEN or None

            if use_gguf:
                self.pipe = self._load_gguf_pipeline(FluxKontextPipeline)
            else:
                self.pipe = FluxKontextPipeline.from_pretrained(
                    settings.FLUX_KONTEXT_MODEL,
                    torch_dtype=self.torch_dtype,
                    token=hf_token,
                )

                # Runtime quantization via optimum-quanto (fallback when GGUF not set).
                if settings.IMAGE_QUANTIZATION != "none":
                    try:
                        from optimum.quanto import freeze, qint4, qint8, quantize

                        weight_dtype = (
                            qint8 if settings.IMAGE_QUANTIZATION == "int8" else qint4
                        )
                        logger.info(
                            f"Quantizing transformer to {settings.IMAGE_QUANTIZATION}..."
                        )
                        quantize(self.pipe.transformer, weights=weight_dtype)
                        freeze(self.pipe.transformer)
                        logger.info(
                            f"✓ Transformer quantized to {settings.IMAGE_QUANTIZATION}"
                        )
                    except ImportError:
                        logger.warning(
                            "optimum-quanto not installed — skipping quantization. "
                            "Run: pip install optimum-quanto"
                        )

            self.cpu_offload_enabled = False

            # Memory management.
            #
            # GGUF (Q8_0 transformer ~12.7 GB + Q5_K_M T5 ~3.4 GB = ~20 GB total)
            # fits in 32 GB on paper, but the attention tensors during the reference
            # scene path (doubled sequence length) push peak MPS usage over the
            # watermark with T5 still resident. Model-level CPU offload solves this:
            # T5 moves to CPU after text encoding, so peak MPS during the 15
            # denoising steps = transformer only (~12.7 GB) + ~3 GB attention = ~16 GB.
            #
            # Full bfloat16 (~34 GB) exceeds 32 GB and requires sequential CPU
            # offload, which pages one layer at a time and is very slow.
            quantized = use_gguf or settings.IMAGE_QUANTIZATION != "none"

            if quantized and self.device == "mps":
                self.pipe.enable_model_cpu_offload(device=self.device)
                self.cpu_offload_enabled = True
                logger.info(
                    "✓ GGUF on MPS: model-level CPU offload enabled "
                    "(T5 moves to CPU after encoding, transformer stays on MPS)"
                )
            elif settings.IMAGE_ENABLE_CPU_OFFLOAD and self.device != "cpu":
                try:
                    if self.device == "mps":
                        self.pipe.enable_sequential_cpu_offload(device=self.device)
                        logger.info(
                            "✓ Enabled sequential CPU offload (minimal-memory mode)"
                        )
                    else:
                        self.pipe.enable_model_cpu_offload(device=self.device)
                        logger.info("✓ Enabled model CPU offload (low-memory mode)")
                    self.cpu_offload_enabled = True
                except Exception as offload_err:
                    logger.warning(
                        f"CPU offload unavailable ({offload_err}); "
                        f"trying direct {self.device} placement"
                    )
                    self.pipe = self.pipe.to(self.device)
            else:
                self.pipe = self.pipe.to(self.device)

            if self.device == "mps" and not self.cpu_offload_enabled:
                self.pipe.enable_attention_slicing()
                logger.info("✓ Enabled attention slicing for MPS")

            # VAE slicing reduces peak memory during decode without changing
            # tensor layout. We deliberately do NOT enable VAE tiling: on MPS,
            # tiling slices the input into non-contiguous spatial views, which
            # makes conv2d in the VAE encoder fail with a "view size is not
            # compatible" error in the reference-image path.
            try:
                self.pipe.vae.enable_slicing()
                if self.device == "mps":
                    # Ensure tiling is off (it may default on in some configs).
                    self.pipe.vae.disable_tiling()
                else:
                    self.pipe.vae.enable_tiling()
            except Exception:
                pass

            logger.info("✓ FLUX Kontext pipeline ready")

        except ImportError as e:
            raise ImportError(
                "FLUX Kontext backend requires a recent diffusers:\n"
                "  pip install -U 'diffusers>=0.34.0' torch transformers accelerate "
                "safetensors sentencepiece protobuf\n"
                f"Error: {e}"
            )

    def _load_gguf_pipeline(self, pipeline_cls):
        """Load pipeline using pre-quantized GGUF files for transformer and T5."""
        import torch
        from diffusers import FluxTransformer2DModel
        from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download
        from transformers import T5EncoderModel

        hf_token = settings.HUGGING_FACE_HUB_TOKEN or None
        t_repo, t_file = settings.FLUX_KONTEXT_GGUF_TRANSFORMER.split(":", 1)
        e_repo, e_file = settings.FLUX_KONTEXT_GGUF_T5.split(":", 1)

        logger.info(f"Loading GGUF transformer: {t_file} from {t_repo}")
        transformer_path = hf_hub_download(repo_id=t_repo, filename=t_file, token=hf_token)
        # FluxTransformer2DModel.load_config needs the transformer subfolder, not the
        # pipeline root. Resolve the local cache path so no extra network call is needed.
        import os
        transformer_config_dir = os.path.dirname(
            hf_hub_download(
                repo_id=settings.FLUX_KONTEXT_MODEL,
                filename="transformer/config.json",
                token=hf_token,
            )
        )
        transformer = FluxTransformer2DModel.from_single_file(
            transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.torch_dtype),
            torch_dtype=self.torch_dtype,
            config=transformer_config_dir,
            token=hf_token,
        )
        # GGUF Kontext models are quantized with in_channels=128 (the actual runtime
        # value after noise+reference concatenation), but FluxKontextPipeline expects
        # in_channels=64 and handles the concatenation internally. Mismatch causes
        # num_channels_latents=32 instead of 16, breaking _pack_latents for reference images.
        if transformer.config.in_channels == 128:
            transformer.config.in_channels = 64
            logger.info("✓ Patched GGUF transformer in_channels: 128 → 64")
        logger.info("✓ GGUF transformer loaded")

        logger.info(f"Loading GGUF T5 encoder: {e_file} from {e_repo}")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            e_repo,
            gguf_file=e_file,
            torch_dtype=self.torch_dtype,
            token=hf_token,
        )
        logger.info("✓ GGUF T5 encoder loaded")

        logger.info("Assembling pipeline (CLIP + VAE from original repo)...")
        pipe = pipeline_cls.from_pretrained(
            settings.FLUX_KONTEXT_MODEL,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=self.torch_dtype,
            token=hf_token,
        )
        logger.info("✓ GGUF pipeline assembled")
        return pipe

    def _get_device(self) -> str:
        """Determine the best device for inference."""
        import torch

        if settings.IMAGE_DEVICE != "auto":
            return settings.IMAGE_DEVICE

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("No GPU detected, using CPU (will be very slow)")
            return "cpu"

    def _make_generator(self):
        """Create a torch generator on the correct device for reproducibility."""
        import torch

        # CPU offload keeps inputs on CPU; otherwise use the pipeline device.
        gen_device = "cpu" if self.cpu_offload_enabled else self.device
        try:
            return torch.Generator(device=gen_device)
        except Exception:
            return torch.Generator()

    @staticmethod
    def _shorten_prompt(text: str, max_words: int = 300) -> str:
        """Trim prompts to stay within T5's 512-token limit (~380 words).

        FLUX uses two encoders: CLIP (77-token hard limit, ~55 words) and T5-XXL
        (512 tokens, ~380 words).  T5 is the primary semantic encoder — all scene
        detail, character descriptions, and action live here.  CLIP truncation is
        harmless: it only loses broad stylistic cues at the end of the prompt.
        Cap at 300 words to leave room for the fixed prefix/suffix we append.
        """
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    @staticmethod
    def _stitch_reference_images(
        reference_images: List[pathlib.Path],
    ) -> Optional[Image.Image]:
        """Combine multiple reference portraits into one conditioning image.

        Kontext [dev] was trained primarily on a single reference image, so the
        robust way to provide several characters is to place them side by side
        in one image. We normalize each portrait to the same height and tile
        them horizontally.
        """
        loaded: List[Image.Image] = []
        for ref_path in reference_images:
            if not ref_path.exists():
                logger.warning(f"Reference image not found: {ref_path}")
                continue
            try:
                loaded.append(Image.open(ref_path).convert("RGB"))
            except Exception as e:
                logger.warning(f"Failed to load reference image {ref_path}: {e}")

        if not loaded:
            return None
        if len(loaded) == 1:
            return loaded[0]

        target_height = 512
        resized: List[Image.Image] = []
        for img in loaded:
            ratio = target_height / img.height
            new_width = max(1, int(img.width * ratio))
            resized.append(
                img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            )

        total_width = sum(img.width for img in resized)
        canvas = Image.new("RGB", (total_width, target_height), (255, 255, 255))
        x_offset = 0
        for img in resized:
            canvas.paste(img, (x_offset, 0))
            x_offset += img.width

        return canvas

    def generate_character_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
    ) -> Optional[pathlib.Path]:
        """Generate a character portrait via text-to-image (no reference)."""
        try:
            enhanced_prompt = (
                f"{self._shorten_prompt(prompt)} | "
                f"full-body or 3/4 view character portrait, neutral background. "
                f"{self.GAME_ART_STYLE}"
            )

            logger.info("FLUX Kontext: Generating character portrait...")

            image = self.pipe(
                prompt=enhanced_prompt,
                guidance_scale=settings.FLUX_KONTEXT_GUIDANCE_SCALE,
                num_inference_steps=settings.IMAGE_NUM_INFERENCE_STEPS,
                height=1024,
                width=768,
                max_sequence_length=256,
                generator=self._make_generator(),
            ).images[0]

            image.save(output_path)
            logger.info(f"✓ FLUX Kontext: Saved character image to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"FLUX Kontext character generation failed: {e}")
            return None

    def generate_scene_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
    ) -> Optional[pathlib.Path]:
        """Generate a scene image, keeping characters consistent via references.

        Multiple reference portraits are stitched side-by-side into a single
        512×512-per-character composite before being passed to the pipeline.
        FluxKontextPipeline only accepts a single image tensor; passing a Python
        list triggers a batch-generation path that crashes in _pack_latents.

        Each portrait is resized to 512×512 so the combined reference stays at
        a FLUX-friendly size (e.g. 1024×512 for two characters).
        """
        try:
            if reference_images:
                SLOT = 512  # px per character slot — keeps total ref image compact
                slots = [
                    Image.open(p).convert("RGB").resize((SLOT, SLOT), Image.LANCZOS)
                    for p in reference_images
                ]
                if len(slots) == 1:
                    reference = slots[0]
                else:
                    # Side-by-side strip; thin dark separator between characters.
                    gap = 4
                    total_w = SLOT * len(slots) + gap * (len(slots) - 1)
                    reference = Image.new("RGB", (total_w, SLOT), (20, 20, 20))
                    x = 0
                    for slot in slots:
                        reference.paste(slot, (x, 0))
                        x += SLOT + gap

                n = len(reference_images)
                enhanced_prompt = (
                    f"Keep all {n} reference character(s) consistent — same face, "
                    "hair, clothing and gear. "
                    f"{self._shorten_prompt(prompt)}. "
                    f"Wide cinematic composition, landscape orientation. {self.GAME_ART_STYLE}"
                )
                logger.info(
                    f"FLUX Kontext: Generating scene with {n} reference image(s) "
                    f"(stitched to {reference.width}×{reference.height})"
                )
                result = self.pipe(
                    image=reference,
                    prompt=enhanced_prompt,
                    guidance_scale=settings.FLUX_KONTEXT_GUIDANCE_SCALE,
                    num_inference_steps=settings.IMAGE_NUM_INFERENCE_STEPS,
                    height=768,
                    width=1024,
                    max_sequence_length=256,
                    generator=self._make_generator(),
                ).images[0]
            else:
                enhanced_prompt = (
                    f"{self._shorten_prompt(prompt)}. "
                    f"Wide cinematic composition, landscape orientation. {self.GAME_ART_STYLE}"
                )
                logger.info("FLUX Kontext: Generating scene (no references)...")
                result = self.pipe(
                    prompt=enhanced_prompt,
                    guidance_scale=settings.FLUX_KONTEXT_GUIDANCE_SCALE,
                    num_inference_steps=settings.IMAGE_NUM_INFERENCE_STEPS,
                    height=768,
                    width=1024,
                    max_sequence_length=256,
                    generator=self._make_generator(),
                ).images[0]

            result.save(output_path)
            logger.info(f"✓ FLUX Kontext: Saved scene image to {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"FLUX Kontext scene generation failed: {e}")
            return None


class FluxKleinImageGenerator(ImageGenerator):
    """Image generator using FLUX.2 Klein [9B or 4B] — native multi-reference.

    Key differences from FLUX.1 Kontext:
    - Native multi-image: pass image=[img1, img2, ...] directly, no stitching.
    - 9B distilled model runs in just 4 steps vs 30 for Kontext.
    - Fits on RTX 3080 with CPU offload (BF16 ~18 GB paged through 10 GB VRAM).
    - For best speed/VRAM: use GGUF Q4_K_M (~5.5 GB, full VRAM residency).
    """

    GAME_ART_STYLE = FluxKontextImageGenerator.GAME_ART_STYLE

    def __init__(self):
        try:
            import torch
            from diffusers import Flux2KleinPipeline

            logger.info(f"Initializing FLUX.2 Klein: {settings.FLUX_KLEIN_MODEL}")
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")
            self.torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

            hf_token = settings.HUGGING_FACE_HUB_TOKEN or None
            self.pipe = Flux2KleinPipeline.from_pretrained(
                settings.FLUX_KLEIN_MODEL,
                torch_dtype=self.torch_dtype,
                token=hf_token,
            )

            # Runtime int8/int4 quantization via optimum-quanto.
            # Klein 9B bfloat16 transformer is ~18 GB; int8 cuts it to ~9.5 GB so the
            # full pipeline (~21 GB with T5) fits in 32 GB unified memory, enabling
            # model-level CPU offload instead of slow sequential layer-paging.
            quantized = settings.IMAGE_QUANTIZATION != "none"
            if quantized:
                try:
                    from optimum.quanto import freeze, qint4, qint8, quantize

                    weight_dtype = (
                        qint8 if settings.IMAGE_QUANTIZATION == "int8" else qint4
                    )
                    logger.info(
                        f"Quantizing Klein transformer to {settings.IMAGE_QUANTIZATION}..."
                    )
                    quantize(self.pipe.transformer, weights=weight_dtype)
                    freeze(self.pipe.transformer)
                    logger.info(
                        f"✓ Klein transformer quantized to {settings.IMAGE_QUANTIZATION}"
                    )
                except ImportError:
                    logger.warning(
                        "optimum-quanto not installed — skipping quantization. "
                        "Run: pip install optimum-quanto"
                    )
                    quantized = False

            self.cpu_offload_enabled = False
            if settings.IMAGE_ENABLE_CPU_OFFLOAD and self.device != "cpu":
                try:
                    # With int8 (~9.5 GB transformer), model-level offload is enough:
                    # T5 moves to CPU after text encoding so peak MPS during the 4
                    # denoising steps is just the quantized transformer + activations.
                    # Without quantization, bfloat16 Klein (~18 GB) also uses model-level
                    # offload (Klein only needs 4 steps so it's still faster than Kontext).
                    self.pipe.enable_model_cpu_offload(device=self.device)
                    self.cpu_offload_enabled = True
                    mode = "int8 model" if quantized else "bfloat16 model"
                    logger.info(f"✓ Model CPU offload enabled ({mode})")
                except Exception as offload_err:
                    logger.warning(
                        f"CPU offload unavailable ({offload_err}); "
                        f"placing directly on {self.device}"
                    )
                    self.pipe = self.pipe.to(self.device)
            else:
                self.pipe = self.pipe.to(self.device)

            try:
                self.pipe.vae.enable_slicing()
                if self.device != "mps":
                    self.pipe.vae.enable_tiling()
            except Exception:
                pass

            logger.info("✓ FLUX.2 Klein pipeline ready")

        except ImportError as e:
            raise ImportError(
                "FLUX.2 Klein requires diffusers>=0.38.0:\n"
                "  pip install -U diffusers\n"
                f"Error: {e}"
            ) from e

    def _get_device(self) -> str:
        import torch

        if settings.IMAGE_DEVICE != "auto":
            return settings.IMAGE_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        logger.warning("No GPU detected, using CPU (will be very slow)")
        return "cpu"

    def _make_generator(self):
        import torch

        gen_device = "cpu" if self.cpu_offload_enabled else self.device
        try:
            return torch.Generator(device=gen_device)
        except Exception:
            return torch.Generator()

    def generate_character_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
    ) -> Optional[pathlib.Path]:
        try:
            enhanced_prompt = (
                f"{prompt} | full-body or 3/4 view character portrait, neutral background. "
                f"{self.GAME_ART_STYLE}"
            )
            logger.info("FLUX.2 Klein: Generating character portrait...")
            image = self.pipe(
                prompt=enhanced_prompt,
                guidance_scale=settings.FLUX_KLEIN_GUIDANCE_SCALE,
                num_inference_steps=settings.FLUX_KLEIN_NUM_INFERENCE_STEPS,
                height=1024,
                width=768,
                generator=self._make_generator(),
            ).images[0]
            image.save(output_path)
            logger.info(f"✓ FLUX.2 Klein: Saved character image to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"FLUX.2 Klein character generation failed: {e}")
            return None

    def generate_scene_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
    ) -> Optional[pathlib.Path]:
        try:
            if reference_images:
                # MPS has no flash-attention kernel and materializes the full
                # attention matrix in one allocation, so multiple 1024px
                # references OOM the GPU — use a smaller reference size there.
                # CUDA stays at full size. Attention cost scales ~quadratically
                # with this, so a smaller size cuts peak memory sharply.
                ref_size = (
                    settings.FLUX_KLEIN_MPS_REFERENCE_SIZE
                    if self.device == "mps"
                    else settings.FLUX_KLEIN_REFERENCE_SIZE
                )
                refs = []
                for p in reference_images:
                    p = pathlib.Path(p)
                    if p.exists():
                        # Klein expects square-ish reference images.
                        refs.append(
                            Image.open(p)
                            .convert("RGB")
                            .resize((ref_size, ref_size), Image.LANCZOS)
                        )
                    else:
                        logger.warning(f"Reference image not found: {p}")

                n = len(refs)
                enhanced_prompt = (
                    f"Keep all {n} reference character(s) consistent — same face, "
                    f"hair, clothing and gear. {prompt}. "
                    f"Wide cinematic composition, landscape orientation. {self.GAME_ART_STYLE}"
                )
                logger.info(
                    f"FLUX.2 Klein: Generating scene with {n} native reference image(s)..."
                )
                result = self.pipe(
                    prompt=enhanced_prompt,
                    image=refs,
                    guidance_scale=settings.FLUX_KLEIN_GUIDANCE_SCALE,
                    num_inference_steps=settings.FLUX_KLEIN_NUM_INFERENCE_STEPS,
                    height=768,
                    width=1024,
                    generator=self._make_generator(),
                ).images[0]
            else:
                enhanced_prompt = (
                    f"{prompt}. Wide cinematic composition, landscape orientation. "
                    f"{self.GAME_ART_STYLE}"
                )
                logger.info("FLUX.2 Klein: Generating scene (no references)...")
                result = self.pipe(
                    prompt=enhanced_prompt,
                    guidance_scale=settings.FLUX_KLEIN_GUIDANCE_SCALE,
                    num_inference_steps=settings.FLUX_KLEIN_NUM_INFERENCE_STEPS,
                    height=768,
                    width=1024,
                    generator=self._make_generator(),
                ).images[0]

            result.save(output_path)
            logger.info(f"✓ FLUX.2 Klein: Saved scene to {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"FLUX.2 Klein scene generation failed: {e}")
            return None


def get_image_generator() -> ImageGenerator:
    """Factory function to get the configured image generator backend.

    Returns:
        ImageGenerator instance based on IMAGE_PROVIDER setting

    Raises:
        ValueError: If IMAGE_PROVIDER is invalid or backend initialization fails
    """
    provider = settings.IMAGE_PROVIDER.lower()

    logger.info(f"Initializing image generator: {provider}")

    if provider == "flux-kontext":
        return FluxKontextImageGenerator()
    elif provider == "flux-klein":
        return FluxKleinImageGenerator()
    elif provider == "gemini":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when IMAGE_PROVIDER='gemini'.")
        return GeminiImageGenerator(api_key=settings.GOOGLE_API_KEY)
    else:
        raise ValueError(
            f"Invalid IMAGE_PROVIDER: {provider}. "
            f"Must be 'flux-kontext', 'flux-klein', or 'gemini'"
        )

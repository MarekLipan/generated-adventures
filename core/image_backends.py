"""Image generation backends for different providers."""

import io
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)


# --- Art styles -----------------------------------------------------------------
# Named style anchors ("Hearthstone", "ArtStation") are far more effective than
# adjective lists because FLUX has seen millions of labeled examples and can
# reproduce the exact aesthetic on demand. Each game picks one of these; the
# chosen suffix is appended to every prompt (portraits and scenes alike) so the
# whole adventure shares one visual language.

PAINTERLY_HERO_STYLE = (
    "Hearthstone card art, World of Warcraft Blizzard Entertainment concept art, "
    "vibrant saturated colors, hand-painted digital illustration, visible painterly "
    "brushstrokes, exaggerated heroic proportions, chunky oversized armor and "
    "weapons, bold clear silhouette with strong outline, warm inviting lighting, "
    "bright warm face illumination, strong rim lighting separating figure from "
    "background, whimsical light-hearted fantasy tone, no photorealism, no text, "
    "no watermarks"
)

ARTSTATION_REALISM_STYLE = (
    "highly detailed semi-realistic digital fantasy painting, ArtStation trending, "
    "cinematic dramatic lighting, realistic materials and rendering, concept art, "
    "intricate detail, lifelike faces, no text, no watermark"
)

ART_STYLES: dict[str, str] = {
    "painterly_hero": PAINTERLY_HERO_STYLE,
    "artstation_realism": ARTSTATION_REALISM_STYLE,
}

DEFAULT_ART_STYLE = "painterly_hero"


def resolve_art_style(art_style: Optional[str]) -> Optional[str]:
    """Map an art-style key to its prompt suffix.

    Returns the suffix string for a known key, or None if `art_style` is None or
    unrecognized (callers then fall back to their own default suffix).
    """
    if not art_style:
        return None
    return ART_STYLES.get(art_style.lower())


def _filter_cloud_references(
    reference_images: Optional[List[pathlib.Path]],
) -> List[pathlib.Path]:
    """Drop reference images before sending them to a CLOUD backend, unless allowed.

    Player profile photos are sensitive; when ALLOW_CLOUD_IMAGE_UPLOAD is False
    (default), a cloud image backend must not upload any reference image. Local
    backends run on-device and must NOT call this.
    """
    refs = list(reference_images or [])
    if refs and not settings.ALLOW_CLOUD_IMAGE_UPLOAD:
        logger.warning(
            "Cloud image backend: dropping %d reference image(s) to keep player "
            "photos on-device (set ALLOW_CLOUD_IMAGE_UPLOAD=true to override).",
            len(refs),
        )
        return []
    return refs


def _load_portrait_reference(path: pathlib.Path, max_side: int = 1024) -> "Image.Image":
    """Load a photo reference for a character portrait.

    Preserves aspect ratio (unlike scene references, which are squared) and caps
    the longest side so facial detail survives while staying FLUX-friendly.
    """
    img = Image.open(path).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
        )
    return img


class ImageGenerator(ABC):
    """Abstract base class for image generation backends."""

    def _style_suffix(self, art_style: Optional[str]) -> str:
        """Resolve the style suffix to append to prompts for a given style key.

        Falls back to this backend's default GAME_ART_STYLE when the key is None
        or unknown. Backends without a GAME_ART_STYLE attribute get an empty
        suffix.
        """
        resolved = resolve_art_style(art_style)
        if resolved is not None:
            return resolved
        return getattr(self, "GAME_ART_STYLE", "")

    @abstractmethod
    def generate_character_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        """Generate a character portrait image.

        Args:
            prompt: Detailed text prompt for image generation
            output_path: Where to save the generated image
            reference_images: Optional reference images (e.g. a player's profile
                photo) so the generated hero resembles the person/subject shown.
            art_style: Optional art-style key ("painterly_hero",
                "artstation_realism"); None uses the backend default.

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
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        """Generate a scene image with optional character references.

        Args:
            prompt: Detailed text prompt for scene composition
            output_path: Where to save the generated image
            reference_images: Optional list of character/asset images for consistency
            art_style: Optional art-style key ("painterly_hero",
                "artstation_realism"); None uses the backend default.

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
        reference_images: Optional[List[pathlib.Path]] = None,
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        """Generate character image using Gemini."""
        try:
            from google.genai import types as genai_types  # type: ignore

            suffix = self._style_suffix(art_style)
            content_parts = [f"{prompt} {suffix}".strip()]

            # Reference images (e.g. a player's profile photo) are only uploaded to
            # this cloud backend when explicitly allowed; otherwise dropped.
            for ref_path in _filter_cloud_references(reference_images):
                ref_path = pathlib.Path(ref_path)
                if ref_path.exists():
                    content_parts.append(
                        genai_types.Part.from_bytes(
                            data=ref_path.read_bytes(), mime_type="image/png"
                        )
                    )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=content_parts,
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
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        """Generate scene image with Gemini (supports reference images)."""
        from google.genai import types as genai_config  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        try:
            suffix = self._style_suffix(art_style)
            content_parts = [f"{prompt} {suffix}".strip()]

            # Reference images are only uploaded to this cloud backend when allowed.
            refs = _filter_cloud_references(reference_images)
            if refs:
                for ref_path in refs:
                    if ref_path.exists():
                        with open(ref_path, "rb") as f:
                            image_data = f.read()
                        content_parts.append(
                            genai_types.Part.from_bytes(
                                data=image_data, mime_type="image/png"
                            )
                        )
                logger.info(f"✓ Gemini: Using {len(refs)} reference images")

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

    # Default art-style anchor (Painterly Hero). A per-game style can override
    # this per call via the `art_style` argument on the generate_* methods.
    GAME_ART_STYLE = PAINTERLY_HERO_STYLE

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
        reference_images: Optional[List[pathlib.Path]] = None,
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        """Generate a character portrait.

        With a reference image (e.g. a player's profile photo), Kontext keeps the
        subject's likeness while restyling them as the described hero. Without
        one, it's plain text-to-image.
        """
        try:
            suffix = self._style_suffix(art_style)
            ref = None
            if reference_images:
                p = pathlib.Path(reference_images[0])
                if p.exists():
                    ref = _load_portrait_reference(p)
                else:
                    logger.warning(f"Reference image not found: {p}")

            enhanced_prompt = (
                f"{self._shorten_prompt(prompt)} | "
                f"full-body or 3/4 view character portrait, neutral background. "
                f"{suffix}"
            )

            logger.info(
                "FLUX Kontext: Generating character portrait%s...",
                " from reference photo" if ref is not None else "",
            )

            kwargs = dict(
                prompt=enhanced_prompt,
                guidance_scale=settings.FLUX_KONTEXT_GUIDANCE_SCALE,
                num_inference_steps=settings.IMAGE_NUM_INFERENCE_STEPS,
                height=1024,
                width=768,
                max_sequence_length=256,
                generator=self._make_generator(),
            )
            if ref is not None:
                kwargs["image"] = ref
            image = self.pipe(**kwargs).images[0]

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
        art_style: Optional[str] = None,
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
            suffix = self._style_suffix(art_style)
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
                    f"Wide cinematic composition, landscape orientation. {suffix}"
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
                    f"Wide cinematic composition, landscape orientation. {suffix}"
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

    Memory strategy (both the 9B transformer and Qwen3-8B text encoder are large):
    - On CUDA, quantize BOTH modules with bitsandbytes (FLUX_KLEIN_QUANTIZATION,
      default "nf4") so model-level CPU offload keeps the active module resident
      — ~12-20 s/image on a 10 GB RTX 3080 at bf16-equivalent quality.
    - Otherwise (MPS, or bitsandbytes unavailable) fall back to a bf16 pipeline
      with model-level offload where a module fits, else sequential offload.
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
            self.cpu_offload_enabled = False

            # Fast path: bitsandbytes-quantized transformer + text encoder on CUDA.
            # Both modules are loaded already quantized, so model-level CPU offload
            # only ever moves a small (~5 GB) module onto the GPU — no crash, and the
            # active module stays resident during compute (bitsandbytes has fast 4-bit
            # CUDA kernels, unlike GGUF/quanto on this build).
            klein_quant = settings.FLUX_KLEIN_QUANTIZATION.lower()
            use_bnb = self.device == "cuda" and klein_quant in ("nf4", "int8")
            if use_bnb:
                try:
                    self.pipe = self._load_bnb_pipeline(
                        Flux2KleinPipeline, klein_quant, hf_token
                    )
                    self.pipe.enable_model_cpu_offload(device=self.device)
                    self.cpu_offload_enabled = True
                    logger.info(
                        f"✓ Klein loaded via bitsandbytes ({klein_quant}) "
                        "with model CPU offload"
                    )
                except Exception as bnb_err:
                    logger.warning(
                        f"bitsandbytes quantization unavailable ({bnb_err}); "
                        "falling back to bfloat16 pipeline"
                    )
                    use_bnb = False

            if use_bnb:
                # Quantized pipeline is fully configured above; skip the bf16 path.
                self._finalize_vae()
                logger.info("✓ FLUX.2 Klein pipeline ready")
                return

            # Fallback path: full bfloat16 pipeline with CPU offload.
            self.pipe = Flux2KleinPipeline.from_pretrained(
                settings.FLUX_KLEIN_MODEL,
                torch_dtype=self.torch_dtype,
                token=hf_token,
            )

            if settings.IMAGE_ENABLE_CPU_OFFLOAD and self.device != "cpu":
                try:
                    # bfloat16 Klein loads TWO oversized modules: the 9B transformer
                    # (~18 GB) and a Qwen3-8B text encoder (~16 GB). model-level offload
                    # moves whole modules onto the GPU one at a time, so it only works
                    # when a single module fits in VRAM. On a small card neither does:
                    # the giant .to() device move overcommits VRAM and hard-crashes
                    # (Windows access violation). enable_sequential_cpu_offload() pages
                    # weights at the layer level, running on ~2 GB VRAM (slower per
                    # step, but Klein needs only 4 steps). For fast generation on a
                    # small card, prefer FLUX_KLEIN_QUANTIZATION=nf4 (the CUDA path
                    # above) over this bf16 fallback.
                    vram_gb = self._device_vram_gb()
                    fits_whole_module = vram_gb is None or vram_gb >= 21.0

                    if not fits_whole_module:
                        self.pipe.enable_sequential_cpu_offload(device=self.device)
                        logger.info(
                            f"✓ Sequential CPU offload enabled (bfloat16; VRAM "
                            f"{vram_gb:.0f} GB too small for whole-module offload)"
                        )
                    else:
                        self.pipe.enable_model_cpu_offload(device=self.device)
                        logger.info("✓ Model CPU offload enabled (bfloat16 model)")
                    self.cpu_offload_enabled = True
                except Exception as offload_err:
                    logger.warning(
                        f"CPU offload unavailable ({offload_err}); "
                        f"placing directly on {self.device}"
                    )
                    self.pipe = self.pipe.to(self.device)
            else:
                self.pipe = self.pipe.to(self.device)

            self._finalize_vae()
            logger.info("✓ FLUX.2 Klein pipeline ready")

        except ImportError as e:
            raise ImportError(
                "FLUX.2 Klein requires diffusers>=0.38.0:\n"
                "  pip install -U diffusers\n"
                f"Error: {e}"
            ) from e

    def _load_bnb_pipeline(self, pipeline_cls, klein_quant: str, hf_token):
        """Load Klein with bitsandbytes-quantized transformer + text encoder.

        Both large modules are quantized on load so that model-level CPU offload
        only ever moves a small (~5 GB) module onto the GPU. The transformer is
        loaded and quantized to the GPU first so its host-RAM copy is freed before
        the 16 GB text encoder loads (avoids OOM on a 32 GB machine).
        """
        import gc

        import torch
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from diffusers import Flux2Transformer2DModel
        from transformers import AutoModel
        from transformers import BitsAndBytesConfig as TransformersBnbConfig

        load_in_4bit = klein_quant == "nf4"

        def _diffusers_cfg():
            if load_in_4bit:
                return DiffusersBnbConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                )
            return DiffusersBnbConfig(load_in_8bit=True)

        def _transformers_cfg():
            if load_in_4bit:
                return TransformersBnbConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                )
            return TransformersBnbConfig(load_in_8bit=True)

        model = settings.FLUX_KLEIN_MODEL

        logger.info(f"Loading Klein transformer ({klein_quant}) via bitsandbytes...")
        transformer = Flux2Transformer2DModel.from_pretrained(
            model,
            subfolder="transformer",
            quantization_config=_diffusers_cfg(),
            torch_dtype=self.torch_dtype,
            token=hf_token,
        )
        gc.collect()

        logger.info(f"Loading Klein text encoder ({klein_quant}) via bitsandbytes...")
        text_encoder = AutoModel.from_pretrained(
            model,
            subfolder="text_encoder",
            quantization_config=_transformers_cfg(),
            torch_dtype=self.torch_dtype,
            token=hf_token,
        )
        gc.collect()

        return pipeline_cls.from_pretrained(
            model,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=self.torch_dtype,
            token=hf_token,
        )

    def _finalize_vae(self) -> None:
        """Enable VAE slicing/tiling to reduce peak memory during decode."""
        try:
            self.pipe.vae.enable_slicing()
            if self.device != "mps":
                self.pipe.vae.enable_tiling()
        except Exception:
            pass

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

    def _device_vram_gb(self) -> Optional[float]:
        """Total VRAM of the active accelerator in GB, or None if unknown.

        Used to decide between model-level and sequential CPU offload. MPS shares
        unified memory with the host, so there is no dedicated-VRAM figure to gate
        on — return None there to keep the existing model-offload behaviour.
        """
        import torch

        try:
            if self.device == "cuda":
                return torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass
        return None

    def generate_character_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        """Generate a character portrait.

        With a reference image (e.g. a player's profile photo), Klein keeps the
        subject's likeness natively while restyling them as the described hero.
        Without one, it's plain text-to-image.
        """
        try:
            suffix = self._style_suffix(art_style)
            ref = None
            if reference_images:
                p = pathlib.Path(reference_images[0])
                if p.exists():
                    ref = _load_portrait_reference(p)
                else:
                    logger.warning(f"Reference image not found: {p}")

            enhanced_prompt = (
                f"{prompt} | full-body or 3/4 view character portrait, neutral background. "
                f"{suffix}"
            )
            logger.info(
                "FLUX.2 Klein: Generating character portrait%s...",
                " from reference photo" if ref is not None else "",
            )
            kwargs = dict(
                prompt=enhanced_prompt,
                guidance_scale=settings.FLUX_KLEIN_GUIDANCE_SCALE,
                num_inference_steps=settings.FLUX_KLEIN_NUM_INFERENCE_STEPS,
                height=1024,
                width=768,
                generator=self._make_generator(),
            )
            if ref is not None:
                kwargs["image"] = [ref]
            image = self.pipe(**kwargs).images[0]
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
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        try:
            suffix = self._style_suffix(art_style)
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
                    f"Wide cinematic composition, landscape orientation. {suffix}"
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
                    f"{suffix}"
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


class HttpImageGenerator(ImageGenerator):
    """Client backend that offloads generation to a remote image server.

    Talks to image_server.py (which hosts a warm local backend) over HTTP, so the
    game process holds no model and no GPU. The server may run on this machine
    (localhost) or another device on the LAN. Implements the same ImageGenerator
    interface, so the rest of the app is unchanged.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 300.0):
        import httpx

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"X-API-Key": api_key} if api_key else {}
        logger.info(f"Using remote image server at {self.base_url}")

    def _post(self, path: str, *, data: dict, files=None) -> Optional[bytes]:
        try:
            resp = self._httpx.post(
                f"{self.base_url}{path}",
                data=data,
                files=files,
                headers=self.headers,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.error(
                    f"Image server {path} returned {resp.status_code}: {resp.text[:200]}"
                )
                return None
            return resp.content
        except Exception as e:
            logger.error(f"Image server request to {path} failed: {e}")
            return None

    def generate_character_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        data = {"prompt": prompt}
        if art_style:
            data["art_style"] = art_style

        files = []
        open_handles = []
        try:
            # Only the first reference (the player photo) drives a portrait.
            for ref in (reference_images or [])[:1]:
                ref = pathlib.Path(ref)
                if not ref.exists():
                    logger.warning(f"Reference image not found, skipping: {ref}")
                    continue
                fh = ref.open("rb")
                open_handles.append(fh)
                files.append(("photo", (ref.name, fh, "image/png")))

            png = self._post(
                "/generate/character",
                data=data,
                files=files or None,
            )
        finally:
            for fh in open_handles:
                fh.close()

        if png is None:
            return None
        pathlib.Path(output_path).write_bytes(png)
        logger.info(f"✓ HTTP: Saved character image to {output_path}")
        return output_path

    def generate_scene_image(
        self,
        prompt: str,
        output_path: pathlib.Path,
        reference_images: Optional[List[pathlib.Path]] = None,
        art_style: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        data = {"prompt": prompt}
        if art_style:
            data["art_style"] = art_style

        files = []
        open_handles = []
        try:
            for ref in reference_images or []:
                ref = pathlib.Path(ref)
                if not ref.exists():
                    logger.warning(f"Reference image not found, skipping: {ref}")
                    continue
                fh = ref.open("rb")
                open_handles.append(fh)
                files.append(("references", (ref.name, fh, "image/png")))

            png = self._post(
                "/generate/scene",
                data=data,
                files=files or None,
            )
        finally:
            for fh in open_handles:
                fh.close()

        if png is None:
            return None
        pathlib.Path(output_path).write_bytes(png)
        logger.info(f"✓ HTTP: Saved scene image to {output_path}")
        return output_path


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
    elif provider == "http":
        return HttpImageGenerator(
            base_url=settings.IMAGE_SERVER_URL,
            api_key=settings.IMAGE_SERVER_API_KEY,
            timeout=settings.IMAGE_SERVER_TIMEOUT,
        )
    elif provider == "gemini":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when IMAGE_PROVIDER='gemini'.")
        return GeminiImageGenerator(api_key=settings.GOOGLE_API_KEY)
    else:
        raise ValueError(
            f"Invalid IMAGE_PROVIDER: {provider}. "
            f"Must be 'flux-kontext', 'flux-klein', 'http', or 'gemini'"
        )


def _local_backend_for_server() -> ImageGenerator:
    """Instantiate the local backend the image server should host.

    The server can't use provider 'http' (that would point at itself), so it maps
    'http' to the recommended local backend and otherwise honors IMAGE_PROVIDER.
    """
    provider = settings.IMAGE_PROVIDER.lower()
    if provider == "http":
        logger.info("IMAGE_PROVIDER='http' on the server; hosting 'flux-klein'.")
        return FluxKleinImageGenerator()
    return get_image_generator()

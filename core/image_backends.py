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
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.bfloat16

            self.pipe = FluxKontextPipeline.from_pretrained(
                settings.FLUX_KONTEXT_MODEL,
                torch_dtype=self.torch_dtype,
            )

            self.cpu_offload_enabled = False

            # Memory management.
            #
            # On Apple Silicon (MPS) the 12B transformer (~24 GB in fp16) plus
            # the T5/CLIP text encoders (~12 GB) cannot all stay resident within
            # the unified-memory budget, and model-level offload does not
            # reliably evict the encoders. Sequential CPU offload keeps only one
            # submodule on the accelerator at a time, giving the lowest possible
            # peak memory (slower, but it fits). On CUDA, the faster model-level
            # offload is sufficient.
            if settings.IMAGE_ENABLE_CPU_OFFLOAD and self.device != "cpu":
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
    def _shorten_prompt(text: str, max_words: int = 45) -> str:
        """Trim prompts to reduce CLIP truncation in reference-image mode."""
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
                f"{prompt} | full-body or 3/4 view character portrait, "
                "painterly fantasy art, high detail, clean simple background"
            )

            logger.info("FLUX Kontext: Generating character portrait...")

            image = self.pipe(
                prompt=enhanced_prompt,
                guidance_scale=settings.FLUX_KONTEXT_GUIDANCE_SCALE,
                num_inference_steps=settings.IMAGE_NUM_INFERENCE_STEPS,
                height=1024,
                width=768,
                max_sequence_length=77,
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
        """Generate a scene image, keeping characters consistent via references."""
        try:
            reference = (
                self._stitch_reference_images(reference_images)
                if reference_images
                else None
            )

            if reference is not None:
                # Ensure a standalone contiguous image object for downstream tensor conversion.
                reference = reference.copy()

                # Instruction-style prompt works best when conditioning on a
                # reference image with Kontext.
                compact_prompt = self._shorten_prompt(prompt)
                enhanced_prompt = (
                    "Use the reference character image and keep face, hair, "
                    "clothing and gear consistent. "
                    f"{compact_prompt}. Wide cinematic fantasy scene, dramatic lighting."
                )
                logger.info(
                    f"FLUX Kontext: Generating scene with "
                    f"{len(reference_images)} reference image(s) for consistency"
                )
                # The reference path concatenates the encoded image latents with
                # the noise latents, doubling the attention sequence length and
                # therefore peak memory. Use a more modest canvas here so it fits
                # within constrained (e.g. Apple Silicon MPS) memory budgets.
                result = self.pipe(
                    image=reference,
                    prompt=enhanced_prompt,
                    guidance_scale=settings.FLUX_KONTEXT_GUIDANCE_SCALE,
                    num_inference_steps=settings.IMAGE_NUM_INFERENCE_STEPS,
                    height=768,
                    width=1024,
                    max_sequence_length=77,
                    generator=self._make_generator(),
                ).images[0]
            else:
                enhanced_prompt = (
                    f"{prompt} | cinematic composition, dramatic lighting, "
                    "landscape orientation, painterly fantasy art"
                )
                logger.info("FLUX Kontext: Generating scene (no references)...")
                result = self.pipe(
                    prompt=enhanced_prompt,
                    guidance_scale=settings.FLUX_KONTEXT_GUIDANCE_SCALE,
                    num_inference_steps=settings.IMAGE_NUM_INFERENCE_STEPS,
                    height=768,
                    width=1344,
                    max_sequence_length=77,
                    generator=self._make_generator(),
                ).images[0]

            result.save(output_path)
            logger.info(f"✓ FLUX Kontext: Saved scene image to {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"FLUX Kontext scene generation failed: {e}")
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
    elif provider == "gemini":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when IMAGE_PROVIDER='gemini'.")
        return GeminiImageGenerator(api_key=settings.GOOGLE_API_KEY)
    else:
        raise ValueError(
            f"Invalid IMAGE_PROVIDER: {provider}. Must be 'flux-kontext', or 'gemini'"
        )

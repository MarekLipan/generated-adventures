#!/usr/bin/env python3
"""
Test script for image generation, focused on the FREE, local FLUX.1 Kontext [dev]
backend that keeps characters consistent across scenes via reference images.

Quick start (one-time setup):
    pip install -e ".[local-image]"

Then run:
    python test_image_generation.py

The most important test is #3 ("Consistency"): it generates a character portrait
and then a scene that reuses that portrait as a reference, so you can visually
confirm the SAME character appears in the scene.

NOTE: FLUX.1 Kontext [dev] is a 12B model. The first run downloads the weights
(~24GB) from Hugging Face and may require accepting the model license at
https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev and logging in with
`huggingface-cli login`. Generation is slow on CPU/Mac; a CUDA GPU is fastest.
"""

import logging
import pathlib
import sys

from core.config import settings
from core.image_backends import get_image_generator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Test output directory
TEST_OUTPUT_DIR = pathlib.Path("test_images")
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Shared paths so tests can build on each other
CHARACTER_PORTRAIT_PATH = TEST_OUTPUT_DIR / "character_portrait_elara.png"

# A realistic character prompt, similar to what the game produces
CHARACTER_PROMPT = (
    "Generate a high quality fantasy character portrait. "
    "Character Name: Elara Moonwhisper. "
    "Scenario: The Frozen Glacier Temple. "
    "Appearance: Tall elf with long silver hair, piercing blue eyes, wearing "
    "midnight blue robes embroidered with silver runes, carrying an ornate staff "
    "topped with a glowing blue crystal. "
    "Personality: Wise and mysterious, calm demeanor. "
    "Notable Equipment: Runed Staff, Crystal Amulet, Spellbook. "
    "Style: painterly, dramatic lighting, 3/4 view, no text."
)


def _get_generator():
    """Build the configured image generator (loads the model   on first call)."""
    return get_image_generator()


def test_character_portrait(generator=None):
    """Generate a sample fantasy character portrait (text-to-image)."""
    logger.info("\n" + "=" * 60)
    logger.info("🧙 TESTING CHARACTER PORTRAIT GENERATION")
    logger.info("=" * 60)

    generator = generator or _get_generator()
    result = generator.generate_character_image(
        prompt=CHARACTER_PROMPT, output_path=CHARACTER_PORTRAIT_PATH
    )

    if result:
        logger.info(f"✅ SUCCESS! Character portrait saved to: {result}")
        logger.info(f"   Open it: open {result}")
    else:
        logger.error("❌ Failed to generate character portrait")
    return result


def test_simple_scene(generator=None):
    """Generate a simple scene (no references) to sanity-check the backend."""
    logger.info("\n" + "=" * 60)
    logger.info("🌲 TESTING SIMPLE SCENE (no references)")
    logger.info("=" * 60)

    prompt = (
        "A mysterious forest clearing at twilight. Ancient oak trees surround a "
        "small stone circle. Magical fireflies float in the air, creating points "
        "of golden light. A narrow path leads into deeper woods. Mystical, "
        "peaceful atmosphere."
    )
    output_path = TEST_OUTPUT_DIR / "scene_forest_clearing.png"

    generator = generator or _get_generator()
    result = generator.generate_scene_image(
        prompt=prompt, output_path=output_path, reference_images=None
    )

    if result:
        logger.info(f"✅ SUCCESS! Simple scene saved to: {result}")
        logger.info(f"   Open it: open {result}")
    else:
        logger.error("❌ Failed to generate simple scene")
    return result


def test_character_consistency(generator=None):
    """THE KEY TEST: generate a scene that reuses the character portrait.

    This proves the backend keeps the SAME character consistent across images,
    which is the whole point of using FLUX.1 Kontext locally.
    """
    logger.info("\n" + "=" * 60)
    logger.info("🎭 TESTING CHARACTER CONSISTENCY (portrait -> scene)")
    logger.info("=" * 60)

    generator = generator or _get_generator()

    # Make sure we have a reference portrait to reuse.
    if not CHARACTER_PORTRAIT_PATH.exists():
        logger.info("No reference portrait yet — generating one first...")
        if not test_character_portrait(generator):
            logger.error("❌ Cannot run consistency test without a portrait")
            return None

    if settings.IMAGE_PROVIDER not in ("flux-kontext", "gemini"):
        logger.warning(
            f"⚠️  Provider '{settings.IMAGE_PROVIDER}' does not support reference "
            "images. Set IMAGE_PROVIDER=flux-kontext for character consistency."
        )

    prompt = (
        "Elara Moonwhisper, the silver-haired elf wizard, stands inside a vast "
        "frozen glacier temple. Massive ice pillars covered in glowing blue runes "
        "frame her. She raises her crystal-topped staff, casting soft blue light "
        "across the smooth ice floor. Ethereal, magical, cold atmosphere."
    )
    output_path = TEST_OUTPUT_DIR / "scene_with_reference_elara_temple.png"

    result = generator.generate_scene_image(
        prompt=prompt,
        output_path=output_path,
        reference_images=[CHARACTER_PORTRAIT_PATH],
    )

    if result:
        logger.info(f"✅ SUCCESS! Scene saved to: {result}")
        logger.info(
            "   Compare the character in this scene with the reference portrait:"
        )
        logger.info(f"     reference: {CHARACTER_PORTRAIT_PATH}")
        logger.info(f"     scene:     {result}")
        logger.info("   The face, hair, robes and staff should match!")
    else:
        logger.error("❌ Failed to generate scene with reference")
    return result


def main():
    """Run image generation tests."""
    print("\n" + "🎨" * 30)
    print("   IMAGE GENERATION TEST SUITE")
    print("🎨" * 30)
    print("\nCurrent Configuration:")
    print(f"  Provider: {settings.IMAGE_PROVIDER}")
    if settings.IMAGE_PROVIDER == "flux-kontext":
        print(f"  Model: {settings.FLUX_KONTEXT_MODEL}")
        print(f"  Guidance Scale: {settings.FLUX_KONTEXT_GUIDANCE_SCALE}")
        print(f"  CPU Offload (low-memory mode): {settings.IMAGE_ENABLE_CPU_OFFLOAD}")

    print(f"  Device: {settings.IMAGE_DEVICE}")
    print(f"  Inference Steps: {settings.IMAGE_NUM_INFERENCE_STEPS}")

    if settings.IMAGE_PROVIDER in ("flux-kontext", "gemini"):
        print("  ✅ Character consistency (reference images): SUPPORTED")
    else:
        print("  ❌ Character consistency: NOT SUPPORTED (use flux-kontext)")

    print(f"\nOutput Directory: {TEST_OUTPUT_DIR.absolute()}")
    print("\n" + "-" * 60)

    print("\nSelect test to run:")
    print("  1. Character Portrait")
    print("  2. Simple Scene (no references)")
    print("  3. Character Consistency (portrait -> scene)  ⭐ recommended")
    print("  4. All Tests")
    print("  q. Quit")

    choice = input("\nYour choice (1-4 or q): ").strip()

    try:
        if choice == "1":
            test_character_portrait()
        elif choice == "2":
            test_simple_scene()
        elif choice == "3":
            test_character_consistency()
        elif choice == "4":
            # Reuse one generator instance so the model loads only once.
            generator = _get_generator()
            test_character_portrait(generator)
            test_simple_scene(generator)
            test_character_consistency(generator)
        elif choice.lower() == "q":
            print("Exiting.")
            sys.exit(0)
        else:
            print(f"Invalid choice: {choice}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✅ TEST COMPLETE!")
        print(f"📁 Check your results in: {TEST_OUTPUT_DIR.absolute()}")
        print(f"🖼️  Open folder: open {TEST_OUTPUT_DIR.absolute()}")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()

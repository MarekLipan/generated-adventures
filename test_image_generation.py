#!/usr/bin/env python3
"""
Test script for image generation, focused on the FREE, local FLUX.1 Kontext [dev]
backend that keeps characters consistent across scenes via reference images.

Quick start (one-time setup):
    pip install -e ".[local-image]"

Then run:
    python test_image_generation.py

The most important tests are:
  #3 — single-character consistency (portrait → scene)
  #5 — party composition grid: generate 3 portraits, build a grid, generate an
        action scene using the grid as the one reference image so all characters
        appear consistently in the same scene.

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

TEST_OUTPUT_DIR = pathlib.Path("test_images")
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# ── Single-character test (legacy) ───────────────────────────────────────────

CHARACTER_PORTRAIT_PATH = TEST_OUTPUT_DIR / "character_portrait_elara.png"

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

# ── Party of 3 ────────────────────────────────────────────────────────────────
# Portrait prompts are processed by T5-XXL (up to ~380 words) as the primary
# encoder.  The backend appends a portrait-style suffix automatically.

PARTY = [
    {
        "name": "Elara Moonwhisper",
        "role": "Mage",
        "portrait_path": TEST_OUTPUT_DIR / "portrait_elara.png",
        "prompt": (
            "Elara Moonwhisper, tall elf mage, long silver hair, piercing blue eyes, "
            "midnight blue robes with glowing silver runes, ornate staff topped with "
            "a large crackling blue crystal, confident magical pose"
        ),
    },
    {
        "name": "Gareth Ironwall",
        "role": "Warrior",
        "portrait_path": TEST_OUTPUT_DIR / "portrait_gareth.png",
        "prompt": (
            "Gareth Ironwall, stocky dwarf warrior, enormous braided red beard, "
            "chunky heavy iron plate armor covered in battle dents, massive "
            "battle-axe resting on shoulder, fierce grinning expression"
        ),
    },
    {
        "name": "Lyra Swiftarrow",
        "role": "Ranger",
        "portrait_path": TEST_OUTPUT_DIR / "portrait_lyra.png",
        "prompt": (
            "Lyra Swiftarrow, agile human ranger, short auburn hair, sharp green eyes, "
            "dark green hooded leather coat with silver buckles, recurve bow drawn "
            "and ready, alert focused expression"
        ),
    },
]

PARTY_GRID_PATH = TEST_OUTPUT_DIR / "party_grid.png"

PARTY_SCENE_PROMPT = (
    "Elara Moonwhisper, Gareth Ironwall, and Lyra Swiftarrow are locked in a fierce "
    "battle inside a crumbling underground dungeon. Elara stands at the center, both "
    "hands raised, hurling a crackling bolt of blue lightning from her crystal-topped "
    "staff at a towering stone golem. Gareth charges from the left, red beard flying, "
    "swinging his massive battle-axe in a wide arc into the golem's leg, sparks and "
    "stone chips flying on impact. Lyra crouches on a crumbling stone balcony above, "
    "drawing her recurve bow with fierce concentration, an arrow already in flight "
    "toward the golem's glowing eye socket. The dungeon is lit by flickering wall "
    "torches and the blue glow of Elara's spell, casting dramatic shadows across "
    "ancient carved stone walls. Dynamic action, motion, high detail, painterly fantasy art."
)


def _get_generator():
    return get_image_generator()


# ── Individual test functions ─────────────────────────────────────────────────

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
    else:
        logger.error("❌ Failed to generate simple scene")
    return result


def test_character_consistency(generator=None):
    """THE KEY TEST: generate a scene that reuses the character portrait."""
    logger.info("\n" + "=" * 60)
    logger.info("🎭 TESTING CHARACTER CONSISTENCY (portrait -> scene)")
    logger.info("=" * 60)

    generator = generator or _get_generator()

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
        logger.info("   Compare character vs reference:")
        logger.info(f"     reference: {CHARACTER_PORTRAIT_PATH}")
        logger.info(f"     scene:     {result}")
    else:
        logger.error("❌ Failed to generate scene with reference")
    return result


def _build_party_grid(portrait_paths: list[pathlib.Path], output_path: pathlib.Path) -> pathlib.Path:
    """Stitch portrait images into a horizontal grid and save it."""
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in portrait_paths]

    # Normalise height so the strip looks balanced.
    target_h = 768
    resized = [img.resize((int(img.width * target_h / img.height), target_h), Image.LANCZOS) for img in images]

    gap = 6
    total_w = sum(img.width for img in resized) + gap * (len(resized) - 1)
    grid = Image.new("RGB", (total_w, target_h), (30, 30, 30))

    x = 0
    for img in resized:
        grid.paste(img, (x, 0))
        x += img.width + gap

    grid.save(output_path)
    logger.info(f"✓ Party grid saved to {output_path}  ({total_w}×{target_h})")
    return output_path


def test_party_composition(generator=None):
    """Generate 3 character portraits, compose them into a grid, then generate
    an action scene using the grid as the single FLUX Kontext reference so all
    characters appear consistently and actively in the same image.

    Flow:
      1. Generate portrait_elara.png  (skip if exists)
      2. Generate portrait_gareth.png (skip if exists)
      3. Generate portrait_lyra.png   (skip if exists)
      4. Stitch the 3 portraits into party_grid.png
      5. Generate a dungeon battle scene with party_grid.png as the reference
    """
    logger.info("\n" + "=" * 60)
    logger.info("⚔️  TESTING PARTY COMPOSITION + ACTION SCENE")
    logger.info("=" * 60)

    generator = generator or _get_generator()

    # Step 1–3: Portraits (skip if already generated, unless --regen passed).
    force_regen = "--regen" in sys.argv
    for member in PARTY:
        path: pathlib.Path = member["portrait_path"]
        if path.exists() and not force_regen:
            logger.info(f"  ↩  {member['name']} portrait already exists, skipping  (use --regen to regenerate)")
            continue
        if path.exists() and force_regen:
            path.unlink()
            logger.info(f"  ♻  Deleted old portrait for {member['name']}")

        logger.info(f"  Generating portrait: {member['name']} ({member['role']})…")
        result = generator.generate_character_image(
            prompt=member["prompt"],
            output_path=path,
        )
        if not result:
            logger.error(f"❌ Failed to generate portrait for {member['name']}")
            return None
        logger.info(f"  ✓ {member['name']} → {path.name}")

    # Step 4: Build the grid.
    logger.info("  Building party grid…")
    portrait_paths = [m["portrait_path"] for m in PARTY]
    if not all(p.exists() for p in portrait_paths):
        logger.error("❌ One or more portraits missing — cannot build grid")
        return None
    _build_party_grid(portrait_paths, PARTY_GRID_PATH)

    # Step 5: Action scene.
    logger.info("  Generating party action scene with grid reference…")
    output_path = TEST_OUTPUT_DIR / "party_action_scene.png"
    result = generator.generate_scene_image(
        prompt=PARTY_SCENE_PROMPT,
        output_path=output_path,
        reference_images=[PARTY_GRID_PATH],
    )

    if result:
        logger.info(f"✅ SUCCESS! Party scene saved to: {result}")
        logger.info("   Check that all 3 characters appear in the scene:")
        for member in PARTY:
            logger.info(f"     • {member['name']} ({member['role']})")
        logger.info(f"   Reference grid: {PARTY_GRID_PATH}")
        logger.info(f"   Action scene:   {result}")
    else:
        logger.error("❌ Failed to generate party action scene")
    return result


# ── Main menu ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("   IMAGE GENERATION TEST SUITE")
    print("=" * 60)
    print("\nCurrent Configuration:")
    print(f"  Provider: {settings.IMAGE_PROVIDER}")
    if settings.IMAGE_PROVIDER == "flux-kontext":
        print(f"  Model: {settings.FLUX_KONTEXT_MODEL}")
        print(f"  CPU Offload: {settings.IMAGE_ENABLE_CPU_OFFLOAD}")
    print(f"  Device: {settings.IMAGE_DEVICE}")
    print(f"  Inference Steps: {settings.IMAGE_NUM_INFERENCE_STEPS}")

    if settings.IMAGE_PROVIDER in ("flux-kontext", "gemini"):
        print("  ✅ Character consistency (reference images): SUPPORTED")
    else:
        print("  ❌ Character consistency: NOT SUPPORTED (use flux-kontext)")

    print(f"\nOutput Directory: {TEST_OUTPUT_DIR.absolute()}")
    print("\n" + "-" * 60)
    print("\nSelect test to run:")
    print("  1. Character Portrait (single character, text-only)")
    print("  2. Simple Scene (no references)")
    print("  3. Character Consistency (portrait → scene)  ⭐ quick reference test")
    print("  4. All Tests (1+2+3)")
    print("  5. Party Composition + Action Scene  ⭐ multi-character grid test")
    print("  q. Quit")

    choice = input("\nYour choice (1-5 or q): ").strip()

    try:
        if choice == "1":
            test_character_portrait()
        elif choice == "2":
            test_simple_scene()
        elif choice == "3":
            test_character_consistency()
        elif choice == "4":
            generator = _get_generator()
            test_character_portrait(generator)
            test_simple_scene(generator)
            test_character_consistency(generator)
        elif choice == "5":
            test_party_composition()
        elif choice.lower() == "q":
            print("Exiting.")
            sys.exit(0)
        else:
            print(f"Invalid choice: {choice!r}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✅ TEST COMPLETE!")
        print(f"📁 Results in: {TEST_OUTPUT_DIR.absolute()}")
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

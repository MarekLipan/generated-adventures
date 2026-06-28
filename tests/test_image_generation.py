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

# Ensure the project root is on the path when running this script directly.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

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


def test_two_character_scene():
    """Pass 2 character portraits as separate reference images (not a grid).

    FluxKontextPipeline natively accepts a list of images — each is VAE-encoded
    independently so FLUX attends to them as distinct characters rather than
    trying to edit a collage.  This is the correct multi-reference approach.

    Run test 5 first to generate the portraits.
    """
    logger.info("\n" + "=" * 60)
    logger.info("👥 TESTING 2-CHARACTER SCENE (separate reference images)")
    logger.info("=" * 60)

    # Use the first two party members — mage + warrior is a good visual contrast.
    char_a, char_b = PARTY[0], PARTY[1]
    for m in (char_a, char_b):
        if not m["portrait_path"].exists():
            logger.error(
                f"❌ Missing portrait: {m['portrait_path'].name} — run test 5 first."
            )
            return None

    prompt = (
        f"{char_a['name']} the {char_a['role']} and {char_b['name']} the {char_b['role']} "
        "stand back-to-back at the entrance of a torchlit dungeon, bracing for battle. "
        f"{char_a['name']} holds her glowing staff aloft, blue light illuminating the stone "
        f"walls. {char_b['name']} grips his battle-axe with both hands, eyes scanning the "
        "shadows ahead. Tense, dramatic atmosphere, warm torchlight against cool magical glow."
    )

    output_path = TEST_OUTPUT_DIR / "scene_two_characters.png"
    generator = _get_generator()

    result = generator.generate_scene_image(
        prompt=prompt,
        output_path=output_path,
        reference_images=[char_a["portrait_path"], char_b["portrait_path"]],
    )

    if result:
        logger.info(f"✅ SUCCESS! Scene saved to: {result}")
        logger.info("   Check whether both characters match their portraits:")
        logger.info(f"     {char_a['name']:20s} → {char_a['portrait_path'].name}")
        logger.info(f"     {char_b['name']:20s} → {char_b['portrait_path'].name}")
        logger.info(f"     scene                → {output_path.name}")
    else:
        logger.error("❌ Failed to generate 2-character scene")
    return result


def test_klein_portrait_and_scene():
    """FLUX.2 Klein: single portrait then consistency scene — quality comparison vs Kontext.

    Generates Elara's portrait with Klein, then generates a scene with that portrait
    as a single reference.  Compare the outputs to test 3 (FLUX.1 Kontext) to judge
    whether Klein's quality is sufficient for the game.

    Klein 9B distilled runs in only 4 inference steps — expect a big speed gain.
    """
    logger.info("\n" + "=" * 60)
    logger.info("🔷 TESTING FLUX.2 KLEIN — Portrait + Single-Character Scene")
    logger.info("=" * 60)

    from core.image_backends import FluxKleinImageGenerator

    gen = FluxKleinImageGenerator()

    portrait_path = TEST_OUTPUT_DIR / "klein_portrait_elara.png"
    logger.info("  [1/2] Generating Klein portrait of Elara...")
    result = gen.generate_character_image(
        prompt=PARTY[0]["prompt"],
        output_path=portrait_path,
    )
    if not result:
        logger.error("❌ Klein portrait generation failed")
        return None
    logger.info(f"  ✓ Portrait → {portrait_path.name}")

    scene_path = TEST_OUTPUT_DIR / "klein_scene_elara_temple.png"
    scene_prompt = (
        "Elara Moonwhisper stands inside a vast frozen glacier temple. "
        "Massive ice pillars covered in glowing blue runes frame her. "
        "She raises her crystal-topped staff, casting soft blue light across "
        "the smooth ice floor. Ethereal, magical, cold atmosphere."
    )
    logger.info("  [2/2] Generating Klein scene with portrait as reference...")
    result = gen.generate_scene_image(
        prompt=scene_prompt,
        output_path=scene_path,
        reference_images=[portrait_path],
    )
    if result:
        logger.info(f"✅ SUCCESS! Compare to FLUX.1 Kontext test 3 output:")
        logger.info(f"   Klein portrait: {portrait_path.name}")
        logger.info(f"   Klein scene:    {scene_path.name}")
        logger.info(f"   Kontext scene:  scene_with_reference_elara_temple.png")
    else:
        logger.error("❌ Klein scene generation failed")
    return result


def test_klein_multi_character():
    """FLUX.2 Klein: native multi-image reference — the key test.

    Generates fresh Klein portraits for all 3 party members, then generates a
    party battle scene passing ALL portraits as a native image list (no stitching).
    This is Klein's primary advantage over FLUX.1 Kontext.

    Compare output to test 5 (FLUX.1 Kontext party grid) to judge consistency.
    Pass --regen to force regeneration of Klein portraits.
    """
    logger.info("\n" + "=" * 60)
    logger.info("⚔️  TESTING FLUX.2 KLEIN — Native Multi-Character Scene")
    logger.info("=" * 60)

    from core.image_backends import FluxKleinImageGenerator

    gen = FluxKleinImageGenerator()
    force_regen = "--regen" in sys.argv

    klein_portraits: list[pathlib.Path] = []
    for member in PARTY:
        klein_path = TEST_OUTPUT_DIR / f"klein_portrait_{member['name'].split()[0].lower()}.png"
        if klein_path.exists() and not force_regen:
            logger.info(f"  ↩  Klein portrait for {member['name']} already exists (--regen to redo)")
        else:
            if klein_path.exists():
                klein_path.unlink()
            logger.info(f"  Generating Klein portrait: {member['name']}...")
            result = gen.generate_character_image(
                prompt=member["prompt"],
                output_path=klein_path,
            )
            if not result:
                logger.error(f"❌ Failed to generate Klein portrait for {member['name']}")
                return None
            logger.info(f"  ✓ {member['name']} → {klein_path.name}")
        klein_portraits.append(klein_path)

    scene_path = TEST_OUTPUT_DIR / "klein_party_scene.png"
    logger.info(
        f"  Generating party scene with {len(klein_portraits)} native references (no stitching)..."
    )
    result = gen.generate_scene_image(
        prompt=PARTY_SCENE_PROMPT,
        output_path=scene_path,
        reference_images=klein_portraits,
    )

    if result:
        logger.info(f"✅ SUCCESS! Compare to FLUX.1 Kontext party scene (test 5):")
        logger.info(f"   Klein scene:   {scene_path.name}")
        logger.info(f"   Kontext scene: party_action_scene.png")
        logger.info("   Check: do characters resemble their Klein portraits more than in Kontext?")
    else:
        logger.error("❌ Klein multi-character scene generation failed")
    return result


def test_ip_adapter_comparison():
    """Side-by-side comparison: SDXL + IP-Adapter vs FLUX Kontext.

    Reuses the three FLUX portraits from test 5 as reference images so both
    backends see identical inputs.  Generates the same party battle scene with
    each backend and saves both results for direct visual comparison.

    Run test 5 first to create the portrait files, then run this test.
    """
    logger.info("\n" + "=" * 60)
    logger.info("🔬 IP-ADAPTER vs FLUX KONTEXT COMPARISON")
    logger.info("=" * 60)

    portraits = [m["portrait_path"] for m in PARTY]
    missing = [p for p in portraits if not p.exists()]
    if missing:
        logger.error(
            f"❌ Missing portrait(s): {[p.name for p in missing]}\n"
            "   Run test 5 first to generate the party portraits."
        )
        return None

    scene_prompt = PARTY_SCENE_PROMPT

    results = {}

    # ── FLUX Kontext scene (reuse existing generator if possible) ──────────────
    logger.info("\n  [1/2] Generating scene with FLUX Kontext (party grid reference)...")
    flux_path = TEST_OUTPUT_DIR / "comparison_flux_kontext.png"
    try:
        flux_gen = get_image_generator()  # uses IMAGE_PROVIDER from .env
        result = flux_gen.generate_scene_image(
            prompt=scene_prompt,
            output_path=flux_path,
            reference_images=portraits,
        )
        if result:
            results["flux-kontext"] = flux_path
            logger.info(f"  ✓ FLUX Kontext → {flux_path.name}")
        else:
            logger.warning("  ⚠  FLUX Kontext scene generation failed")
    except Exception as e:
        logger.warning(f"  ⚠  FLUX Kontext failed: {e}")

    # ── SDXL + IP-Adapter scene ────────────────────────────────────────────────
    logger.info("\n  [2/2] Generating scene with SDXL + IP-Adapter...")
    sdxl_path = TEST_OUTPUT_DIR / "comparison_sdxl_ip_adapter.png"
    try:
        from core.image_backends import SDXLIPAdapterImageGenerator
        sdxl_gen = SDXLIPAdapterImageGenerator()
        result = sdxl_gen.generate_scene_image(
            prompt=scene_prompt,
            output_path=sdxl_path,
            reference_images=portraits,
        )
        if result:
            results["sdxl-ip-adapter"] = sdxl_path
            logger.info(f"  ✓ SDXL IP-Adapter → {sdxl_path.name}")
        else:
            logger.warning("  ⚠  SDXL IP-Adapter scene generation failed")
    except Exception as e:
        logger.warning(f"  ⚠  SDXL IP-Adapter failed: {e}")

    if results:
        logger.info("\n✅ Comparison complete — open both files side by side:")
        for name, path in results.items():
            logger.info(f"   {name:20s} → {path}")
        logger.info("\n  Ask yourself:")
        logger.info("  • Do the characters look like the portraits?")
        logger.info("  • Which art style looks more like Hearthstone?")
        logger.info("  • Is the quality difference acceptable for the game?")
    return results


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
    print("  5. Party Composition + Action Scene  (grid reference)")
    print("  6. Two-Character Scene  ⭐ separate reference images, no grid")
    print("  7. IP-Adapter vs FLUX Kontext comparison  (requires test 5 portraits)")
    print("  8. FLUX.2 Klein — Portrait + Single-Character Scene  ⭐ quality vs speed")
    print("  9. FLUX.2 Klein — Native Multi-Character Scene  ⭐ no stitching")
    print("  q. Quit")

    choice = input("\nYour choice (1-9 or q): ").strip()

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
        elif choice == "6":
            test_two_character_scene()
        elif choice == "7":
            test_ip_adapter_comparison()
        elif choice == "8":
            test_klein_portrait_and_scene()
        elif choice == "9":
            test_klein_multi_character()
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

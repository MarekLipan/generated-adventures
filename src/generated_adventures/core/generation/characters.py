"""Party members: pre-made characters, hero archetypes, and custom heroes with lore and a portrait."""

import asyncio
import logging
import pathlib

from ..llm import retry_on_overload, text_agent
from ..media import (
    IMAGE_DIR,
    generate_hero_portrait_sync,
    shared_image_generator,
)
from ..models import (
    Character,
    GeneratedArchetype,
    GeneratedArchetypeList,
    GeneratedCharacter,
)
from ..prompts import (
    SCENARIO_SCALES,
    avoid_names_section,
    setting_only,
)

logger = logging.getLogger(__name__)


async def generate_archetypes(
    scenario_name: str,
    scenario_details: str | None = None,
    num_archetypes: int = 5,
    scale: str = "any",
) -> list[GeneratedArchetype]:
    """Generate scenario-tailored hero archetypes for players to choose from.

    This is a cheap text-only call. Portraits and full lore are only generated
    later, after a player commits to one of these archetypes.

    Args:
        scenario_name: Name of the chosen scenario
        scenario_details: Markdown DM notes (setting/plot/quest/NPCs) for context
        num_archetypes: How many distinct archetypes to offer
        scale: Stakes/scale key from SCENARIO_SCALES — sets how legendary vs green
               the heroes should feel (aspiring locals vs world-saving legends).

    Returns:
        List of GeneratedArchetype options.
    """
    logger.info(
        f"Generating {num_archetypes} archetypes for scenario: {scenario_name} (scale={scale})"
    )
    agent = text_agent(GeneratedArchetypeList)

    # Only the non-spoiler SETTING reaches this player-facing generation — never
    # the plot, quest, villains or secrets from the DM notes.
    setting = setting_only(scenario_details)
    context_section = ""
    if setting:
        context_section = f"""
WORLD & SETTING (non-spoiler background — the tone and locale the players begin in):
{setting}

"""

    scale_key = scale if scale in SCENARIO_SCALES else "any"
    _scale_label, scale_steering = SCENARIO_SCALES[scale_key]
    stature = {
        "grounded": (
            "LEVEL-1 NOVICES. These are ordinary locals taking their FIRST steps as adventurers — "
            "a farmhand, a poacher, an apprentice, a militia recruit, a road-worn pedlar. They have "
            "no renown, no titles, no mastery.\n"
            "  * Titles/roles must sound humble and ordinary (e.g. 'The Poacher', 'The Apprentice', "
            "'The Cooper's Daughter') — NOT grand epithets like 'Warden', 'Master', 'Champion', "
            "'Blademaster', 'Arcanist' or 'of the ...' honorifics.\n"
            "  * GEAR IS POOR AND PRACTICAL: patched wool, homespun linen, worn leather, a hooded "
            "cloak, hobnail boots, a rope belt. Weapons are simple and often improvised or "
            "second-hand — a hunting bow, a woodsman's axe, a sling, a cheap short sword, a staff.\n"
            "  * FORBIDDEN in the visual concept: plate/scale/ornate armor, gilding, filigree, "
            "silks, jewels, runes, glowing/enchanted items, flowing masterwork cloaks, heraldry, "
            "trophies, or anything that reads as expensive, magical or 'master-tier'.\n"
            "  * They should look like they scraped their kit together, not like seasoned heroes."
        ),
        "heroic": (
            "CAPABLE, SEASONED adventurers — competent professionals, but not legendary. "
            "Well-used, functional kit: sturdy leather or mail, a decent weapon, practical gear. "
            "No artifact-tier or ostentatious equipment."
        ),
        "epic": (
            "LEGENDARY, powerful heroes fit for world-shaking deeds — renowned, battle-hardened, "
            "carrying signature, storied equipment."
        ),
    }.get(scale_key, "")
    scale_section = (
        f"\nHERO STATURE — match this level EXACTLY:\n{stature}\n" if stature else ""
    )

    prompt = f"""
You are designing playable hero archetypes for a D&D-style adventure named '{scenario_name}'.
{context_section}{scale_section}Generate {num_archetypes} DISTINCT hero archetypes that fit this world's SETTING and
TONE. Avoid generic filler — each should feel like it belongs in this world and covers a different
party role (e.g. front-line fighter, stealth/skill specialist, arcane/support, face/social,
ranged/scout). Make the set complementary so any pick leads to a fun, viable hero.

CRITICAL — NO SPOILERS: These archetypes are shown to players BEFORE the adventure begins. Describe
ONLY who the hero is — their fantasy, role, and playstyle — grounded in the world's setting and tone.
Do NOT reveal or even hint at the plot, the quest's objective, any villain or their name, twists, or
secrets the players are meant to discover in play. Sell the CHARACTER, never the story.

For each archetype provide:
- name: An evocative title (NOT a personal name — a role identity, e.g. 'The Ashen Warden')
- role: A short party-function label (e.g. 'Front-line bruiser', 'Stealth & sabotage', 'Arcane support')
- hook: ONE enticing sentence about the fantasy of PLAYING this archetype — their role and flavor in
  this world. No plot, no quest objective, no villain names, no secrets.
- concept: A vivid VISUAL concept for the portrait — costume, armor/clothing, signature equipment,
  silhouette and overall vibe (2-3 sentences). Describe gear and style ONLY. Do NOT describe facial
  features, age, or ethnicity — the player's own face will be used for the portrait.
  * The concept MUST obey the HERO STATURE above — its clothing and equipment have to look like
    that tier. Do not dress a novice in impressive or ornate gear.

Make them diverse, world-appropriate, and exciting — WITHOUT spoiling anything.
"""
    result = await retry_on_overload(agent.run, prompt)
    archetypes = result.output.archetypes
    logger.info(f"=== {len(archetypes)} Archetypes Generated ===")
    logger.info(f"Archetypes: {[a.name for a in archetypes]}")
    return archetypes


async def _generate_hero_lore(
    scenario_name: str,
    scenario_details: str | None,
    archetype: GeneratedArchetype,
    has_photo: bool,
    custom_name: str | None = None,
    gender: str = "unspecified",
    avoid_names: list[str] | None = None,
    scale: str = "any",
) -> GeneratedCharacter:
    """Generate stats + lore for a single hero fitted to scenario and archetype."""
    agent = text_agent(GeneratedCharacter)

    context_section = ""
    # Only the non-spoiler setting reaches hero lore — the backstory is shown to the
    # player and must not leak the plot/villains/secrets they should discover in play.
    setting = setting_only(scenario_details)
    if setting:
        context_section = f"""
WORLD & SETTING (non-spoiler background):
{setting}

"""

    if custom_name:
        name_rule = f"- name: Use EXACTLY this name for the hero: '{custom_name}'"
    else:
        avoid = avoid_names_section(
            avoid_names,
            "Do NOT reuse any of these names already used in other adventures (pick something clearly different):",
        ).strip()
        name_rule = (
            "- name: Invent a fitting, distinctive full name (with an optional title/nickname). "
            + (avoid if avoid else "Avoid generic, overused fantasy names.")
        )
    appearance_rule = (
        "- appearance: Describe their COSTUME, equipment, and bearing to match the archetype concept. "
        + (
            "Do NOT describe facial features, hair or skin — those come from the player's own photo."
            if has_photo
            else "Include distinctive physical and clothing details."
        )
    )
    gender_line = (
        f"- The hero is {gender}. Use {gender} pronouns and gendered language "
        "throughout the backstory, personality, and appearance.\n"
        if gender and gender != "unspecified"
        else ""
    )
    tier_section = {
        "grounded": (
            "\nPOWER TIER — LEVEL 1 NOVICE (obey strictly):\n"
            "This hero is just STARTING OUT: an ordinary local, no renown, no mastery, no titles.\n"
            "- Stats: keep them modest and human — roughly 8-13, with ONE modest strength up to 14. "
            "No 16+ anywhere.\n"
            "- Skills: humble, practical, everyday competences (mending, foraging, haggling, "
            "a bit of hunting) — NOT mastery, NOT rare arcane arts.\n"
            "- Inventory: cheap, worn, scavenged everyday kit. NO enchanted, glowing, runed, "
            "masterwork, gilded or heirloom-artifact items.\n"
            "- Appearance: patched, homespun, weather-worn clothing and second-hand gear. "
            "NO ornate armor, silks, jewels, filigree, heraldry or anything expensive-looking.\n"
            "- Backstory: small, local and personal (a village, a trade, a debt, a lost sibling) — "
            "no prophecies, chosen-one destinies, famous deeds or legendary bloodlines.\n"
        ),
        "heroic": (
            "\nPOWER TIER — SEASONED (not legendary):\n"
            "- Stats mostly 10-15. Competent, well-used gear; nothing artifact-tier or ostentatious.\n"
        ),
        "epic": (
            "\nPOWER TIER — LEGENDARY:\n"
            "- Stats may reach 16-18. Storied, signature equipment and renowned deeds fit here.\n"
        ),
    }.get(scale if scale in SCENARIO_SCALES else "any", "")

    prompt = f"""
Create ONE fully-realized hero for a D&D-style adventure named '{scenario_name}'.
{context_section}The hero is built from this archetype (make the character a specific, named embodiment of it):
- Archetype: {archetype.name} ({archetype.role})
- Fantasy: {archetype.hook}
- Visual concept: {archetype.concept}
{gender_line}{tier_section}
Provide:
{name_rule}
- strength, intelligence, agility: Stats between 1-20 that reflect this archetype's role
  (respecting the POWER TIER above)
- backstory: 2-3 sentences of personal history and motivation grounded in this WORLD. Do NOT
  reference the adventure's plot, quest objective, villains, twists, or secrets — those are for the
  player to discover in play
- personality: 2-3 sentences on their traits, mannerisms, and how they interact with others
- skills: 3-5 concrete skills matching the archetype and stats
- inventory: 3-4 starting items. Each item needs a PLAIN, non-cryptic name AND a one-sentence
  purpose (what it does and when the player would use it). Favor a few clearly-useful items
  (a signature weapon, a tool tied to one of their skills, one situational item) over opaque
  flavor trinkets, so the player always knows what each item is for.
{appearance_rule}

Make the hero distinctive, memorable, and a natural fit for both the archetype and the scenario.
"""
    result = await retry_on_overload(agent.run, prompt)
    return result.output


async def generate_hero(
    game_id: str,
    scenario_name: str,
    scenario_details: str | None,
    archetype: GeneratedArchetype,
    art_style: str,
    player_index: int,
    photo_path: pathlib.Path | None = None,
    custom_name: str | None = None,
    gender: str = "unspecified",
    avoid_names: list[str] | None = None,
    scale: str = "any",
) -> Character:
    """Generate a single hero (lore + portrait) from a chosen archetype.

    Lore (text) and the portrait (image) are generated concurrently. When a
    player photo is provided, the portrait keeps their likeness while restyling
    them as the archetype hero.

    Args:
        game_id: Game identifier (for image storage)
        scenario_name: Name of the scenario
        scenario_details: Markdown DM notes for context
        archetype: The archetype the player chose
        art_style: Art-style key for imagery ("painterly_hero" / "artstation_realism")
        player_index: 1-based index of the player this hero belongs to
        photo_path: Optional path to the player's uploaded photo
        custom_name: Optional player-chosen hero name

    Returns:
        A fully-populated Character.
    """
    has_photo = bool(photo_path and pathlib.Path(photo_path).exists())
    logger.info(
        f"Generating hero for player {player_index}: archetype '{archetype.name}', "
        f"style '{art_style}', photo={'yes' if has_photo else 'no'}"
    )

    game_dir = IMAGE_DIR / game_id
    game_dir.mkdir(parents=True, exist_ok=True)
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(shared_image_generator)

    # Kick off lore + portrait concurrently.
    lore_task = _generate_hero_lore(
        scenario_name,
        scenario_details,
        archetype,
        has_photo,
        custom_name,
        gender,
        avoid_names,
        scale,
    )
    portrait_task = asyncio.to_thread(
        generate_hero_portrait_sync,
        generator,
        custom_name or archetype.name,
        archetype,
        art_style,
        pathlib.Path(photo_path) if has_photo else None,
        game_dir,
        player_index,
        scale,
    )

    lore, image_file_path = await asyncio.gather(
        lore_task, portrait_task, return_exceptions=True
    )

    if isinstance(lore, BaseException):
        logger.error(f"Hero lore generation failed: {lore}")
        raise lore
    if isinstance(image_file_path, BaseException):
        logger.error(f"Hero portrait generation failed: {image_file_path}")
        image_file_path = None

    hero_name = custom_name or lore.name

    photo_web_path = None
    if has_photo:
        photo_web_path = f"/static/photos/{game_id}/{pathlib.Path(photo_path).name}"

    return Character(
        name=hero_name,
        gender=gender if gender in ("male", "female", "nonbinary") else "unspecified",
        strength=lore.strength,
        intelligence=lore.intelligence,
        agility=lore.agility,
        maximum_health=100,
        current_health=100,
        backstory=lore.backstory,
        appearance=lore.appearance,
        personality=lore.personality,
        skills=lore.skills,
        inventory=lore.inventory,
        image_path=(
            f"/static/characters/{game_id}/{image_file_path.name}"
            if image_file_path
            else None
        ),
        archetype=archetype.name,
        player_photo_path=photo_web_path,
    )

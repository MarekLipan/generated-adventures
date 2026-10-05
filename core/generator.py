"""
Handles content generation via external APIs.

This file currently contains placeholder deterministic generators. Replace
these with real model calls (Gemini, OpenAI, etc.) later.
"""

import asyncio
import logging
import pathlib

from .character_updates import apply_character_updates, format_inventory
from .image_backends import ImageGenerator
from .llm import retry_on_overload, text_agent
from .media import (
    IMAGE_DIR,
    generate_asset_image_sync,
    generate_character_image_sync,
    generate_hero_portrait_sync,
    generate_scene_image_sync,
    generate_scene_voiceover,
    resolve_image_subject_ids,
    shared_image_generator,
)
from .models import (
    Asset,
    Character,
    GameStatus,
    GeneratedArchetype,
    GeneratedArchetypeList,
    GeneratedCharacter,
    GeneratedCharacterList,
    GeneratedLocationList,
    GeneratedScenarioTemplate,
    GeneratedScene,
    Location,
    LocationReference,
    RecapResponse,
    ScenarioTemplate,
    Scene,
    SceneSummary,
)
from .names import names_match, npc_names_match
from .prompts import (
    SCENARIO_SCALES,
    SCENARIO_THEMES,
    avoid_names_section,
    build_dice_check_resolution_rules,
    build_impossible_action_rules,
    build_scene_generation_prompt_rules,
    format_existing_assets,
    format_existing_locations,
    scene_scale_section,
    setting_only,
)

logger = logging.getLogger()  # Use root logger so Uvicorn/NiceGUI picks up logs
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)


async def generate_characters(
    game_id: str,
    scenario_name: str,
    num_characters: int = 6,
    scenario_details: str | None = None,
) -> list[Character]:
    """Generates a list of characters with stats and portraits for a given scenario.

    Args:
        game_id: Unique identifier for the game (used for image storage)
        scenario_name: Name of the scenario
        num_characters: Number of characters to generate
        scenario_details: Optional markdown details about the scenario (setting, plot, NPCs, etc.)
    """
    # Step 1: Generate character concepts (name, stats)

    logger.info(f"Generating {num_characters} characters for scenario: {scenario_name}")
    agent = text_agent(GeneratedCharacterList)

    # Build context-aware prompt
    context_section = ""
    if scenario_details:
        context_section = f"""
SCENARIO CONTEXT:
{scenario_details}

Use the above scenario details to inform your character creation. Characters should fit naturally into this world, 
relate to the plot/quest, and potentially interact with the mentioned NPCs and locations.

"""

    prompt = f"""
Generate {num_characters} unique and compelling fantasy characters for a D&D-style adventure named '{scenario_name}'.
Provide a diverse set of classic fantasy archetypes.

{context_section}For each character, provide:
- name: Full name with title/nickname
- strength, intelligence, agility: Stats between 1-20
- backstory: 2-3 sentences about their history and what drives them (reference scenario context when relevant)
- appearance: 2-3 sentences describing their physical look, clothing, and distinctive features
- personality: 2-3 sentences about their personality traits, mannerisms, how they speak, and how they interact with others
- skills: 3-5 specific skills or abilities (e.g., "Swordsmanship", "Lockpicking", "Persuasion", "Arcane Knowledge", "Tracking", "Healing", "Stealth")
- inventory: 3-5 items they carry (weapons, tools, magical items, personal effects)

Make each character distinctive and memorable with rich details that fit the scenario.
Skills should match the character's background, stats, and archetype.
"""
    concept_result = await retry_on_overload(agent.run, prompt)
    generated_concepts = concept_result.output.characters
    logger.info(f"=== {num_characters} Characters Generated ===")
    logger.info(f"Characters: {concept_result.output.model_dump_json(indent=2)}")

    # Step 2: Synchronous image generation using backend abstraction
    # Create game-specific directory for character images
    game_dir = IMAGE_DIR / game_id
    game_dir.mkdir(parents=True, exist_ok=True)

    generator = await asyncio.to_thread(shared_image_generator)

    # Step 3: Generate images asynchronously (in thread pool to avoid blocking event loop)
    final_characters: list[Character] = []

    # Generate all images concurrently using asyncio.to_thread
    image_tasks = [
        asyncio.to_thread(
            generate_character_image_sync,
            generator,
            concept,
            scenario_name,
            game_id,
            game_dir,
            idx,
        )
        for idx, concept in enumerate(generated_concepts, start=1)
    ]

    logger.info("Starting concurrent image generation for all characters...")
    image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
    logger.info("All image generation tasks completed")

    # Build final character list with image paths
    for concept, result in zip(generated_concepts, image_results, strict=True):
        # A failed task comes back as the exception itself, not a (name, path)
        # tuple, so check before unpacking: one bad portrait must not sink the party.
        if isinstance(result, BaseException):
            logger.error(f"Image generation failed for {concept.name}: {result}")
            image_file_path = None
        else:
            _, image_file_path = result

        new_char = Character(
            name=concept.name,
            strength=concept.strength,
            intelligence=concept.intelligence,
            agility=concept.agility,
            maximum_health=100,
            current_health=100,
            backstory=concept.backstory,
            appearance=concept.appearance,
            personality=concept.personality,
            skills=concept.skills,
            inventory=concept.inventory,
            image_path=(
                f"/static/characters/{game_id}/{image_file_path.name}"
                if image_file_path
                else None
            ),
        )
        final_characters.append(new_char)

    logger.info(f"Final character list: {[c.name for c in final_characters]}")
    return final_characters


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


async def generate_scenario_template(
    existing_scenarios: list[ScenarioTemplate],
    theme: str = "any",
    scale: str = "any",
    avoid_names: list[str] | None = None,
) -> ScenarioTemplate:
    """Generates a complete new scenario template with contrastive prompting.

    Args:
        existing_scenarios: List of all existing scenario templates (both played and unplayed)
                          to ensure the new scenario is distinct and original
        theme: Genre steering key from SCENARIO_THEMES. When a specific genre is chosen,
               novelty is sought WITHIN that genre rather than by switching setting/era.
        scale: Stakes/scale steering key from SCENARIO_SCALES (local/heroic/epic), shaping
               how big the threat and scope are.

    Returns:
        ScenarioTemplate object with name, one_liner, and full DM notes
    """
    agent = text_agent(GeneratedScenarioTemplate)

    theme_key = theme if theme in SCENARIO_THEMES else "any"
    _label, theme_steering = SCENARIO_THEMES[theme_key]
    themed = bool(theme_steering)

    scale_key = scale if scale in SCENARIO_SCALES else "any"
    _scale_label, scale_steering = SCENARIO_SCALES[scale_key]

    logger.info(
        f"Generating new scenario template (theme={theme_key}, scale={scale_key}, "
        f"contrasting with {len(existing_scenarios)} existing scenarios)..."
    )

    theme_section = (
        f"\nGENRE / THEME — follow this closely:\n{theme_steering}\n" if themed else ""
    )
    scale_section = (
        f"\nSTAKES / SCALE — follow this closely:\n{scale_steering}\n"
        if scale_steering
        else ""
    )
    avoid_section = avoid_names_section(
        avoid_names,
        "- These character and place names are ALREADY USED by other adventures — do NOT "
        "reuse any of them or close variants:",
    )

    # Build contrastive prompt with existing scenarios
    contrastive_context = ""
    if existing_scenarios:
        scenario_summaries = []
        for i, scenario in enumerate(
            existing_scenarios[:10], 1
        ):  # Limit to 10 most recent
            scenario_summaries.append(
                f"**Scenario {i}: {scenario.name}**\n"
                f"Hook: {scenario.one_liner}\n"
                f"Times Played: {scenario.times_played}\n"
                f"DM Notes Preview: {scenario.dm_notes[:300]}..."
            )
        summaries = chr(10).join(scenario_summaries)

        if themed:
            # Stay in the chosen genre; find novelty WITHIN it (do not change era/world-type).
            contrastive_context = f"""
EXISTING SCENARIOS (make your new one a genuinely fresh adventure, but STAY in the genre above):

{summaries}

Keep the GENRE, era and world-type fixed — that is deliberate, not a limitation. Put ALL of your
originality into the things that actually make an adventure feel new:
- a distinct PLOT and story arc (never a reskin of an existing one),
- a different QUEST STRUCTURE (e.g. rescue, heist, mystery, escort, siege, ritual, hunt, exploration),
- fresh CHARACTERS and factions, and a new ANTAGONIST with their own motive,
- new KEY LOCATIONS and a different central CONFLICT and mood — still within the genre.
Do NOT reach for novelty by changing the setting type, era, or genre.
"""
        else:
            contrastive_context = f"""
EXISTING SCENARIOS (create something DISTINCTLY DIFFERENT):

{summaries}

CRITICAL: Your new scenario MUST differ in at least 3 major aspects:
1. Setting type/era (medieval village, space station, underwater city, floating islands, etc.)
2. Quest structure (rescue mission, mystery investigation, treasure hunt, political intrigue, survival, etc.)
3. Antagonist type (villain, natural disaster, curse, rival faction, internal conflict, etc.)
4. Tone/atmosphere (horror, heroic fantasy, noir mystery, comedy, epic, gritty, whimsical, etc.)
5. Core conflict (combat-focused, social/political, exploration, resource management, moral dilemma, etc.)

Analyze the patterns in existing scenarios and deliberately create something novel.
"""

    prompt = f"""
You are a creative fantasy adventure designer creating scenarios for a D&D-style tabletop RPG.
{theme_section}
{scale_section}
{contrastive_context}

Generate a COMPLETE scenario with the following components:

0. **genre**: A short 1-3 word genre/tone label for the scenario you are creating (e.g. "Classic
   Fantasy", "Dark Fantasy", "Fantasy Mystery", "Steampunk"). It must match the GENRE/THEME above
   when one is given.


1. **name**: A compelling, memorable scenario name (3-7 words)
   - Examples: "The Curse of Ravenmoor", "Heist at the Skyport", "Whispers from the Deep"

2. **one_liner**: A short, enticing hook (1-2 sentences) that attracts players WITHOUT spoiling the plot
   - Should create mystery, excitement, or intrigue
   - NO spoilers about villains, plot twists, or endings
   - Examples: 
     * "A cursed village where children vanish under the blood moon"
     * "Rival treasure hunters race to claim an artifact in ancient sky ruins"
     * "Political intrigue threatens to tear apart the kingdom's fragile peace"

3. **setting**: Detailed description of the world/locale (2-4 rich paragraphs)
   - Geography, climate, culture, history
   - Key locations and landmarks
   - Visual and atmospheric details

4. **plot**: The narrative arc and story structure (2-4 paragraphs)
   - Setup, complications, potential paths
   - Key events and decision points
   - How the story could unfold

5. **main_quest**: The primary objective for the party (2-4 paragraphs)
   - What the party needs to accomplish
   - Why it matters
   - Challenges they'll face
   - Possible approaches and outcomes

6. **important_npcs**: Key non-player characters (2-4 paragraphs)
   - Describe 3-5 important NPCs with names, roles, motivations
   - Include allies, neutrals, and antagonists
   - Make them memorable and useful for roleplay

NAME & PLACE VARIETY (do not reuse names across adventures):
- Invent FRESH character and place names, and list every named character/place you use in the
  character_names field so future adventures can avoid repeating them.
{avoid_section}- Draw on genuinely varied naming traditions and sound-shapes; avoid generic, repetitive fantasy
  names so different adventures don't feel same-y.

Create a scenario that is fresh, engaging, and distinctly different from existing ones.
Make it playable, fun, and full of opportunities for player choice and creativity.
"""

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Log generated scenario
    logger.info("=== Scenario Template Generated ===")
    logger.info(f"Name: {generated.name}")
    logger.info(f"One-liner: {generated.one_liner}")

    # Convert to markdown DM notes
    dm_notes = (
        f"## The Setting\n\n{generated.setting}\n\n"
        f"## The Plot\n\n{generated.plot}\n\n"
        f"## Main Quest\n\n{generated.main_quest}\n\n"
        f"## Important NPCs\n\n{generated.important_npcs}\n"
    )

    # Genre label: when a specific theme was chosen, that IS the genre; otherwise
    # use the tag the model assigned to what it actually produced.
    genre = _label if themed else (getattr(generated, "genre", "") or "").strip()

    # Create ScenarioTemplate object
    scenario_template = ScenarioTemplate(
        name=generated.name,
        genre=genre,
        scale=scale_key,
        one_liner=generated.one_liner,
        dm_notes=dm_notes,
        character_names=list(getattr(generated, "character_names", []) or []),
    )

    logger.info(
        f"Genre: {genre} | Scale: {scale_key} | names: {len(scenario_template.character_names)}"
    )
    return scenario_template


async def generate_initial_locations(
    scenario_name: str,
    scenario_details: str | None = None,
) -> dict[str, Location]:
    """Generates initial set of locations for the adventure.

    Args:
        scenario_name: Name of the scenario
        scenario_details: Optional markdown details about the scenario

    Returns:
        Dictionary mapping location IDs to Location objects
    """
    agent = text_agent(GeneratedLocationList)

    logger.info(f"Generating initial locations for scenario: {scenario_name}")

    # Build context-aware prompt
    context_section = ""
    if scenario_details:
        context_section = f"""
SCENARIO CONTEXT:
{scenario_details}

Use the above scenario details to create locations that fit naturally into this world and support the plot/quest.

"""

    prompt = f"""
Generate 4-6 diverse and memorable locations for a fantasy adventure named '{scenario_name}'.
These locations will serve as settings for scenes throughout the adventure.

{context_section}For each location, provide:
- name: Memorable name (e.g., "The Crimson Tavern", "Shadowfen Swamp", "Dragon's Peak Summit")
- location_type: One of: indoor, outdoor, underground, aerial, aquatic, mystical
- description: Extensive physical description (3-5 sentences) including architecture/terrain, materials, scale, layout
- key_features: 3-5 distinctive features that make this location unique and memorable
- atmosphere: The general mood and feel (1-2 sentences)
- lighting_default: Default lighting conditions (e.g., "torch-lit", "moonlit", "bright daylight", "bioluminescent glow")

Make each location visually distinct and narratively useful. Include a mix of location types (indoor/outdoor/etc).
Ensure locations support different story beats: safe havens, dangerous areas, mysterious places, social hubs, etc.
"""
    result = await retry_on_overload(agent.run, prompt)
    generated_locations = result.output.locations

    logger.info(f"=== {len(generated_locations)} Initial Locations Generated ===")
    logger.info(f"Locations: {result.output.model_dump_json(indent=2)}")

    # Convert to Location objects with IDs
    locations_dict: dict[str, Location] = {}
    for gen_loc in generated_locations:
        location = Location(
            name=gen_loc.name,
            location_type=gen_loc.location_type,
            description=gen_loc.description,
            key_features=gen_loc.key_features,
            atmosphere=gen_loc.atmosphere,
            lighting_default=gen_loc.lighting_default,
        )
        locations_dict[location.id] = location
        logger.info(
            f"Created location '{location.name}' (ID: {location.id}, Type: {location.location_type})"
        )

    return locations_dict


def _make_scene(
    scene_id: int,
    scene_text: str,
    prompt,
    image_path: str | None = None,
    game_status: GameStatus = "ongoing",
    visible_asset_ids: list[str] = None,
    health_changes: list = None,
    inventory_changes: list = None,
    skill_changes: list = None,
    stat_changes: list = None,
) -> Scene:
    """Helper to create a Scene object from generated data.

    Args:
        scene_id: Unique scene identifier
        scene_text: The narrative text of the scene
        prompt: PromptType object from the generated scene
        image_path: Optional path to the scene image
        game_status: Status of the game ('ongoing', 'completed', 'failed')
        visible_asset_ids: List of asset IDs visible in this scene
        health_changes: List of CharacterHealthChange objects from this scene
        inventory_changes: List of CharacterInventoryChange objects from this scene
        skill_changes: List of CharacterSkillChange objects from this scene
        stat_changes: List of CharacterStatChange objects from this scene
    """
    return Scene(
        id=scene_id,
        text=scene_text,
        prompt=prompt,
        image_path=image_path,
        voiceover_path=None,
        game_status=game_status,
        visible_asset_ids=visible_asset_ids or [],
        health_changes=health_changes or [],
        inventory_changes=inventory_changes or [],
        skill_changes=skill_changes or [],
        stat_changes=stat_changes or [],
    )


def _process_scene_assets(
    game_id: str,
    scene_id: int,
    assets_present: list,
    existing_assets: dict[str, Asset],
    generator: ImageGenerator,
    art_style: str | None = None,
) -> tuple[dict[str, Asset], list[str]]:
    """Process assets from generated scene, create new ones, and generate images.

    Args:
        game_id: Game identifier for storage
        scene_id: Scene identifier
        assets_present: List of AssetReference objects from the generated scene
        existing_assets: Dictionary of existing assets
        generator: Local image backend used to render new asset images
        art_style: Art-style key for the game

    Returns:
        Tuple of (updated assets dictionary, list of visible asset IDs for this scene)
    """
    updated_assets = existing_assets.copy()
    visible_asset_ids = []

    # Create asset directory if it doesn't exist
    asset_dir = pathlib.Path(f"webapp/static/assets/{game_id}")
    asset_dir.mkdir(parents=True, exist_ok=True)

    for asset_ref in assets_present:
        # Find if this asset already exists (match by name, case-insensitive)
        existing_asset_id = None
        for asset_id, asset in existing_assets.items():
            if asset.name.lower() == asset_ref.name.lower():
                existing_asset_id = asset_id
                break

        # If no exact match, try a tolerant match. Party portraits carry the
        # player's likeness (photo reference), so match them loosely (subset) to
        # always reuse them even if the hero was named slightly differently. NPCs
        # and objects match on normalized token-set EQUALITY, which reunites
        # naming drift ("The Echo Child" vs "Echo-Child") into ONE asset/portrait
        # while keeping distinct NPCs ("Guard" vs "Guard Captain") apart.
        if not existing_asset_id:
            for asset_id, asset in existing_assets.items():
                is_party = asset_id.startswith("player_")
                matched = (
                    names_match(asset_ref.name, asset.name)
                    if is_party
                    else npc_names_match(asset_ref.name, asset.name)
                )
                if matched:
                    existing_asset_id = asset_id
                    logger.info(
                        f"✓ Matched '{asset_ref.name}' to existing "
                        f"'{asset.name}' (tolerant, {'party' if is_party else 'npc'})"
                    )
                    break

        if existing_asset_id:
            # Asset exists, use existing one
            if asset_ref.is_visible:
                visible_asset_ids.append(existing_asset_id)
            logger.info(
                f"✓ Reusing existing asset: '{asset_ref.name}' (matched with existing '{existing_assets[existing_asset_id].name}')"
            )
        else:
            # New asset - log all existing assets for debugging
            if existing_assets:
                existing_names = [f"'{a.name}'" for a in existing_assets.values()]
                logger.info(
                    f"✗ Creating NEW asset: '{asset_ref.name}' (type: {asset_ref.type}) - Did not match any existing: {', '.join(existing_names)}"
                )
            else:
                logger.info(
                    f"✓ Creating first asset: '{asset_ref.name}' (type: {asset_ref.type})"
                )

            new_asset = Asset(
                name=asset_ref.name,
                type=asset_ref.type,
                description=asset_ref.description,
            )

            # Generate image for the new asset
            try:
                asset_image_path = generate_asset_image_sync(
                    generator,
                    game_id,
                    new_asset.id,
                    new_asset.name,
                    new_asset.description,
                    art_style,
                )
                if asset_image_path:
                    new_asset.image_path = (
                        f"/static/assets/{game_id}/{asset_image_path.name}"
                    )
                logger.info("  → Asset image generated successfully")
            except Exception as e:
                logger.error(
                    f"Failed to generate image for asset {new_asset.name}: {e}"
                )

            updated_assets[new_asset.id] = new_asset

            if asset_ref.is_visible:
                visible_asset_ids.append(new_asset.id)

    return updated_assets, visible_asset_ids


def _process_scene_location(
    location_reference: LocationReference | None,
    existing_locations: dict[str, Location],
) -> tuple[dict[str, Location], LocationReference | None]:
    """Process location from generated scene and create new location if needed.

    Args:
        location_reference: LocationReference from the generated scene (may be None)
        existing_locations: Dictionary of existing locations

    Returns:
        Tuple of (updated locations dictionary, processed location reference)
    """
    if location_reference is None:
        return existing_locations, None

    updated_locations = existing_locations.copy()

    # Check if location_id is provided and exists
    if location_reference.location_id in existing_locations:
        # Reusing existing location
        existing_loc = existing_locations[location_reference.location_id]
        logger.info(
            f"✓ Reusing existing location: '{existing_loc.name}' (ID: {location_reference.location_id})"
        )
        return updated_locations, location_reference

    # Location ID not found - this shouldn't normally happen as LLM should provide valid IDs
    # But if it does, we'll just return without the location rather than creating incomplete data
    logger.warning(
        f"⚠ Location ID {location_reference.location_id} not found in existing locations. Scene will have no location reference."
    )
    return updated_locations, None


async def generate_opening_scene(
    game_id: str,
    scenario_name: str,
    scenario_details: str,
    characters: list[Character],
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
    scene_id: int = 1,
    art_style: str | None = None,
    scale: str = "any",
) -> tuple[Scene, list[Character], dict[str, Asset], dict[str, Location]]:
    """Generate the opening scene for the chosen scenario.

    Args:
        game_id: Unique identifier for the game (for image storage)
        scenario_name: Name of the chosen scenario
        scenario_details: Full markdown details about the scenario
        characters: List of Character objects in the party (with full stats, inventory, backstory)
        existing_assets: Dictionary of existing assets (NPCs/objects) for consistency
        existing_locations: Dictionary of existing locations for consistency
        scene_id: Scene identifier (default 1 for opening)

    Returns:
        Tuple of (Scene object, Updated character list, Updated assets dictionary, Updated locations dictionary)
    """
    agent = text_agent(GeneratedScene)

    logger.info(f"Generating opening scene for: {scenario_name}")

    # Build detailed character information for context
    character_sheets = "\n\n".join(
        [
            f"**{char.name}**\n"
            f"- Gender: {getattr(char, 'gender', 'unspecified')} (use matching pronouns)\n"
            f"- Stats: Strength {char.strength}, Intelligence {char.intelligence}, Agility {char.agility}\n"
            f"- Health: {char.current_health}/{char.maximum_health}\n"
            f"- Backstory: {char.backstory}\n"
            f"- Appearance: {char.appearance}\n"
            f"- Personality: {char.personality}\n"
            f"- Skills: {', '.join(char.skills) if char.skills else 'None'}\n"
            f"- Inventory: {format_inventory(char.inventory)}"
            for char in characters
        ]
    )

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}
{scene_scale_section(scale)}
PARTY CHARACTER SHEETS:
{character_sheets}

{format_existing_assets(existing_assets)}

{format_existing_locations(existing_locations)}

Generate the opening scene for this adventure. Create an engaging, atmospheric introduction that:

CRITICAL FOR OPENING SCENE:
1. **Ground how the party arrived HERE**: Before anything happens, establish how these characters
   came to be at this exact place — their journey in, why they're passing through, or the ordinary
   moment they are in as the scene opens. Use their backstories for natural connections (travelling
   together, hired on, drawn by rumor, chance). 2-4 sentences of grounded on-ramp so the start feels
   lived-in and earned, not dropped in cold.

2. **Ease into the situation — do NOT cold-open with a quest-giver**: Avoid starting with an elder,
   mayor, or official who immediately tells the party to go somewhere and do something — that feels
   forced. Let the place and a natural moment draw them in first: something they notice, overhear,
   stumble into, or are asked in passing. The hook should emerge from lived experience, not an
   exposition dump.

3. **Introduce the goal naturally and at the right pace**: Players don't know the scenario details,
   so reveal the objective gradually through what they see and do — not a briefing. Give a reason to
   care and a direction, but a smaller, closer hook first is good; let any larger stakes surface over
   time rather than all at once.

4. **Set the scene and mood**: Create atmosphere with vivid descriptions
   - Describe the environment, sounds, smells, lighting
   - Build tension or intrigue appropriate to the scenario

5. **Introduce party members naturally**: Reference characters by name, appearance, and personality
   - Show their personalities through brief actions or reactions
   - Consider their backstories when describing how they interact
   - Be aware of their stats, health, skills, and inventory

6. **End with a clear hook**: Give players an immediate decision or action to take

REMEMBER: This is the first thing players experience. They need to understand:
- Who they are (as a group)
- Where they are
- What their goal/quest is
- What they should do next

{build_scene_generation_prompt_rules()}

Make the scene immersive, clear, and exciting!
"""

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Log generated scene details
    logger.info(f"=== Opening Scene Generated (Scene {scene_id}) ===")
    logger.info(
        f"Generated scene model: {generated.model_dump_json(indent=2, exclude={'scene_text'})}"
    )

    # Apply character updates from the scene
    updated_characters = apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread (local backend).
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(shared_image_generator)

    asset_task = asyncio.to_thread(
        _process_scene_assets,
        game_id,
        scene_id,
        generated.assets_present,
        existing_assets,
        generator,
        art_style,
    )

    # Wait for asset processing to complete so we have asset images for scene generation
    updated_assets, visible_asset_ids = await asset_task
    if isinstance(updated_assets, Exception):
        logger.error(f"Asset processing failed: {updated_assets}")
        updated_assets = existing_assets
        visible_asset_ids = []

    # Reference only the art-directed in-frame subjects (reference isolation) so the
    # image fits the beat and other figures don't copy a referenced character's look.
    image_ref_ids, subject_directions = resolve_image_subject_ids(
        generated.image_subjects, updated_assets, visible_asset_ids
    )

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation (generator acquired above)
    image_task = asyncio.to_thread(
        generate_scene_image_sync,
        generator,
        game_id,
        scene_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        image_ref_ids,
        art_style,
        subject_directions,
    )

    voiceover_task = generate_scene_voiceover(
        game_id,
        scene_id,
        generated.scene_text,
        updated_characters,
        updated_assets,
        generated.narration_segments,
    )

    # Wait for both to complete
    image_file_path, voiceover_file_path = await asyncio.gather(
        image_task, voiceover_task, return_exceptions=True
    )

    # Handle exceptions
    if isinstance(image_file_path, BaseException):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, BaseException):
        logger.error(f"Voiceover generation failed: {voiceover_file_path}")
        voiceover_file_path = None

    # Convert to web paths if generated
    image_web_path = None
    if image_file_path:
        image_web_path = f"/static/scenes/{game_id}/{image_file_path.name}"

    voiceover_web_path = None
    if voiceover_file_path:
        voiceover_web_path = f"/static/voiceovers/{game_id}/{voiceover_file_path.name}"

    scene = _make_scene(
        scene_id,
        generated.scene_text,
        generated.prompt,
        image_web_path,
        generated.game_status,
        visible_asset_ids,
        health_changes=generated.health_changes,
        inventory_changes=generated.inventory_changes,
        skill_changes=generated.skill_changes,
        stat_changes=generated.stat_changes,
    )
    scene.voiceover_path = voiceover_web_path
    scene.location_reference = processed_location_ref
    return scene, updated_characters, updated_assets, updated_locations


async def generate_next_scene(
    game_id: str,
    scenario_name: str,
    scenario_details: str,
    characters: list[Character],
    last_scene_id: int,
    player_action: str | None,
    conversation_history: list[dict],
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
    art_style: str | None = None,
    scale: str = "any",
) -> tuple[Scene, list[Character], dict[str, Asset], dict[str, Location]]:
    """Generate the next scene based on the story so far and player action.

    Args:
        game_id: Unique identifier for the game (for image storage)
        scenario_name: Name of the scenario
        scenario_details: Full markdown details about the scenario
        characters: List of Character objects in the party (with full stats, inventory, backstory)
        last_scene_id: ID of the previous scene
        player_action: The player's response to the last prompt
        conversation_history: List of dicts containing scene_text, prompt, and player_action for full context
        existing_assets: Dictionary of existing assets (NPCs/objects) for consistency
        existing_locations: Dictionary of existing locations for consistency

    Returns:
        Tuple of (Scene object, Updated character list, Updated assets dictionary, Updated locations dictionary)
    """
    agent = text_agent(GeneratedScene)

    next_id = last_scene_id + 1
    logger.info(f"Generating scene {next_id} for: {scenario_name}")

    # Build detailed character information for context
    # IMPORTANT: These are CURRENT states as they ENTER this scene, not changes to apply
    character_sheets = "\n\n".join(
        [
            f"**{char.name}** (Current State Entering Scene {next_id})\n"
            f"- Gender: {getattr(char, 'gender', 'unspecified')} (use matching pronouns)\n"
            f"- Stats: Strength {char.strength}, Intelligence {char.intelligence}, Agility {char.agility}\n"
            f"- Health: {char.current_health}/{char.maximum_health} (this is their CURRENT health, don't subtract from it again)\n"
            f"- Backstory: {char.backstory}\n"
            f"- Appearance: {char.appearance}\n"
            f"- Personality: {char.personality}\n"
            f"- Skills: {', '.join(char.skills) if char.skills else 'None'}\n"
            f"- Inventory: {format_inventory(char.inventory)} (current items, don't remove again unless used in THIS scene)"
            for char in characters
        ]
    )

    # Build complete conversation history showing the full exchange between DM and players
    history_context = ""
    if conversation_history:
        history_parts = []
        for entry in conversation_history:
            scene_part = f"**Scene {entry['scene_id']}:**\n{entry['scene_text']}"

            # Add the prompt that was given to the player
            if entry.get("prompt"):
                prompt_obj = entry["prompt"]
                target = (
                    f"{prompt_obj.target_character}"
                    if prompt_obj.target_character
                    else "Party"
                )
                prompt_type_icon = (
                    "🎲"
                    if prompt_obj.type == "dice_check"
                    else "💬"
                    if prompt_obj.type == "dialogue"
                    else "⚔️"
                )
                # Include dice type for dice checks so the AI knows what die was rolled
                prompt_detail = prompt_obj.prompt_text
                if prompt_obj.type == "dice_check" and prompt_obj.dice_type:
                    prompt_detail = (
                        f"{prompt_obj.prompt_text} (roll {prompt_obj.dice_type})"
                    )

                scene_part += (
                    f"\n\n{prompt_type_icon} **DM prompts {target}:** {prompt_detail}"
                )

            # Add the player's response
            if entry.get("player_action"):
                scene_part += f"\n\n**Player responded:** {entry['player_action']}"

            history_parts.append(scene_part)

        history_context = "FULL CONVERSATION HISTORY:\n\n" + "\n\n---\n\n".join(
            history_parts
        )

    # Current action context (this is the response to the last prompt)
    action_context = (
        f"\n\nCURRENT PLAYER ACTION: {player_action}" if player_action else ""
    )

    # Check if the last prompt was a dice_check to include resolution rules
    dice_check_rules = ""
    if conversation_history and len(conversation_history) > 0:
        last_entry = conversation_history[-1]
        if last_entry.get("prompt") and last_entry["prompt"].type == "dice_check":
            dice_check_rules = f"\n\n{build_dice_check_resolution_rules()}"

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}
{scene_scale_section(scale)}
PARTY CHARACTER SHEETS:
{character_sheets}

{format_existing_assets(existing_assets)}

{format_existing_locations(existing_locations)}

{history_context}{action_context}{dice_check_rules}

Generate the next scene in this adventure. The scene should:
1. Respond naturally to the player's action (if provided)
2. Advance the story toward the main quest
3. Introduce new challenges, discoveries, or NPCs as appropriate
4. Maintain tension and engagement
5. Reference party members when relevant (consider their backstories, appearance, personality, skills, and capabilities)
6. Be aware of their CURRENT stats, health, skills, and inventory - check if they actually have items they try to use
7. Adjust challenge difficulty based on party's current health and capabilities
8. Create opportunities for characters to use their unique skills

{build_impossible_action_rules()}

{build_scene_generation_prompt_rules()}

Continue the adventure!
"""

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Log generated scene details
    logger.info(f"=== Next Scene Generated (Scene {next_id}) ===")
    logger.info(
        f"Generated scene model: {generated.model_dump_json(indent=2, exclude={'scene_text'})}"
    )

    # Apply character updates from the scene
    updated_characters = apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread (local backend).
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(shared_image_generator)

    asset_task = asyncio.to_thread(
        _process_scene_assets,
        game_id,
        next_id,
        generated.assets_present,
        existing_assets,
        generator,
        art_style,
    )

    # Wait for asset processing to complete so we have asset images for scene generation
    updated_assets, visible_asset_ids = await asset_task
    if isinstance(updated_assets, Exception):
        logger.error(f"Asset processing failed: {updated_assets}")
        updated_assets = existing_assets
        visible_asset_ids = []

    # Reference only the art-directed in-frame subjects (reference isolation) so the
    # image fits the beat and other figures don't copy a referenced character's look.
    image_ref_ids, subject_directions = resolve_image_subject_ids(
        generated.image_subjects, updated_assets, visible_asset_ids
    )

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation (generator acquired above)
    image_task = asyncio.to_thread(
        generate_scene_image_sync,
        generator,
        game_id,
        next_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        image_ref_ids,
        art_style,
        subject_directions,
    )

    voiceover_task = generate_scene_voiceover(
        game_id,
        next_id,
        generated.scene_text,
        updated_characters,
        updated_assets,
        generated.narration_segments,
    )

    # Wait for both to complete
    image_file_path, voiceover_file_path = await asyncio.gather(
        image_task, voiceover_task, return_exceptions=True
    )

    # Handle exceptions
    if isinstance(image_file_path, BaseException):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, BaseException):
        logger.error(f"Voiceover generation failed: {voiceover_file_path}")
        voiceover_file_path = None

    # Convert to web paths if generated
    image_web_path = None
    if image_file_path:
        image_web_path = f"/static/scenes/{game_id}/{image_file_path.name}"

    voiceover_web_path = None
    if voiceover_file_path:
        voiceover_web_path = f"/static/voiceovers/{game_id}/{voiceover_file_path.name}"

    scene = _make_scene(
        next_id,
        generated.scene_text,
        generated.prompt,
        image_web_path,
        generated.game_status,
        visible_asset_ids,
        health_changes=generated.health_changes,
        inventory_changes=generated.inventory_changes,
        skill_changes=generated.skill_changes,
        stat_changes=generated.stat_changes,
    )
    scene.voiceover_path = voiceover_web_path
    scene.location_reference = processed_location_ref
    return scene, updated_characters, updated_assets, updated_locations


async def generate_scene_recap(
    game_id: str,
    scenario_name: str,
    scenes: list[Scene],
    characters: list[Character],
    current_scene_index: int,
) -> RecapResponse:
    """Generate a narrative recap with per-scene summaries.

    Args:
        game_id: Unique identifier for the game
        scenario_name: Name of the scenario being played
        scenes: List of all scenes in the game
        characters: Current party characters
        current_scene_index: Index of the current scene (0-based)

    Returns:
        RecapResponse containing overall recap text and individual scene summaries
    """
    agent = text_agent(RecapResponse)

    logger.info(
        f"Generating recap for game {game_id}, scenes 1-{current_scene_index + 1}"
    )

    # Build scene summaries for context
    scene_summaries = []
    for scene in scenes[: current_scene_index + 1]:
        summary = f"**Scene {scene.id}:**\n{scene.text[:500]}..."
        if scene.prompt:
            summary += "\n*Player choice was needed here*"
        scene_summaries.append(summary)

    scenes_context = "\n\n".join(scene_summaries)

    # Build character status
    character_status = "\n".join(
        [
            f"- **{char.name}**: HP {char.current_health}/{char.maximum_health}, "
            f"Skills: {', '.join(char.skills[:3]) if char.skills else 'None'}"
            for char in characters
        ]
    )

    prompt = f"""You are a skilled storyteller creating a recap for players returning to their adventure after a break.

SCENARIO: {scenario_name}

CURRENT PARTY STATUS:
{character_status}

SCENES SO FAR:
{scenes_context}

Generate TWO things:

1. **recap_text**: An engaging, concise overall recap (200-300 words) that:
   - Reminds players of the adventure's main goal
   - Summarizes the key events that have happened so far
   - Highlights any important choices the party made
   - Notes any significant character developments (injuries, items gained, etc.)
   - Sets the stage for where the story is now
   - Uses an exciting, narrative tone as if read by a Dungeon Master
   - Written in past tense, like "Previously on [Adventure Name]..."

2. **scene_summaries**: For EACH scene (scenes 1-{current_scene_index + 1}), provide:
   - scene_id: The scene number
   - summary: A concise one-sentence summary (15-20 words) capturing the key event/development
   
   Make each scene summary punchy and informative. Focus on:
   - What significant thing happened (discovery, combat, decision, meeting)
   - Who was involved (specific characters if relevant)
   - The immediate consequence or outcome
   
   Examples of good scene summaries:
   - "The party discovered an ancient temple hidden in the misty mountains."
   - "Theron negotiated with the bandit chief, securing safe passage through the forest."
   - "A fierce battle with goblins left the party wounded but victorious."
   - "The mysterious wizard revealed the location of the legendary artifact."

Return your response in the RecapResponse format with both recap_text and scene_summaries array.
"""

    try:
        result = await retry_on_overload(agent.run, prompt)
        recap_response = result.output
        logger.info(
            f"Successfully generated recap ({len(recap_response.recap_text)} chars, {len(recap_response.scene_summaries)} scene summaries)"
        )
        return recap_response
    except Exception as e:
        logger.error(f"Failed to generate recap: {e}")
        # Return a simple fallback recap
        fallback_summaries = [
            SceneSummary(
                scene_id=scene.id, summary=f"Scene {scene.id}: Adventure continues..."
            )
            for scene in scenes[: current_scene_index + 1]
        ]
        return RecapResponse(
            recap_text=f"""## Previously in {scenario_name}...

Your party has journeyed through {current_scene_index + 1} scene(s) of adventure. 
The story continues with your brave heroes facing new challenges.

Current party status:
{character_status}
""",
            scene_summaries=fallback_summaries,
        )

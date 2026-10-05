"""Scenario templates and their opening locations."""

import logging

from ..llm import retry_on_overload, text_agent
from ..models import (
    GeneratedLocationList,
    GeneratedScenarioTemplate,
    Location,
    ScenarioTemplate,
)
from ..prompts import (
    SCENARIO_SCALES,
    SCENARIO_THEMES,
    avoid_names_section,
)

logger = logging.getLogger(__name__)


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

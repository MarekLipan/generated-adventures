"""
Handles content generation via external APIs.

This file currently contains placeholder deterministic generators. Replace
these with real model calls (Gemini, OpenAI, etc.) later.
"""

import asyncio
import logging
import pathlib
from io import BytesIO
from typing import List, Optional

from google import genai as genai_client  # type: ignore
from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from .config import settings
from .models import (
    Character,
    GeneratedCharacterList,
    GeneratedScenarioDetails,
    GeneratedScenarios,
    GeneratedScene,
    Scene,
)

logger = logging.getLogger()  # Use root logger so Uvicorn/NiceGUI picks up logs
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)

# NOTE: google.generativeai high-level configure may not export consistently in current lib version; using provider-based auth only.

IMAGE_DIR = pathlib.Path("webapp/static/characters")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

SCENE_IMAGE_DIR = pathlib.Path("webapp/static/scenes")
SCENE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _generate_character_image_sync(
    client,
    concept,
    scenario_name: str,
    game_id: str,
    game_dir: pathlib.Path,
    idx: int,
) -> tuple[str, pathlib.Path | None]:
    """Synchronous helper to generate a single character image.

    Returns: (character_name, image_file_path or None)
    """
    # Build detailed prompt using all character information
    prompt_parts = [
        "Generate a high quality fantasy character portrait.",
        f"Character Name: {concept.name}",
        f"Scenario: {scenario_name}",
        f"Appearance: {concept.appearance}",
        f"Personality: {concept.personality}",
        f"Backstory Context: {concept.backstory}",
    ]

    # Add key inventory items that would be visible
    if concept.inventory:
        visible_items = ", ".join(
            concept.inventory[:3]
        )  # First 3 items most likely to be visible
        prompt_parts.append(f"Notable Equipment: {visible_items}")

    # Add character stats (let the LLM interpret physical implications)
    prompt_parts.append(
        f"Character Stats (scale 1-20): "
        f"Strength {concept.strength}, "
        f"Intelligence {concept.intelligence}, "
        f"Agility {concept.agility}"
    )

    prompt_parts.append(
        "Style: painterly, dramatic lighting, 3/4 view, no text, clean background, high detail."
    )

    prompt_img = " ".join(prompt_parts)
    image_file_path = None
    logger.info(f"Generating image for character: {concept.name}")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt_img],
        )
    except (RuntimeError, ValueError, OSError):  # request-level failure
        response = None
        logger.warning(f"Image generation request failed for {concept.name}")

    if response is not None:
        # Iterate parts, save first inline image
        try:
            for part in response.candidates[0].content.parts:  # type: ignore[index]
                if getattr(part, "inline_data", None) and getattr(
                    part.inline_data, "data", None
                ):
                    img = Image.open(BytesIO(part.inline_data.data))
                    safe_name = (
                        "".join(c for c in concept.name if c.isalnum() or c in " _-")
                        .rstrip()
                        .replace(" ", "_")
                    )
                    filename = f"{idx:02d}_{safe_name or 'character'}.png"
                    image_file_path = game_dir / filename
                    img.save(image_file_path)
                    logger.info(f"Saved image for {concept.name} at {image_file_path}")
                    break
        except (OSError, ValueError):  # image decode/save issues
            image_file_path = None
            logger.warning(f"Failed to decode/save image for {concept.name}")

    return concept.name, image_file_path


def _generate_scene_image_sync(
    client,
    game_id: str,
    scene_id: int,
    scene_text: str,
    characters: List[Character],
    scenario_name: str,
    previous_scene_image_path: Optional[pathlib.Path] = None,
) -> pathlib.Path | None:
    """Synchronous helper to generate a scene image.

    Args:
        client: Google GenAI client
        game_id: Game identifier for directory structure
        scene_id: Scene number
        scene_text: The narrative text of the scene
        characters: List of characters in the party (for reference images)
        scenario_name: Name of the scenario
        previous_scene_image_path: Optional path to the previous scene's image for visual continuity

    Returns:
        Path to saved scene image or None if generation failed
    """
    from google.genai import types as genai_types  # type: ignore

    # Create scene directory for this game
    scene_dir = SCENE_IMAGE_DIR / game_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt with scene context
    prompt_parts = [
        "CRITICAL: Generate image in LANDSCAPE/HORIZONTAL orientation - width MUST be greater than height (16:9 aspect ratio).",
        "Generate a high quality fantasy scene illustration.",
        "Based on this narrative:",
        f"\n{scene_text}\n",
        f"Scenario: {scenario_name}",
        "\nIMPORTANT CHARACTER INSTRUCTIONS:",
        "- Characters should have DYNAMIC poses and body language that match the action/emotion of the scene",
        "- Show appropriate facial expressions and emotions (fear, determination, excitement, etc.)",
        "- Position characters naturally engaged in the scene's action, not standing static",
        "- If combat/action: show fighting stances, movement, tension",
        "- If dialogue/social: show gestures, interactions between characters",
        "- If exploration: show characters examining, pointing, reacting to environment",
        "- Match each character's personality traits, mannerisms, and nature to their pose, expression, and body language",
        "- Use the character reference images for appearance (the images have personality context embedded)",
    ]

    # Add note about previous scene if available
    if previous_scene_image_path:
        prompt_parts.append(
            "- Use the previous scene image as a reference for visual continuity in style, lighting, and environment"
        )

    prompt_parts.extend(
        [
            "\nStyle: cinematic landscape composition, painterly, dramatic lighting, atmospheric, fantasy art, detailed environment, wide angle view.",
            "\nCRITICAL FORMAT REQUIREMENT: Image MUST be in LANDSCAPE/HORIZONTAL orientation (wider than tall). Aspect ratio 16:9 or similar wide format.",
        ]
    )

    prompt_text = " ".join(prompt_parts)

    # Prepare content with text prompt and reference images
    content_parts = [prompt_text]

    # Add previous scene image for visual continuity (if available)
    if previous_scene_image_path and previous_scene_image_path.exists():
        try:
            with open(previous_scene_image_path, "rb") as f:
                prev_image_data = f.read()
            content_parts.append(
                genai_types.Part.from_bytes(data=prev_image_data, mime_type="image/png")
            )
            logger.info(
                f"Added previous scene image for continuity: {previous_scene_image_path.name}"
            )
        except Exception as e:
            logger.warning(f"Could not load previous scene image: {e}")

    # Add character images as reference (if they exist)
    for char in characters:
        if char.image_path:
            # Convert web path to file path
            char_image_file = pathlib.Path("webapp") / char.image_path.lstrip("/")
            if char_image_file.exists():
                try:
                    with open(char_image_file, "rb") as f:
                        image_data = f.read()
                    content_parts.append(
                        genai_types.Part.from_bytes(
                            data=image_data, mime_type="image/png"
                        )
                    )
                    logger.info(f"Added reference image for {char.name}")
                except Exception as e:
                    logger.warning(
                        f"Could not load character image for {char.name}: {e}"
                    )

    logger.info(f"Generating scene image for scene {scene_id} in game {game_id}")

    try:
        from google.genai import types as genai_config  # type: ignore

        # Try to configure landscape aspect ratio
        config = genai_config.GenerateContentConfig(
            response_modalities=["image"],
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=content_parts,
            config=config,
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(
            f"Scene image generation request failed for scene {scene_id}: {e}"
        )
        return None

    if response is not None:
        try:
            for part in response.candidates[0].content.parts:  # type: ignore[index]
                if getattr(part, "inline_data", None) and getattr(
                    part.inline_data, "data", None
                ):
                    img = Image.open(BytesIO(part.inline_data.data))
                    filename = f"scene_{scene_id:03d}.png"
                    image_file_path = scene_dir / filename
                    img.save(image_file_path)
                    logger.info(f"Saved scene image at {image_file_path}")
                    return image_file_path
        except (OSError, ValueError) as e:
            logger.warning(
                f"Failed to decode/save scene image for scene {scene_id}: {e}"
            )
            return None

    return None


async def generate_scenarios() -> list[str]:
    """Calls Google Gemini to generate scenarios."""
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedScenarios)

    logger.info("Generating scenarios...")
    result = await agent.run(
        "Generate three distinct fantasy adventure scenarios. Provide only the names.",
    )

    return result.output.scenarios


async def generate_characters(
    game_id: str,
    scenario_name: str,
    num_characters: int = 6,
    scenario_details: Optional[str] = None,
) -> List[Character]:
    """Generates a list of characters with stats and portraits for a given scenario.

    Args:
        game_id: Unique identifier for the game (used for image storage)
        scenario_name: Name of the scenario
        num_characters: Number of characters to generate
        scenario_details: Optional markdown details about the scenario (setting, plot, NPCs, etc.)
    """
    # Step 1: Generate character concepts (name, stats)

    logger.info(f"Generating {num_characters} characters for scenario: {scenario_name}")
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedCharacterList)

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
    concept_result = await agent.run(prompt)
    generated_concepts = concept_result.output.characters
    logger.info(f"Character concepts generated: {[c.name for c in generated_concepts]}")

    # Step 2: Synchronous image generation using working google.genai client
    # Create game-specific directory for character images
    game_dir = IMAGE_DIR / game_id
    game_dir.mkdir(parents=True, exist_ok=True)

    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)  # type: ignore[attr-defined]

    # Step 3: Generate images asynchronously (in thread pool to avoid blocking event loop)
    final_characters: List[Character] = []

    # Generate all images concurrently using asyncio.to_thread
    image_tasks = [
        asyncio.to_thread(
            _generate_character_image_sync,
            client,
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
    for concept, (char_name, image_file_path) in zip(generated_concepts, image_results):
        if isinstance(image_file_path, Exception):
            logger.error(
                f"Image generation failed for {concept.name}: {image_file_path}"
            )
            image_file_path = None

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


async def generate_scenario_details(scenario_name: str) -> str:
    """Generates the detailed story for the selected scenario.

    Returns a markdown string intended for DM notes.
    """
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedScenarioDetails)

    logger.info(f"Generating scenario details for: {scenario_name}")
    prompt = f"""
You are a creative fantasy adventure designer.
Return ONLY JSON matching this pydantic model fields (no markdown, no code fences):
{{
  "setting": "...",
  "plot": "...",
  "main_quest": "...",
  "important_npcs": "..."
}}
Each field should be 2-4 paragraphs of rich, game-master-usable content for the scenario '{scenario_name}'.
"""
    result = await agent.run(prompt)
    data = result.output
    markdown = (
        f"## The Setting\n\n{data.setting}\n\n"
        f"## The Plot\n\n{data.plot}\n\n"
        f"## Main Quest\n\n{data.main_quest}\n\n"
        f"## Important NPCs\n\n{data.important_npcs}\n"
    )
    return markdown


def _apply_character_updates(
    characters: List[Character], updated_generated_chars: List
) -> List[Character]:
    """Apply character state updates from regenerated character data.

    Args:
        characters: List of current character objects (with image_path, etc.)
        updated_generated_chars: List of GeneratedCharacter objects from the LLM

    Returns:
        Updated list of characters with modifications applied
    """
    # Create a dictionary for quick character lookup by name
    char_dict = {char.name: char for char in characters}

    for updated_char in updated_generated_chars:
        if updated_char.name not in char_dict:
            logger.warning(
                f"Character update for unknown character: {updated_char.name}. Skipping."
            )
            continue

        char = char_dict[updated_char.name]
        changes = []

        # Track changes for logging
        if char.strength != updated_char.strength:
            changes.append(f"STR: {char.strength} -> {updated_char.strength}")
            char.strength = updated_char.strength

        if char.intelligence != updated_char.intelligence:
            changes.append(f"INT: {char.intelligence} -> {updated_char.intelligence}")
            char.intelligence = updated_char.intelligence

        if char.agility != updated_char.agility:
            changes.append(f"AGI: {char.agility} -> {updated_char.agility}")
            char.agility = updated_char.agility

        # Update health (clamp to maximum)
        if char.current_health != updated_char.current_health:
            old_health = char.current_health
            char.current_health = max(
                0, min(updated_char.current_health, char.maximum_health)
            )
            changes.append(f"Health: {old_health} -> {char.current_health}")

        # Update inventory (replace entirely with new inventory)
        old_inventory = set(char.inventory)
        new_inventory = set(updated_char.inventory)

        added = new_inventory - old_inventory
        removed = old_inventory - new_inventory

        if added:
            changes.append(f"Gained: {', '.join(added)}")
        if removed:
            changes.append(f"Lost: {', '.join(removed)}")

        char.inventory = updated_char.inventory

        # Update skills (track changes like inventory)
        old_skills = set(char.skills)
        new_skills = set(updated_char.skills)

        gained_skills = new_skills - old_skills
        lost_skills = old_skills - new_skills

        if gained_skills:
            changes.append(f"Learned: {', '.join(gained_skills)}")
        if lost_skills:
            changes.append(f"Forgot: {', '.join(lost_skills)}")

        char.skills = updated_char.skills

        # Update backstory, appearance, and personality if they changed (shouldn't normally change)
        if char.backstory != updated_char.backstory:
            char.backstory = updated_char.backstory
        if char.appearance != updated_char.appearance:
            char.appearance = updated_char.appearance
        if char.personality != updated_char.personality:
            char.personality = updated_char.personality

        if changes:
            logger.info(f"Updated {char.name}: {', '.join(changes)}")

    return characters


def _make_scene(
    scene_id: int, scene_text: str, prompt, image_path: str | None = None
) -> Scene:
    """Helper to create a Scene object from generated data.

    Args:
        scene_id: Unique scene identifier
        scene_text: The narrative text of the scene
        prompt: PromptType object from the generated scene
        image_path: Optional path to the scene image
    """
    return Scene(
        id=scene_id,
        text=scene_text,
        prompt=prompt,
        image_path=image_path,
        voiceover_path=None,
    )


def _build_scene_generation_prompt_rules() -> str:
    """Returns the common rules section for scene generation prompts."""
    return """
After the scene narrative, provide a prompt for player interaction. The prompt should be:
- Either for the entire party or a specific character
- One of three types:
  * dialogue: Character should speak (player enters text in quotes)
  * action: Player describes what they do (for simple, non-risky actions)
  * dice_check: Player must roll dice (for ANY risky action, test of skill, or combat)

CRITICAL RULES FOR PROMPT TYPES - VARY THE PROMPTS:
- DO NOT use dice_check for every scene - mix dialogue, action, and dice_check
- Use dialogue when: characters need to talk to NPCs, negotiate, roleplay conversations
- Use action for: simple tasks, exploration without immediate danger, planning, easy skill checks
- Use dice_check for: combat, risky actions, difficult skill checks, life-or-death situations

DICE CHECK RULES (when using dice_check type):
- Choose dice type based on difficulty:
  * d6: Simple to moderate challenges (lockpicking, climbing, basic combat)
  * d10: Challenging situations (difficult combat, dangerous magic, persuading hostile NPCs)
- Choose dice count based on the situation:
  * 1 die: Single character making one attempt
  * 2-3 dice: Multiple attempts, team effort, or more dramatic moments
  * 4+ dice: Epic moments, multiple characters acting together
- Vary your dice choices - don't always use the same type and count!
- Remember: The next scene will interpret the roll with stat modifiers (+2 for 16-20, +1 for 11-15, +0 for 6-10, -1 for 1-5) and skill bonuses

TARGETING RULES:
- Use target_character to address a single specific party member when their unique skills/abilities are relevant
- Use target_characters (array) for dice_check prompts when multiple specific characters need to roll simultaneously
  * Example: Combat where multiple characters attack at once
  * Format: target_characters: ["Character1", "Character2"], target_character: null
  * The prompt_text should mention each character and what they're rolling for
- Use null for target_character (and omit target_characters) when entire party responds or any character could act
- Consider each character's skills when creating prompts and challenges

CRITICAL RULES FOR CHARACTER UPDATES:
- ALWAYS return updated_characters array with ALL party members
- Re-generate complete character data (stats, health, skills, inventory) based on what happened in the scene
- Update current_health for characters who took damage or were healed (keep stats same unless permanently changed)
- Update skills if a character learns a new ability or loses one (rare, but possible)
- Update inventory to reflect items gained, lost, used, or consumed
- If nothing happened to a character, return them unchanged
- Keep backstory, appearance, and personality the same (these don't change during adventures)

Return your response in this JSON structure:
{
  "scene_text": "The vivid narrative of the scene...",
  "prompt": {
    "type": "dialogue" | "action" | "dice_check",
    "dice_type": "d6" | "d10" (only if type is dice_check),
    "dice_count": 1 to 6 (only if type is dice_check),
    "target_character": "Character Name" or null (use for single character or entire party),
    "target_characters": ["Char1", "Char2"] or null (use for multi-character dice_check only),
    "prompt_text": "The question or instruction for the player(s)"
  },
  "updated_characters": [
    {
      "name": "Character Name",
      "strength": stat_value,
      "intelligence": stat_value,
      "agility": stat_value,
      "current_health": current_health_value,
      "backstory": "same as before",
      "appearance": "same as before",
      "personality": "same as before",
      "skills": ["current", "skills", "list"],
      "inventory": ["current", "items", "list"]
    }
  ]
}
"""


def _build_impossible_action_rules() -> str:
    """Returns rules for handling impossible player actions."""
    return """
CRITICAL RULES FOR HANDLING PLAYER ACTIONS:
- If player attempts an IMPOSSIBLE action (using items they don't have, doing something beyond their character's capabilities, breaking world logic):
  * DO NOT allow the action to succeed
  * In scene_text, explain why it's not possible
  * Prompt them to try something different (use "action" type prompt)
  * Example: "You reach for the magic sword, but you don't have such an item. What do you do instead?"

- If player describes an action that should undergo a test or challenge:
  * DO NOT resolve it immediately
  * Prompt for dice_check to determine success/failure
  * Examples: attacking, dodging, climbing, lockpicking, persuading, searching, casting difficult spells

- COMBAT and FIGHTING scenarios MUST use dice_check prompts:
  * Any attack, defense, or combat maneuver requires dice
  * Multiple characters in combat should each get dice_check prompts
  * Never auto-resolve combat without dice rolls
"""


def _build_dice_check_resolution_rules() -> str:
    """Returns rules for resolving dice check results."""
    return """
CRITICAL RULES FOR DICE CHECK RESOLUTION:
When the previous prompt was a dice_check and the player provides their roll result(s):

NOTE: For multi-character dice checks, the player will provide rolls in format: "CharacterName1: 7, CharacterName2: 4"
Parse each character's roll separately and apply their individual modifiers.

1. INTERPRET THE ROLL with character stat modifiers:
   - For PHYSICAL actions (combat, climbing, swimming, breaking): Apply STRENGTH modifier
     * Strength 16-20: +2 to roll
     * Strength 11-15: +1 to roll
     * Strength 6-10: +0 to roll
     * Strength 1-5: -1 to roll
   
   - For MENTAL actions (arcane knowledge, puzzle solving, investigation): Apply INTELLIGENCE modifier
     * Intelligence 16-20: +2 to roll
     * Intelligence 11-15: +1 to roll
     * Intelligence 6-10: +0 to roll
     * Intelligence 1-5: -1 to roll
   
   - For DEXTERITY actions (lockpicking, dodging, stealth, acrobatics): Apply AGILITY modifier
     * Agility 16-20: +2 to roll
     * Agility 11-15: +1 to roll
     * Agility 6-10: +0 to roll
     * Agility 1-5: -1 to roll
   
   - For SOCIAL actions (persuasion, intimidation, deception): Apply INTELLIGENCE modifier (wit/charisma)

2. APPLY SKILL BONUSES:
   - If the character has a relevant skill, add +1 to the final result
   - Examples: "Lockpicking" skill for picking locks, "Swordsmanship" for melee combat, "Persuasion" for social checks

3. DETERMINE SUCCESS:
   - Calculate: Roll + Stat Modifier + Skill Bonus (if applicable)
   - For d6 checks: 4+ is success, 6+ is critical success, 2 or less is critical failure
   - For d10 checks: 6+ is success, 9+ is critical success, 3 or less is critical failure
   - Multiple dice: Sum all dice, then apply modifiers once to the total

4. NARRATE THE RESULT:
   - Explicitly mention the stat modifier and skill bonus in your narrative
   - Example format: "You rolled a 4. With your high Strength (+2) and relevant skill (+1), that's a total of 7 - a solid success!"
   - Example format: "You rolled a 3. Despite your Agility modifier (+1), you barely manage..."
   - Make outcomes dramatic and descriptive
   - Critical successes should have extra benefits
   - Critical failures should have interesting consequences

5. UPDATE CHARACTER STATE:
   - Successful combat: Enemy takes damage, may gain items/information
   - Failed combat: Character loses health (appropriate to danger level)
   - Successful skill checks: Progress story, gain items/allies/knowledge
   - Failed skill checks: Setbacks, complications, but not instant death unless extremely dangerous
"""


async def generate_opening_scene(
    game_id: str,
    scenario_name: str,
    scenario_details: str,
    characters: List[Character],
    scene_id: int = 1,
) -> tuple[Scene, List[Character]]:
    """Generate the opening scene for the chosen scenario.

    Args:
        game_id: Unique identifier for the game (for image storage)
        scenario_name: Name of the chosen scenario
        scenario_details: Full markdown details about the scenario
        characters: List of Character objects in the party (with full stats, inventory, backstory)
        scene_id: Scene identifier (default 1 for opening)

    Returns:
        Tuple of (Scene object with narrative text and player prompt, Updated character list)
    """
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedScene)

    logger.info(f"Generating opening scene for: {scenario_name}")

    # Build detailed character information for context
    character_sheets = "\n\n".join(
        [
            f"**{char.name}**\n"
            f"- Stats: Strength {char.strength}, Intelligence {char.intelligence}, Agility {char.agility}\n"
            f"- Health: {char.current_health}/{char.maximum_health}\n"
            f"- Backstory: {char.backstory}\n"
            f"- Appearance: {char.appearance}\n"
            f"- Personality: {char.personality}\n"
            f"- Skills: {', '.join(char.skills) if char.skills else 'None'}\n"
            f"- Inventory: {', '.join(char.inventory) if char.inventory else 'None'}"
            for char in characters
        ]
    )

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}

PARTY CHARACTER SHEETS:
{character_sheets}

Generate the opening scene for this adventure. Create an engaging, atmospheric introduction that:
1. Sets the scene and mood
2. Introduces the immediate situation
3. Gives players a clear hook into the adventure
4. References the party members naturally (consider their backstories, appearance, personality, skills, and capabilities)
5. Be aware of their current stats, health, skills, and inventory when setting up situations

{_build_scene_generation_prompt_rules()}

Make the scene immersive and exciting!
"""

    result = await agent.run(prompt)
    generated = result.output

    # Apply character updates from the scene
    updated_characters = _apply_character_updates(
        characters, generated.updated_characters
    )

    # Generate scene image in background thread
    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)  # type: ignore[attr-defined]
    image_file_path = await asyncio.to_thread(
        _generate_scene_image_sync,
        client,
        game_id,
        scene_id,
        generated.scene_text,
        characters,
        scenario_name,
        None,  # No previous scene for opening scene
    )

    # Convert to web path if image was generated
    image_web_path = None
    if image_file_path:
        image_web_path = f"/static/scenes/{game_id}/{image_file_path.name}"

    scene = _make_scene(
        scene_id, generated.scene_text, generated.prompt, image_web_path
    )
    return scene, updated_characters


async def generate_next_scene(
    game_id: str,
    scenario_name: str,
    scenario_details: str,
    characters: List[Character],
    last_scene_id: int,
    player_action: Optional[str],
    conversation_history: List[dict],
) -> tuple[Scene, List[Character]]:
    """Generate the next scene based on the story so far and player action.

    Args:
        game_id: Unique identifier for the game (for image storage)
        scenario_name: Name of the scenario
        scenario_details: Full markdown details about the scenario
        characters: List of Character objects in the party (with full stats, inventory, backstory)
        last_scene_id: ID of the previous scene
        player_action: The player's response to the last prompt
        conversation_history: List of dicts containing scene_text, prompt, and player_action for full context

    Returns:
        Tuple of (Scene object with narrative text and player prompt, Updated character list)
    """
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedScene)

    next_id = last_scene_id + 1
    logger.info(f"Generating scene {next_id} for: {scenario_name}")

    # Build detailed character information for context
    character_sheets = "\n\n".join(
        [
            f"**{char.name}**\n"
            f"- Stats: Strength {char.strength}, Intelligence {char.intelligence}, Agility {char.agility}\n"
            f"- Health: {char.current_health}/{char.maximum_health}\n"
            f"- Backstory: {char.backstory}\n"
            f"- Appearance: {char.appearance}\n"
            f"- Personality: {char.personality}\n"
            f"- Skills: {', '.join(char.skills) if char.skills else 'None'}\n"
            f"- Inventory: {', '.join(char.inventory) if char.inventory else 'None'}"
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
                scene_part += f"\n\n{prompt_type_icon} **DM prompts {target}:** {prompt_obj.prompt_text}"

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
            dice_check_rules = f"\n\n{_build_dice_check_resolution_rules()}"

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}

PARTY CHARACTER SHEETS:
{character_sheets}

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

{_build_impossible_action_rules()}

{_build_scene_generation_prompt_rules()}

Continue the adventure!
"""

    result = await agent.run(prompt)
    generated = result.output

    # Apply character updates from the scene
    updated_characters = _apply_character_updates(
        characters, generated.updated_characters
    )

    # Generate scene image in background thread
    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)  # type: ignore[attr-defined]

    # Construct path to previous scene image for visual continuity
    previous_scene_image_path = None
    if last_scene_id >= 1:
        previous_scene_image_path = (
            SCENE_IMAGE_DIR / game_id / f"scene_{last_scene_id:03d}.png"
        )

    image_file_path = await asyncio.to_thread(
        _generate_scene_image_sync,
        client,
        game_id,
        next_id,
        generated.scene_text,
        characters,
        scenario_name,
        previous_scene_image_path,
    )

    # Convert to web path if image was generated
    image_web_path = None
    if image_file_path:
        image_web_path = f"/static/scenes/{game_id}/{image_file_path.name}"

    scene = _make_scene(next_id, generated.scene_text, generated.prompt, image_web_path)
    return scene, updated_characters

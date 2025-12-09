"""
Handles content generation via external APIs.

This file currently contains placeholder deterministic generators. Replace
these with real model calls (Gemini, OpenAI, etc.) later.
"""

import asyncio
import logging
import pathlib
import wave
from io import BytesIO
from typing import Callable, List, Optional

from google import genai as genai_client  # type: ignore
from google.genai import types as genai_types  # type: ignore
from google.genai.errors import ClientError, ServerError  # type: ignore
from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from .config import settings
from .models import (
    Asset,
    Character,
    GameStatus,
    GeneratedCharacterList,
    GeneratedLocationList,
    GeneratedScenarioTemplate,
    GeneratedScene,
    Location,
    LocationReference,
    ScenarioTemplate,
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

VOICEOVER_DIR = pathlib.Path("webapp/static/voiceovers")
VOICEOVER_DIR.mkdir(parents=True, exist_ok=True)


# Retry configuration for API calls
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds
RETRY_BACKOFF = 2  # exponential backoff multiplier


async def retry_on_overload(func: Callable, *args, **kwargs) -> any:
    """Retry async function calls when the API is overloaded.

    Retries on 503 (Service Unavailable) and 429 (Too Many Requests) errors
    with exponential backoff.
    """
    last_exception = None
    delay = RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)
        except (ServerError, ClientError) as e:
            last_exception = e
            # Check if it's a retryable error (503, 429, or 500)
            if hasattr(e, "status_code"):
                status_code = e.status_code
            elif hasattr(e, "code"):
                status_code = e.code
            else:
                # Unknown error format, don't retry
                raise

            if status_code in (503, 429, 500):
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"API error {status_code} on attempt {attempt + 1}/{MAX_RETRIES}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    delay *= RETRY_BACKOFF  # exponential backoff
                else:
                    logger.error(
                        f"API error {status_code} after {MAX_RETRIES} attempts. Giving up."
                    )
                    raise
            else:
                # Non-retryable error (e.g., 400 Bad Request)
                raise
        except Exception as e:
            # Unexpected error, don't retry
            logger.error(f"Unexpected error during API call: {type(e).__name__}: {e}")
            raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


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
    visual_description: str,
    characters: List[Character],
    scenario_name: str,
    game_status: GameStatus = "ongoing",
    assets: dict[str, Asset] = None,
    visible_asset_ids: List[str] = None,
) -> pathlib.Path | None:
    """Synchronous helper to generate a scene image.

    Args:
        client: Google GenAI client
        game_id: Game identifier for directory structure
        scene_id: Scene number
        scene_text: The narrative text of the scene
        visual_description: Detailed visual description for image composition
        characters: List of characters in the party (for reference images)
        scenario_name: Name of the scenario
        game_status: Status of the game ('ongoing', 'completed', 'failed')
        assets: Dictionary of all assets (NPCs and objects)
        visible_asset_ids: List of asset IDs that should appear in this scene

    Returns:
        Path to saved scene image or None if generation failed
    """
    from google.genai import types as genai_types  # type: ignore

    # Create scene directory for this game
    scene_dir = SCENE_IMAGE_DIR / game_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt with scene context and visual description
    prompt_parts = [
        "🎯 CRITICAL PRIORITY #1 - CHARACTER & ASSET VISUAL CONSISTENCY:",
        "- REFERENCE IMAGES ARE PROVIDED BELOW FOR ALL VISIBLE CHARACTERS AND ASSETS - YOU MUST USE THEM",
        "- Every character (party members + NPCs) MUST look EXACTLY like their reference image",
        "- Every object/NPC MUST match their reference image exactly",
        "- This is THE MOST IMPORTANT requirement - consistency of appearance across all scenes",
        "- Match: facial features, hairstyle, hair color, clothing style, clothing colors, equipment, accessories",
        "- If a character/asset appears, they MUST be instantly recognizable from their reference image",
        "",
        "📐 FORMAT REQUIREMENT:",
        "- Generate in LANDSCAPE/HORIZONTAL orientation (16:9 aspect ratio, width > height)",
        "",
        "🎨 SCENE COMPOSITION:",
        f"{visual_description}",
        "",
        "📖 NARRATIVE CONTEXT:",
        f"{scene_text[:500]}..."
        if len(scene_text) > 500
        else scene_text,  # Limit context to reduce complexity
        "",
        "✅ ADDITIONAL GUIDELINES:",
        "- Only include characters/objects mentioned in the visual description",
        "- Natural poses and angles matching the composition (not just frontal portraits)",
        "- Painterly fantasy art style, cinematic, dramatic lighting",
        f"- Environment: {scenario_name} fantasy setting",
    ]

    # Add special instructions for ending scenes
    if game_status == "completed":
        prompt_parts.append("")
        prompt_parts.append(
            "🏆 VICTORY SCENE: Triumphant conclusion with uplifting atmosphere"
        )
    elif game_status == "failed":
        prompt_parts.append("")
        prompt_parts.append("💀 DEFEAT SCENE: Dramatic defeat with somber atmosphere")

    prompt_text = "\n".join(prompt_parts)

    # Prepare content with text prompt and reference images
    content_parts = [prompt_text]

    # Add asset images as reference (if they exist and are visible in this scene)
    # NOTE: Party characters are now also in the assets system (converted before first scene)
    # so this loop handles BOTH party members AND other NPCs/objects with one unified approach
    asset_count = 0
    if assets and visible_asset_ids:
        for asset_id in visible_asset_ids:
            asset = assets.get(asset_id)
            if asset and asset.image_path:
                # Convert web path to file path
                asset_image_file = pathlib.Path("webapp") / asset.image_path.lstrip("/")
                if asset_image_file.exists():
                    try:
                        # Add text label before the image
                        content_parts.append(
                            f"\n📸 {asset.type.upper()} REFERENCE - {asset.name.upper()}:"
                        )
                        content_parts.append(
                            f"This is {asset.name}. Use THIS appearance in the scene."
                        )

                        with open(asset_image_file, "rb") as f:
                            image_data = f.read()
                        content_parts.append(
                            genai_types.Part.from_bytes(
                                data=image_data, mime_type="image/png"
                            )
                        )
                        asset_count += 1
                        logger.info(
                            f"✓ Added reference image for asset: {asset.name} ({asset.type})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not load asset image for {asset.name}: {e}"
                        )

    if asset_count > 0:
        content_parts.append(
            f"\n✅ {asset_count} reference image(s) provided above (party members + other assets). USE THEM for visual consistency."
        )

    logger.info(
        f"Generating scene image for scene {scene_id} with {asset_count} total reference images (party + assets)"
    )

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


def _generate_scene_voiceover_sync(
    client,
    game_id: str,
    scene_id: int,
    scene_text: str,
) -> pathlib.Path | None:
    """Synchronous helper to generate a scene voiceover narration using Gemini TTS.

    Args:
        client: Google GenAI client
        game_id: Game identifier for directory structure
        scene_id: Scene number
        scene_text: The narrative text of the scene to narrate

    Returns:
        Path to saved voiceover WAV file or None if generation failed
    """
    # Create voiceover directory for this game
    voiceover_dir = VOICEOVER_DIR / game_id
    voiceover_dir.mkdir(parents=True, exist_ok=True)

    filename = f"scene_{scene_id:03d}.wav"
    voiceover_file_path = voiceover_dir / filename

    logger.info(f"Generating voiceover for scene {scene_id} in game {game_id}")

    # Build enhanced prompt with narration guidance
    narration_prompt = f"""You are an expert fantasy audiobook narrator and Dungeon Master bringing an adventure to life. 
Read the following scene with appropriate emotion, pacing, and dramatic flair.

NARRATION GUIDELINES:
- Use a storytelling tone that draws listeners into the fantasy world
- Vary your pacing: slow down for dramatic moments, speed up for action
- Emphasize emotional content: excitement during discoveries, tension during danger, solemnity for serious moments
- Add dramatic pauses where appropriate for impact (use natural speech rhythm)
- Convey the atmosphere: mysterious for intrigue, ominous for danger, warm for friendly encounters
- When describing dialogue or character actions, subtly shift tone to match the character's personality
- Build tension gradually in suspenseful moments
- Express wonder and awe for magical or epic descriptions
- Maintain energy and engagement throughout - avoid monotone delivery

SCENE TO NARRATE:

{scene_text}"""

    try:
        config = genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name="Algieba"
                    )
                )
            ),
        )

        response = client.models.generate_content(
            model="models/gemini-2.5-flash-preview-tts",
            contents=narration_prompt,
            config=config,
        )

        # Extract PCM audio data from response and save as WAV
        if response and hasattr(response, "candidates"):
            for part in response.candidates[0].content.parts:  # type: ignore[index]
                if getattr(part, "inline_data", None) and getattr(
                    part.inline_data, "data", None
                ):
                    # Convert PCM data to WAV format
                    pcm_data = part.inline_data.data

                    # Save as WAV with proper headers (24kHz, 16-bit, mono)
                    with wave.open(str(voiceover_file_path), "wb") as wav_file:
                        wav_file.setnchannels(1)  # mono
                        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                        wav_file.setframerate(24000)  # 24kHz sample rate
                        wav_file.writeframes(pcm_data)

                    logger.info(f"Saved voiceover at {voiceover_file_path}")
                    return voiceover_file_path

        logger.warning(f"No audio content in response for scene {scene_id}")
        return None

    except Exception as e:
        logger.warning(f"Voiceover generation failed for scene {scene_id}: {e}")
        return None


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
    concept_result = await retry_on_overload(agent.run, prompt)
    generated_concepts = concept_result.output.characters
    logger.info(f"=== {num_characters} Characters Generated ===")
    logger.info(f"Characters: {concept_result.output.model_dump_json(indent=2)}")

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


async def generate_scenario_template(
    existing_scenarios: List[ScenarioTemplate],
) -> ScenarioTemplate:
    """Generates a complete new scenario template with contrastive prompting.

    Args:
        existing_scenarios: List of all existing scenario templates (both played and unplayed)
                          to ensure the new scenario is distinct and original

    Returns:
        ScenarioTemplate object with name, one_liner, and full DM notes
    """
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedScenarioTemplate)

    logger.info(
        f"Generating new scenario template (contrasting with {len(existing_scenarios)} existing scenarios)..."
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

        contrastive_context = f"""
EXISTING SCENARIOS (create something DISTINCTLY DIFFERENT):

{chr(10).join(scenario_summaries)}

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

{contrastive_context}

Generate a COMPLETE scenario with the following components:

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

    # Create ScenarioTemplate object
    scenario_template = ScenarioTemplate(
        name=generated.name,
        one_liner=generated.one_liner,
        dm_notes=dm_notes,
    )

    return scenario_template


async def generate_initial_locations(
    scenario_name: str,
    scenario_details: Optional[str] = None,
) -> dict[str, Location]:
    """Generates initial set of locations for the adventure.

    Args:
        scenario_name: Name of the scenario
        scenario_details: Optional markdown details about the scenario

    Returns:
        Dictionary mapping location IDs to Location objects
    """
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedLocationList)

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


def _apply_character_updates(
    characters: List[Character], generated_scene
) -> List[Character]:
    """Apply character state updates from explicit changes in the generated scene.

    Args:
        characters: List of current character objects (with image_path, etc.)
        generated_scene: GeneratedScene object with health_changes, inventory_changes, etc.

    Returns:
        Updated list of characters with modifications applied
    """
    # Create a dictionary for quick character lookup by name
    char_dict = {char.name: char for char in characters}

    # Apply health changes
    for health_change in generated_scene.health_changes:
        if health_change.character_name not in char_dict:
            logger.warning(
                f"Health change for unknown character: {health_change.character_name}. Skipping."
            )
            continue

        char = char_dict[health_change.character_name]
        old_health = char.current_health
        char.current_health = max(
            0,
            min(char.current_health + health_change.health_change, char.maximum_health),
        )

        change_desc = f"Health: {old_health} -> {char.current_health}"
        if health_change.health_change < 0:
            change_desc += (
                f" ({health_change.health_change} damage: {health_change.reason})"
            )
        elif health_change.health_change > 0:
            change_desc += (
                f" (+{health_change.health_change} healing: {health_change.reason})"
            )

        logger.info(f"Updated {char.name}: {change_desc}")

    # Apply inventory changes
    for inv_change in generated_scene.inventory_changes:
        if inv_change.character_name not in char_dict:
            logger.warning(
                f"Inventory change for unknown character: {inv_change.character_name}. Skipping."
            )
            continue

        char = char_dict[inv_change.character_name]
        changes = []

        # Add new items
        for item in inv_change.items_added:
            if item not in char.inventory:
                char.inventory.append(item)
                changes.append(f"+{item}")

        # Remove items
        for item in inv_change.items_removed:
            if item in char.inventory:
                char.inventory.remove(item)
                changes.append(f"-{item}")

        if changes:
            logger.info(f"Updated {char.name} inventory: {', '.join(changes)}")

    # Apply skill changes
    for skill_change in generated_scene.skill_changes:
        if skill_change.character_name not in char_dict:
            logger.warning(
                f"Skill change for unknown character: {skill_change.character_name}. Skipping."
            )
            continue

        char = char_dict[skill_change.character_name]
        changes = []

        # Learn new skills
        for skill in skill_change.skills_learned:
            if skill not in char.skills:
                char.skills.append(skill)
                changes.append(f"Learned: {skill}")

        # Lose skills
        for skill in skill_change.skills_lost:
            if skill in char.skills:
                char.skills.remove(skill)
                changes.append(f"Lost: {skill}")

        if changes:
            logger.info(f"Updated {char.name} skills: {', '.join(changes)}")

    # Apply stat changes (rare)
    for stat_change in generated_scene.stat_changes:
        if stat_change.character_name not in char_dict:
            logger.warning(
                f"Stat change for unknown character: {stat_change.character_name}. Skipping."
            )
            continue

        char = char_dict[stat_change.character_name]
        changes = []

        if stat_change.strength_change != 0:
            old_str = char.strength
            char.strength = max(0, min(20, char.strength + stat_change.strength_change))
            changes.append(f"STR: {old_str} -> {char.strength}")

        if stat_change.intelligence_change != 0:
            old_int = char.intelligence
            char.intelligence = max(
                0, min(20, char.intelligence + stat_change.intelligence_change)
            )
            changes.append(f"INT: {old_int} -> {char.intelligence}")

        if stat_change.agility_change != 0:
            old_agi = char.agility
            char.agility = max(0, min(20, char.agility + stat_change.agility_change))
            changes.append(f"AGI: {old_agi} -> {char.agility}")

        if changes:
            logger.info(
                f"Updated {char.name} stats: {', '.join(changes)} (Reason: {stat_change.reason})"
            )

    return characters


def _make_scene(
    scene_id: int,
    scene_text: str,
    prompt,
    image_path: str | None = None,
    game_status: GameStatus = "ongoing",
    visible_asset_ids: List[str] = None,
    health_changes: List = None,
    inventory_changes: List = None,
    skill_changes: List = None,
    stat_changes: List = None,
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


def _format_existing_assets(assets: dict[str, Asset]) -> str:
    """Format existing assets for inclusion in prompts.

    Args:
        assets: Dictionary mapping asset IDs to Asset objects

    Returns:
        Formatted string listing existing assets
    """
    if not assets:
        return "No existing assets yet (this is the first scene)."

    # Separate party members from other assets for clarity
    party_assets = []
    other_assets = []

    for asset_id, asset in assets.items():
        asset_str = f"- **{asset.name}** ({asset.type}): {asset.description}"
        if asset_id.startswith("player_"):
            party_assets.append(asset_str)
        else:
            other_assets.append(asset_str)

    result_parts = []

    if party_assets:
        result_parts.append(
            "PARTY MEMBERS (your player characters - ALWAYS include in assets_present when visible):\n"
            + "\n".join(party_assets)
        )

    if other_assets:
        result_parts.append(
            "OTHER EXISTING ASSETS (NPCs and objects already introduced - REUSE THESE NAMES AND DESCRIPTIONS):\n"
            + "\n".join(other_assets)
        )

    return "\n\n".join(result_parts) if result_parts else "No existing assets yet."


def _format_existing_locations(locations: dict[str, Location]) -> str:
    """Format existing locations for inclusion in prompts.

    Args:
        locations: Dictionary mapping location IDs to Location objects

    Returns:
        Formatted string listing available locations
    """
    if not locations:
        return "No available locations yet."

    location_lines = []
    for loc_id, location in locations.items():
        features_str = ", ".join(location.key_features[:3])  # First 3 features
        location_lines.append(
            f"- **{location.name}** (ID: `{loc_id}`, Type: {location.location_type})\n"
            f"  Description: {location.description}\n"
            f"  Key Features: {features_str}\n"
            f"  Atmosphere: {location.atmosphere}\n"
            f"  Lighting: {location.lighting_default}"
        )

    return (
        "AVAILABLE LOCATIONS (choose from these for location_reference):\n"
        + "\n\n".join(location_lines)
    )


def _process_scene_assets(
    game_id: str,
    scene_id: int,
    assets_present: list,
    existing_assets: dict[str, Asset],
    client,
) -> tuple[dict[str, Asset], List[str]]:
    """Process assets from generated scene, create new ones, and generate images.

    Args:
        game_id: Game identifier for storage
        scene_id: Scene identifier
        assets_present: List of AssetReference objects from the generated scene
        existing_assets: Dictionary of existing assets
        client: Google GenAI client for image generation

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
                asset_image_path = _generate_asset_image_sync(
                    client, game_id, new_asset.id, new_asset.name, new_asset.description
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
    location_reference: Optional[LocationReference],
    existing_locations: dict[str, Location],
) -> tuple[dict[str, Location], Optional[LocationReference]]:
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


def _generate_asset_image_sync(
    client, game_id: str, asset_id: str, asset_name: str, asset_description: str
) -> pathlib.Path | None:
    """Generate and save an image for an asset (NPC or object).

    Args:
        client: Google GenAI client
        game_id: Game identifier
        asset_id: Asset identifier
        asset_name: Name of the asset
        asset_description: Physical description

    Returns:
        Path to the saved image file, or None if generation failed
    """
    # Create asset directory for this game
    asset_dir = pathlib.Path(f"webapp/static/assets/{game_id}")
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt for asset image
    prompt = f"""Generate a high-quality fantasy art image of: {asset_name}

Description: {asset_description}

CRITICAL: Generate image in SQUARE orientation (1:1 aspect ratio) - width and height MUST be equal.

Style requirements:
- Painterly fantasy art style
- Detailed, cinematic lighting, atmospheric
- Clear, centered view of the character/object as described
- Consistent with fantasy adventure game aesthetics
- DO NOT include text or labels in the image
- Focus on the subject - minimal background, subject should be prominent

CRITICAL FORMAT REQUIREMENT: Image MUST be in SQUARE orientation (1:1 aspect ratio).
"""

    logger.info(f"Generating asset image for: {asset_name}")

    try:
        from google.genai import types as genai_config  # type: ignore

        # Configure for square aspect ratio
        config = genai_config.GenerateContentConfig(
            response_modalities=["image"],
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt],
            config=config,
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Asset image generation request failed for {asset_name}: {e}")
        return None

    if response is not None:
        try:
            for part in response.candidates[0].content.parts:  # type: ignore[index]
                if getattr(part, "inline_data", None) and getattr(
                    part.inline_data, "data", None
                ):
                    img = Image.open(BytesIO(part.inline_data.data))
                    file_path = asset_dir / f"{asset_id}.png"
                    img.save(file_path, format="PNG")
                    logger.info(f"Asset image saved: {file_path}")
                    return file_path
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to decode/save asset image for {asset_name}: {e}")
            return None

    return None


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
- Use action for: simple tasks, exploration without immediate danger, planning, easy skill checks, or coordinated team efforts
- Use dice_check for: combat, risky actions, difficult skill checks, life-or-death situations

DICE CHECK RULES (when using dice_check type):
- ALWAYS single die roll (either d6 or d10)
- Choose dice type based on difficulty:
  * d6: Simple to moderate challenges (lockpicking, climbing, basic combat)
  * d10: Challenging situations (difficult combat, dangerous magic, persuading hostile NPCs)
- Remember: The next scene will interpret the roll with stat modifiers (+2 for 16-20, +1 for 11-15, +0 for 6-10, -1 for 1-5) and skill bonuses
- Each character targeted will roll their own single die

TARGETING RULES (CRITICAL - READ CAREFULLY):
For dice_check prompts specifically:
  * ALWAYS specify target_character (single name) OR target_characters (array of names)
  * NEVER use target_character: null for dice_check prompts - this causes confusion about who rolls
  * If one character should attempt the check, use target_character with their name
    - Example: "target_character": "Theron the Brave" (use the ACTUAL character name from the party)
  * If multiple specific characters should roll simultaneously, use target_characters array
    - Example: "target_characters": ["ActualName1", "ActualName2"] (use ACTUAL names from the party roster)
  * If the situation is "any character could volunteer," pick the most suitable character based on their skills/stats
    - Example: For persuasion, pick the character with highest Intelligence or relevant skills
  * Make your prompt_text match your targeting:
    - Single target: "[CharacterName], roll d10 to persuade the elder" (use their actual name)
    - Multiple targets: "[Name1] and [Name2], both roll d6 for your attacks" (use their actual names)

For action prompts:
  * Can use target_character for single character actions
  * Can use target_characters for coordinated multi-character actions (e.g., "[Name1] searches left, [Name2] searches right" - use actual party member names)
  * Use target_character: null when the entire party acts together

For dialogue prompts:
  * Use target_character with a name to address a specific character
  * Use target_character: null when the entire party speaks/responds
  
CRITICAL: For ANY dice_check prompt, you MUST set either target_character (string) or target_characters (array). Never null.

CRITICAL RULES FOR CHARACTER CHANGES (SIMPLIFIED APPROACH):
- The character sheets show CURRENT states as characters ENTER this scene
- DO NOT apply previous scene's changes again! 
- Report ONLY NEW changes that occur in THIS SPECIFIC scene
- Use the change arrays: health_changes, inventory_changes, skill_changes, stat_changes
- Empty arrays mean no changes of that type occurred in THIS scene

HEALTH CHANGES (health_changes array):
  * Report ONLY NEW damage or healing that occurs in THIS SPECIFIC scene
  * NEVER repeat health changes from conversation history - those are already applied
  * Character current_health already reflects past changes - don't subtract them again
  * Use negative values for NEW damage, positive for NEW healing
  * CRITICAL: These are CHANGES (deltas) from what happens NOW, not what happened before
  
  * CORRECT EXAMPLES:
    - Character takes NEW 15 damage in this scene: {"character_name": "Theron", "health_change": -15, "reason": "fell from cliff"}
    - Character gets NEW healing in this scene: {"character_name": "Elara", "health_change": 20, "reason": "healing spell"}
    - Character unharmed in this scene: Don't include them in the array at all
  
  * WRONG EXAMPLES (DON'T DO THIS):
    - ❌ Applying damage from a previous scene again
    - ❌ Using conversation history damage in this scene's health_changes
    - ❌ Seeing "took 10 damage" in history and putting -10 in THIS scene's changes
  
  * Remember: Only report what happens in THIS SPECIFIC scene based on current action

INVENTORY CHANGES (inventory_changes array):
  * Report items gained or lost in THIS scene only, based on what happens NOW
  * Current inventory list shows what they have entering this scene
  * Don't remove items that were used in previous scenes - they're already gone
  * EXAMPLES:
    - Gained NEW items: {"character_name": "Theron", "items_added": ["Magic Sword"], "items_removed": []}
    - Lost/used item NOW: {"character_name": "Elara", "items_added": [], "items_removed": ["Rope"]}
    - No change: Don't include character in array

SKILL CHANGES (skill_changes array):
  * Report skills learned or lost in THIS scene (rare)
  * EXAMPLES:
    - Learned skill: {"character_name": "Theron", "skills_learned": ["Advanced Combat"], "skills_lost": []}
    - Lost skill: {"character_name": "Elara", "skills_learned": [], "skills_lost": ["Stealth"]} (due to injury/curse)
    - No change: Empty array (most scenes)

STAT CHANGES (stat_changes array):
  * Report permanent stat changes in THIS scene (very rare)
  * Use for curses, blessings, transformations, permanent effects
  * EXAMPLES:
    - Cursed: {"character_name": "Theron", "strength_change": -2, "intelligence_change": 0, "agility_change": 0, "reason": "witch's curse"}
    - Blessed: {"character_name": "Elara", "strength_change": 0, "intelligence_change": 3, "agility_change": 0, "reason": "ancient tome"}
    - No change: Empty array (almost all scenes)

CRITICAL: NARRATE ALL CHARACTER SHEET CHANGES IN SCENE TEXT:
- If a character takes damage, EXPLICITLY state how much health they lose in the narrative
  * Example: "The goblin's blade slashes across your arm, dealing 15 damage. You feel your strength waning."
  * Example: "The fall bruises you badly - you lose 8 health."
- If a character is healed, EXPLICITLY state the healing amount
  * Example: "The healing potion surges through you, restoring 20 health."
  * Example: "The cleric's spell mends your wounds, recovering 12 health."
- If a character gains an item, EXPLICITLY mention it in the narrative
  * Example: "You pick up the enchanted dagger and add it to your pack."
  * Example: "The grateful merchant hands you a Potion of Invisibility."
- If a character loses/uses an item, EXPLICITLY describe it
  * Example: "You consume the Health Potion, feeling its effects immediately."
  * Example: "Your rope snaps under the weight and falls into the chasm below."
- If a character learns a new skill, describe how they acquired it
  * Example: "Through intense practice, you've mastered the art of Stealth."
- Players should NEVER be surprised by character sheet changes - everything must be clear in the story

CRITICAL: SPECIFY EXACT COUNTS IN SCENE TEXT:
- When groups of enemies, NPCs, or creatures appear, ALWAYS mention the exact count in the narrative
  * Example: "Two bandits step out from behind the rocks, weapons drawn."
  * Example: "Three wolves emerge from the shadows, circling your party."
  * Example: "Four guards block the entrance to the throne room."
- DO NOT use vague terms like "some enemies", "several bandits", "a group of wolves"
- The count in scene_text MUST match the count in visual_description and assets_present
- This ensures players understand exactly what they're facing and images match the narrative

CRITICAL RULES FOR GAME STATUS (REQUIRED FIELD):
- ALWAYS include game_status in your response - it is a REQUIRED field
- Set game_status to "ongoing" (default) - continue the adventure normally
- Set game_status to "failed" if ANY character reaches 0 current_health (party death/defeat):
  * This ends the game immediately with a game over
  * The scene should describe their defeat or death dramatically
  * Still provide a prompt field (required by schema), but it won't be shown to players
- Set game_status to "completed" ONLY when the main quest objective is definitively achieved:
  * The primary goal from scenario_details must be fulfilled
  * This ends the game with a victory screen
  * The scene should describe their triumph and resolution
  * Still provide a prompt field (required by schema), but it won't be shown to players
- Do NOT set completed prematurely - minor victories are still "ongoing"
- Do NOT set completed just because things are going well - only when main quest is done

CRITICAL RULES FOR VISUAL ASSETS (NPCs AND OBJECTS):
- Include assets_present array listing ALL important NPCs and objects in the scene
- Important assets are: Named NPCs, significant objects, key locations that should be visually consistent
- CRITICAL: PARTY MEMBERS ARE ASSETS TOO:
  * ALL party member characters are pre-registered as assets (type: "npc")
  * When party members appear in a scene, ALWAYS include them in assets_present
  * Use their EXACT character name from the party roster
  * Set is_visible=true when they're present in the scene
  * This ensures their visual consistency (reference images) across all scenes
- For each asset, specify:
  * name: The name of the NPC or object
  * type: "npc" for characters (including party members!), "object" for items/places/things
  * description: Physical description for image generation
  * is_visible: MUST be true ONLY if the NPC/object is PHYSICALLY PRESENT in the current scene location
  * is_visible: MUST be false if the NPC/object is only mentioned, talked about, in another location, or not directly visible
- CRITICAL is_visible RULES:
  * Set is_visible=true: NPC is in the room/area with the party, object is visible in the scene
  * Set is_visible=false: NPC is somewhere else, object is mentioned but not present, talked about but not seen
  * Example: If party discusses a villain who is in a distant location, is_visible=false
  * Example: If party sees a door/portal directly in front of them, is_visible=true
  * Example: If an NPC mentions an item in another room that party can't see, is_visible=false
- CRITICAL ASSET REUSE RULES (PREVENT DUPLICATES):
  * ALWAYS check the EXISTING ASSETS list first before creating a new asset
  * If an NPC or object was introduced in a previous scene, you MUST reuse it
  * Use the EXACT SAME name (character-by-character identical, including capitalization, spacing, and punctuation)
  * Use the EXACT SAME physical description (copy it word-for-word from EXISTING ASSETS)
  * DO NOT create variations of the same name (e.g., if existing asset is "Guard Captain", don't create "The Guard Captain" or "Captain of the Guard")
  * DO NOT add adjectives or modifiers to existing names (e.g., if existing is "Ancient Door", don't create "Large Ancient Door")
  * When in doubt, if it could be the same NPC/object, REUSE the existing one
  * Only create a NEW asset if it's genuinely a different NPC/object that has never appeared before
- For genuinely new NPCs/objects appearing for the first time:
  * Choose a clear, unique name that won't conflict with existing assets
  * Create a clear, detailed physical description
  * This description will be used to generate their consistent image
- Examples:
  * NPC present: {"name": "[NPC's actual name]", "type": "npc", "description": "[Physical description of NPC]", "is_visible": true}
  * Object present: {"name": "[Object's actual name]", "type": "object", "description": "[Physical description of object]", "is_visible": true}
  * NPC mentioned/elsewhere: {"name": "[NPC's actual name]", "type": "npc", "description": "[Physical description of NPC]", "is_visible": false}
  * Object mentioned/not visible: {"name": "[Object's actual name]", "type": "object", "description": "[Physical description of object]", "is_visible": false}

CRITICAL: VISUAL DESCRIPTION FOR IMAGE GENERATION:
- Include a visual_description field separate from scene_text
- This description is ONLY for the image generator - players will never see it
- MUST be comprehensive and detailed (8-12 sentences minimum) - treat this as precise direction for a fantasy artist
- REQUIRED elements to describe in detail:
  
  1. CAMERA & FRAMING:
     * Specify exact camera angle (wide shot, medium shot, close-up, bird's eye, low angle, etc.)
     * Aspect ratio reminder: LANDSCAPE/HORIZONTAL (16:9)
     * What is in foreground, midground, background
  
  2. EACH CHARACTER IN SCENE (describe ALL party members present):
     * Name each character explicitly
     * Their exact position in the scene (left/right/center, foreground/background, distance from camera)
     * Body position and stance (standing/crouching/kneeling, facing direction, body angle to camera)
     * What they're doing with their hands/arms (weapon drawn, casting spell, examining object, gesturing)
     * Facial expression and where they're looking
     * Body language conveying emotion/intent
  
  3. IMPORTANT OBJECTS/NPCs (describe ALL visible assets):
     * CRITICAL: If describing groups (enemies, guards, creatures), specify EXACT COUNT (e.g., "two [enemy type]", "three [creature type]", "four [NPC type]")
     * Name each important object or NPC explicitly
     * If there are multiple similar NPCs (enemies, guards), describe each one's position individually
     * Exact placement in the scene for each individual
     * Size relative to characters
     * State/condition (glowing, damaged, active, dormant, etc.)
     * How characters are interacting with them (if applicable)
  
  4. ENVIRONMENT:
     * Specific location type and architectural/natural features
     * Time of day and weather conditions
     * Key environmental details (textures, materials, vegetation, terrain)
     * Spatial layout (how elements are arranged in 3D space)
  
  5. LIGHTING & ATMOSPHERE:
     * Primary light source(s) and direction
     * Quality of light (harsh/soft, warm/cool, colored/white)
     * Shadows and how they fall
     * Atmospheric effects (fog, dust, magic glow, particles, etc.)
     * Overall mood and emotional tone
  
  6. FOCAL POINTS:
     * What draws the viewer's eye first
     * Visual hierarchy of elements
  
- Write as clear, specific directions - avoid vague terms
- CRITICAL for groups: Always specify exact counts (e.g., "two [enemies]", not "[enemies]" or "several [enemies]")
- CRITICAL: Only describe what should actually appear in the image - image generator follows this literally
- Example format with counted NPCs: "Medium wide shot from [angle]. In the foreground left, [Character A] stands with [action/pose], facing right toward two [enemy/NPC type], [expression/stance]. The first [enemy] is in the right foreground, [specific pose/action]. The second [enemy] is [position], [specific pose/action]. Center background, [Character B] [action/pose]. Far right, [Character C] [action/pose]. In the midground, [important object] with [details], [size/placement]. The scene is set in [location type] with [environmental features]. Lighting is [light description] creating [shadow effects]. [Additional atmospheric elements]. Atmosphere is [mood], with [final atmospheric details]."

Return your response in this JSON structure (ALL FIELDS REQUIRED):
{
  "scene_text": "The vivid narrative of the scene...",
  "visual_description": "COMPREHENSIVE visual description for image generation (8-12+ sentences with detailed camera angle, each character's position/pose/expression, all visible objects/NPCs placement, environment details, lighting setup, and atmosphere)...",
  "prompt": {
    "type": "dialogue" | "action" | "dice_check",
    "dice_type": "d6" | "d10" (only if type is dice_check, always single die roll),
    "target_character": "Character Name" (REQUIRED for single dice_check, optional for action/dialogue) or null,
    "target_characters": ["Char1", "Char2"] (for multi-character dice_check or action) or null,
    "prompt_text": "The question or instruction for the player(s)"
  },
  "health_changes": [
    {
      "character_name": "Character Name",
      "health_change": -15 (negative for damage, positive for healing),
      "reason": "brief explanation"
    }
  ],
  "inventory_changes": [
    {
      "character_name": "Character Name",
      "items_added": ["New Item 1", "New Item 2"],
      "items_removed": ["Used Item"]
    }
  ],
  "skill_changes": [
    {
      "character_name": "Character Name",
      "skills_learned": ["New Skill"],
      "skills_lost": []
    }
  ],
  "stat_changes": [
    {
      "character_name": "Character Name",
      "strength_change": 0,
      "intelligence_change": 2,
      "agility_change": 0,
      "reason": "blessed by ancient tome"
    }
  ],
  "assets_present": [
    {
      "name": "NPC or Object Name",
      "type": "npc" | "object",
      "description": "Physical description...",
      "is_visible": true | false
    }
  ],
  "location_reference": {
    "location_id": "UUID of the location from AVAILABLE LOCATIONS list",
    "time_of_day": "dawn" | "midday" | "afternoon" | "dusk" | "night" | null (optional variation),
    "weather": "clear" | "rainy" | "stormy" | "foggy" | "snowing" | null (optional, for outdoor locations),
    "state_changes": ["description of any changes to location", "..."] (optional, e.g., ["tables overturned", "fire damage"]),
    "camera_angle": "wide establishing shot" | "close interior view" | "aerial view" | "low angle" | null (optional),
    "focus_area": "specific area to emphasize" | null (optional, e.g., "the bar area", "the throne", "the entrance")
  } | null,
  "game_status": "ongoing" | "completed" | "failed"
}

CRITICAL RULES FOR LOCATIONS:
- ALWAYS include location_reference for every scene (or null if truly no specific location applies)
- Choose from the AVAILABLE LOCATIONS list provided in the prompt
- REUSE existing locations when narratively appropriate:
  * If party returns to a tavern, reuse the tavern location_id
  * If continuing in same area, reuse the same location_id
  * Locations provide visual consistency across scenes in same place

IMPORTANT - RELATIONSHIP BETWEEN location_reference AND visual_description:
- The location_reference provides CONTEXT that should be incorporated into visual_description
- When writing visual_description, use the location's description, key_features, and variations
- Workflow: 
  1. Set location_reference with location_id and any variations
  2. Write visual_description that naturally incorporates:
     - Location's physical description and key features
     - Time of day (affects lighting: dawn=soft, midday=bright, dusk=golden, night=dark)
     - Weather (affects atmosphere: rainy=wet/gloomy, stormy=dramatic, foggy=mysterious)
     - State changes (damaged furniture, broken doors, scorch marks, etc.)
     - Camera angle suggestion (but feel free to adjust for best composition)
     - Focus area (emphasize that part while maintaining overall scene)
- Example: If location_reference has time_of_day='dusk', camera_angle='wide shot', focus_area='entrance'
  Then visual_description might be: "Wide shot of the tavern entrance bathed in golden dusk light..."
- The visual_description is the complete, unified description used for image generation
- Location variations add diversity while maintaining consistent features

- Use the location variations to add visual diversity:
  * time_of_day: Change time to show passage of time (dawn -> midday -> dusk -> night)
  * weather: For outdoor locations, vary weather conditions
  * state_changes: After battles or events, describe damage/changes to location
  * camera_angle: Vary the view (wide shot, close-up, different angles)
  * focus_area: Emphasize different parts of the same location
- Examples of good location reuse with variation:
  * Same tavern, different time: morning crowd vs late night
  * Same forest, different weather: sunny morning vs stormy afternoon  
  * Same throne room, different focus: wide view of whole room vs close on throne
  * Same location after battle: add state_changes like "broken furniture", "scorch marks"
- This allows visual consistency (same location features) while avoiding repetitive images
- For NEW locations not in the list, choose the CLOSEST match from available locations
  * If moving to new area, pick the most appropriate existing location type
  * Don't request new locations - work with what's available
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
   - Note: Always single die roll, so modifiers are very impactful

4. NARRATE THE RESULT:
   - ALWAYS mention the dice type that was rolled (d6 or d10) to provide context
   - Show the roll value as a fraction to make success/failure clear to players
   - Explicitly mention the stat modifier and skill bonus in your narrative
   - Example formats:
     * "You rolled 4/6. With your high Strength (+2) and Swordsmanship skill (+1), that's a total of 7 - a solid success!"
     * "You rolled 3/10. Despite your Agility modifier (+1), bringing you to 4, you barely manage..."
     * "A natural 6/6! With your Intelligence bonus (+2), that's a perfect 8 - critical success!"
     * "You rolled 2/10. Even with your skill bonus (+1), you only reach 3 - a critical failure..."
   - The fraction format (roll/max) helps players immediately understand if it was a good or bad roll
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
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
    scene_id: int = 1,
) -> tuple[Scene, List[Character], dict[str, Asset], dict[str, Location]]:
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

{_format_existing_assets(existing_assets)}

{_format_existing_locations(existing_locations)}

Generate the opening scene for this adventure. Create an engaging, atmospheric introduction that:

CRITICAL FOR OPENING SCENE:
1. **Establish how the party came together**: Briefly narrate how these characters met and why they're working together
   - Use their backstories to create natural connections (shared goals, chance meeting, hired together, etc.)
   - Make it believable based on their personalities and backgrounds
   - Keep this part concise (2-3 sentences)

2. **Explain the current situation**: Describe what brings them to this specific moment
   - Where are they? (refer to the setting from scenario details)
   - Why are they here? (connect to the main quest)
   - What immediate situation do they find themselves in?

3. **Introduce the main quest naturally**: Players don't know the scenario details, so reveal the quest/goal through the narrative
   - Don't assume players know anything from the DM notes
   - Make the objective clear through dialogue, events, or circumstances
   - Give them a reason to care and a clear direction

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

{_build_scene_generation_prompt_rules()}

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
    updated_characters = _apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread
    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)  # type: ignore[attr-defined]

    asset_task = asyncio.to_thread(
        _process_scene_assets,
        game_id,
        scene_id,
        generated.assets_present,
        existing_assets,
        client,
    )

    # Wait for asset processing to complete so we have asset images for scene generation
    updated_assets, visible_asset_ids = await asset_task
    if isinstance(updated_assets, Exception):
        logger.error(f"Asset processing failed: {updated_assets}")
        updated_assets = existing_assets
        visible_asset_ids = []

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation
    image_task = asyncio.to_thread(
        _generate_scene_image_sync,
        client,
        game_id,
        scene_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        visible_asset_ids,
    )

    voiceover_task = asyncio.to_thread(
        _generate_scene_voiceover_sync,
        client,
        game_id,
        scene_id,
        generated.scene_text,
    )

    # Wait for both to complete
    image_file_path, voiceover_file_path = await asyncio.gather(
        image_task, voiceover_task, return_exceptions=True
    )

    # Handle exceptions
    if isinstance(image_file_path, Exception):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, Exception):
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
    characters: List[Character],
    last_scene_id: int,
    player_action: Optional[str],
    conversation_history: List[dict],
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
) -> tuple[Scene, List[Character], dict[str, Asset], dict[str, Location]]:
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
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedScene)

    next_id = last_scene_id + 1
    logger.info(f"Generating scene {next_id} for: {scenario_name}")

    # Build detailed character information for context
    # IMPORTANT: These are CURRENT states as they ENTER this scene, not changes to apply
    character_sheets = "\n\n".join(
        [
            f"**{char.name}** (Current State Entering Scene {next_id})\n"
            f"- Stats: Strength {char.strength}, Intelligence {char.intelligence}, Agility {char.agility}\n"
            f"- Health: {char.current_health}/{char.maximum_health} (this is their CURRENT health, don't subtract from it again)\n"
            f"- Backstory: {char.backstory}\n"
            f"- Appearance: {char.appearance}\n"
            f"- Personality: {char.personality}\n"
            f"- Skills: {', '.join(char.skills) if char.skills else 'None'}\n"
            f"- Inventory: {', '.join(char.inventory) if char.inventory else 'None'} (current items, don't remove again unless used in THIS scene)"
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
            dice_check_rules = f"\n\n{_build_dice_check_resolution_rules()}"

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}

PARTY CHARACTER SHEETS:
{character_sheets}

{_format_existing_assets(existing_assets)}

{_format_existing_locations(existing_locations)}

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

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Log generated scene details
    logger.info(f"=== Next Scene Generated (Scene {next_id}) ===")
    logger.info(
        f"Generated scene model: {generated.model_dump_json(indent=2, exclude={'scene_text'})}"
    )

    # Apply character updates from the scene
    updated_characters = _apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread
    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)  # type: ignore[attr-defined]

    asset_task = asyncio.to_thread(
        _process_scene_assets,
        game_id,
        next_id,
        generated.assets_present,
        existing_assets,
        client,
    )

    # Wait for asset processing to complete so we have asset images for scene generation
    updated_assets, visible_asset_ids = await asset_task
    if isinstance(updated_assets, Exception):
        logger.error(f"Asset processing failed: {updated_assets}")
        updated_assets = existing_assets
        visible_asset_ids = []

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation
    image_task = asyncio.to_thread(
        _generate_scene_image_sync,
        client,
        game_id,
        next_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        visible_asset_ids,
    )

    voiceover_task = asyncio.to_thread(
        _generate_scene_voiceover_sync,
        client,
        game_id,
        next_id,
        generated.scene_text,
    )

    # Wait for both to complete
    image_file_path, voiceover_file_path = await asyncio.gather(
        image_task, voiceover_task, return_exceptions=True
    )

    # Handle exceptions
    if isinstance(image_file_path, Exception):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, Exception):
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
    scenes: List[Scene],
    characters: List[Character],
    current_scene_index: int,
) -> str:
    """Generate a narrative recap of all scenes up to the current point.

    Args:
        game_id: Unique identifier for the game
        scenario_name: Name of the scenario being played
        scenes: List of all scenes in the game
        characters: Current party characters
        current_scene_index: Index of the current scene (0-based)

    Returns:
        Narrative recap text in markdown format
    """
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-flash", provider=provider)
    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)

    logger.info(
        f"Generating recap for game {game_id}, scenes 1-{current_scene_index + 1}"
    )

    # Build scene summaries for context
    scene_summaries = []
    for i, scene in enumerate(scenes[: current_scene_index + 1]):
        summary = f"**Scene {scene.id}:**\n{scene.text[:500]}..."
        if scene.prompt:
            summary += f"\n*Player choice was needed here*"
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

Create an engaging, concise recap (200-300 words) that:
1. Reminds players of the adventure's main goal
2. Summarizes the key events that have happened so far
3. Highlights any important choices the party made
4. Notes any significant character developments (injuries, items gained, etc.)
5. Sets the stage for where the story is now
6. Uses an exciting, narrative tone as if read by a Dungeon Master

Write in past tense, as if narrating a story. Make it feel like "Previously on [Adventure Name]..."
"""

    try:
        response = await retry_on_overload(
            client.aio.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
        )
        recap_text = response.text.strip()
        logger.info(f"Successfully generated recap ({len(recap_text)} chars)")
        return recap_text
    except Exception as e:
        logger.error(f"Failed to generate recap: {e}")
        # Return a simple fallback recap
        return f"""## Previously in {scenario_name}...

Your party has journeyed through {current_scene_index + 1} scene(s) of adventure. 
The story continues with your brave heroes facing new challenges.

Current party status:
{character_status}
"""

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
    Character,
    GameStatus,
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
    characters: List[Character],
    scenario_name: str,
    previous_scene_image_path: Optional[pathlib.Path] = None,
    game_status: GameStatus = "ongoing",
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
        game_status: Status of the game ('ongoing', 'completed', 'failed')

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
        "- Characters should appear NATURALLY in the scene with varied angles and perspectives",
        "- DO NOT default to frontal/portrait poses - use three-quarter views, side angles, back views, or dynamic angles as appropriate",
        "- Characters should have DYNAMIC poses and body language that match the action/emotion of the scene",
        "- Show appropriate facial expressions when faces are visible (fear, determination, excitement, etc.)",
        "- Position characters naturally engaged in the scene's action, not standing static or posed for camera",
        "- Vary character positioning: some closer, some farther, some partially visible, creating depth",
        "- If combat/action: show fighting stances, movement, tension, characters in mid-action",
        "- If dialogue/social: show characters facing each other, gesturing, natural conversation poses",
        "- If exploration: show characters examining objects, pointing, reacting to environment, looking around",
        "- Match each character's personality traits, mannerisms, and nature to their pose, expression, and body language",
        "- Characters can be shown from any angle that serves the scene: side view, back view, three-quarter, profile, etc.",
        "- Use the character reference images ONLY for appearance details (face, clothing, build) - NOT for pose or angle",
        "- Create a cinematic, immersive scene where characters are part of the action, not posing for portraits",
    ]

    # Add note about previous scene if available
    if previous_scene_image_path:
        prompt_parts.append(
            "- Use the previous scene image as a reference for visual continuity in style, lighting, and environment"
        )

    # Add special instructions for ending scenes
    if game_status == "completed":
        prompt_parts.append(
            "\n🏆 VICTORY SCENE: This is the triumphant conclusion! Show epic victory, celebration, or peaceful resolution. Use dramatic, uplifting lighting (golden hour, radiant light). Emphasize heroic achievement and success."
        )
    elif game_status == "failed":
        prompt_parts.append(
            "\n💀 DEFEAT SCENE: This is the tragic ending. Show dramatic defeat, somber atmosphere, or fallen heroes. Use darker, dramatic lighting (shadows, dusk, stormy). Convey the weight of failure or sacrifice."
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
            contents=scene_text,
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


async def generate_scenarios(previously_played: list[str] | None = None) -> list[str]:
    """Calls Google Gemini to generate scenarios.

    Args:
        previously_played: Optional list of scenario names that were already played,
                          to ensure new scenarios are unique and distinct.
    """
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedScenarios)

    # Build prompt with context about previously played scenarios
    prompt = (
        "Generate three distinct fantasy adventure scenarios. Provide only the names."
    )

    if previously_played:
        previously_played_list = "\n".join(
            f"- {scenario}" for scenario in previously_played
        )
        prompt = f"""Generate three NEW and distinct fantasy adventure scenarios.

IMPORTANT: The following scenarios have ALREADY been played. Your new scenarios must be completely different and unique:

{previously_played_list}

Generate three fresh, creative fantasy adventure scenarios that are distinct from the ones listed above. 
Consider different themes, settings, conflict types, and narrative hooks.
Provide only the scenario names."""

    logger.info(
        f"Generating scenarios (avoiding {len(previously_played or [])} previously played)..."
    )
    result = await retry_on_overload(agent.run, prompt)

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
    concept_result = await retry_on_overload(agent.run, prompt)
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
    result = await retry_on_overload(agent.run, prompt)
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
    scene_id: int,
    scene_text: str,
    prompt,
    image_path: str | None = None,
    game_status: GameStatus = "ongoing",
) -> Scene:
    """Helper to create a Scene object from generated data.

    Args:
        scene_id: Unique scene identifier
        scene_text: The narrative text of the scene
        prompt: PromptType object from the generated scene
        image_path: Optional path to the scene image
        game_status: Status of the game ('ongoing', 'completed', 'failed')
    """
    return Scene(
        id=scene_id,
        text=scene_text,
        prompt=prompt,
        image_path=image_path,
        voiceover_path=None,
        game_status=game_status,
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

CRITICAL RULES FOR CHARACTER UPDATES:
- ALWAYS return updated_characters array with ALL party members
- Re-generate complete character data (stats, health, skills, inventory) based on what happened in the scene
- Update current_health for characters who took damage or were healed (keep stats same unless permanently changed)
- Update skills if a character learns a new ability or loses one (rare, but possible)
- Update inventory to reflect items gained, lost, used, or consumed
- If nothing happened to a character, return them unchanged
- Keep backstory, appearance, and personality the same (these don't change during adventures)

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

Return your response in this JSON structure (ALL FIELDS REQUIRED):
{
  "scene_text": "The vivid narrative of the scene...",
  "prompt": {
    "type": "dialogue" | "action" | "dice_check",
    "dice_type": "d6" | "d10" (only if type is dice_check, always single die roll),
    "target_character": "Character Name" (REQUIRED for single dice_check, optional for action/dialogue) or null,
    "target_characters": ["Char1", "Char2"] (for multi-character dice_check or action) or null,
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
  ],
  "game_status": "ongoing" | "completed" | "failed"
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

    # Apply character updates from the scene
    updated_characters = _apply_character_updates(
        characters, generated.updated_characters
    )

    # Generate scene image and voiceover in background threads (concurrently)
    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)  # type: ignore[attr-defined]

    image_task = asyncio.to_thread(
        _generate_scene_image_sync,
        client,
        game_id,
        scene_id,
        generated.scene_text,
        characters,
        scenario_name,
        None,  # No previous scene for opening scene
        generated.game_status,
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
    )
    scene.voiceover_path = voiceover_web_path
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

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Apply character updates from the scene
    updated_characters = _apply_character_updates(
        characters, generated.updated_characters
    )

    # Generate scene image and voiceover in background threads (concurrently)
    client = genai_client.Client(api_key=settings.GOOGLE_API_KEY)  # type: ignore[attr-defined]

    # Construct path to previous scene image for visual continuity
    previous_scene_image_path = None
    if last_scene_id >= 1:
        previous_scene_image_path = (
            SCENE_IMAGE_DIR / game_id / f"scene_{last_scene_id:03d}.png"
        )

    image_task = asyncio.to_thread(
        _generate_scene_image_sync,
        client,
        game_id,
        next_id,
        generated.scene_text,
        characters,
        scenario_name,
        previous_scene_image_path,
        generated.game_status,
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
    )
    scene.voiceover_path = voiceover_web_path
    return scene, updated_characters

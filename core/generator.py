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


def _generate_character_image_sync(
    client,
    concept,
    scenario_name: str,
    safe_scenario: str,
    scenario_dir: pathlib.Path,
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
                    image_file_path = scenario_dir / filename
                    img.save(image_file_path)
                    logger.info(f"Saved image for {concept.name} at {image_file_path}")
                    break
        except (OSError, ValueError):  # image decode/save issues
            image_file_path = None
            logger.warning(f"Failed to decode/save image for {concept.name}")

    return concept.name, image_file_path


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
    scenario_name: str, num_characters: int = 6, scenario_details: Optional[str] = None
) -> List[Character]:
    """Generates a list of characters with stats and portraits for a given scenario.

    Args:
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
- inventory: 3-5 items they carry (weapons, tools, magical items, personal effects)

Make each character distinctive and memorable with rich details that fit the scenario.
"""
    concept_result = await agent.run(prompt)
    generated_concepts = concept_result.output.characters
    logger.info(f"Character concepts generated: {[c.name for c in generated_concepts]}")

    # Step 2: Synchronous image generation using working google.genai client
    # Create scenario-specific directory
    safe_scenario = (
        "".join(c for c in scenario_name if c.isalnum() or c in " _-")
        .strip()
        .replace(" ", "_")
        or "scenario"
    )
    scenario_dir = IMAGE_DIR / safe_scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

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
            safe_scenario,
            scenario_dir,
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
            inventory=concept.inventory,
            image_path=(
                f"/static/characters/{safe_scenario}/{image_file_path.name}"
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


def _make_scene(scene_id: int, title: str, body: str) -> Scene:
    text = f"## Scene {scene_id}: {title}\n\n{body}"
    return Scene(id=scene_id, text=text, image_path=None, voiceover_path=None)


def generate_opening_scene(scenario_name: str, scene_id: int = 1) -> Scene:  # noqa: ARG001 - currently unused in placeholder body
    """Generate the opening scene for the chosen scenario.

    Placeholder implementation returns a short description. Replace with
    LLM-driven content later.
    """
    title = f"Opening of {scenario_name}"
    body = (
        "The party gathers at the edge of the Whispering Village as dusk falls. "
        "Smoke rises in the distance and villagers whisper of shadows moving in the marshes."
    )
    return _make_scene(scene_id, title, body)


def generate_next_scene(
    scenario_name: str, last_scene_id: int, player_action: Optional[str]
) -> Scene:
    """Generate the next scene based on the last scene and a player action.

    This placeholder just increments the scene id and produces a canned
    continuation. A real implementation would use the scenario, scene history,
    and the player's action to generate a coherent next scene.
    """
    next_id = last_scene_id + 1
    # scenario_name could be incorporated here in future for contextual generation
    title = (
        f"Aftermath {next_id} - {scenario_name}"
        if scenario_name
        else f"Aftermath {next_id}"
    )
    action_line = f"The party's action: '{player_action}'.\n\n" if player_action else ""
    body = (
        action_line
        + f"Pressing onward in the scope of the '{scenario_name}' narrative, the party moves deeper into the marsh. "
        + "Strange lights dance between the twisted trees and a distant tower gleams black against the sky."
    )
    return _make_scene(next_id, title, body)

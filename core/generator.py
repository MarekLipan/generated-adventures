"""
Handles content generation via external APIs.

This file currently contains placeholder deterministic generators. Replace
these with real model calls (Gemini, OpenAI, etc.) later.
"""

import pathlib
from io import BytesIO

from google import genai as genai_client  # type: ignore
import logging

from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from typing import Optional, List
from core.models import (
    Scene,
    Character,
    GeneratedScenarios,
    GeneratedScenarioDetails,
    GeneratedCharacterList,
)
from core.config import settings


logger = logging.getLogger()  # Use root logger so Uvicorn/NiceGUI picks up logs
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)

# NOTE: google.generativeai high-level configure may not export consistently in current lib version; using provider-based auth only.

IMAGE_DIR = pathlib.Path("webapp/static/characters")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


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
    scenario_name: str, num_characters: int = 6
) -> List[Character]:
    """Generates a list of characters with stats and portraits for a given scenario."""
    # Step 1: Generate character concepts (name, stats)

    logger.info(f"Generating {num_characters} characters for scenario: {scenario_name}")
    provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
    model = GoogleModel("gemini-2.5-pro", provider=provider)
    agent = Agent(model=model, output_type=GeneratedCharacterList)

    prompt = f"""
Generate {num_characters} unique and compelling fantasy characters for a D&D-style adventure named '{scenario_name}'.
Provide a diverse set of classic fantasy archetypes.
For each character, provide a name, strength, intelligence, and agility score between 1 and 20.
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

    final_characters: List[Character] = []

    for idx, concept in enumerate(generated_concepts, start=1):
        prompt_img = (
            "Generate a high quality fantasy character portrait. "
            f"Name: {concept.name}. Scenario: {scenario_name}. "
            "Style: painterly, dramatic lighting, 3/4 view, no text, clean background."
        )
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
                            "".join(
                                c for c in concept.name if c.isalnum() or c in " _-"
                            )
                            .rstrip()
                            .replace(" ", "_")
                        )
                        filename = f"{idx:02d}_{safe_name or 'character'}.png"
                        image_file_path = scenario_dir / filename
                        img.save(image_file_path)
                        logger.info(
                            f"Saved image for {concept.name} at {image_file_path}"
                        )
                        break
            except (OSError, ValueError):  # image decode/save issues
                image_file_path = None
                logger.warning(f"Failed to decode/save image for {concept.name}")

        new_char = Character(
            name=concept.name,
            strength=concept.strength,
            intelligence=concept.intelligence,
            agility=concept.agility,
            maximum_health=100,
            current_health=100,
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


def generate_opening_scene(
    scenario_name: str, scene_id: int = 1
) -> Scene:  # noqa: ARG001 - currently unused in placeholder body
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

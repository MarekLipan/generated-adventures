"""
Handles content generation via external APIs.

This file currently contains placeholder deterministic generators. Replace
these with real model calls (Gemini, OpenAI, etc.) later.
"""

from typing import Optional
from core.models import Scene


def generate_scenarios() -> list[str]:
    """Calls an external service to generate scenarios."""
    # In a real app, call your LLM here.
    return [
        "Rescue in Frostvale",
        "Siege of Emberhold",
        "The Lost Tomb of Varaxis",
    ]


def generate_characters() -> list[str]:
    """Calls an external service to generate characters."""
    # In a real app, call your LLM here.
    return [
        "Thalion the Ranger",
        "Mira the Mage",
        "Gruk the Barbarian",
        "Elowen the Druid",
        "Sera the Rogue",
        "Borin the Cleric",
    ]


def generate_scenario_details(scenario_name: str) -> str:
    """Generates the detailed story for the selected scenario.

    Returns a markdown string intended for DM notes.
    """
    return f"""
# {scenario_name}

## The Setting
Twilight and cold creep across the land. Strange omens and ruined watchtowers dot the roads.

## The Plot
The party must recover the Sunstone from the Sorcerer Malakor before the land freezes over.

## Main Quest
Journey to the Obsidian Spire, confront Malakor, and retrieve the Sunstone.

## Important NPCs
- **Elara, the Seer** – offers visions.
- **Garrick, the Disgraced Knight** – knows a hidden pass.

"""


def _make_scene(scene_id: int, title: str, body: str) -> Scene:
    text = f"## Scene {scene_id}: {title}\n\n{body}"
    return Scene(id=scene_id, text=text)


def generate_opening_scene(scenario_name: str, scene_id: int = 1) -> Scene:
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
    title = f"Aftermath {next_id}"
    action_line = f"The party's action: '{player_action}'.\n\n" if player_action else ""
    body = (
        action_line
        + "The party moves deeper into the marsh. Strange lights dance between the twisted trees and a distant tower gleams black against the sky."
    )
    return _make_scene(next_id, title, body)

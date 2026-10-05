"""Recaps for players returning to an adventure after a break."""

import logging

from ..llm import retry_on_overload, text_agent
from ..models import (
    Character,
    RecapResponse,
    Scene,
    SceneSummary,
)

logger = logging.getLogger(__name__)


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

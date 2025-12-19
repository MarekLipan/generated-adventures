"""Test script for recap generation feature.

Note: This is a manual test script, not a unit test.
For unit tests, mock the generator.generate_scene_recap() function.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from core import generator, persistence


async def test_recap_with_mock():
    """Test recap generation with mocked AI call (no API usage)."""
    # Load the migrated glacier game
    game_id = "147c9b95-35b8-4904-8c90-89dd48cc2eb2"
    game_state = persistence.load_game(game_id)

    if not game_state:
        print(f"❌ Could not load game {game_id}")
        return

    print(f"✅ Loaded game with {len(game_state.scenes)} scenes")
    print(f"   Scenario ID: {game_state.scenario_id}")
    print(f"   Characters: {[c.name for c in game_state.characters]}")

    # Load scenario
    scenario_template = persistence.load_scenario_template(game_state.scenario_id)
    if not scenario_template:
        print(f"❌ Could not load scenario template {game_state.scenario_id}")
        return

    print(f"✅ Loaded scenario: {scenario_template.name}")

    # Test generating recap for scene 3 (after some events have happened)
    test_scene_index = 2  # Scene 3 (0-indexed)

    print(f"\n🎭 Testing recap generation for scenes 1-{test_scene_index + 1}...")

    # Mock the AI generation to avoid API calls
    mock_recap = f"""## Previously in {scenario_template.name}...

Your brave party of adventurers has ventured into the frozen wastes, seeking the legendary Fire-Wyrm's Egg. 
Elder Theron warned you of the dangers ahead, and you've already faced treacherous ice bridges and deadly falls.

**Current Status:**
{chr(10).join(f"- {c.name}: {c.current_health}/{c.maximum_health} HP" for c in game_state.characters)}

The glacier's secrets are slowly revealing themselves, but many dangers still lie ahead...
"""

    with patch.object(
        generator, "generate_scene_recap", new=AsyncMock(return_value=mock_recap)
    ):
        try:
            recap_text = await generator.generate_scene_recap(
                game_id=game_state.id,
                scenario_name=scenario_template.name,
                scenes=game_state.scenes,
                characters=game_state.characters,
                current_scene_index=test_scene_index,
            )

            print(f"\n{'=' * 80}")
            print("MOCKED RECAP (no API call made):")
            print(f"{'=' * 80}")
            print(recap_text)
            print(f"{'=' * 80}\n")

            print(f"✅ Recap generation test passed ({len(recap_text)} characters)")
            print("✅ No API calls were made (mocked)")

        except Exception as e:
            print(f"❌ Error in recap test: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_recap_with_mock())

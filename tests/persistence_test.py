#!/usr/bin/env python3
"""
Quick test script to verify save/load functionality works correctly.
"""

import asyncio

from core import game, persistence
from core.models import Character


async def test_save_load():
    """Test creating, saving, and loading a game."""
    print("Testing game save/load functionality...")

    # Create a new game
    print("\n1. Creating new game...")
    game_id = game.create_new_game(players=3)
    print(f"   Created game with ID: {game_id}")

    # Select a scenario
    print("\n2. Selecting scenario...")
    game.select_scenario_for_game(game_id, "Test Scenario")
    print("   Scenario selected: Test Scenario")

    # Add a character
    print("\n3. Adding test character...")
    test_char = Character(
        name="Test Hero",
        strength=15,
        intelligence=12,
        agility=10,
        maximum_health=100,
        current_health=100,
        image_path="/static/test.png",
    )
    game.add_character_to_game(game_id, test_char)
    print(f"   Added character: {test_char.name}")

    # Clear from memory
    print("\n4. Clearing game from memory...")
    del game.games[game_id]
    print("   Game removed from memory")

    # Load from disk
    print("\n5. Loading game from disk...")
    loaded_game = game.load_game(game_id)

    if not loaded_game:
        print("   ❌ FAILED: Could not load game")
        return False

    print(f"   ✓ Game loaded successfully")
    print(f"   Scenario: {loaded_game.scenario_name}")
    print(f"   Players: {loaded_game.players}")
    print(f"   Characters: {len(loaded_game.characters)}")

    if loaded_game.characters:
        print(f"   First character: {loaded_game.characters[0].name}")

    # Verify data
    print("\n6. Verifying loaded data...")
    if loaded_game.scenario_name == "Test Scenario":
        print("   ✓ Scenario name matches")
    else:
        print(f"   ❌ Scenario mismatch: {loaded_game.scenario_name}")
        return False

    if loaded_game.players == 3:
        print("   ✓ Player count matches")
    else:
        print(f"   ❌ Player count mismatch: {loaded_game.players}")
        return False

    if len(loaded_game.characters) == 1:
        print("   ✓ Character count matches")
    else:
        print(f"   ❌ Character count mismatch: {len(loaded_game.characters)}")
        return False

    if loaded_game.characters[0].name == "Test Hero":
        print("   ✓ Character name matches")
    else:
        print(f"   ❌ Character name mismatch: {loaded_game.characters[0].name}")
        return False

    print("\n7. Listing saved games...")
    saved_games = persistence.list_saved_games()
    print(f"   Found {len(saved_games)} saved game(s)")
    for gid, scenario, summary in saved_games:
        print(f"   - {scenario}: {summary}")

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    asyncio.run(test_save_load())

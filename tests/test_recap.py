"""Test script for recap generation feature."""

import asyncio
from core import generator, persistence

async def test_recap():
    """Test recap generation for an existing game."""
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
    
    print(f"\n🎭 Generating recap for scenes 1-{test_scene_index + 1}...")
    
    try:
        recap_text = await generator.generate_scene_recap(
            game_id=game_state.id,
            scenario_name=scenario_template.name,
            scenes=game_state.scenes,
            characters=game_state.characters,
            current_scene_index=test_scene_index,
        )
        
        print(f"\n{'='*80}")
        print("GENERATED RECAP:")
        print(f"{'='*80}")
        print(recap_text)
        print(f"{'='*80}\n")
        
        # Test caching
        print(f"📝 Caching recap in scene {test_scene_index + 1}...")
        game_state.scenes[test_scene_index].recap_text = recap_text
        persistence.save_game(game_state)
        print("✅ Recap cached successfully!")
        
        # Reload and verify cache
        reloaded_game = persistence.load_game(game_id)
        if reloaded_game.scenes[test_scene_index].recap_text:
            print("✅ Cached recap verified after reload!")
        else:
            print("❌ Cached recap not found after reload")
        
    except Exception as e:
        print(f"❌ Error generating recap: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_recap())

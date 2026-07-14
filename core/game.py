"""
Core game state management.
This module is UI-agnostic.

It manages an in-memory mapping of games and exposes helpers to advance the
scene-based story by calling the generator.
"""

from datetime import datetime
from typing import Optional

from . import generator, persistence
from .models import Asset, Character, Game, ScenarioTemplate, Scene

# In-memory storage for active games, mapping game_id to a Game object.
games: dict[str, Game] = {}


def create_new_game(players: int) -> str:
    """Initializes a new game object and returns its ID."""
    new_game = Game(players=players)
    games[new_game.id] = new_game
    persistence.save_game(new_game)
    return new_game.id


def load_game(game_id: str) -> Optional[Game]:
    """Load a game from disk into memory. Returns the Game if found, None otherwise."""
    game = persistence.load_game(game_id)
    if game:
        games[game_id] = game
    return game


def select_scenario_for_game(game_id: str, scenario_id: str):
    """Updates the game state with the selected scenario template.

    Args:
        game_id: ID of the game
        scenario_id: ID of the ScenarioTemplate to use
    """
    if game_id in games:
        # Load the scenario template to update its play count
        scenario = persistence.load_scenario_template(scenario_id)
        if scenario:
            scenario.times_played += 1
            scenario.last_played_at = datetime.now()
            persistence.save_scenario_template(scenario)

        games[game_id].scenario_id = scenario_id
        persistence.save_game(games[game_id])


def set_art_style(game_id: str, art_style: str):
    """Set the visual art style for all generated imagery in this game."""
    if game_id in games:
        games[game_id].art_style = art_style
        persistence.save_game(games[game_id])


def add_character_to_game(game_id: str, character: Character):
    """Adds a new character to the game state."""
    if game_id in games:
        games[game_id].characters.append(character)
        persistence.save_game(games[game_id])


def convert_party_characters_to_assets(game_id: str):
    """Converts all party characters into assets for visual consistency.

    This ensures that player characters are treated as assets with reference images
    in scene generation, maintaining visual consistency throughout the adventure.
    Should be called once after the party is finalized and before the first scene.
    """
    game = games.get(game_id)
    if not game:
        return

    for character in game.characters:
        # Create an asset for each player character if not already present
        # Use a deterministic ID based on character name to avoid duplicates
        asset_id = f"player_{character.name.lower().replace(' ', '_')}"

        if asset_id not in game.assets:
            asset = Asset(
                id=asset_id,
                name=character.name,
                type="npc",
                description=character.appearance,
                image_path=character.image_path,
            )
            game.assets[asset_id] = asset

    persistence.save_game(game)


async def generate_opening_scene(game_id: str):
    """Generates the opening scene for the adventure.

    This should be called after the party has been selected and characters
    have been generated. It will create the first scene and update character
    states if needed.
    """
    game = games.get(game_id)
    if not game or not game.scenario_id or not game.characters:
        return

    # Load scenario template to get details
    scenario = persistence.load_scenario_template(game.scenario_id)
    if not scenario:
        return

    # Convert party characters to assets for visual consistency
    convert_party_characters_to_assets(game_id)

    # Generate initial locations for the adventure if not already done
    if not game.locations:
        game.locations = await generator.generate_initial_locations(
            scenario.name,
            scenario.dm_notes,
        )

    # Generate the very first scene for the players (and get updated character states, assets, and locations)
    (
        opening_scene,
        updated_characters,
        updated_assets,
        updated_locations,
    ) = await generator.generate_opening_scene(
        game_id,
        scenario.name,
        scenario.dm_notes,
        game.characters,
        game.assets,
        game.locations,
        art_style=game.art_style,
    )
    game.scenes.append(opening_scene)
    game.characters = updated_characters  # Update character states
    game.assets = updated_assets  # Update assets
    game.locations = updated_locations  # Update locations

    persistence.save_game(game)


def get_game_state(game_id: str) -> Optional[Game]:
    """Retrieves the current state of a game as a Pydantic object."""
    return games.get(game_id)


def get_scenario_from_game(game_id: str) -> Optional[ScenarioTemplate]:
    """Retrieves the scenario template for a game."""
    game = get_game_state(game_id)
    if not game or not game.scenario_id:
        return None
    return persistence.load_scenario_template(game.scenario_id)


def get_current_scene(game_id: str) -> Optional[Scene]:
    """Return the last scene in the game's scene list, if any."""
    game_state = get_game_state(game_id)
    if not game_state or not game_state.scenes:
        return None
    return game_state.scenes[-1]


async def advance_scene(game_id: str, player_action: Optional[str]) -> Optional[Scene]:
    """Generate and append the next scene based on a player's action.

    Returns the newly-created Scene, or None on error.
    """
    game_state = get_game_state(game_id)
    if not game_state:
        return None

    last_scene = get_current_scene(game_id)
    last_id = last_scene.id if last_scene else 0

    # Record action for history
    if player_action is not None:
        game_state.player_actions.append(player_action)

    # Build complete conversation history with scenes, prompts, and player actions
    # This gives the AI full context of the entire exchange
    conversation_history = []
    for i, scene in enumerate(game_state.scenes):
        conversation_history.append(
            {
                "scene_id": scene.id,
                "scene_text": scene.text,
                "prompt": scene.prompt,
                "player_action": game_state.player_actions[i]
                if i < len(game_state.player_actions)
                else None,
            }
        )

    # Get scenario details from template
    scenario = get_scenario_from_game(game_id)
    if not scenario:
        return None

    (
        next_scene,
        updated_characters,
        updated_assets,
        updated_locations,
    ) = await generator.generate_next_scene(
        game_id=game_id,
        scenario_name=scenario.name,
        scenario_details=scenario.dm_notes,
        characters=game_state.characters,
        last_scene_id=last_id,
        player_action=player_action,
        conversation_history=conversation_history,
        existing_assets=game_state.assets,
        existing_locations=game_state.locations,
        art_style=game_state.art_style,
    )
    game_state.scenes.append(next_scene)
    game_state.characters = updated_characters  # Update character states
    game_state.assets = updated_assets  # Update assets
    game_state.locations = updated_locations  # Update locations
    persistence.save_game(game_state)
    return next_scene

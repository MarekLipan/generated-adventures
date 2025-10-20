"""
Core game state management.
This module is UI-agnostic.

It manages an in-memory mapping of games and exposes helpers to advance the
scene-based story by calling the generator.
"""

from typing import Optional

from . import generator, persistence
from .models import Character, Game, Scene

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


def select_scenario_for_game(game_id: str, scenario_name: str):
    """Updates the game state with the selected scenario."""
    if game_id in games:
        games[game_id].scenario_name = scenario_name
        persistence.save_game(games[game_id])


def add_character_to_game(game_id: str, character: Character):
    """Adds a new character to the game state."""
    if game_id in games:
        games[game_id].characters.append(character)
        persistence.save_game(games[game_id])


async def generate_and_set_scenario_details(game_id: str):
    """Generates the scenario notes (DM) only.

    This will populate `game.scenario_details` with the background information
    for the Dungeon Master. The opening scene should be generated separately
    after the party is selected.
    """
    game = games.get(game_id)
    if not game or not game.scenario_name:
        return

    # Generate the main story details for the DM
    details = await generator.generate_scenario_details(game.scenario_name)
    game.scenario_details = details

    persistence.save_game(game)


async def generate_opening_scene(game_id: str):
    """Generates the opening scene for the adventure.

    This should be called after the party has been selected and characters
    have been generated. It will create the first scene and update character
    states if needed.
    """
    game = games.get(game_id)
    if not game or not game.scenario_name or not game.characters:
        return

    # Generate the very first scene for the players (and get updated character states)
    opening_scene, updated_characters = await generator.generate_opening_scene(
        game_id, game.scenario_name, game.scenario_details or "", game.characters
    )
    game.scenes.append(opening_scene)
    game.characters = updated_characters  # Update character states

    persistence.save_game(game)


def get_game_state(game_id: str) -> Optional[Game]:
    """Retrieves the current state of a game as a Pydantic object."""
    return games.get(game_id)


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

    next_scene, updated_characters = await generator.generate_next_scene(
        game_id=game_id,
        scenario_name=game_state.scenario_name or "",
        scenario_details=game_state.scenario_details or "",
        characters=game_state.characters,
        last_scene_id=last_id,
        player_action=player_action,
        conversation_history=conversation_history,
    )
    game_state.scenes.append(next_scene)
    game_state.characters = updated_characters  # Update character states
    persistence.save_game(game_state)
    return next_scene

"""
Core game state management.
This module is UI-agnostic.

It manages an in-memory mapping of games and exposes helpers to advance the
scene-based story by calling the generator.
"""

from typing import Optional
from .models import Game, Character, Scene
from . import generator

# In-memory storage for active games, mapping game_id to a Game object.
games: dict[str, Game] = {}


def create_new_game(players: int) -> str:
    """Initializes a new game object and returns its ID."""
    new_game = Game(players=players)
    games[new_game.id] = new_game
    return new_game.id


def select_scenario_for_game(game_id: str, scenario_name: str):
    """Updates the game state with the selected scenario."""
    if game_id in games:
        games[game_id].scenario_name = scenario_name


def add_character_to_game(game_id: str, character: Character):
    """Adds a new character to the game state."""
    if game_id in games:
        games[game_id].characters.append(character)


async def generate_and_set_scenario_details(game_id: str):
    """Generates the scenario notes (DM) and the opening scene.

    This will populate `game.scenario_details` and append the opening
    `Scene` into `game.scenes` so the UI can display it immediately.
    """
    game = games.get(game_id)
    if not game or not game.scenario_name:
        return

    # Generate the main story details for the DM
    details = await generator.generate_scenario_details(game.scenario_name)
    game.scenario_details = details

    # Generate the very first scene for the players
    opening_scene = generator.generate_opening_scene(game.scenario_name)
    game.scenes.append(opening_scene)


def get_game_state(game_id: str) -> Optional[Game]:
    """Retrieves the current state of a game as a Pydantic object."""
    return games.get(game_id)


def get_current_scene(game_id: str) -> Optional[Scene]:
    """Return the last scene in the game's scene list, if any."""
    game_state = get_game_state(game_id)
    if not game_state or not game_state.scenes:
        return None
    return game_state.scenes[-1]


def advance_scene(game_id: str, player_action: Optional[str]) -> Optional[Scene]:
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

    next_scene = generator.generate_next_scene(
        game_state.scenario_name or "", last_id, player_action
    )
    game_state.scenes.append(next_scene)
    return next_scene

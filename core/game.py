"""
Core game state management.
This module is UI-agnostic.
"""

from typing import Optional
from .models import Game, Character
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


def add_character_to_game(game_id: str, character_name: str):
    """Adds a new character to the game state."""
    if game_id in games:
        # This is a simplification. In a real scenario, you'd generate
        # the full character stats here.
        new_character = Character(
            name=character_name,
            strength=10,
            intelligence=10,
            agility=10,
            maximum_health=100,
            current_health=100,
        )
        games[game_id].characters.append(new_character)


def generate_and_set_scenario_details(game_id: str):
    """Generates the scenario story and saves it to the game state."""
    game_state = get_game_state(game_id)
    if game_state and game_state.scenario_name:
        scenario_details = generator.generate_scenario_details(game_state.scenario_name)
        game_state.scenario_details = scenario_details


def get_game_state(game_id: str) -> Optional[Game]:
    """Retrieves the current state of a game as a Pydantic object."""
    return games.get(game_id)

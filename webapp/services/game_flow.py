"""Service layer orchestrating game and generation logic for the UI components.

Purpose:
- Provide thin async wrappers around core.generator and core.game functions
  so UI components depend on this service instead of core modules directly.
- Offer helper accessors for game state.
"""

from __future__ import annotations

from typing import List, Optional

from core import game, generator, persistence
from core.models import Character, Game, ScenarioTemplate, Scene  # type: ignore

# --- Scenario management ----------------------------------------------------------


async def get_scenario_pool() -> List[ScenarioTemplate]:  # type: ignore
    """Load all available scenario templates."""
    return persistence.load_all_scenario_templates()


async def generate_new_scenario() -> ScenarioTemplate:  # type: ignore
    """Generate a new scenario template with contrastive prompting.

    Returns the newly created ScenarioTemplate.
    """
    # Load all existing scenarios for contrast
    existing_scenarios = persistence.load_all_scenario_templates()

    # Generate new scenario
    new_scenario = await generator.generate_scenario_template(existing_scenarios)

    # Save it
    persistence.save_scenario_template(new_scenario)

    return new_scenario


def select_scenario(game_id: str, scenario_id: str) -> None:
    """Select a scenario template for the game."""
    game.select_scenario_for_game(game_id, scenario_id)


def get_scenario_for_game(game_id: str) -> Optional[ScenarioTemplate]:  # type: ignore
    """Get the scenario template for a game."""
    return game.get_scenario_from_game(game_id)


async def generate_opening_scene(game_id: str) -> None:
    await game.generate_opening_scene(game_id)


# --- Game state access ------------------------------------------------------------


def create_new_game(players: int) -> str:
    return game.create_new_game(players)


def get_game_state(game_id: str) -> Optional[Game]:  # type: ignore
    return game.get_game_state(game_id)


# --- Characters ------------------------------------------------------------------


async def generate_characters(
    game_id: str,
    scenario_name: str,
    num_characters: int,
    scenario_details: str | None = None,
) -> List[Character]:
    return await generator.generate_characters(
        game_id=game_id,
        scenario_name=scenario_name,
        num_characters=num_characters,
        scenario_details=scenario_details,
    )  # type: ignore


def add_character(game_id: str, character: Character) -> None:
    game.add_character_to_game(game_id, character)


# --- Scenes / Adventure Loop ------------------------------------------------------


def get_current_scene(game_id: str) -> Optional[Scene]:  # type: ignore
    return game.get_current_scene(game_id)


async def advance_scene(game_id: str, player_action: str) -> Optional[Scene]:  # type: ignore
    return await game.advance_scene(game_id, player_action)


# --- Game persistence -------------------------------------------------------------


def load_game(game_id: str) -> Optional[Game]:  # type: ignore
    """Load a saved game from disk into memory."""
    return game.load_game(game_id)


def list_saved_games() -> list[tuple[str, str, str]]:
    """Return list of (game_id, scenario_name, summary) for all saved games."""
    return persistence.list_saved_games()

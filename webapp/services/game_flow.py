"""Service layer orchestrating game and generation logic for the UI components.

Purpose:
- Provide thin async wrappers around core.generator and core.game functions
  so UI components depend on this service instead of core modules directly.
- Centralize environment flag checks (e.g. SHOW_DM_NOTES).
- Offer helper accessors for game state.
"""

from __future__ import annotations

from typing import List, Optional
import os

from core import game, generator
from core.models import Character, Game, Scene  # type: ignore

# --- Environment / Config helpers -------------------------------------------------


def show_dm_notes_enabled() -> bool:
    return os.getenv("SHOW_DM_NOTES", "0") == "1"


# --- Scenario generation ----------------------------------------------------------


async def generate_scenarios() -> List[str]:
    return await generator.generate_scenarios()


async def generate_and_set_details(game_id: str) -> None:
    await game.generate_and_set_scenario_details(game_id)


# --- Game state access ------------------------------------------------------------


def create_new_game(players: int) -> str:
    return game.create_new_game(players)


def get_game_state(game_id: str) -> Optional[Game]:  # type: ignore
    return game.get_game_state(game_id)


def select_scenario(game_id: str, scenario_name: str) -> None:
    game.select_scenario_for_game(game_id, scenario_name)


# --- Characters ------------------------------------------------------------------


async def generate_characters(
    scenario_name: str, num_characters: int
) -> List[Character]:
    return await generator.generate_characters(scenario_name, num_characters=num_characters)  # type: ignore


def add_character(game_id: str, character: Character) -> None:
    game.add_character_to_game(game_id, character)


# --- Scenes / Adventure Loop ------------------------------------------------------


def get_current_scene(game_id: str) -> Optional[Scene]:  # type: ignore
    return game.get_current_scene(game_id)


def advance_scene(game_id: str, player_action: str) -> Optional[Scene]:  # type: ignore
    return game.advance_scene(game_id, player_action)

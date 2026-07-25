"""Service layer orchestrating game and generation logic for the UI components.

Purpose:
- Provide thin async wrappers around core.generator and core.game functions
  so UI components depend on this service instead of core modules directly.
- Offer helper accessors for game state.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

from core import game, generator, persistence
from core.models import (  # type: ignore
    Character,
    Game,
    GeneratedArchetype,
    ScenarioTemplate,
    Scene,
)

# Where uploaded player photos are stored (served at /static/photos/...).
PHOTO_DIR = pathlib.Path("webapp/static/photos")

# --- Scenario management ----------------------------------------------------------


async def get_scenario_pool() -> List[ScenarioTemplate]:  # type: ignore
    """Load all available scenario templates."""
    return persistence.load_all_scenario_templates()


def scenario_theme_options() -> dict[str, str]:
    """{key: label} genre choices for the scenario picker."""
    return generator.scenario_theme_options()


def scenario_scale_options() -> dict[str, str]:
    """{key: label} stakes/scale choices for the scenario picker."""
    return generator.scenario_scale_options()


async def generate_new_scenario(
    theme: str = "any", scale: str = "any"
) -> ScenarioTemplate:  # type: ignore
    """Generate a new scenario template, optionally steered to a genre and scale.

    Returns the newly created ScenarioTemplate.
    """
    # Load all existing scenarios for contrast
    existing_scenarios = persistence.load_all_scenario_templates()

    # Generate new scenario
    new_scenario = await generator.generate_scenario_template(
        existing_scenarios, theme=theme, scale=scale, avoid_names=_used_names()
    )

    # Save it
    persistence.save_scenario_template(new_scenario)

    return new_scenario


def select_scenario(game_id: str, scenario_id: str) -> None:
    """Select a scenario template for the game."""
    game.select_scenario_for_game(game_id, scenario_id)


def set_art_style(game_id: str, art_style: str) -> None:
    """Set the visual art style for the game."""
    game.set_art_style(game_id, art_style)


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


# --- Archetypes & photo-based heroes ----------------------------------------------


def _used_names() -> list[str]:
    """Names already used by saved scenarios and games (characters, NPCs, places),
    so new generations can avoid repeating them across adventures. Best-effort."""
    names: set[str] = set()
    try:
        for tpl in persistence.load_all_scenario_templates():
            names.update(getattr(tpl, "character_names", []) or [])
    except Exception:  # noqa: BLE001 - name de-dup is best-effort, never fatal
        pass
    try:
        for gid, _name, _summary in persistence.list_saved_games():
            g = persistence.load_game(gid)
            if not g:
                continue
            names.update(c.name for c in g.characters if c.name)
            names.update(
                a.name
                for a in g.assets.values()
                if getattr(a, "type", None) == "npc" and a.name
            )
    except Exception:  # noqa: BLE001
        pass
    return sorted(n for n in names if n and n.strip())


async def generate_archetypes(
    scenario_name: str,
    scenario_details: str | None = None,
    num_archetypes: int = 5,
    scale: str = "any",
) -> List[GeneratedArchetype]:  # type: ignore
    """Generate scenario-tailored hero archetypes for players to pick from."""
    return await generator.generate_archetypes(
        scenario_name=scenario_name,
        scenario_details=scenario_details,
        num_archetypes=num_archetypes,
        scale=scale,
    )


def save_player_photo(
    game_id: str, player_index: int, content: bytes, suffix: str = ".jpg"
) -> pathlib.Path:
    """Persist an uploaded player photo and return its file path.

    Stored under webapp/static/photos/<game_id>/player_<index>.<ext> so it is
    both usable as a generation reference and servable to the UI.
    """
    game_photo_dir = PHOTO_DIR / game_id
    game_photo_dir.mkdir(parents=True, exist_ok=True)
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    photo_path = game_photo_dir / f"player_{player_index}{ext}"
    photo_path.write_bytes(content)
    return photo_path


async def generate_hero(
    game_id: str,
    scenario_name: str,
    archetype: GeneratedArchetype,  # type: ignore
    art_style: str,
    player_index: int,
    scenario_details: str | None = None,
    photo_path: pathlib.Path | None = None,
    custom_name: str | None = None,
    gender: str = "unspecified",
) -> Character:  # type: ignore
    """Generate a single hero (lore + portrait) from a chosen archetype."""
    return await generator.generate_hero(
        game_id=game_id,
        scenario_name=scenario_name,
        scenario_details=scenario_details,
        archetype=archetype,
        art_style=art_style,
        player_index=player_index,
        photo_path=photo_path,
        custom_name=custom_name,
        gender=gender,
        avoid_names=_used_names(),
    )


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

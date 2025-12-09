"""
Game persistence - save/load game state to/from disk.

Stores games as JSON files in the data/games directory.
Stores scenario templates as JSON files in the data/scenarios/templates directory.
"""

import json
import pathlib
from typing import List, Optional

from .models import Game, ScenarioTemplate

GAMES_DIR = pathlib.Path("data/games")
GAMES_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS_DIR = pathlib.Path("data/scenarios/templates")
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)


def _game_file_path(game_id: str) -> pathlib.Path:
    """Return the path to the JSON file for a given game ID."""
    return GAMES_DIR / f"{game_id}.json"


def save_game(game: Game) -> None:
    """Save a game state to disk as JSON."""
    file_path = _game_file_path(game.id)
    with open(file_path, "w") as f:
        json.dump(game.model_dump(), f, indent=2)


def load_game(game_id: str) -> Optional[Game]:
    """Load a game state from disk. Returns None if not found."""
    file_path = _game_file_path(game_id)
    if not file_path.exists():
        return None

    with open(file_path, "r") as f:
        data = json.load(f)

    return Game.model_validate(data)


def list_saved_games() -> List[tuple[str, str, str]]:
    """Return list of (game_id, scenario_name, summary) for all saved games.

    Returns tuples of (game_id, scenario_name or 'Unknown', brief summary).
    Note: scenario_name is loaded from the referenced ScenarioTemplate.
    """
    games = []
    for game_file in GAMES_DIR.glob("*.json"):
        try:
            with open(game_file, "r") as f:
                data = json.load(f)

            game_id = data.get("id", game_file.stem)

            # Get scenario name from referenced template
            scenario_id = data.get("scenario_id")
            scenario_name = "Unknown Scenario"
            if scenario_id:
                scenario = load_scenario_template(scenario_id)
                if scenario:
                    scenario_name = scenario.name

            num_chars = len(data.get("characters", []))
            num_scenes = len(data.get("scenes", []))

            summary = f"{num_chars} characters, {num_scenes} scenes"
            games.append((game_id, scenario_name, summary))
        except (json.JSONDecodeError, KeyError):
            # Skip corrupted files
            continue

    return games


def delete_game(game_id: str) -> bool:
    """Delete a saved game file. Returns True if deleted, False if not found."""
    file_path = _game_file_path(game_id)
    if file_path.exists():
        file_path.unlink()
        return True
    return False


# ============================================================================
# Scenario Template Persistence
# ============================================================================


def _scenario_file_path(scenario_id: str) -> pathlib.Path:
    """Return the path to the JSON file for a given scenario template ID."""
    return SCENARIOS_DIR / f"{scenario_id}.json"


def save_scenario_template(scenario: ScenarioTemplate) -> None:
    """Save a scenario template to disk as JSON."""
    file_path = _scenario_file_path(scenario.id)
    # Convert datetime to ISO format for JSON serialization
    data = scenario.model_dump()
    if data.get("created_at"):
        data["created_at"] = data["created_at"].isoformat()
    if data.get("last_played_at"):
        data["last_played_at"] = data["last_played_at"].isoformat()

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_scenario_template(scenario_id: str) -> Optional[ScenarioTemplate]:
    """Load a scenario template from disk. Returns None if not found."""
    file_path = _scenario_file_path(scenario_id)
    if not file_path.exists():
        return None

    with open(file_path, "r") as f:
        data = json.load(f)

    return ScenarioTemplate.model_validate(data)


def load_all_scenario_templates() -> List[ScenarioTemplate]:
    """Load all scenario templates from disk, sorted by creation date (newest first)."""
    scenarios = []
    for scenario_file in SCENARIOS_DIR.glob("*.json"):
        try:
            with open(scenario_file, "r") as f:
                data = json.load(f)
            scenario = ScenarioTemplate.model_validate(data)
            scenarios.append(scenario)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Skip corrupted files
            continue

    # Sort by creation date, newest first
    scenarios.sort(key=lambda s: s.created_at, reverse=True)
    return scenarios


def delete_scenario_template(scenario_id: str) -> bool:
    """Delete a scenario template file. Returns True if deleted, False if not found."""
    file_path = _scenario_file_path(scenario_id)
    if file_path.exists():
        file_path.unlink()
        return True
    return False

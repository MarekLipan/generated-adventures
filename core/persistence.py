"""
Game persistence - save/load game state to/from disk.

Stores games as JSON files in the data/games directory.
"""

import json
import pathlib
from typing import List, Optional

from .models import Game

GAMES_DIR = pathlib.Path("data/games")
GAMES_DIR.mkdir(parents=True, exist_ok=True)


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
    """
    games = []
    for game_file in GAMES_DIR.glob("*.json"):
        try:
            with open(game_file, "r") as f:
                data = json.load(f)

            game_id = data.get("id", game_file.stem)
            scenario = data.get("scenario_name") or "Unknown Scenario"
            num_chars = len(data.get("characters", []))
            num_scenes = len(data.get("scenes", []))

            summary = f"{num_chars} characters, {num_scenes} scenes"
            games.append((game_id, scenario, summary))
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

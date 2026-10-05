"""Save/load round trip through core.game and core.persistence."""

import pytest

from generated_adventures.core import game, persistence


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point persistence at temp dirs so tests never touch real saves."""
    games_dir = tmp_path / "games"
    scenarios_dir = tmp_path / "scenarios"
    games_dir.mkdir()
    scenarios_dir.mkdir()
    monkeypatch.setattr(persistence, "GAMES_DIR", games_dir)
    monkeypatch.setattr(persistence, "SCENARIOS_DIR", scenarios_dir)


def test_save_load_round_trip(make_character):
    game_id = game.create_new_game(players=3)
    game.select_scenario_for_game(game_id, "test-scenario")
    game.add_character_to_game(game_id, make_character())

    del game.games[game_id]  # force the next load to come from disk
    loaded = game.load_game(game_id)

    assert loaded is not None
    assert loaded.scenario_id == "test-scenario"
    assert loaded.players == 3
    assert [c.name for c in loaded.characters] == ["Test Hero"]


def test_saved_game_is_listed(make_character):
    game_id = game.create_new_game(players=1)
    game.add_character_to_game(game_id, make_character())

    listed = {
        gid: (name, summary) for gid, name, summary in persistence.list_saved_games()
    }

    # No template was saved for this game, so the name falls back.
    assert listed[game_id] == ("Unknown Scenario", "1 characters, 0 scenes")

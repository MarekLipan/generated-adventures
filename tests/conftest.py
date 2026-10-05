"""Shared test fixtures."""

import pytest

from core.models import Character


@pytest.fixture
def make_character():
    """Build a valid Character; pass keyword overrides for the fields under test."""

    def _make(name: str = "Test Hero", **overrides) -> Character:
        fields = {
            "name": name,
            "backstory": "A former caravan guard.",
            "appearance": "Tall, scarred, in a weathered leather coat.",
            "personality": "Dry humor, fiercely loyal.",
            "strength": 12,
            "intelligence": 12,
            "agility": 12,
            "maximum_health": 100,
            "current_health": 100,
        }
        fields.update(overrides)
        return Character(**fields)

    return _make

"""Shared test fixtures."""

import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from core import generator
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


@pytest.fixture
def fake_llm(monkeypatch):
    """Route the generator's text model to a function; returns the captured prompts.

    Call it with a zero-argument function that returns a ModelResponse (or raises).
    """
    prompts: list[str] = []

    def install(reply):
        def fn(messages, info):
            prompts.append(messages[-1].parts[-1].content)
            return reply()

        # The generator uses NativeOutput, which needs a model that advertises
        # JSON-schema output.
        model = FunctionModel(
            fn, profile=ModelProfile(supports_json_schema_output=True)
        )
        monkeypatch.setattr(generator, "_get_text_model", lambda: model)
        return prompts

    return install

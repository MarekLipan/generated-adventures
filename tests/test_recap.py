"""generate_scene_recap: what reaches the prompt, output parsing, and the fallback.

The LLM is replaced by a pydantic-ai FunctionModel, so these tests run offline.
"""

import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from core import generator
from core.models import Scene


@pytest.fixture
def fake_llm(monkeypatch):
    """Route the generator's text model to a function; returns the captured prompts."""
    prompts: list[str] = []

    def install(reply):
        def fn(messages, info):
            prompts.append(messages[-1].parts[-1].content)
            return reply()

        # NativeOutput needs a model that advertises JSON-schema output.
        model = FunctionModel(fn, profile=ModelProfile(supports_json_schema_output=True))
        monkeypatch.setattr(generator, "_get_text_model", lambda: model)
        return prompts

    return install


SCENES = [
    Scene(id=1, text="The party crosses the frozen pass."),
    Scene(id=2, text="An ice bridge cracks beneath them."),
    Scene(id=3, text="They reach the wyrm's lair."),
]


async def _recap(characters, current_scene_index=1):
    return await generator.generate_scene_recap(
        game_id="g1",
        scenario_name="The Frozen Pass",
        scenes=SCENES,
        characters=characters,
        current_scene_index=current_scene_index,
    )


async def test_prompt_covers_scenes_up_to_current_only(fake_llm, make_character):
    reply = {"recap_text": "So far...", "scene_summaries": []}
    prompts = fake_llm(lambda: ModelResponse(parts=[TextPart(json.dumps(reply))]))

    await _recap([make_character("Lyra", current_health=80)])

    (prompt,) = prompts
    assert "frozen pass" in prompt and "ice bridge" in prompt
    assert "wyrm's lair" not in prompt  # scene 3 is after the current scene
    assert "**Lyra**: HP 80/100" in prompt


async def test_returns_parsed_model_output(fake_llm, make_character):
    reply = {
        "recap_text": "Previously, the party braved the pass.",
        "scene_summaries": [{"scene_id": 1, "summary": "Crossed the pass."}],
    }
    fake_llm(lambda: ModelResponse(parts=[TextPart(json.dumps(reply))]))

    recap = await _recap([make_character()])

    assert recap.recap_text == reply["recap_text"]
    assert [(s.scene_id, s.summary) for s in recap.scene_summaries] == [(1, "Crossed the pass.")]


async def test_falls_back_when_model_fails(fake_llm, make_character):
    def boom():
        raise RuntimeError("model unavailable")

    fake_llm(boom)

    recap = await _recap([make_character("Lyra")])

    assert "The Frozen Pass" in recap.recap_text
    assert "Lyra" in recap.recap_text
    assert [s.scene_id for s in recap.scene_summaries] == [1, 2]

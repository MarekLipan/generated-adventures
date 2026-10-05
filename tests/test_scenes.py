"""The scene pipeline: one LLM reply becomes a playable scene and an updated party.

The LLM is a FunctionModel and rendering is stubbed, so this runs offline and
writes nothing to disk. The bookkeeping in between (character changes, location
and subject resolution, scene assembly) runs for real.
"""

import pathlib

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from generated_adventures.core.generation import scenes
from generated_adventures.core.models import (
    CharacterHealthChange,
    GeneratedScene,
    PromptType,
)


@pytest.fixture
def no_rendering(monkeypatch):
    """Stub image, narration and asset-portrait rendering; record what was rendered."""
    rendered = {}

    def image(generator, game_id, scene_id, *args):
        rendered["image"] = (game_id, scene_id)
        return pathlib.Path(f"scene_{scene_id:03d}.png")

    async def voiceover(game_id, scene_id, *args):
        rendered["voice"] = (game_id, scene_id)
        return pathlib.Path(f"scene_{scene_id:03d}.wav")

    def assets(game_id, scene_id, present, existing, generator, art_style):
        return existing, []

    monkeypatch.setattr(scenes, "shared_image_generator", lambda: object())
    monkeypatch.setattr(scenes, "generate_scene_image_sync", image)
    monkeypatch.setattr(scenes, "generate_scene_voiceover", voiceover)
    monkeypatch.setattr(scenes, "_process_scene_assets", assets)
    return rendered


def _reply(scene: GeneratedScene):
    return lambda: ModelResponse(parts=[TextPart(scene.model_dump_json())])


async def test_next_scene_applies_changes_and_renders(
    fake_llm, make_character, no_rendering
):
    reply = GeneratedScene(
        scene_text="The ice bridge cracks and falling shards graze Lyra.",
        visual_description="A ranger on a cracking ice bridge.",
        prompt=PromptType(type="action", prompt_text="What does Lyra do?"),
        health_changes=[
            CharacterHealthChange(
                character_name="Lyra", health_change=-10, reason="falling ice"
            )
        ],
    )
    prompts = fake_llm(_reply(reply))

    scene, party, _, _ = await scenes.generate_next_scene(
        game_id="g1",
        scenario_name="The Frozen Pass",
        scenario_details="## The Setting\nA glacier.",
        characters=[make_character("Lyra")],
        last_scene_id=1,
        player_action="Lyra crosses the ice bridge.",
        conversation_history=[
            {
                "scene_id": 1,
                "scene_text": "You reach a frozen pass.",
                "prompt": PromptType(type="action", prompt_text="What now?"),
                "player_action": "Lyra crosses the ice bridge.",
            }
        ],
        existing_assets={},
        existing_locations={},
    )

    (prompt,) = prompts
    assert "Lyra crosses the ice bridge." in prompt
    assert (scene.id, scene.text) == (2, reply.scene_text)
    assert party[0].current_health == 90
    assert scene.image_path == "/static/scenes/g1/scene_002.png"
    assert scene.voiceover_path == "/static/voiceovers/g1/scene_002.wav"
    assert no_rendering == {"image": ("g1", 2), "voice": ("g1", 2)}


async def test_opening_scene_uses_the_given_id(fake_llm, make_character, no_rendering):
    reply = GeneratedScene(
        scene_text="Snow drifts across the pass.",
        visual_description="A snowy mountain pass at dawn.",
        prompt=PromptType(type="party_choice", prompt_text="Where do you go?"),
    )
    fake_llm(_reply(reply))

    scene, party, _, _ = await scenes.generate_opening_scene(
        game_id="g1",
        scenario_name="The Frozen Pass",
        scenario_details="## The Setting\nA glacier.",
        characters=[make_character("Lyra")],
        existing_assets={},
        existing_locations={},
    )

    assert (scene.id, scene.text, scene.prompt) == (1, reply.scene_text, reply.prompt)
    assert party[0].current_health == 100
    assert no_rendering == {"image": ("g1", 1), "voice": ("g1", 1)}

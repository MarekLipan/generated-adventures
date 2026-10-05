"""generate_characters: concepts from the LLM, portraits rendered concurrently."""

from pydantic_ai.messages import ModelResponse, TextPart

from core.generation import characters
from core.models import GeneratedCharacter, GeneratedCharacterList


def _concept(name: str) -> GeneratedCharacter:
    return GeneratedCharacter(
        name=name,
        strength=12,
        intelligence=12,
        agility=12,
        backstory="Grew up on the frontier.",
        appearance="Lean, weathered, quick eyes.",
        personality="Wry and watchful.",
        skills=["Tracking"],
        inventory=["Rope"],
    )


async def test_one_failed_portrait_does_not_sink_the_party(
    fake_llm, monkeypatch, tmp_path
):
    concepts = GeneratedCharacterList(characters=[_concept("Ada"), _concept("Brann")])
    fake_llm(lambda: ModelResponse(parts=[TextPart(concepts.model_dump_json())]))

    def render(generator_, concept, scenario_name, game_id, game_dir, idx):
        if concept.name == "Brann":
            raise RuntimeError("GPU fell over")
        path = game_dir / f"{idx}.png"
        path.write_bytes(b"png")
        return concept.name, path

    monkeypatch.setattr(characters, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(characters, "shared_image_generator", lambda: object())
    monkeypatch.setattr(characters, "generate_character_image_sync", render)

    party = await characters.generate_characters(
        "g1", "The Frozen Pass", num_characters=2
    )

    assert [c.name for c in party] == ["Ada", "Brann"]
    assert party[0].image_path == "/static/characters/g1/1.png"
    assert party[1].image_path is None

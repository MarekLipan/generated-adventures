from typing import List
from nicegui import ui
import asyncio
from core.models import Character
from webapp.services import game_flow  # type: ignore

from .party import show_character_overview  # type: ignore


async def show_characters(main_container, game_id: str, scenario_name: str):
    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        return

    all_characters: List[Character] = await game_flow.generate_characters(scenario_name, num_characters=6)  # type: ignore
    selected = {c.name for c in game_state.characters}
    available = [c for c in all_characters if c.name not in selected]

    main_container.clear()
    with main_container:
        ui.label(
            f"Choose Character ({len(game_state.characters)}/{game_state.players})"
        ).classes("text-h4")
        with ui.grid(columns=3).classes("w-full gap-4 mt-4"):
            for character in available:
                with ui.card().classes("w-60"):
                    if character.image_path:
                        ui.image(character.image_path).classes(
                            "w-full h-48 object-cover"
                        )
                    ui.label(character.name).classes("text-h6 text-center mt-2")
                    ui.label(f"Strength: {character.strength}").classes("text-sm")
                    ui.label(f"Intelligence: {character.intelligence}").classes(
                        "text-sm"
                    )
                    ui.label(f"Agility: {character.agility}").classes("text-sm")
                    ui.label(f"Health: {character.current_health}").classes("text-sm")
                    ui.button(
                        "Select",
                        on_click=lambda _e=None, c=character: asyncio.create_task(
                            select_character(main_container, game_id, scenario_name, c)
                        ),
                    ).classes("mt-2 w-full")


async def select_character(
    main_container, game_id: str, scenario_name: str, character: Character
):
    game_flow.add_character(game_id, character)
    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        return
    if len(game_state.characters) < game_state.players:
        await show_characters(main_container, game_id, scenario_name)
    else:
        show_character_overview(main_container, game_id)

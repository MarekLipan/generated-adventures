from nicegui import ui
from webapp.services import game_flow  # type: ignore
from .adventure import start_adventure  # type: ignore
import asyncio


def show_character_overview(main_container, game_id: str):
    """Display the selected party and provide a button to start the adventure."""
    main_container.clear()
    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        return
    with main_container:
        ui.label("Your Party is Ready!").classes("text-h4")
        with ui.grid(columns=game_state.players).classes("w-full gap-4"):
            for character in game_state.characters:
                with ui.card():
                    ui.label(character.name).classes("text-h6")
                    ui.label(f"Strength: {character.strength}")
                    ui.label(f"Intelligence: {character.intelligence}")
                    ui.label(f"Agility: {character.agility}")
                    ui.label(f"Health: {character.current_health}")
        ui.button(
            "Start Adventure!",
            on_click=lambda _e=None: asyncio.create_task(
                start_adventure(main_container, game_id)
            ),
        ).classes("mt-4")

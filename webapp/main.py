"""Web application entry point wiring FastAPI and NiceGUI using modular components."""

from fastapi import FastAPI
from nicegui import ui
from webapp.services import game_flow  # type: ignore

# Component imports (UI flow steps)
from .components.scenarios import show_scenarios  # type: ignore

app = FastAPI()


@ui.page("/")
def main_page():
    """Defines the user interface shell and delegates flow to component functions."""
    ui.label("Generated Adventures").classes("text-h2 text-primary")
    main_container = ui.column().classes("w-full items-center")

    async def new_game_dialog():
        """Dialog for new game setup then launches scenario selection."""
        with ui.dialog() as dialog, ui.card():
            players_input = ui.number(label="Number of Players", value=1, min=1)
            with ui.row():
                ui.button(
                    "Start",
                    on_click=lambda _e=None: dialog.submit(players_input.value),
                )
                ui.button("Cancel", on_click=dialog.close)

        num_players = await dialog
        if num_players:
            game_id = game_flow.create_new_game(num_players)
            await show_scenarios(main_container, game_id)

    with main_container:
        ui.button("New Game", on_click=new_game_dialog).classes("text-lg")


ui.run_with(app)

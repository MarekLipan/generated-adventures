"""Web application entry point wiring FastAPI and NiceGUI using modular components."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .components.resume import resume_game  # type: ignore

# Component imports (UI flow steps)
from .components.scenarios import show_scenarios  # type: ignore

app = FastAPI()

# Mount static files directory for character images
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@ui.page("/")
def main_page():
    """Defines the user interface shell and delegates flow to component functions."""
    # Add custom CSS
    ui.add_head_html('<link rel="stylesheet" href="/static/css/fantasy-theme.css">')
    ui.add_head_html(
        '<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap" rel="stylesheet">'
    )

    # Set dark mode
    ui.dark_mode().enable()

    # Game Title - centered
    with ui.row().classes("w-full justify-center mb-8"):
        ui.label("⚔️ Generated Adventures ⚔️").classes("game-title")
    main_container = ui.column().classes("w-full items-center")

    async def new_game_dialog():
        """Dialog for new game setup then launches scenario selection."""
        with ui.dialog() as dialog, ui.card().classes("fantasy-panel"):
            ui.label("⚔️ Begin Your Quest").classes("text-h5 mb-4")
            players_input = ui.number(label="Number of Players", value=1, min=1)
            with ui.row().classes("gap-2 mt-4"):
                ui.button(
                    "Start Adventure",
                    on_click=lambda _e=None: dialog.submit(players_input.value),
                )
                ui.button("Cancel", on_click=dialog.close)

        num_players = await dialog
        if num_players:
            game_id = game_flow.create_new_game(num_players)
            await show_scenarios(main_container, game_id)

    async def load_game_dialog():
        """Dialog to select and load a saved game."""
        saved_games = game_flow.list_saved_games()

        if not saved_games:
            with ui.dialog() as dialog, ui.card().classes("fantasy-panel"):
                ui.label("📜 No saved games found.").classes("text-lg")
                ui.button("OK", on_click=dialog.close).classes("mt-4")
            await dialog
            return

        with ui.dialog() as dialog, ui.card().classes("fantasy-panel"):
            ui.label("📜 Select a Quest to Resume").classes("text-h5 mb-4")

            selected_game_id = None

            def select_game(game_id: str):
                nonlocal selected_game_id
                selected_game_id = game_id
                dialog.submit(game_id)

            for game_id, scenario_name, summary in saved_games:
                with ui.card().classes("w-full cursor-pointer mb-2"):
                    ui.button(
                        f"⚔️ {scenario_name}\n{summary}",
                        on_click=lambda _e=None, gid=game_id: select_game(gid),
                    ).classes("w-full")

            ui.button("Cancel", on_click=dialog.close).classes("mt-4")

        selected_id = await dialog
        if selected_id:
            loaded_game = game_flow.load_game(selected_id)
            if loaded_game:
                await resume_game(main_container, selected_id)

    with main_container:
        with ui.card().classes("fantasy-panel mt-8"):
            ui.label("Welcome, Adventurer!").classes("text-h4 mb-4 text-center")
            ui.label("Choose your path...").classes(
                "text-body1 fantasy-text-muted mb-6 text-center"
            )
            ui.button("⚔️ New Game", on_click=new_game_dialog).classes("text-lg w-64")
            ui.button("📜 Load Game", on_click=load_game_dialog).classes(
                "text-lg mt-4 w-64"
            )


ui.run_with(app)

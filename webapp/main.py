"""Web application entry point (FastAPI and NiceGUI)."""

from fastapi import FastAPI
from nicegui import ui
from core import game, generator
import os

app = FastAPI()


@ui.page("/")
def main_page():
    """Defines the user interface."""
    ui.label("Generated Adventures").classes("text-h2 text-primary")
    main_container = ui.column().classes("w-full items-center")

    def show_scenarios(game_id: str):
        """Clears UI and shows scenario selection."""
        scenarios = generator.generate_scenarios()
        main_container.clear()
        with main_container:
            ui.label("Choose a Scenario").classes("text-h4")
            with ui.row():
                for scenario in scenarios:
                    ui.button(
                        scenario,
                        on_click=lambda _event=None, s=scenario: show_characters(
                            game_id, s
                        ),
                    )

    def show_characters(game_id: str, scenario_name: str):
        """Clears UI and shows character selection, tracking selected characters."""
        game.select_scenario_for_game(game_id, scenario_name)
        game_state = game.get_game_state(game_id)
        if not game_state:
            return  # Or handle error

        selected_character_names = {char.name for char in game_state.characters}
        available_characters = [
            char
            for char in generator.generate_characters()
            if char not in selected_character_names
        ]

        main_container.clear()
        with main_container:
            ui.label(
                f"Choose Character ({len(game_state.characters)}/{game_state.players})"
            ).classes("text-h4")

            with ui.row():
                for character_name in available_characters:
                    ui.button(
                        character_name,
                        on_click=lambda _event=None, c=character_name: select_character(
                            game_id, scenario_name, c
                        ),
                    )

    def select_character(game_id: str, scenario_name: str, character_name: str):
        """Adds a character and checks if the team is full."""
        game.add_character_to_game(game_id, character_name)
        game_state = game.get_game_state(game_id)
        if not game_state:
            return

        if len(game_state.characters) < game_state.players:
            # Refresh the character selection screen
            show_characters(game_id, scenario_name)
        else:
            # All characters selected, proceed to overview
            show_character_overview(game_id)

    def show_character_overview(game_id: str):
        """Displays the selected characters and their stats."""
        main_container.clear()
        game_state = game.get_game_state(game_id)
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
                on_click=lambda _event=None, gid=game_id: start_adventure(gid),
            ).classes("mt-4")
            # This button will eventually trigger the first game scene.

    def start_adventure(game_id: str):
        """Generates the story and displays it.

        DM-only notes (game_state.scenario_details) are only rendered when the
        environment variable SHOW_DM_NOTES is set to "1". Otherwise a generic
        player-facing message is shown.
        """
        game.generate_and_set_scenario_details(game_id)
        game_state = game.get_game_state(game_id)
        if not game_state:
            return

        main_container.clear()
        with main_container:
            if os.getenv("SHOW_DM_NOTES", "0") == "1":
                # Render full DM notes (markdown)
                if game_state.scenario_details:
                    ui.markdown(game_state.scenario_details).classes("w-full text-left")
                else:
                    ui.label("No scenario details available.")
            else:
                # Player-facing minimal view
                ui.label("The adventure has begun! DM notes are hidden.").classes(
                    "text-h5"
                )
                if game_state.scenario_name:
                    ui.label(f"Scenario: {game_state.scenario_name}")

    async def new_game_dialog():
        """Shows a dialog to start a new game."""
        with ui.dialog() as dialog, ui.card():
            players_input = ui.number(label="Number of Players", value=1, min=1)
            with ui.row():
                ui.button(
                    "Start",
                    on_click=lambda _event=None: dialog.submit(players_input.value),
                )
                ui.button("Cancel", on_click=dialog.close)

        num_players = await dialog
        if num_players:
            game_id = game.create_new_game(num_players)
            show_scenarios(game_id)

    with main_container:
        ui.button("New Game", on_click=new_game_dialog).classes("text-lg")


ui.run_with(app)

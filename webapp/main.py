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

    async def show_scenarios(game_id: str):
        """Clears UI and shows scenario selection."""
        scenarios = await generator.generate_scenarios()
        main_container.clear()
        with main_container:
            ui.label("Choose a Scenario").classes("text-h4")
            with ui.row():
                for scenario in scenarios:
                    ui.button(
                        scenario,
                        on_click=lambda _event=None, s=scenario: handle_scenario_selection(
                            game_id, s
                        ),
                    )

    def handle_scenario_selection(game_id: str, scenario_name: str):
        """Handles scenario selection, generates story, then proceeds to character selection."""
        game.select_scenario_for_game(game_id, scenario_name)
        game.generate_and_set_scenario_details(game_id)

        if os.getenv("SHOW_DM_NOTES", "0") == "1":
            show_dm_notes_before_characters(game_id, scenario_name)
        else:
            show_characters(game_id, scenario_name)

    def show_dm_notes_before_characters(game_id: str, scenario_name: str):
        """Displays DM notes and then a button to proceed to character selection."""
        main_container.clear()
        game_state = game.get_game_state(game_id)
        if not game_state or not game_state.scenario_details:
            # Fallback if something went wrong
            show_characters(game_id, scenario_name)
            return

        with main_container:
            ui.markdown(game_state.scenario_details).classes("w-full text-left")
            ui.button(
                "Continue to Character Selection",
                on_click=lambda: show_characters(game_id, scenario_name),
            ).classes("mt-4")

    def show_characters(game_id: str, scenario_name: str):
        """Clears UI and shows character selection, tracking selected characters."""
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
        """Start the interactive scene loop: render current scene and prompt action."""

        def render_scene(scene):
            main_container.clear()
            with main_container:
                ui.label(f"Scene {scene.id}").classes("text-h5")
                ui.markdown(scene.text).classes("w-full text-left")

                # Action input
                action_input = ui.input(
                    label="Your action", placeholder="What does your party do?"
                )

                def on_submit(_event=None):
                    player_action = action_input.value
                    # Advance scene and re-render
                    next_scene = game.advance_scene(game_id, player_action)
                    if next_scene:
                        render_scene(next_scene)
                    else:
                        ui.label("No further scenes.")

                ui.button("Submit Action", on_click=on_submit).classes("mt-4")

        # Render the current scene (should exist because generate_and_set_scenario_details
        # created the opening scene). If not present, attempt to generate.
        current = game.get_current_scene(game_id)
        if not current:
            # try to generate initial details and scene
            game.generate_and_set_scenario_details(game_id)
            current = game.get_current_scene(game_id)

        if current:
            render_scene(current)
        else:
            main_container.clear()
            ui.label("Unable to start the adventure: no scene available.")

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
            await show_scenarios(game_id)

    with main_container:
        ui.button("New Game", on_click=new_game_dialog).classes("text-lg")


ui.run_with(app)

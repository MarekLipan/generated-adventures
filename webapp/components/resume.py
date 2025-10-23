"""Resume game flow - detects game state and routes to appropriate component."""

from nicegui import ui

from webapp.services import game_flow  # type: ignore
from webapp.utils import show_api_error  # type: ignore

from .adventure import start_adventure  # type: ignore
from .characters import show_characters  # type: ignore
from .dm_notes import show_dm_notes_before_characters  # type: ignore
from .party import show_character_overview  # type: ignore
from .scenarios import show_scenarios  # type: ignore


async def resume_game(main_container, game_id: str):
    """Detect game state and resume at the appropriate step."""
    game_state = game_flow.get_game_state(game_id)

    if not game_state:
        main_container.clear()
        with main_container:
            with ui.card().classes("fantasy-panel"):
                ui.label("⚠️ Error: Could not load game state.").classes(
                    "text-h5 fantasy-accent-red"
                )
        return

    # Determine where to resume based on game state

    # If no scenario selected, show scenario selection
    if not game_state.scenario_name:
        await show_scenarios(main_container, game_id)
        return

    # If scenario selected but no details generated, generate and continue
    if not game_state.scenario_details:
        try:
            await game_flow.generate_and_set_details(game_id)
            game_state = game_flow.get_game_state(game_id)  # Refresh state
        except Exception as e:
            show_api_error(
                main_container,
                error=e,
                title="Error Generating Scenario Details",
                message="The Dungeon Master encountered an issue while preparing the scenario.",
                retry_callback=None,  # No retry - user can refresh or go back
            )
            return

    # If we have scenes, game has started - go to adventure
    if game_state.scenes:
        await start_adventure(main_container, game_id)
        return

    # If we have characters selected, show party overview
    if game_state.characters:
        show_character_overview(main_container, game_id)
        return

    # If we have scenario and details but no characters, show character selection
    # Check if DM notes should be shown first
    if game_flow.show_dm_notes_enabled():
        await show_dm_notes_before_characters(
            main_container, game_id, game_state.scenario_name
        )
    else:
        await show_characters(main_container, game_id, game_state.scenario_name)

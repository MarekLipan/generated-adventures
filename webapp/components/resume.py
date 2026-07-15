"""Resume game flow - detects game state and routes to appropriate component."""

from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .adventure import start_adventure  # type: ignore
from .hero_creation import start_hero_creation  # type: ignore
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
    if not game_state.scenario_id:
        await show_scenarios(main_container, game_id)
        return

    # Get scenario details from template
    scenario = game_flow.get_scenario_for_game(game_id)
    if not scenario:
        # If scenario ID exists but template not found, show error
        main_container.clear()
        with main_container:
            with ui.card().classes("fantasy-panel"):
                ui.label("⚠️ Error: Scenario template not found.").classes(
                    "text-h5 fantasy-accent-red"
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

    # If we have scenario but no characters, start hero creation
    await start_hero_creation(main_container, game_id, scenario.name)

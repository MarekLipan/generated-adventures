import asyncio
import logging

from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .hero_creation import start_hero_creation  # type: ignore

logger = logging.getLogger()


async def show_dm_notes_before_characters(
    main_container, game_id: str, scenario_name: str
):
    logger.info(f"show_dm_notes_before_characters called for game {game_id}")
    main_container.clear()

    # Get scenario template from game
    scenario = game_flow.get_scenario_for_game(game_id)
    if not scenario:
        logger.warning("No scenario found, skipping to hero creation")
        await start_hero_creation(main_container, game_id, scenario_name)
        return

    logger.info("Displaying DM notes")
    with main_container:
        with ui.card().classes("fantasy-panel w-full max-w-4xl"):
            ui.label("📜 Dungeon Master's Notes").classes("text-h4 mb-6")
            ui.markdown(scenario.dm_notes).classes("markdown w-full text-left mb-4")
            ui.button(
                "⚔️ Continue to Hero Creation",
                on_click=lambda: asyncio.create_task(
                    start_hero_creation(main_container, game_id, scenario_name)
                ),
            ).classes("mt-6 w-full")

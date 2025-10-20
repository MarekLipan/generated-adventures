import asyncio
import logging

from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .characters import show_characters  # type: ignore
from .dm_notes import show_dm_notes_before_characters  # type: ignore

logger = logging.getLogger()


async def show_scenarios(main_container, game_id: str):
    # Show loading indicator while generating scenarios
    main_container.clear()
    with main_container:
        ui.label("Generating scenarios...").classes("text-h5 mb-4")
        ui.spinner(size="lg")
        ui.label("🗺️ Creating adventure scenarios...").classes("text-gray-600 mt-4")

    scenarios = await game_flow.generate_scenarios()
    main_container.clear()
    with main_container:
        ui.label("Choose a Scenario").classes("text-h4")
        with ui.row():
            for scenario in scenarios:
                ui.button(
                    scenario,
                    on_click=lambda _event=None, s=scenario: asyncio.create_task(
                        handle_scenario_selection(main_container, game_id, s)
                    ),
                )


async def handle_scenario_selection(main_container, game_id: str, scenario_name: str):
    logger.info(f"Scenario selected: {scenario_name} for game {game_id}")
    game_flow.select_scenario(game_id, scenario_name)

    # Show loading indicator while generating scenario details
    main_container.clear()
    with main_container:
        ui.label("Generating scenario details...").classes("text-h5 mb-4")
        ui.spinner(size="lg")
        ui.label("📜 The Dungeon Master is preparing the story...").classes(
            "text-gray-600 mt-4"
        )

    logger.info("Generating scenario details...")
    await game_flow.generate_and_set_details(game_id)
    logger.info("Scenario details generated")

    if game_flow.show_dm_notes_enabled():
        logger.info("DM notes enabled, showing DM notes first")
        await show_dm_notes_before_characters(main_container, game_id, scenario_name)
    else:
        logger.info("DM notes disabled, going directly to character selection")
        await show_characters(main_container, game_id, scenario_name)

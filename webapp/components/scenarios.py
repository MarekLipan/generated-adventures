import asyncio
import logging

from nicegui import ui

from webapp.services import game_flow  # type: ignore
from webapp.utils import show_api_error, show_loading  # type: ignore

from .characters import show_characters  # type: ignore
from .dm_notes import show_dm_notes_before_characters  # type: ignore

logger = logging.getLogger()


async def show_scenarios(main_container, game_id: str):
    # Show loading indicator while generating scenarios
    show_loading(
        main_container,
        "🎲 Generating Scenarios...",
        "The Dungeon Master is preparing epic tales...",
    )

    try:
        scenarios = await game_flow.generate_scenarios()
    except Exception as e:
        show_api_error(
            main_container,
            error=e,
            title="Error Generating Scenarios",
            message="The Dungeon Master encountered an issue while preparing adventures.",
            retry_callback=lambda: asyncio.create_task(
                show_scenarios(main_container, game_id)
            ),
        )
        return

    main_container.clear()
    with main_container:
        with ui.card().classes("fantasy-panel"):
            ui.label("🗺️ Choose Your Adventure").classes("text-h4 mb-6")
            with ui.column().classes("gap-3"):
                for scenario in scenarios:
                    ui.button(
                        f"⚔️ {scenario}",
                        on_click=lambda _event=None, s=scenario: asyncio.create_task(
                            handle_scenario_selection(main_container, game_id, s)
                        ),
                    ).classes("w-full text-left")


async def handle_scenario_selection(main_container, game_id: str, scenario_name: str):
    logger.info(f"Scenario selected: {scenario_name} for game {game_id}")
    game_flow.select_scenario(game_id, scenario_name)

    # Show loading indicator while generating scenario details
    show_loading(
        main_container,
        "📜 Preparing Your Quest...",
        "The Dungeon Master is weaving your tale...",
    )

    logger.info("Generating scenario details...")
    try:
        await game_flow.generate_and_set_details(game_id)
        logger.info("Scenario details generated")
    except Exception as e:
        logger.error(f"Error generating scenario details: {e}", exc_info=True)
        show_api_error(
            main_container,
            error=e,
            title="Error Generating Quest Details",
            message="The Dungeon Master encountered an issue while preparing your quest.",
            retry_callback=lambda: asyncio.create_task(
                handle_scenario_selection(main_container, game_id, scenario_name)
            ),
        )
        return

    if game_flow.show_dm_notes_enabled():
        logger.info("DM notes enabled, showing DM notes first")
        await show_dm_notes_before_characters(main_container, game_id, scenario_name)
    else:
        logger.info("DM notes disabled, going directly to character selection")
        await show_characters(main_container, game_id, scenario_name)

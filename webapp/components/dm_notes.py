import asyncio
import logging

from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .characters import show_characters  # type: ignore

logger = logging.getLogger()


async def show_dm_notes_before_characters(
    main_container, game_id: str, scenario_name: str
):
    logger.info(f"show_dm_notes_before_characters called for game {game_id}")
    main_container.clear()
    game_state = game_flow.get_game_state(game_id)
    if not game_state or not game_state.scenario_details:
        logger.warning(
            "No game state or scenario details, skipping to character selection"
        )
        await show_characters(main_container, game_id, scenario_name)
        return

    logger.info("Displaying DM notes")
    with main_container:
        with ui.card().classes("fantasy-panel w-full max-w-4xl"):
            ui.label("📜 Dungeon Master's Notes").classes("text-h4 mb-6")
            ui.markdown(game_state.scenario_details).classes(
                "markdown w-full text-left mb-4"
            )
            ui.button(
                "⚔️ Continue to Character Selection",
                on_click=lambda: asyncio.create_task(
                    show_characters(main_container, game_id, scenario_name)
                ),
            ).classes("mt-6 w-full")

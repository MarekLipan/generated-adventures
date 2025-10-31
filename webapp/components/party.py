import asyncio
import logging

from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .adventure import start_adventure  # type: ignore
from .character_display import render_character_cards

logger = logging.getLogger()


def show_character_overview(main_container, game_id: str):
    """Display the selected party and provide a button to start the adventure."""
    logger.info(f"show_character_overview called for game {game_id}")
    main_container.clear()
    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        logger.error("No game state found in show_character_overview")
        return

    logger.info(f"Displaying party with {len(game_state.characters)} characters")
    with main_container:
        with ui.card().classes("fantasy-panel w-full max-w-4xl"):
            ui.label("⚔️ Your Party is Ready!").classes("text-h4 mb-6")

            # Render character cards
            render_character_cards(game_state.characters, game_id)

            ui.button(
                "🎲 Begin the Adventure!",
                on_click=lambda _e=None: asyncio.create_task(
                    start_adventure(main_container, game_id)
                ),
            ).classes("mt-6 text-lg w-full")

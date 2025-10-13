import asyncio
import logging

from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .adventure import start_adventure  # type: ignore

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
        ui.label("Your Party is Ready!").classes("text-h4 mb-4")

        for character in game_state.characters:
            with ui.card().classes("w-full mb-4"):
                with ui.row().classes("w-full gap-4"):
                    # Left: Image
                    if character.image_path:
                        ui.image(character.image_path).classes(
                            "w-32 h-40 object-cover rounded"
                        )

                    # Right: Info
                    with ui.column().classes("flex-1"):
                        ui.label(character.name).classes("text-h6 font-bold")

                        # Stats
                        with ui.row().classes("gap-4 my-2"):
                            ui.label(f"💪 {character.strength}").classes("text-sm")
                            ui.label(f"🧠 {character.intelligence}").classes("text-sm")
                            ui.label(f"⚡ {character.agility}").classes("text-sm")
                            ui.label(
                                f"❤️ {character.current_health}/{character.maximum_health}"
                            ).classes("text-sm")

                        # Collapsible sections
                        with ui.expansion("Details", icon="info").classes("w-full"):
                            ui.label("Appearance:").classes("font-bold text-xs mt-2")
                            ui.label(character.appearance).classes("text-xs mb-2")

                            ui.label("Backstory:").classes("font-bold text-xs")
                            ui.label(character.backstory).classes("text-xs mb-2")

                            ui.label("Inventory:").classes("font-bold text-xs")
                            for item in character.inventory:
                                ui.label(f"• {item}").classes("text-xs ml-2")

        ui.button(
            "Start Adventure!",
            on_click=lambda _e=None: asyncio.create_task(
                start_adventure(main_container, game_id)
            ),
        ).classes("mt-4 text-lg")

import asyncio
import logging
from typing import List

from nicegui import ui

from core.models import Character
from webapp.services import game_flow  # type: ignore

from .party import show_character_overview  # type: ignore

logger = logging.getLogger()


async def show_characters(
    main_container,
    game_id: str,
    scenario_name: str,
    cached_characters: List[Character] | None = None,
):
    logger.info(f"show_characters called for game {game_id}, scenario: {scenario_name}")
    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        logger.error(f"Game state not found for game {game_id}")
        return

    # Only generate characters if not cached (first time)
    if cached_characters is None:
        try:
            logger.info("Starting character generation with scenario context...")
            all_characters: List[Character] = await game_flow.generate_characters(
                scenario_name,
                num_characters=6,
                scenario_details=game_state.scenario_details,
            )  # type: ignore
            logger.info(
                f"Character generation complete: {len(all_characters)} characters"
            )
        except Exception as e:
            logger.error(f"Error generating characters: {e}", exc_info=True)
            main_container.clear()
            with main_container:
                ui.label("Error generating characters").classes("text-h4 text-red")
                ui.label(str(e)).classes("text-sm")
            return
    else:
        logger.info(f"Using cached characters: {len(cached_characters)} available")
        all_characters = cached_characters

    selected = {c.name for c in game_state.characters}
    available = [c for c in all_characters if c.name not in selected]

    logger.info(f"Displaying {len(available)} available characters for selection")
    main_container.clear()
    with main_container:
        ui.label(
            f"Choose Character ({len(game_state.characters)}/{game_state.players})"
        ).classes("text-h4")
        with ui.grid(columns=2).classes("w-full gap-6 mt-4"):
            for character in available:
                with ui.card().classes("w-full max-w-2xl"):
                    with ui.row().classes("w-full gap-4"):
                        # Left column: Image
                        if character.image_path:
                            ui.image(character.image_path).classes(
                                "w-48 h-64 object-cover rounded"
                            )
                        else:
                            ui.label("No Image").classes(
                                "w-48 h-64 flex items-center justify-center bg-gray-200 rounded"
                            )

                        # Right column: Character details
                        with ui.column().classes("flex-1"):
                            ui.label(character.name).classes("text-h6 font-bold mb-2")

                            # Stats row
                            with ui.row().classes("gap-4 mb-3"):
                                ui.label(f"💪 STR: {character.strength}").classes(
                                    "text-sm font-semibold"
                                )
                                ui.label(f"🧠 INT: {character.intelligence}").classes(
                                    "text-sm font-semibold"
                                )
                                ui.label(f"⚡ AGI: {character.agility}").classes(
                                    "text-sm font-semibold"
                                )
                                ui.label(f"❤️ HP: {character.current_health}").classes(
                                    "text-sm font-semibold"
                                )

                            # Appearance
                            with ui.expansion("Appearance", icon="visibility").classes(
                                "w-full mb-2"
                            ):
                                ui.label(character.appearance).classes("text-sm")

                            # Backstory
                            with ui.expansion("Backstory", icon="book").classes(
                                "w-full mb-2"
                            ):
                                ui.label(character.backstory).classes("text-sm")

                            # Inventory
                            with ui.expansion("Inventory", icon="backpack").classes(
                                "w-full mb-2"
                            ):
                                for item in character.inventory:
                                    ui.label(f"• {item}").classes("text-sm ml-2")

                            # Select button
                            ui.button(
                                "Select This Character",
                                on_click=lambda _e=None,
                                c=character: asyncio.create_task(
                                    select_character(
                                        main_container,
                                        game_id,
                                        scenario_name,
                                        c,
                                        all_characters,
                                    )
                                ),
                            ).classes("mt-3 w-full")


async def select_character(
    main_container,
    game_id: str,
    scenario_name: str,
    character: Character,
    all_characters: List[Character],
):
    logger.info(f"Character selected: {character.name}")
    game_flow.add_character(game_id, character)
    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        logger.error("Game state not found after adding character")
        return

    logger.info(
        f"Characters selected: {len(game_state.characters)}/{game_state.players}"
    )
    if len(game_state.characters) < game_state.players:
        logger.info(
            "More characters needed, showing character selection again with cached characters"
        )
        await show_characters(
            main_container, game_id, scenario_name, cached_characters=all_characters
        )
    else:
        logger.info("All characters selected, showing party overview")
        show_character_overview(main_container, game_id)

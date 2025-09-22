from nicegui import ui
import asyncio
from webapp.services import game_flow  # type: ignore
from .characters import show_characters  # type: ignore


async def show_dm_notes_before_characters(
    main_container, game_id: str, scenario_name: str
):
    main_container.clear()
    game_state = game_flow.get_game_state(game_id)
    if not game_state or not game_state.scenario_details:
        await show_characters(main_container, game_id, scenario_name)
        return

    with main_container:
        ui.markdown(game_state.scenario_details).classes("w-full text-left")
        ui.button(
            "Continue to Character Selection",
            on_click=lambda: asyncio.create_task(
                show_characters(main_container, game_id, scenario_name)
            ),
        ).classes("mt-4")

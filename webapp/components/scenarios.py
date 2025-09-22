from nicegui import ui
import asyncio
from webapp.services import game_flow  # type: ignore
from .dm_notes import show_dm_notes_before_characters  # type: ignore
from .characters import show_characters  # type: ignore


async def show_scenarios(main_container, game_id: str):
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
    game_flow.select_scenario(game_id, scenario_name)
    await game_flow.generate_and_set_details(game_id)
    if game_flow.show_dm_notes_enabled():
        await show_dm_notes_before_characters(main_container, game_id, scenario_name)
    else:
        await show_characters(main_container, game_id, scenario_name)

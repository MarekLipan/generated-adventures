import asyncio
import logging

from nicegui import ui

from core.config import settings  # type: ignore
from webapp.services import game_flow  # type: ignore
from webapp.utils import show_api_error, show_loading  # type: ignore

from .hero_creation import start_hero_creation  # type: ignore

logger = logging.getLogger()


async def show_scenarios(main_container, game_id: str):
    """Display the scenario pool with one-liners and option to generate new scenarios."""
    # Load existing scenario pool
    show_loading(
        main_container,
        "🗺️ Loading Adventures...",
        "Retrieving available quests...",
    )

    try:
        scenario_pool = await game_flow.get_scenario_pool()
    except Exception as e:
        show_api_error(
            main_container,
            error=e,
            title="Error Loading Scenarios",
            message="Could not load the scenario pool.",
            retry_callback=lambda: asyncio.create_task(
                show_scenarios(main_container, game_id)
            ),
        )
        return

    main_container.clear()
    with main_container:
        with ui.card().classes("fantasy-panel w-full"):
            with ui.row().classes("w-full items-center justify-between mb-6"):
                ui.label("🗺️ Choose Your Adventure").classes("text-h4")
                with ui.row().classes("items-center gap-3"):
                    theme_options = game_flow.scenario_theme_options()
                    default_theme = (
                        settings.SCENARIO_DEFAULT_THEME
                        if settings.SCENARIO_DEFAULT_THEME in theme_options
                        else "any"
                    )
                    scale_options = game_flow.scenario_scale_options()
                    default_scale = (
                        settings.SCENARIO_DEFAULT_SCALE
                        if settings.SCENARIO_DEFAULT_SCALE in scale_options
                        else "any"
                    )
                    genre_select = (
                        ui.select(theme_options, value=default_theme, label="Genre")
                        .props("outlined dense options-dense")
                        .classes("w-48")
                    )
                    scale_select = (
                        ui.select(scale_options, value=default_scale, label="Stakes")
                        .props("outlined dense options-dense")
                        .classes("w-48")
                    )
                    ui.button(
                        "✨ Generate New Scenario",
                        on_click=lambda: asyncio.create_task(
                            generate_and_add_new_scenario(
                                main_container,
                                game_id,
                                genre_select.value,
                                scale_select.value,
                            )
                        ),
                    ).classes("fantasy-accent-gold")

            if not scenario_pool:
                ui.label(
                    "No scenarios available yet. Generate your first adventure!"
                ).classes("text-lg fantasy-text-muted mb-4")
            else:
                with ui.column().classes("gap-4 w-full"):
                    for scenario in scenario_pool:
                        with ui.card().classes(
                            "scenario-card w-full hover:shadow-lg transition-shadow"
                        ):
                            with ui.row().classes("items-center gap-2 mb-2"):
                                ui.label(scenario.name).classes(
                                    "text-h6 font-bold fantasy-text-gold"
                                )
                                genre = getattr(scenario, "genre", "") or ""
                                if genre:
                                    ui.label(genre).classes("genre-badge")
                                scale_key = getattr(scenario, "scale", "") or ""
                                if scale_key and scale_key != "any":
                                    ui.label(
                                        scale_options.get(scale_key, scale_key)
                                    ).classes("scale-badge")
                            ui.label(scenario.one_liner).classes("text-base mb-3")

                            with ui.row().classes("items-center gap-4"):
                                if scenario.times_played > 0:
                                    ui.label(
                                        f"🎲 Played {scenario.times_played} time{'s' if scenario.times_played > 1 else ''}"
                                    ).classes("text-sm fantasy-text-muted")
                                ui.button(
                                    "⚔️ Select This Adventure",
                                    on_click=lambda _event=None,
                                    s=scenario: asyncio.create_task(
                                        handle_scenario_selection(
                                            main_container, game_id, s.id
                                        )
                                    ),
                                ).classes("ml-auto")


async def generate_and_add_new_scenario(
    main_container, game_id: str, theme: str = "any", scale: str = "any"
):
    """Generate a new scenario (steered to a genre + stakes/scale) and add it to the pool."""
    show_loading(
        main_container,
        "✨ Generating New Scenario...",
        "The Dungeon Master is crafting a unique adventure...",
    )

    try:
        new_scenario = await game_flow.generate_new_scenario(theme=theme, scale=scale)
        logger.info(
            f"New scenario generated (theme={theme}, scale={scale}): {new_scenario.name}"
        )

        # Refresh the scenario list (which will show the new scenario)
        await show_scenarios(main_container, game_id)

    except Exception as e:
        logger.error(f"Error generating new scenario: {e}", exc_info=True)
        show_api_error(
            main_container,
            error=e,
            title="Error Generating Scenario",
            message="The Dungeon Master encountered an issue while creating a new adventure.",
            retry_callback=lambda: asyncio.create_task(
                generate_and_add_new_scenario(main_container, game_id, theme, scale)
            ),
        )


async def handle_scenario_selection(main_container, game_id: str, scenario_id: str):
    """Handle scenario selection and proceed to character generation."""
    logger.info(f"Scenario selected: {scenario_id} for game {game_id}")
    game_flow.select_scenario(game_id, scenario_id)

    # Get scenario details
    scenario = game_flow.get_scenario_for_game(game_id)
    if not scenario:
        logger.error(f"Could not load scenario {scenario_id}")
        ui.notify("Error loading scenario", type="negative")
        return

    # Proceed to art-style choice and photo-based hero creation
    logger.info("Proceeding to hero creation")
    await start_hero_creation(main_container, game_id, scenario.name)

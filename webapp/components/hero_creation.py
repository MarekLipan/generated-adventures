"""Hero-creation flow: art style -> per-player photo + archetype -> generated hero.

Replaces the old "generate 6 full characters, pick N" flow. Instead:
  1. The player picks a visual art style for the whole game (once).
  2. For each player, we show cheap scenario-tailored archetype cards. The player
     optionally uploads a profile photo, then picks an archetype.
  3. Only then do we spend a portrait render + a lore call to build that one hero,
     whose face resembles the uploaded photo (when provided).
  4. The player can rename or regenerate before confirming.
"""

import asyncio
import logging
import pathlib

from nicegui import ui

from webapp.services import game_flow  # type: ignore
from webapp.utils import show_api_error, show_loading  # type: ignore

from .character_display import render_character_cards
from .party import show_character_overview  # type: ignore

logger = logging.getLogger()

# Art styles offered at the start of a scenario. Keys must match core.models.ArtStyle.
ART_STYLE_OPTIONS = [
    (
        "painterly_hero",
        "🎨 Painterly Hero",
        "Bold, warm, hand-painted fantasy — like premium card art. Punchy and readable. (Default)",
    ),
    (
        "artstation_realism",
        "🖼️ ArtStation Realism",
        "Cinematic, semi-realistic and detailed. The strongest likeness when using your photo.",
    ),
]


async def start_hero_creation(main_container, game_id: str, scenario_name: str):
    """Entry point: choose the art style, then build the party."""
    main_container.clear()
    with main_container:
        with ui.card().classes("fantasy-panel w-full max-w-3xl"):
            ui.label("🎨 Choose Your Art Style").classes("text-h4 mb-2")
            ui.label(
                "This sets the look of every portrait and scene in your adventure. "
                "If you upload a photo later, your hero will resemble you in this style."
            ).classes("text-sm fantasy-text-muted mb-6")

            with ui.column().classes("gap-4 w-full"):
                for style_key, title, desc in ART_STYLE_OPTIONS:
                    with ui.card().classes(
                        "scenario-card w-full hover:shadow-lg transition-shadow"
                    ):
                        ui.label(title).classes(
                            "text-h6 font-bold fantasy-text-gold mb-1"
                        )
                        ui.label(desc).classes("text-sm mb-3")
                        ui.button(
                            "Use This Style",
                            on_click=lambda _e=None, s=style_key: asyncio.create_task(
                                _on_style_chosen(
                                    main_container, game_id, scenario_name, s
                                )
                            ),
                        ).classes("w-full")


async def _on_style_chosen(
    main_container, game_id: str, scenario_name: str, art_style: str
):
    """Persist the chosen style, generate archetypes, then start the party loop."""
    game_flow.set_art_style(game_id, art_style)

    show_loading(
        main_container,
        "🗺️ Preparing Heroes...",
        "The Dungeon Master is dreaming up heroes suited to this adventure...",
    )

    try:
        scenario = game_flow.get_scenario_for_game(game_id)
        scenario_details = scenario.dm_notes if scenario else None
        archetypes = await game_flow.generate_archetypes(
            scenario_name=scenario_name,
            scenario_details=scenario_details,
        )
    except Exception as e:
        logger.error(f"Error generating archetypes: {e}", exc_info=True)
        show_api_error(
            main_container,
            error=e,
            title="Error Preparing Heroes",
            message="Could not generate hero archetypes for this scenario.",
            retry_callback=lambda: asyncio.create_task(
                _on_style_chosen(main_container, game_id, scenario_name, art_style)
            ),
        )
        return

    await create_hero_for_player(
        main_container,
        game_id,
        scenario_name,
        archetypes,
        player_index=1,
        chosen_archetypes=set(),
    )


async def create_hero_for_player(
    main_container,
    game_id: str,
    scenario_name: str,
    archetypes,
    player_index: int,
    chosen_archetypes: set,
):
    """Show archetype cards + optional photo upload for one player."""
    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        logger.error("Game state not found in create_hero_for_player")
        return

    total_players = game_state.players
    # Photo bytes uploaded for this player (mutable holder shared with the handler).
    photo_holder = {"bytes": None, "suffix": ".jpg", "name": None}

    available = [a for a in archetypes if a.name not in chosen_archetypes]

    main_container.clear()
    with main_container:
        with ui.card().classes("fantasy-panel w-full"):
            ui.label(
                f"⚔️ Create Your Hero — Player {player_index} of {total_players}"
            ).classes("text-h4 mb-2")
            ui.label(
                "Optionally upload a profile photo so your hero looks like you, "
                "then choose an archetype that fits the adventure."
            ).classes("text-sm fantasy-text-muted mb-4")

            # --- Photo upload ---
            with ui.card().classes("w-full mb-6 p-4"):
                ui.label("📸 Your Photo (optional)").classes(
                    "text-h6 fantasy-text-gold mb-2"
                )
                ui.label(
                    "A clear, front-facing photo works best. Skip this to get a "
                    "freshly-imagined hero face instead."
                ).classes("text-xs fantasy-text-muted mb-3")

                status_label = ui.label("No photo uploaded — your hero's face will be invented.").classes(
                    "text-xs fantasy-text-muted mb-2"
                )

                def _on_upload(e):
                    data = e.content.read()
                    photo_holder["bytes"] = data
                    photo_holder["name"] = e.name
                    suffix = pathlib.Path(e.name or "").suffix.lower()
                    photo_holder["suffix"] = suffix if suffix else ".jpg"
                    status_label.set_text(
                        f"✓ Photo ready: {e.name} — your hero will resemble you."
                    )
                    status_label.classes(
                        replace="text-xs fantasy-text-gold mb-2"
                    )

                def _on_remove(_e=None):
                    photo_holder["bytes"] = None
                    photo_holder["name"] = None
                    status_label.set_text(
                        "No photo uploaded — your hero's face will be invented."
                    )
                    status_label.classes(replace="text-xs fantasy-text-muted mb-2")

                ui.upload(
                    on_upload=_on_upload,
                    on_rejected=lambda: ui.notify(
                        "Please upload an image file.", type="warning"
                    ),
                    auto_upload=True,
                    max_files=1,
                ).props('accept="image/*" flat').classes("w-full")

            # --- Archetype cards ---
            ui.label("🛡️ Choose an Archetype").classes(
                "text-h5 fantasy-text-gold mb-3"
            )
            with ui.grid(columns=2).classes("w-full gap-4"):
                for archetype in available:
                    with ui.card().classes("character-card w-full"):
                        ui.label(archetype.name).classes(
                            "text-h6 font-bold fantasy-text-gold"
                        )
                        ui.label(archetype.role).classes(
                            "text-xs stat-label mb-2 uppercase"
                        )
                        ui.label(archetype.hook).classes("text-sm mb-2")
                        with ui.expansion("Concept", icon="visibility").classes(
                            "w-full mb-2"
                        ):
                            ui.label(archetype.concept).classes("text-xs")
                        ui.button(
                            "⚔️ Choose This Archetype",
                            on_click=lambda _e=None, a=archetype: asyncio.create_task(
                                _do_generate_hero(
                                    main_container,
                                    game_id,
                                    scenario_name,
                                    archetypes,
                                    player_index,
                                    chosen_archetypes,
                                    a,
                                    photo_holder,
                                )
                            ),
                        ).classes("w-full mt-2")


async def _do_generate_hero(
    main_container,
    game_id: str,
    scenario_name: str,
    archetypes,
    player_index: int,
    chosen_archetypes: set,
    archetype,
    photo_holder: dict,
    custom_name: str | None = None,
):
    """Generate one hero from a chosen archetype and show a confirmation preview."""
    show_loading(
        main_container,
        "✨ Forging Your Hero...",
        f"Bringing '{archetype.name}' to life in your chosen style...",
    )

    # Persist the uploaded photo (once) so it can be used as a generation reference.
    photo_path = None
    if photo_holder.get("bytes"):
        try:
            photo_path = game_flow.save_player_photo(
                game_id,
                player_index,
                photo_holder["bytes"],
                photo_holder.get("suffix", ".jpg"),
            )
        except Exception as e:
            logger.error(f"Failed to save player photo: {e}", exc_info=True)
            photo_path = None

    game_state = game_flow.get_game_state(game_id)
    art_style = game_state.art_style if game_state else "painterly_hero"
    scenario = game_flow.get_scenario_for_game(game_id)
    scenario_details = scenario.dm_notes if scenario else None

    try:
        hero = await game_flow.generate_hero(
            game_id=game_id,
            scenario_name=scenario_name,
            archetype=archetype,
            art_style=art_style,
            player_index=player_index,
            scenario_details=scenario_details,
            photo_path=photo_path,
            custom_name=custom_name,
        )
    except Exception as e:
        logger.error(f"Error generating hero: {e}", exc_info=True)
        show_api_error(
            main_container,
            error=e,
            title="Error Forging Hero",
            message="The Dungeon Master stumbled while creating your hero.",
            retry_callback=lambda: asyncio.create_task(
                _do_generate_hero(
                    main_container,
                    game_id,
                    scenario_name,
                    archetypes,
                    player_index,
                    chosen_archetypes,
                    archetype,
                    photo_holder,
                    custom_name,
                )
            ),
        )
        return

    _show_hero_preview(
        main_container,
        game_id,
        scenario_name,
        archetypes,
        player_index,
        chosen_archetypes,
        archetype,
        photo_holder,
        hero,
    )


def _show_hero_preview(
    main_container,
    game_id: str,
    scenario_name: str,
    archetypes,
    player_index: int,
    chosen_archetypes: set,
    archetype,
    photo_holder: dict,
    hero,
):
    """Preview a generated hero with rename / regenerate / confirm controls."""
    game_state = game_flow.get_game_state(game_id)
    total_players = game_state.players if game_state else player_index

    main_container.clear()
    with main_container:
        with ui.card().classes("fantasy-panel w-full max-w-4xl"):
            ui.label(
                f"✨ Meet Your Hero — Player {player_index} of {total_players}"
            ).classes("text-h4 mb-2")
            ui.label(
                f"Archetype: {archetype.name}"
            ).classes("text-sm fantasy-text-muted mb-4")

            render_character_cards([hero], game_id)

            # Rename
            name_input = ui.input(
                label="Hero Name", value=hero.name
            ).classes("w-full max-w-md mt-4")

            with ui.row().classes("gap-3 mt-4 w-full"):
                ui.button(
                    "✅ Confirm This Hero",
                    on_click=lambda _e=None: asyncio.create_task(
                        _confirm_hero(
                            main_container,
                            game_id,
                            scenario_name,
                            archetypes,
                            player_index,
                            chosen_archetypes,
                            archetype,
                            hero,
                            name_input.value,
                        )
                    ),
                ).classes("text-lg")
                ui.button(
                    "🔄 Regenerate",
                    on_click=lambda _e=None: asyncio.create_task(
                        _do_generate_hero(
                            main_container,
                            game_id,
                            scenario_name,
                            archetypes,
                            player_index,
                            chosen_archetypes,
                            archetype,
                            photo_holder,
                            name_input.value or None,
                        )
                    ),
                ).props("outline"),
                ui.button(
                    "↩ Pick a Different Archetype",
                    on_click=lambda _e=None: asyncio.create_task(
                        create_hero_for_player(
                            main_container,
                            game_id,
                            scenario_name,
                            archetypes,
                            player_index,
                            chosen_archetypes,
                        )
                    ),
                ).props("flat")


async def _confirm_hero(
    main_container,
    game_id: str,
    scenario_name: str,
    archetypes,
    player_index: int,
    chosen_archetypes: set,
    archetype,
    hero,
    final_name: str,
):
    """Lock in a hero and move to the next player or the party overview."""
    if final_name and final_name.strip():
        hero.name = final_name.strip()

    game_flow.add_character(game_id, hero)
    chosen_archetypes.add(archetype.name)

    game_state = game_flow.get_game_state(game_id)
    if not game_state:
        logger.error("Game state not found after confirming hero")
        return

    if len(game_state.characters) < game_state.players:
        await create_hero_for_player(
            main_container,
            game_id,
            scenario_name,
            archetypes,
            player_index + 1,
            chosen_archetypes,
        )
    else:
        show_character_overview(main_container, game_id)

"""Recap display component for showing story summary and scene gallery."""

import logging

from nicegui import ui

from core import generator, models  # type: ignore
from webapp.services import game_flow  # type: ignore

logger = logging.getLogger(__name__)


async def show_recap_dialog(game_state: models.Game, current_scene_index: int):
    """Display a recap dialog with story summary and scene gallery.

    Args:
        game_state: The current game state
        current_scene_index: Index of the current scene (0-based)
    """
    current_scene = game_state.scenes[current_scene_index]

    # Get scenario name
    scenario = game_flow.get_scenario_for_game(game_state.id)
    scenario_name = scenario.name if scenario else "Unknown Quest"

    # Check if recap is already cached in memory (session only, not persisted)
    # Note: We can only cache recap_text in the Scene model, not voiceover path
    recap_text = current_scene.recap_text if current_scene.recap_text else None
    voiceover_path = None  # Will be generated fresh or remain None

    if not recap_text:
        # Generate new recap (but don't save to disk - viewing recap shouldn't modify game state)
        with ui.dialog() as loading_dialog, ui.card().classes("fantasy-panel"):
            ui.label("📖 Generating Recap").classes("text-h5 mb-4")
            ui.spinner(size="lg")
            ui.label("Summarizing your adventure...").classes("loading-message mt-4")

        loading_dialog.open()

        try:
            # Generate recap with scene summaries
            recap_response = await generator.generate_scene_recap(
                game_id=game_state.id,
                scenario_name=scenario_name,
                scenes=game_state.scenes,
                characters=game_state.characters,
                current_scene_index=current_scene_index,
            )

            # Cache recap text in memory for this session only (no game state persistence)
            recap_text = recap_response.recap_text
            current_scene.recap_text = recap_text

            # Store scene summaries for use in gallery (local variable, not persisted)
            scene_summaries_dict = {
                summary.scene_id: summary.summary
                for summary in recap_response.scene_summaries
            }

            # Generate recap voiceover via the configured TTS backend (Kokoro local
            # by default). File is saved temporarily but not tracked in game state.
            voiceover_file_path = await generator.generate_recap_voiceover(
                game_state.id, current_scene.id, recap_text
            )

            # Convert to web path (keep in local variable, don't store in scene model)
            if voiceover_file_path:
                voiceover_path = (
                    f"/static/voiceovers/{game_state.id}/{voiceover_file_path.name}"
                )

        except Exception as e:
            loading_dialog.close()
            with (
                ui.dialog() as error_dialog,
                ui.card().classes("fantasy-panel border-2 border-red-500"),
            ):
                ui.label("⚠️ Error Generating Recap").classes(
                    "text-h5 text-red-400 mb-4"
                )
                ui.label(f"Error: {str(e)}").classes(
                    "text-sm text-red-300 mb-4 font-mono"
                )
                ui.button("Close", on_click=error_dialog.close).classes("mt-2")
            error_dialog.open()
            return
        finally:
            loading_dialog.close()
    else:
        # Recap was cached, but we don't have scene summaries
        # Use fallback: empty dict means we'll show text preview instead
        scene_summaries_dict = {}

    # Create the recap dialog
    with (
        ui.dialog() as recap_dialog,
        ui.card().classes(
            "fantasy-panel w-full max-w-5xl max-h-[90vh] overflow-y-auto"
        ),
    ):
        # Header
        with ui.row().classes("w-full justify-between items-center mb-4"):
            ui.label("📖 Story Recap").classes("text-h4 fantasy-text-gold")
            ui.button(icon="close", on_click=recap_dialog.close).props(
                "flat round"
            ).classes("text-white")

        ui.html('<div class="ornate-divider"></div>')

        # Scenario title
        ui.label(scenario_name).classes("text-h5 text-center mb-4 text-yellow-300")

        # Voiceover player (if available)
        if voiceover_path:
            with ui.card().classes("bg-slate-800/50 p-4 mb-4"):
                ui.label("🎧 Listen to the Recap").classes(
                    "text-sm font-bold mb-2 text-yellow-300"
                )
                ui.audio(voiceover_path).props("controls").classes("w-full")

        # Recap narrative
        with ui.card().classes("bg-slate-800/50 p-4 mb-6"):
            ui.markdown(recap_text).classes("text-lg leading-relaxed")

        ui.html('<div class="ornate-divider"></div>')

        # Scene gallery
        ui.label("🎭 Your Journey So Far").classes("text-h6 mb-4 fantasy-text-gold")

        with ui.column().classes("w-full gap-4"):
            for i, scene in enumerate(game_state.scenes[: current_scene_index + 1]):
                with ui.card().classes("bg-slate-800/30 p-4"):
                    # Scene header
                    ui.label(f"Scene {scene.id}").classes(
                        "text-h6 fantasy-text-gold mb-2"
                    )

                    # Scene image
                    if scene.image_path:
                        ui.image(scene.image_path).classes(
                            "w-full max-w-2xl rounded-lg shadow-lg mb-3"
                        )

                    # Scene summary (if available) or fallback to text preview
                    if scene_summaries_dict and scene.id in scene_summaries_dict:
                        summary_text = scene_summaries_dict[scene.id]
                        ui.label(summary_text).classes(
                            "text-sm text-yellow-200 font-semibold"
                        )
                    else:
                        # Fallback: brief scene text preview (first 200 chars)
                        preview_text = (
                            scene.text[:200] + "..."
                            if len(scene.text) > 200
                            else scene.text
                        )
                        ui.label(preview_text).classes("text-sm text-gray-300 italic")

        # Footer
        ui.html('<div class="ornate-divider mt-6"></div>')
        with ui.row().classes("w-full justify-center mt-4"):
            ui.button("Continue Adventure →", on_click=recap_dialog.close).classes(
                "text-lg"
            )

    recap_dialog.open()

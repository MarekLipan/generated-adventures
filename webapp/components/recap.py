"""Recap display component for showing story summary and scene gallery."""

from nicegui import ui

from core import generator, models, persistence  # type: ignore
from webapp.services import game_flow  # type: ignore
from webapp.utils import show_api_error, show_loading  # type: ignore


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
    
    # Check if recap is already cached
    if current_scene.recap_text:
        recap_text = current_scene.recap_text
    else:
        # Generate new recap
        loading_dialog = show_loading("Generating recap of your adventure...")
        try:
            recap_text = await generator.generate_scene_recap(
                game_id=game_state.id,
                scenario_name=scenario_name,
                scenes=game_state.scenes,
                characters=game_state.characters,
                current_scene_index=current_scene_index,
            )
            
            # Cache the recap
            current_scene.recap_text = recap_text
            persistence.save_game(game_state)
            
        except Exception as e:
            loading_dialog.close()
            show_api_error(e)
            return
        finally:
            loading_dialog.close()
    
    # Create the recap dialog
    with ui.dialog() as recap_dialog, ui.card().classes(
        "fantasy-panel w-full max-w-5xl max-h-[90vh] overflow-y-auto"
    ):
        # Header
        with ui.row().classes("w-full justify-between items-center mb-4"):
            ui.label("📖 Story Recap").classes("text-h4 fantasy-text-gold")
            ui.button(
                icon="close",
                on_click=recap_dialog.close
            ).props("flat round").classes("text-white")
        
        ui.html('<div class="ornate-divider"></div>')
        
        # Scenario title
        ui.label(scenario_name).classes("text-h5 text-center mb-4 text-yellow-300")
        
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
                    
                    # Brief scene text (first 200 chars)
                    preview_text = scene.text[:200] + "..." if len(scene.text) > 200 else scene.text
                    ui.label(preview_text).classes("text-sm text-gray-300 italic")
        
        # Footer
        ui.html('<div class="ornate-divider mt-6"></div>')
        with ui.row().classes("w-full justify-center mt-4"):
            ui.button(
                "Continue Adventure →",
                on_click=recap_dialog.close
            ).classes("text-lg")
    
    recap_dialog.open()

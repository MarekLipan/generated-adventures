from nicegui import ui
from webapp.services import game_flow  # type: ignore


async def start_adventure(main_container, game_id: str):
    def render_scene(scene):
        main_container.clear()
        with main_container:
            ui.label(f"Scene {scene.id}").classes("text-h5")
            ui.markdown(scene.text).classes("w-full text-left")
            action_input = ui.input(
                label="Your action", placeholder="What does your party do?"
            )

            def on_submit(_e=None):
                player_action = action_input.value
                next_scene = game_flow.advance_scene(game_id, player_action)
                if next_scene:
                    render_scene(next_scene)
                else:
                    ui.label("No further scenes.")

            ui.button("Submit Action", on_click=on_submit).classes("mt-4")

    current = game_flow.get_current_scene(game_id)
    if not current:
        await game_flow.generate_and_set_details(game_id)
        current = game_flow.get_current_scene(game_id)
    if current:
        render_scene(current)
    else:
        main_container.clear()
        ui.label("Unable to start the adventure: no scene available.")

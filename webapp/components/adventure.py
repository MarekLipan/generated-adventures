from nicegui import ui

from webapp.services import game_flow  # type: ignore

from .character_display import render_character_cards  # type: ignore


async def start_adventure(main_container, game_id: str):
    def render_scene(scene):
        main_container.clear()
        with main_container:
            # Scene content
            ui.label(f"Scene {scene.id}").classes("text-h5 mb-4")
            ui.markdown(scene.text).classes("w-full text-left mb-6")

            # Action input - dynamically rendered based on prompt type
            ui.separator().classes("my-4")

            # Get prompt information
            prompt = scene.prompt
            if not prompt:
                # Fallback if no prompt data
                prompt_type = "action"
                prompt_text = "What do you do?"
                target_character = None
            else:
                prompt_type = prompt.type
                prompt_text = prompt.prompt_text
                target_character = prompt.target_character

            # Display the prompt
            target_label = f"{target_character}" if target_character else "Party"
            ui.label(f"{target_label}: {prompt_text}").classes(
                "text-lg font-semibold mb-2"
            )

            # Render appropriate input based on prompt type
            action_input = None

            if prompt_type == "dice_check":
                # Dice roll input - number only
                dice_display = ""
                if prompt and prompt.dice_count and prompt.dice_type:
                    dice_display = (
                        f"{prompt.dice_count}{prompt.dice_type}"
                        if prompt.dice_count > 1
                        else prompt.dice_type
                    )
                else:
                    dice_display = "dice"

                ui.label(f"🎲 Roll {dice_display} and enter the result:").classes(
                    "text-sm text-gray-600 mb-2"
                )
                action_input = ui.number(
                    label="Dice Result",
                    placeholder="Enter the sum of your dice roll",
                    min=1,
                ).classes("w-full")

            elif prompt_type == "dialogue":
                # Dialogue input - will be wrapped in quotes
                ui.label(
                    "💬 Enter what your character says (quotes will be added automatically):"
                ).classes("text-sm text-gray-600 mb-2")
                action_input = ui.input(
                    label="Dialogue", placeholder="What do you say?"
                ).classes("w-full")

            else:  # action
                # Standard action input
                ui.label("⚔️ Describe your action:").classes(
                    "text-sm text-gray-600 mb-2"
                )
                action_input = ui.input(
                    label="Action", placeholder="What do you do?"
                ).classes("w-full")

            async def on_submit(_e=None):
                player_input = action_input.value

                # Format the input based on type
                if prompt_type == "dialogue" and player_input:
                    player_action = f'"{player_input}"'
                elif prompt_type == "dice_check" and player_input is not None:
                    player_action = str(int(player_input))
                else:
                    player_action = player_input

                next_scene = await game_flow.advance_scene(game_id, player_action)
                if next_scene:
                    render_scene(next_scene)
                else:
                    ui.label("No further scenes.")

            ui.button("Submit", on_click=on_submit).classes("mt-4 mb-6")

            # Party Status section - after action input
            game_state = game_flow.get_game_state(game_id)
            if game_state and game_state.characters:
                ui.separator().classes("my-6")
                ui.label("Party Status").classes("text-h6 mb-3")

                # Render character cards with expandable sections
                render_character_cards(game_state.characters)

    current = game_flow.get_current_scene(game_id)
    if not current:
        await game_flow.generate_opening_scene(game_id)
        current = game_flow.get_current_scene(game_id)
    if current:
        render_scene(current)
    else:
        main_container.clear()
        ui.label("Unable to start the adventure: no scene available.")

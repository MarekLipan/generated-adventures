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

            # Display scene image if available
            if scene.image_path:
                ui.image(scene.image_path).classes(
                    "w-full max-w-3xl rounded-lg shadow-lg mb-6"
                )

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
            character_dice_inputs = {}  # For multi-character dice checks

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

                # Check if this is a multi-character dice check
                has_multiple_targets = (
                    prompt
                    and hasattr(prompt, "target_characters")
                    and prompt.target_characters
                    and len(prompt.target_characters) > 1
                )

                if has_multiple_targets:
                    # Multiple character dice inputs
                    ui.label(f"🎲 Each character rolls {dice_display}:").classes(
                        "text-sm text-gray-600 mb-2"
                    )
                    for char_name in prompt.target_characters:
                        character_dice_inputs[char_name] = ui.number(
                            label=f"{char_name}'s Roll",
                            placeholder=f"Enter {char_name}'s dice result",
                            min=1,
                        ).classes("w-full mb-2")
                else:
                    # Single dice input
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
                # Handle multi-character dice checks
                if prompt_type == "dice_check" and character_dice_inputs:
                    # Format: "Korgath: 7, Sister Anya: 4"
                    rolls = []
                    for char_name, input_field in character_dice_inputs.items():
                        if input_field.value is not None:
                            rolls.append(f"{char_name}: {int(input_field.value)}")
                    player_action = ", ".join(rolls) if rolls else ""
                else:
                    player_input = action_input.value

                    # Format the input based on type
                    if prompt_type == "dialogue" and player_input:
                        player_action = f'"{player_input}"'
                    elif prompt_type == "dice_check" and player_input is not None:
                        player_action = str(int(player_input))
                    else:
                        player_action = player_input

                # Show loading indicator
                main_container.clear()
                with main_container:
                    ui.label("Generating next scene...").classes("text-h5 mb-4")
                    ui.spinner(size="lg")
                    ui.label("🎲 The Dungeon Master is crafting your story...").classes(
                        "text-gray-600 mt-4"
                    )

                next_scene = await game_flow.advance_scene(game_id, player_action)
                if next_scene:
                    render_scene(next_scene)
                else:
                    main_container.clear()
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
        # Show loading indicator for opening scene generation
        main_container.clear()
        with main_container:
            ui.label("Starting your adventure...").classes("text-h5 mb-4")
            ui.spinner(size="lg")
            ui.label("🎲 The Dungeon Master is preparing the opening scene...").classes(
                "text-gray-600 mt-4"
            )

        await game_flow.generate_opening_scene(game_id)
        current = game_flow.get_current_scene(game_id)
    if current:
        render_scene(current)
    else:
        main_container.clear()
        ui.label("Unable to start the adventure: no scene available.")

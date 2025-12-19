import asyncio

from nicegui import ui

from webapp.services import game_flow  # type: ignore
from webapp.utils import show_api_error, show_loading  # type: ignore

from .character_changes import render_character_changes  # type: ignore
from .character_display import render_character_cards  # type: ignore
from .recap import show_recap_dialog  # type: ignore


def render_scene_navigation(game_state, scene_index, render_scene_callback):
    """Render navigation buttons for moving between scenes.

    Args:
        game_state: The game state containing all scenes
        scene_index: Current scene index being viewed
        render_scene_callback: Function to call when navigating to a different scene
    """
    if game_state and len(game_state.scenes) > 1:
        ui.html('<div class="ornate-divider"></div>')
        with ui.row().classes("w-full max-w-4xl justify-between items-center mb-4"):
            # Previous scene button
            if scene_index > 0:
                ui.button(
                    "◀ Previous Scene",
                    on_click=lambda idx=scene_index - 1: render_scene_callback(
                        game_state.scenes[idx], idx
                    ),
                )
            else:
                # Invisible placeholder to maintain spacing
                ui.label("").classes("w-32")

            # Center column: Scene indicator and Recap button
            with ui.column().classes("items-center gap-2"):
                ui.label(
                    f"Scene {scene_index + 1} of {len(game_state.scenes)}"
                ).classes("text-center fantasy-text-gold")
                ui.button(
                    "📖 Recap",
                    on_click=lambda: show_recap_dialog(game_state, scene_index),
                ).classes("text-sm")

            # Next scene button (if not on the last scene)
            if scene_index < len(game_state.scenes) - 1:
                ui.button(
                    "Next Scene ▶",
                    on_click=lambda idx=scene_index + 1: render_scene_callback(
                        game_state.scenes[idx], idx
                    ),
                )
            else:
                # Invisible placeholder to maintain spacing
                ui.label("").classes("w-32")


def render_game_completed(
    main_container, scene, game_state, scene_index, render_scene_callback
):
    """Render the victory screen when the quest is completed."""
    main_container.clear()

    # Get scenario name from template
    scenario = game_flow.get_scenario_for_game(game_state.id)
    scenario_name = scenario.name if scenario else "Unknown Quest"

    with main_container:
        # Victory banner
        with ui.card().classes(
            "fantasy-panel w-full max-w-4xl border-4 border-yellow-500 shadow-2xl"
        ):
            # Scenario title
            ui.label(f"📜 {scenario_name}").classes(
                "text-h4 mb-2 fantasy-text-gold text-center"
            )
            ui.html('<div class="ornate-divider"></div>')

            # Victory title
            ui.label("🏆 QUEST COMPLETED! 🏆").classes(
                "text-h3 mb-4 text-yellow-400 text-center font-bold"
            )

            # Final scene narrative
            ui.label(f"⚔️ Scene {scene.id} - The Conclusion").classes(
                "text-h5 mb-4 fantasy-text-gold"
            )
            ui.markdown(scene.text).classes("markdown w-full text-left mb-6")

            # Voice-over if available
            if scene.voiceover_path:
                ui.html('<div class="ornate-divider"></div>')
                with ui.row().classes(
                    "items-center gap-3 mb-4 p-3 rounded-lg bg-gray-800/50"
                ):
                    ui.icon("volume_up", size="lg").classes("text-yellow-500")
                    ui.label("Final Narration:").classes(
                        "text-sm fantasy-text-gold font-semibold"
                    )
                    ui.audio(scene.voiceover_path).classes("flex-grow")
                ui.html('<div class="ornate-divider"></div>')

        # Display scene image if available
        if scene.image_path:
            ui.image(scene.image_path).classes(
                "w-full max-w-3xl rounded-lg shadow-lg mb-6"
            )

        # Scene navigation buttons
        render_scene_navigation(game_state, scene_index, render_scene_callback)

        # Victory message
        with ui.card().classes("fantasy-panel w-full max-w-4xl bg-yellow-900/20"):
            ui.label("✨ The Adventure Concludes in Glory! ✨").classes(
                "text-h5 text-center text-yellow-300 mb-4"
            )
            ui.label(
                "Your party has successfully completed their quest. Their names will be remembered in legend!"
            ).classes("text-center text-lg mb-4")

            ui.button(
                "🏠 Return to Main Menu", on_click=lambda: ui.navigate.to("/")
            ).classes("mx-auto")

        # Final party status
        if game_state and game_state.characters:
            ui.html('<div class="ornate-divider"></div>')
            ui.label("👥 Victorious Heroes").classes(
                "text-h6 mb-4 fantasy-text-gold text-center"
            )
            render_character_cards(game_state.characters, game_state.id)


def render_game_failed(
    main_container, scene, game_state, scene_index, render_scene_callback
):
    """Render the game over screen when the party is defeated."""
    main_container.clear()

    # Get scenario name from template
    scenario = game_flow.get_scenario_for_game(game_state.id)
    scenario_name = scenario.name if scenario else "Unknown Quest"

    with main_container:
        # Defeat banner
        with ui.card().classes(
            "fantasy-panel w-full max-w-4xl border-4 border-red-500 shadow-2xl"
        ):
            # Scenario title
            ui.label(f"📜 {scenario_name}").classes(
                "text-h4 mb-2 fantasy-text-gold text-center"
            )
            ui.html('<div class="ornate-divider"></div>')

            # Game over title
            ui.label("💀 GAME OVER 💀").classes(
                "text-h3 mb-4 text-red-400 text-center font-bold"
            )

            # Final scene narrative
            ui.label(f"⚔️ Scene {scene.id} - The End").classes(
                "text-h5 mb-4 fantasy-text-gold"
            )
            ui.markdown(scene.text).classes("markdown w-full text-left mb-6")

            # Voice-over if available
            if scene.voiceover_path:
                ui.html('<div class="ornate-divider"></div>')
                with ui.row().classes(
                    "items-center gap-3 mb-4 p-3 rounded-lg bg-gray-800/50"
                ):
                    ui.icon("volume_up", size="lg").classes("text-yellow-500")
                    ui.label("Final Narration:").classes(
                        "text-sm fantasy-text-gold font-semibold"
                    )
                    ui.audio(scene.voiceover_path).classes("flex-grow")
                ui.html('<div class="ornate-divider"></div>')

        # Display scene image if available
        if scene.image_path:
            ui.image(scene.image_path).classes(
                "w-full max-w-3xl rounded-lg shadow-lg mb-6"
            )

        # Scene navigation buttons
        render_scene_navigation(game_state, scene_index, render_scene_callback)

        # Defeat message
        with ui.card().classes("fantasy-panel w-full max-w-4xl bg-red-900/20"):
            ui.label("⚰️ The Party Has Fallen ⚰️").classes(
                "text-h5 text-center text-red-300 mb-4"
            )
            ui.label(
                "The adventure has ended in tragedy. Perhaps another band of heroes will take up the quest..."
            ).classes("text-center text-lg mb-4")

            ui.button(
                "🏠 Return to Main Menu", on_click=lambda: ui.navigate.to("/")
            ).classes("mx-auto")

        # Final party status (fallen heroes)
        if game_state and game_state.characters:
            ui.html('<div class="ornate-divider"></div>')
            ui.label("💀 Fallen Heroes").classes(
                "text-h6 mb-4 text-red-400 text-center"
            )
            render_character_cards(game_state.characters, game_state.id)


async def start_adventure(main_container, game_id: str):
    # State for scene navigation
    viewing_scene_index = {"value": None}  # None means show current/latest scene

    def render_scene(scene, scene_index=None):
        """Render a scene. If scene_index is provided, it's a historical scene being reviewed."""
        main_container.clear()

        # Get game state for scenario name and all scenes
        game_state = game_flow.get_game_state(game_id)

        # Determine if this is a historical scene or the current scene
        is_current_scene = (
            scene_index is None or scene_index == len(game_state.scenes) - 1
        )

        # If scene_index wasn't provided, calculate it
        if scene_index is None:
            scene_index = len(game_state.scenes) - 1

        # Update viewing state
        viewing_scene_index["value"] = scene_index

        # Check if game has ended (victory or defeat)
        game_status = getattr(scene, "game_status", "ongoing")
        if game_status == "completed":
            render_game_completed(
                main_container, scene, game_state, scene_index, render_scene
            )
            return
        elif game_status == "failed":
            render_game_failed(
                main_container, scene, game_state, scene_index, render_scene
            )
            return

        # Continue with normal scene rendering for ongoing games
        with main_container:
            with ui.card().classes("fantasy-panel w-full max-w-4xl"):
                # Scenario title at the top
                scenario = game_flow.get_scenario_for_game(game_id)
                scenario_name = scenario.name if scenario else "Unknown Quest"
                ui.label(f"📜 {scenario_name}").classes(
                    "text-h4 mb-2 fantasy-text-gold text-center"
                )
                ui.html('<div class="ornate-divider"></div>')

                # Scene content
                ui.label(f"⚔️ Scene {scene.id}").classes(
                    "text-h5 mb-4 fantasy-text-gold"
                )

                ui.markdown(scene.text).classes("markdown w-full text-left mb-6")

                # Voice-over audio player if available
                if scene.voiceover_path:
                    ui.html('<div class="ornate-divider"></div>')
                    with ui.row().classes(
                        "items-center gap-3 mb-4 p-3 rounded-lg bg-gray-800/50"
                    ):
                        ui.icon("volume_up", size="lg").classes("text-yellow-500")
                        ui.label("Scene Narration:").classes(
                            "text-sm fantasy-text-gold font-semibold"
                        )
                        ui.audio(scene.voiceover_path).classes("flex-grow")
                    ui.html('<div class="ornate-divider"></div>')

            # Display scene image if available
            if scene.image_path:
                ui.image(scene.image_path).classes(
                    "w-full max-w-3xl rounded-lg shadow-lg mb-6"
                )

            # Display character changes from this scene
            render_character_changes(scene)

            # Scene navigation buttons
            render_scene_navigation(game_state, scene_index, render_scene)

            # Action input or player response display
            ui.html('<div class="ornate-divider"></div>')

            # For historical scenes, show what the player did
            if not is_current_scene:
                # Get prompt information
                prompt = scene.prompt
                if prompt:
                    prompt_type = prompt.type
                    prompt_text = prompt.prompt_text
                    target_character = prompt.target_character
                    target_label = (
                        f"{target_character}" if target_character else "Party"
                    )

                    # Display the original prompt
                    ui.label(f"🎯 {target_label}: {prompt_text}").classes(
                        "text-lg font-semibold mb-2 fantasy-text-gold"
                    )

                # Display the player's action
                if scene_index < len(game_state.player_actions):
                    player_response = game_state.player_actions[scene_index]
                    with ui.card().classes("bg-gray-800/50 p-4 w-full max-w-4xl"):
                        ui.label("Player's Action:").classes(
                            "text-sm text-gray-400 mb-2"
                        )
                        ui.label(player_response).classes("text-lg text-white")

                # Skip the input rendering for historical scenes
            else:
                # Current scene - show interactive prompt
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
                ui.label(f"🎯 {target_label}: {prompt_text}").classes(
                    "text-lg font-semibold mb-4 fantasy-text-gold"
                )

                # Render appropriate input based on prompt type
                action_input = None
                character_dice_inputs = {}  # For multi-character dice checks

                if prompt_type == "dice_check":
                    # Dice roll input - single die only
                    dice_display = (
                        prompt.dice_type if prompt and prompt.dice_type else "dice"
                    )

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
                            "text-sm fantasy-text-muted mb-3 stat-label"
                        )
                        for char_name in prompt.target_characters:
                            character_dice_inputs[char_name] = ui.number(
                                label=f"{char_name}'s Roll",
                                placeholder=f"Enter {char_name}'s dice result",
                                min=1,
                            ).classes("w-full mb-2")
                    else:
                        # Single dice input
                        ui.label(
                            f"🎲 Roll {dice_display} and enter the result:"
                        ).classes("text-sm fantasy-text-muted mb-3 stat-label")
                        action_input = ui.number(
                            label="Dice Result",
                            placeholder="Enter the sum of your dice roll",
                            min=1,
                        ).classes("w-full")

                elif prompt_type == "dialogue":
                    # Dialogue input - will be wrapped in quotes
                    ui.label(
                        "💬 Enter what your character says (quotes will be added automatically):"
                    ).classes("text-sm fantasy-text-muted mb-3 stat-label")
                    action_input = ui.input(
                        label="Dialogue", placeholder="What do you say?"
                    ).classes("w-full")

                else:  # action
                    # Check if this is a multi-character action
                    has_multiple_targets = (
                        prompt
                        and hasattr(prompt, "target_characters")
                        and prompt.target_characters
                        and len(prompt.target_characters) > 1
                    )

                    if has_multiple_targets:
                        # Multiple character action inputs
                        ui.label("⚔️ Each character describes their action:").classes(
                            "text-sm fantasy-text-muted mb-3 stat-label"
                        )
                        for char_name in prompt.target_characters:
                            character_dice_inputs[char_name] = ui.input(
                                label=f"{char_name}'s Action",
                                placeholder=f"What does {char_name} do?",
                            ).classes("w-full mb-2")
                    else:
                        # Standard single action input
                        ui.label("⚔️ Describe your action:").classes(
                            "text-sm fantasy-text-muted mb-3 stat-label"
                        )
                        action_input = ui.input(
                            label="Action", placeholder="What do you do?"
                        ).classes("w-full")

                async def on_submit(_e=None):
                    # Handle multi-character prompts (dice checks or actions)
                    if character_dice_inputs:
                        # Format: "CharacterName1: value, CharacterName2: value"
                        responses = []
                        for char_name, input_field in character_dice_inputs.items():
                            if input_field.value is not None:
                                # For dice checks, convert to int; for actions, keep as string
                                value = (
                                    int(input_field.value)
                                    if prompt_type == "dice_check"
                                    else input_field.value
                                )
                                responses.append(f"{char_name}: {value}")
                        player_action = ", ".join(responses) if responses else ""
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
                    show_loading(
                        main_container,
                        "🎲 Generating Next Scene...",
                        "The Dungeon Master is crafting your story...",
                    )

                    try:
                        next_scene = await game_flow.advance_scene(
                            game_id, player_action
                        )
                        if next_scene:
                            render_scene(next_scene)
                        else:
                            main_container.clear()
                            with ui.card().classes("fantasy-panel"):
                                ui.label("📜 The Tale Concludes").classes("text-h5")
                                ui.label("No further scenes.").classes(
                                    "loading-message mt-2"
                                )
                    except Exception as e:
                        show_api_error(
                            main_container,
                            error=e,
                            title="Error Generating Scene",
                            message="The Dungeon Master encountered an issue while crafting the story.",
                            retry_callback=lambda: render_scene(game_state.scenes[-1]),
                        )

                ui.button("⚔️ Submit", on_click=on_submit).classes("mt-4 mb-6 w-full")

            # Party Status section - after action input
            game_state = game_flow.get_game_state(game_id)
            if game_state and game_state.characters:
                ui.html('<div class="ornate-divider"></div>')
                ui.label("👥 Party Status").classes("text-h6 mb-4 fantasy-text-gold")

                # Render character cards with expandable sections
                render_character_cards(game_state.characters, game_id)

    current = game_flow.get_current_scene(game_id)
    if not current:
        # Show loading indicator for opening scene generation
        show_loading(
            main_container,
            "🎲 Starting Your Adventure...",
            "The Dungeon Master is preparing the opening scene...",
        )

        try:
            await game_flow.generate_opening_scene(game_id)
            current = game_flow.get_current_scene(game_id)
        except Exception as e:
            show_api_error(
                main_container,
                error=e,
                title="Error Starting Adventure",
                message="The Dungeon Master encountered an issue while preparing the opening scene.",
                retry_callback=lambda: asyncio.create_task(
                    start_adventure(main_container, game_id)
                ),
            )
            return

    if current:
        render_scene(current)
    else:
        main_container.clear()
        with main_container:
            with ui.card().classes("fantasy-panel"):
                ui.label("⚠️ Unable to Start").classes("text-h5")
                ui.label("Unable to start the adventure: no scene available.").classes(
                    "loading-message mt-2"
                )

"""Reusable character card display utilities."""

from nicegui import ui


def render_character_cards(characters):
    """Render character cards with images and expandable sections.

    Args:
        characters: List of Character objects to display
    """
    for character in characters:
        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("w-full gap-4"):
                # Left: Image
                if character.image_path:
                    ui.image(character.image_path).classes(
                        "w-32 h-40 object-cover rounded"
                    )

                # Right: Info
                with ui.column().classes("flex-1"):
                    ui.label(character.name).classes("text-h6 font-bold")

                    # Stats
                    with ui.row().classes("gap-4 my-2"):
                        ui.label(f"💪 {character.strength}").classes("text-sm")
                        ui.label(f"🧠 {character.intelligence}").classes("text-sm")
                        ui.label(f"⚡ {character.agility}").classes("text-sm")
                        ui.label(
                            f"❤️ {character.current_health}/{character.maximum_health}"
                        ).classes("text-sm")

                    # Collapsible sections
                    with ui.expansion("Details", icon="info").classes("w-full"):
                        ui.label("Appearance:").classes("font-bold text-xs mt-2")
                        ui.label(character.appearance).classes("text-xs mb-2")

                        ui.label("Backstory:").classes("font-bold text-xs")
                        ui.label(character.backstory).classes("text-xs mb-2")

                        ui.label("Inventory:").classes("font-bold text-xs")
                        for item in character.inventory:
                            ui.label(f"• {item}").classes("text-xs ml-2")

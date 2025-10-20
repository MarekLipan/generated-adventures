"""Reusable character card display utilities."""

from nicegui import ui


def render_character_cards(characters):
    """Render character cards with images and expandable sections.

    Args:
        characters: List of Character objects to display
    """
    for character in characters:
        with ui.card().classes("character-card w-full mb-4"):
            with ui.row().classes("w-full gap-4"):
                # Left: Image
                if character.image_path:
                    ui.image(character.image_path).classes(
                        "w-32 h-40 object-cover rounded"
                    )

                # Right: Info
                with ui.column().classes("flex-1"):
                    ui.label(character.name).classes(
                        "text-h6 font-bold fantasy-text-gold"
                    )

                    # Stats
                    with ui.row().classes("gap-4 my-2"):
                        ui.label(f"💪 {character.strength}").classes(
                            "text-sm stat-label"
                        )
                        ui.label(f"🧠 {character.intelligence}").classes(
                            "text-sm stat-label"
                        )
                        ui.label(f"⚡ {character.agility}").classes(
                            "text-sm stat-label"
                        )

                        # Health with color coding
                        health_class = "text-sm stat-label"
                        if character.current_health <= character.maximum_health * 0.3:
                            health_class += " health-critical"
                        elif character.current_health >= character.maximum_health * 0.8:
                            health_class += " health-good"

                        ui.label(
                            f"❤️ {character.current_health}/{character.maximum_health}"
                        ).classes(health_class)

                    # Collapsible sections
                    with ui.expansion("📜 Details", icon="info").classes("w-full"):
                        ui.label("Appearance:").classes(
                            "font-bold text-xs mt-2 fantasy-text-gold"
                        )
                        ui.label(character.appearance).classes("text-xs mb-2")

                        ui.label("Personality:").classes(
                            "font-bold text-xs fantasy-text-gold"
                        )
                        ui.label(character.personality).classes("text-xs mb-2")

                        ui.label("Backstory:").classes(
                            "font-bold text-xs fantasy-text-gold"
                        )
                        ui.label(character.backstory).classes("text-xs mb-2")

                        ui.label("Skills:").classes(
                            "font-bold text-xs fantasy-text-gold"
                        )
                        if character.skills:
                            for skill in character.skills:
                                ui.label(f"⚡ {skill}").classes("text-xs ml-2")
                        else:
                            ui.label("None").classes("text-xs ml-2 fantasy-text-muted")

                        ui.label("Inventory:").classes(
                            "font-bold text-xs mt-2 fantasy-text-gold"
                        )
                        for item in character.inventory:
                            ui.label(f"🎒 {item}").classes("text-xs ml-2")

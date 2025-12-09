"""Component for displaying character changes from a scene."""

from nicegui import ui

from core.models import Scene


def render_character_changes(scene: Scene):
    """Display character changes that occurred in this scene.

    Args:
        scene: The Scene object containing change information
    """
    # Check if there are any changes to display
    has_changes = (
        scene.health_changes
        or scene.inventory_changes
        or scene.skill_changes
        or scene.stat_changes
    )

    if not has_changes:
        return

    ui.html('<div class="ornate-divider"></div>')

    with ui.card().classes(
        "w-full max-w-4xl bg-gray-800/30 border border-yellow-600/30"
    ):
        ui.label("📊 Character Changes").classes("text-h6 mb-3 fantasy-text-gold")

        # Health Changes
        if scene.health_changes:
            for change in scene.health_changes:
                icon = "❤️" if change.health_change > 0 else "💔"
                color_class = (
                    "text-green-400" if change.health_change > 0 else "text-red-400"
                )
                change_text = (
                    f"+{change.health_change}"
                    if change.health_change > 0
                    else str(change.health_change)
                )

                with ui.row().classes("items-center gap-2 mb-2"):
                    ui.label(icon).classes("text-lg")
                    ui.label(f"{change.character_name}:").classes("font-bold")
                    ui.label(f"{change_text} HP").classes(
                        f"{color_class} font-semibold"
                    )
                    ui.label(f"({change.reason})").classes(
                        "text-sm fantasy-text-muted italic"
                    )

        # Inventory Changes
        if scene.inventory_changes:
            for change in scene.inventory_changes:
                if change.items_added:
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("🎒").classes("text-lg")
                        ui.label(f"{change.character_name}:").classes("font-bold")
                        ui.label("gained").classes("text-green-400")
                        ui.label(", ".join(change.items_added)).classes("font-semibold")

                if change.items_removed:
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("🎒").classes("text-lg")
                        ui.label(f"{change.character_name}:").classes("font-bold")
                        ui.label("lost").classes("text-red-400")
                        ui.label(", ".join(change.items_removed)).classes(
                            "font-semibold"
                        )

        # Skill Changes
        if scene.skill_changes:
            for change in scene.skill_changes:
                if change.skills_learned:
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("⭐").classes("text-lg")
                        ui.label(f"{change.character_name}:").classes("font-bold")
                        ui.label("learned").classes("text-green-400")
                        ui.label(", ".join(change.skills_learned)).classes(
                            "font-semibold"
                        )

                if change.skills_lost:
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("⭐").classes("text-lg")
                        ui.label(f"{change.character_name}:").classes("font-bold")
                        ui.label("lost").classes("text-red-400")
                        ui.label(", ".join(change.skills_lost)).classes("font-semibold")

        # Stat Changes
        if scene.stat_changes:
            for change in scene.stat_changes:
                changes_list = []
                if change.strength_change != 0:
                    changes_list.append(f"STR {change.strength_change:+d}")
                if change.intelligence_change != 0:
                    changes_list.append(f"INT {change.intelligence_change:+d}")
                if change.agility_change != 0:
                    changes_list.append(f"AGI {change.agility_change:+d}")

                if changes_list:
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.label("💪").classes("text-lg")
                        ui.label(f"{change.character_name}:").classes("font-bold")
                        ui.label(", ".join(changes_list)).classes(
                            "text-yellow-400 font-semibold"
                        )
                        if change.reason:
                            ui.label(f"({change.reason})").classes(
                                "text-sm fantasy-text-muted italic"
                            )

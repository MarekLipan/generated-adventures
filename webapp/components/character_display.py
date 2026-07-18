"""Reusable character card display utilities."""

import html

from nicegui import ui

from webapp.utils.pdf_generator import generate_character_sheet_pdf


def _hp_state(current: int, maximum: int) -> str:
    """Class suffix for the HP bar / label based on the health ratio."""
    ratio = (current / maximum) if maximum else 0
    if ratio <= 0.3:
        return "crit"
    if ratio >= 0.8:
        return "good"
    return "mid"


def _stat_chip(icon: str, value, key: str) -> None:
    """A compact, always-visible stat pill (STR / INT / AGI)."""
    ui.html(
        f'<div class="stat-chip"><span class="stat-ico">{icon}</span>'
        f'<span class="stat-val">{html.escape(str(value))}</span>'
        f'<span class="stat-key">{html.escape(key)}</span></div>'
    )


def render_character_cards(characters, game_id: str = None):
    """Render character sheets: stats, HP and inventory always visible; the
    narrative lore (appearance / personality / backstory) tucked behind a toggle.

    Args:
        characters: List of Character objects to display
    """
    for character in characters:
        with ui.card().classes("character-card w-full mb-4"):
            with ui.row().classes("w-full gap-4 items-start no-wrap"):
                # Left: Portrait
                if character.image_path:
                    ui.image(character.image_path).classes(
                        "w-32 h-40 object-cover rounded flex-shrink-0"
                    )

                # Right: Sheet
                with ui.column().classes("flex-1 gap-2 min-w-0"):
                    ui.label(character.name).classes(
                        "text-h6 font-bold fantasy-text-gold"
                    )
                    archetype = getattr(character, "archetype", None)
                    if archetype:
                        ui.label(archetype).classes("text-xs fantasy-text-muted -mt-1")

                    # HP bar (always visible)
                    cur, mx = character.current_health, character.maximum_health
                    pct = max(0, min(100, int(100 * cur / mx))) if mx else 0
                    state = _hp_state(cur, mx)
                    ui.html(
                        f'<div class="hp-bar"><div class="hp-bar-fill {state}" '
                        f'style="width:{pct}%"></div>'
                        f'<span class="hp-bar-text">❤️ {cur} / {mx} HP</span></div>'
                    ).classes("w-full")

                    # Stat chips (always visible)
                    with ui.row().classes("gap-2 mt-1 flex-wrap"):
                        _stat_chip("💪", character.strength, "STR")
                        _stat_chip("🧠", character.intelligence, "INT")
                        _stat_chip("⚡", character.agility, "AGI")

                    # Inventory (always visible) — name + what it's for, so the
                    # player knows how/when to use each item.
                    ui.label("🎒 Inventory").classes("sheet-section-label mt-1")
                    if character.inventory:
                        with ui.column().classes("gap-1 w-full"):
                            for item in character.inventory:
                                item_name = getattr(item, "name", str(item))
                                purpose = getattr(item, "purpose", "") or ""
                                with ui.row().classes("items-baseline gap-2"):
                                    ui.html(
                                        f'<span class="inv-chip">🎒 {html.escape(item_name)}</span>'
                                    )
                                    if purpose:
                                        ui.label(purpose).classes(
                                            "text-xs fantasy-text-muted"
                                        )
                    else:
                        ui.label("Empty-handed").classes("inv-empty")

                    # Skills (always visible — they drive dice checks)
                    ui.label("⚡ Skills").classes("sheet-section-label mt-1")
                    with ui.row().classes("gap-2 flex-wrap"):
                        if character.skills:
                            for skill in character.skills:
                                ui.html(
                                    f'<span class="skill-chip">⚡ {html.escape(str(skill))}</span>'
                                )
                        else:
                            ui.label("None").classes("inv-empty")

                    # Lore behind a toggle (appearance / personality / backstory)
                    with ui.expansion("📖 Lore & Appearance", icon="auto_stories").classes(
                        "w-full mt-1"
                    ):
                        ui.label("Appearance").classes(
                            "font-bold text-xs mt-2 fantasy-text-gold"
                        )
                        ui.label(character.appearance).classes("text-xs mb-2")

                        ui.label("Personality").classes(
                            "font-bold text-xs fantasy-text-gold"
                        )
                        ui.label(character.personality).classes("text-xs mb-2")

                        ui.label("Backstory").classes(
                            "font-bold text-xs fantasy-text-gold"
                        )
                        ui.label(character.backstory).classes("text-xs mb-2")

                    # PDF Export Button
                    def create_pdf_download(char):
                        """Create a download handler for this character's PDF."""

                        def download_pdf():
                            pdf_bytes = generate_character_sheet_pdf(char, game_id)
                            # Safe filename
                            filename = (
                                f"{char.name.replace(' ', '_')}_character_sheet.pdf"
                            )
                            ui.download(pdf_bytes, filename)

                        return download_pdf

                    ui.button(
                        "📄 Export PDF", on_click=create_pdf_download(character)
                    ).classes("mt-2").props("flat color=primary size=sm")

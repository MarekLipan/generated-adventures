"""PDF generation for character sheets."""

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.models import Character


def generate_character_sheet_pdf(character: Character, game_id: str = None) -> bytes:
    """Generate a PDF character sheet for a single character.

    Args:
        character: Character object to generate sheet for
        game_id: Optional game ID for accessing character image

    Returns:
        PDF file content as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
    )
    story = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CharacterTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#8B4513"),
        spaceAfter=8,
        alignment=1,  # Center
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#654321"),
        spaceAfter=3,
        spaceBefore=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=9, spaceAfter=3, leading=11
    )

    # Title
    story.append(Paragraph(character.name, title_style))
    story.append(Spacer(1, 0.1 * inch))

    # Create a table for image and stats side by side
    # Left column: Character Image
    left_content = []
    if character.image_path:
        try:
            image_path = Path("webapp") / character.image_path.lstrip("/")
            if image_path.exists():
                img = Image(str(image_path), width=1.8 * inch, height=2.25 * inch)
                left_content.append(img)
        except Exception:
            pass

    if not left_content:
        left_content.append(Paragraph("", body_style))  # Empty placeholder

    # Right column: Stats Table
    stats_data = [
        ["Attribute", "Value"],
        ["Strength", str(character.strength)],
        ["Intelligence", str(character.intelligence)],
        ["Agility", str(character.agility)],
        ["Current Health", str(character.current_health)],
        ["Maximum Health", str(character.maximum_health)],
    ]
    stats_table = Table(stats_data, colWidths=[1.4 * inch, 1.1 * inch])
    stats_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B4513")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    # Combine image and stats in a table
    top_table = Table(
        [[left_content[0], stats_table]], colWidths=[2 * inch, 2.8 * inch]
    )
    top_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(top_table)
    story.append(Spacer(1, 0.15 * inch))

    # Two-column layout for the text sections
    # Left column: Appearance, Personality
    left_col = []
    left_col.append(Paragraph("Appearance", heading_style))
    left_col.append(Paragraph(character.appearance, body_style))
    left_col.append(Spacer(1, 0.05 * inch))
    left_col.append(Paragraph("Personality", heading_style))
    left_col.append(Paragraph(character.personality, body_style))

    # Right column: Backstory
    right_col = []
    right_col.append(Paragraph("Backstory", heading_style))
    right_col.append(Paragraph(character.backstory, body_style))

    # Create two-column table
    text_table = Table([[left_col, right_col]], colWidths=[3.5 * inch, 3.5 * inch])
    text_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 6),
                ("LEFTPADDING", (1, 0), (1, 0), 6),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ]
        )
    )
    story.append(text_table)
    story.append(Spacer(1, 0.1 * inch))

    # Skills and Inventory side by side
    skills_content = []
    skills_content.append(Paragraph("Skills", heading_style))
    if character.skills:
        skills_text = "<br/>".join([f"• {skill}" for skill in character.skills])
        skills_content.append(Paragraph(skills_text, body_style))
    else:
        skills_content.append(Paragraph("No special skills", body_style))

    inventory_content = []
    inventory_content.append(Paragraph("Inventory", heading_style))
    if character.inventory:
        inventory_text = "<br/>".join([f"• {item}" for item in character.inventory])
        inventory_content.append(Paragraph(inventory_text, body_style))
    else:
        inventory_content.append(Paragraph("No items", body_style))

    bottom_table = Table(
        [[skills_content, inventory_content]], colWidths=[3.5 * inch, 3.5 * inch]
    )
    bottom_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 6),
                ("LEFTPADDING", (1, 0), (1, 0), 6),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ]
        )
    )
    story.append(bottom_table)

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def generate_party_sheet_pdf(characters: list[Character], game_id: str = None) -> bytes:
    """Generate a PDF with all character sheets.

    Args:
        characters: List of Character objects
        game_id: Optional game ID for accessing character images

    Returns:
        PDF file content as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch)
    story = []

    for idx, character in enumerate(characters):
        # Generate individual character sheet content
        character_pdf = _build_character_content(character, game_id)
        story.extend(character_pdf)

        # Add page break between characters (except for the last one)
        if idx < len(characters) - 1:
            story.append(PageBreak())

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def _build_character_content(character: Character, game_id: str = None) -> list:
    """Build story elements for a single character (helper for party PDF).

    Args:
        character: Character object
        game_id: Optional game ID

    Returns:
        List of reportlab story elements
    """
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CharacterTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#8B4513"),
        spaceAfter=12,
        alignment=1,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#654321"),
        spaceAfter=6,
        spaceBefore=12,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, spaceAfter=6
    )

    # Title
    story.append(Paragraph(character.name, title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Character Image
    if character.image_path:
        try:
            image_path = Path("webapp") / character.image_path.lstrip("/")
            if image_path.exists():
                img = Image(str(image_path), width=2 * inch, height=2.5 * inch)
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))
        except Exception:
            pass

    # Stats Table
    stats_data = [
        ["Attribute", "Value"],
        ["Strength", str(character.strength)],
        ["Intelligence", str(character.intelligence)],
        ["Agility", str(character.agility)],
        ["Current Health", str(character.current_health)],
        ["Maximum Health", str(character.maximum_health)],
    ]
    stats_table = Table(stats_data, colWidths=[2 * inch, 1.5 * inch])
    stats_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B4513")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(stats_table)
    story.append(Spacer(1, 0.2 * inch))

    # Appearance
    story.append(Paragraph("Appearance", heading_style))
    story.append(Paragraph(character.appearance, body_style))

    # Personality
    story.append(Paragraph("Personality", heading_style))
    story.append(Paragraph(character.personality, body_style))

    # Backstory
    story.append(Paragraph("Backstory", heading_style))
    story.append(Paragraph(character.backstory, body_style))

    # Skills
    story.append(Paragraph("Skills", heading_style))
    if character.skills:
        skills_text = "<br/>".join([f"• {skill}" for skill in character.skills])
        story.append(Paragraph(skills_text, body_style))
    else:
        story.append(Paragraph("No special skills", body_style))

    # Inventory
    story.append(Paragraph("Inventory", heading_style))
    if character.inventory:
        inventory_text = "<br/>".join([f"• {item}" for item in character.inventory])
        story.append(Paragraph(inventory_text, body_style))
    else:
        story.append(Paragraph("No items", body_style))

    return story

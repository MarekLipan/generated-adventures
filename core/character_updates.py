"""Apply the stat, health and inventory changes a scene reports to the party."""

import logging

from .models import Character
from .names import names_match

logger = logging.getLogger(__name__)


def _resolve_char(name: str, char_dict: dict) -> Character | None:
    """Find the character a change refers to, tolerating name variations."""
    if name in char_dict:
        return char_dict[name]
    for cname, c in char_dict.items():
        if names_match(name, cname):
            return c
    return None


def _item_name(item) -> str:
    """Item display name, tolerant of legacy bare-string items."""
    return getattr(item, "name", str(item))


def format_inventory(items) -> str:
    """Render inventory as 'Name (purpose); ...' for prompts ('None' if empty)."""
    if not items:
        return "None"
    parts = []
    for it in items:
        purpose = getattr(it, "purpose", "") or ""
        parts.append(f"{_item_name(it)} ({purpose})" if purpose else _item_name(it))
    return "; ".join(parts)


def apply_character_updates(
    characters: list[Character], generated_scene
) -> list[Character]:
    """Apply character state updates from explicit changes in the generated scene.

    Args:
        characters: List of current character objects (with image_path, etc.)
        generated_scene: GeneratedScene object with health_changes, inventory_changes, etc.

    Returns:
        Updated list of characters with modifications applied
    """
    # Create a dictionary for quick character lookup by name
    char_dict = {char.name: char for char in characters}

    # Apply health changes
    for health_change in generated_scene.health_changes:
        char = _resolve_char(health_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Health change for unknown character: {health_change.character_name}. Skipping."
            )
            continue

        old_health = char.current_health
        char.current_health = max(
            0,
            min(char.current_health + health_change.health_change, char.maximum_health),
        )

        change_desc = f"Health: {old_health} -> {char.current_health}"
        if health_change.health_change < 0:
            change_desc += (
                f" ({health_change.health_change} damage: {health_change.reason})"
            )
        elif health_change.health_change > 0:
            change_desc += (
                f" (+{health_change.health_change} healing: {health_change.reason})"
            )

        logger.info(f"Updated {char.name}: {change_desc}")

    # Apply inventory changes
    for inv_change in generated_scene.inventory_changes:
        char = _resolve_char(inv_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Inventory change for unknown character: {inv_change.character_name}. Skipping."
            )
            continue

        changes = []
        have = {_item_name(it).strip().lower() for it in char.inventory}

        # Add new items (structured), skipping ones already carried (by name).
        for item in inv_change.items_added:
            key = _item_name(item).strip().lower()
            if key and key not in have:
                char.inventory.append(item)
                have.add(key)
                changes.append(f"+{_item_name(item)}")

        # Remove items by name, tolerating minor naming variation
        # ("lockpicks" vs "Clockwork Lockpicks").
        for name in inv_change.items_removed:
            key = str(name).strip().lower()
            if not key:
                continue
            match = next(
                (
                    it
                    for it in char.inventory
                    if (n := _item_name(it).strip().lower()) == key
                    or key in n
                    or n in key
                ),
                None,
            )
            if match is not None:
                char.inventory.remove(match)
                changes.append(f"-{_item_name(match)}")

        if changes:
            logger.info(f"Updated {char.name} inventory: {', '.join(changes)}")

    # Apply skill changes
    for skill_change in generated_scene.skill_changes:
        char = _resolve_char(skill_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Skill change for unknown character: {skill_change.character_name}. Skipping."
            )
            continue

        changes = []

        # Learn new skills
        for skill in skill_change.skills_learned:
            if skill not in char.skills:
                char.skills.append(skill)
                changes.append(f"Learned: {skill}")

        # Lose skills
        for skill in skill_change.skills_lost:
            if skill in char.skills:
                char.skills.remove(skill)
                changes.append(f"Lost: {skill}")

        if changes:
            logger.info(f"Updated {char.name} skills: {', '.join(changes)}")

    # Apply stat changes (rare)
    for stat_change in generated_scene.stat_changes:
        char = _resolve_char(stat_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Stat change for unknown character: {stat_change.character_name}. Skipping."
            )
            continue

        changes = []

        if stat_change.strength_change != 0:
            old_str = char.strength
            char.strength = max(0, min(20, char.strength + stat_change.strength_change))
            changes.append(f"STR: {old_str} -> {char.strength}")

        if stat_change.intelligence_change != 0:
            old_int = char.intelligence
            char.intelligence = max(
                0, min(20, char.intelligence + stat_change.intelligence_change)
            )
            changes.append(f"INT: {old_int} -> {char.intelligence}")

        if stat_change.agility_change != 0:
            old_agi = char.agility
            char.agility = max(0, min(20, char.agility + stat_change.agility_change))
            changes.append(f"AGI: {old_agi} -> {char.agility}")

        if changes:
            logger.info(
                f"Updated {char.name} stats: {', '.join(changes)} (Reason: {stat_change.reason})"
            )

    return characters

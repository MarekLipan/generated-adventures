"""Test the new change-based character update system."""

from core.generator import _apply_character_updates
from core.models import (
    Character,
    CharacterHealthChange,
    CharacterInventoryChange,
    GeneratedScene,
    PromptType,
)


def test_health_change_damage():
    """Test that damage is applied correctly as a negative change."""
    # Setup character with 100 health
    char = Character(
        name="Theron",
        strength=15,
        intelligence=12,
        agility=14,
        maximum_health=100,
        current_health=100,
        backstory="Test backstory",
        appearance="Test appearance",
        personality="Test personality",
        skills=["Combat"],
        inventory=["Sword"],
    )

    # Create a scene with 15 damage
    scene = GeneratedScene(
        scene_text="You take 15 damage from a goblin attack.",
        visual_description="Test visual description",
        prompt=PromptType(
            type="action",
            target_character=None,
            prompt_text="What do you do?",
        ),
        health_changes=[
            CharacterHealthChange(
                character_name="Theron",
                health_change=-15,
                reason="goblin attack",
            )
        ],
        inventory_changes=[],
        skill_changes=[],
        stat_changes=[],
        assets_present=[],
        game_status="ongoing",
    )

    # Apply changes
    updated = _apply_character_updates([char], scene)

    # Verify health decreased by 15
    assert updated[0].current_health == 85
    assert updated[0].maximum_health == 100


def test_health_change_healing():
    """Test that healing is applied correctly as a positive change."""
    # Setup character with 60 health
    char = Character(
        name="Elara",
        strength=10,
        intelligence=18,
        agility=12,
        maximum_health=100,
        current_health=60,
        backstory="Test backstory",
        appearance="Test appearance",
        personality="Test personality",
        skills=["Healing"],
        inventory=["Staff"],
    )

    # Create a scene with 25 healing
    scene = GeneratedScene(
        scene_text="You drink a healing potion and recover 25 health.",
        visual_description="Test visual description",
        prompt=PromptType(
            type="action",
            target_character=None,
            prompt_text="What do you do next?",
        ),
        health_changes=[
            CharacterHealthChange(
                character_name="Elara",
                health_change=25,
                reason="healing potion",
            )
        ],
        inventory_changes=[],
        skill_changes=[],
        stat_changes=[],
        assets_present=[],
        game_status="ongoing",
    )

    # Apply changes
    updated = _apply_character_updates([char], scene)

    # Verify health increased by 25
    assert updated[0].current_health == 85


def test_health_change_caps_at_maximum():
    """Test that healing doesn't exceed maximum health."""
    # Setup character with 90 health
    char = Character(
        name="Gareth",
        strength=16,
        intelligence=10,
        agility=13,
        maximum_health=100,
        current_health=90,
        backstory="Test backstory",
        appearance="Test appearance",
        personality="Test personality",
        skills=["Defense"],
        inventory=["Shield"],
    )

    # Create a scene with 30 healing (would exceed max)
    scene = GeneratedScene(
        scene_text="You are fully healed by a powerful spell.",
        visual_description="Test visual description",
        prompt=PromptType(
            type="action",
            target_character=None,
            prompt_text="Continue?",
        ),
        health_changes=[
            CharacterHealthChange(
                character_name="Gareth",
                health_change=30,
                reason="powerful healing spell",
            )
        ],
        inventory_changes=[],
        skill_changes=[],
        stat_changes=[],
        assets_present=[],
        game_status="ongoing",
    )

    # Apply changes
    updated = _apply_character_updates([char], scene)

    # Verify health capped at maximum
    assert updated[0].current_health == 100
    assert updated[0].maximum_health == 100


def test_inventory_changes():
    """Test that inventory additions and removals work correctly."""
    char = Character(
        name="Lyra",
        strength=11,
        intelligence=14,
        agility=16,
        maximum_health=100,
        current_health=100,
        backstory="Test backstory",
        appearance="Test appearance",
        personality="Test personality",
        skills=["Stealth"],
        inventory=["Dagger", "Rope", "Torch"],
    )

    # Create a scene where character gains and loses items
    scene = GeneratedScene(
        scene_text="You use your rope and find a magic amulet.",
        visual_description="Test visual description",
        prompt=PromptType(
            type="action",
            target_character=None,
            prompt_text="What next?",
        ),
        health_changes=[],
        inventory_changes=[
            CharacterInventoryChange(
                character_name="Lyra",
                items_added=["Magic Amulet"],
                items_removed=["Rope"],
            )
        ],
        skill_changes=[],
        stat_changes=[],
        assets_present=[],
        game_status="ongoing",
    )

    # Apply changes
    updated = _apply_character_updates([char], scene)

    # Verify inventory updated correctly
    assert "Magic Amulet" in updated[0].inventory
    assert "Rope" not in updated[0].inventory
    assert "Dagger" in updated[0].inventory  # unchanged items remain
    assert "Torch" in updated[0].inventory


def test_no_changes():
    """Test that characters remain unchanged when no changes are specified."""
    char = Character(
        name="Marcus",
        strength=17,
        intelligence=11,
        agility=12,
        maximum_health=100,
        current_health=75,
        backstory="Test backstory",
        appearance="Test appearance",
        personality="Test personality",
        skills=["Intimidation"],
        inventory=["Axe", "Armor"],
    )

    # Create a scene with no changes
    scene = GeneratedScene(
        scene_text="You observe the situation carefully.",
        visual_description="Test visual description",
        prompt=PromptType(
            type="dialogue",
            target_character="Marcus",
            prompt_text="What do you say?",
        ),
        health_changes=[],
        inventory_changes=[],
        skill_changes=[],
        stat_changes=[],
        assets_present=[],
        game_status="ongoing",
    )

    # Apply changes
    updated = _apply_character_updates([char], scene)

    # Verify nothing changed
    assert updated[0].current_health == 75
    assert updated[0].inventory == ["Axe", "Armor"]
    assert updated[0].skills == ["Intimidation"]
    assert updated[0].strength == 17


def test_multiple_character_changes():
    """Test that changes can be applied to multiple characters in one scene."""
    char1 = Character(
        name="Alice",
        strength=14,
        intelligence=13,
        agility=15,
        maximum_health=100,
        current_health=100,
        backstory="Test",
        appearance="Test",
        personality="Test",
        skills=["Combat"],
        inventory=["Sword"],
    )

    char2 = Character(
        name="Bob",
        strength=16,
        intelligence=10,
        agility=11,
        maximum_health=100,
        current_health=80,
        backstory="Test",
        appearance="Test",
        personality="Test",
        skills=["Defense"],
        inventory=["Shield"],
    )

    # Scene where Alice takes damage and Bob heals
    scene = GeneratedScene(
        scene_text="Alice gets hit while Bob drinks a potion.",
        visual_description="Test visual description",
        prompt=PromptType(
            type="action",
            target_character=None,
            prompt_text="Continue?",
        ),
        health_changes=[
            CharacterHealthChange(
                character_name="Alice",
                health_change=-20,
                reason="enemy attack",
            ),
            CharacterHealthChange(
                character_name="Bob",
                health_change=15,
                reason="healing potion",
            ),
        ],
        inventory_changes=[
            CharacterInventoryChange(
                character_name="Bob",
                items_added=[],
                items_removed=["Healing Potion"],
            )
        ],
        skill_changes=[],
        stat_changes=[],
        assets_present=[],
        game_status="ongoing",
    )

    # Apply changes
    updated = _apply_character_updates([char1, char2], scene)

    # Verify both characters updated correctly
    alice = next(c for c in updated if c.name == "Alice")
    bob = next(c for c in updated if c.name == "Bob")

    assert alice.current_health == 80  # 100 - 20
    assert bob.current_health == 95  # 80 + 15


if __name__ == "__main__":
    print("Running change-based update tests...")
    test_health_change_damage()
    print("✓ Damage test passed")
    test_health_change_healing()
    print("✓ Healing test passed")
    test_health_change_caps_at_maximum()
    print("✓ Health cap test passed")
    test_inventory_changes()
    print("✓ Inventory test passed")
    test_no_changes()
    print("✓ No changes test passed")
    test_multiple_character_changes()
    print("✓ Multiple character changes test passed")
    print("\n✅ All tests passed!")

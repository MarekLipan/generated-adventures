"""Tests for the Pydantic model validators."""

import pytest
from pydantic import ValidationError

from core.models import PromptType


def test_dice_check_requires_target():
    """Test that dice_check prompts must have either target_character or target_characters."""
    # Should raise error when both are None
    with pytest.raises(ValidationError) as exc_info:
        PromptType(
            type="dice_check",
            dice_type="d10",
            dice_count=1,
            target_character=None,
            target_characters=None,
            prompt_text="Roll for persuasion",
        )

    assert "must specify either 'target_character'" in str(exc_info.value)


def test_dice_check_single_target_valid():
    """Test that dice_check with single target_character is valid."""
    prompt = PromptType(
        type="dice_check",
        dice_type="d10",
        dice_count=1,
        target_character="Kaelen",
        target_characters=None,
        prompt_text="Kaelen, roll d10 for persuasion",
    )
    assert prompt.target_character == "Kaelen"
    assert prompt.target_characters is None


def test_dice_check_multi_target_valid():
    """Test that dice_check with target_characters array is valid."""
    prompt = PromptType(
        type="dice_check",
        dice_type="d6",
        dice_count=1,
        target_character=None,
        target_characters=["Kaelen", "Sera"],
        prompt_text="Kaelen and Sera, both roll d6",
    )
    assert prompt.target_character is None
    assert prompt.target_characters == ["Kaelen", "Sera"]


def test_dice_check_cannot_have_both_targets():
    """Test that dice_check cannot have both target_character and target_characters."""
    with pytest.raises(ValidationError) as exc_info:
        PromptType(
            type="dice_check",
            dice_type="d6",
            dice_count=1,
            target_character="Kaelen",
            target_characters=["Sera", "Theron"],
            prompt_text="Roll for combat",
        )

    assert "cannot specify both" in str(exc_info.value)


def test_dialogue_allows_null_target():
    """Test that dialogue prompts can have null target (entire party)."""
    prompt = PromptType(
        type="dialogue",
        target_character=None,
        target_characters=None,
        prompt_text="What do you say to the merchant?",
    )
    assert prompt.target_character is None
    assert prompt.target_characters is None


def test_action_allows_null_target():
    """Test that action prompts can have null target (entire party)."""
    prompt = PromptType(
        type="action",
        target_character=None,
        target_characters=None,
        prompt_text="What does the party do?",
    )
    assert prompt.target_character is None
    assert prompt.target_characters is None

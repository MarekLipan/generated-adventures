"""Pydantic models for the game state."""

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class Character(BaseModel):
    """Represents a single character in the game."""

    name: str
    strength: int = Field(
        ..., description="Character's strength attribute", ge=0, le=20
    )
    intelligence: int = Field(
        ..., description="Character's intelligence attribute", ge=0, le=20
    )
    agility: int = Field(..., description="Character's agility attribute", ge=0, le=20)
    maximum_health: int = Field(..., description="Character's maximum health", ge=1)
    current_health: int = Field(..., description="Character's current health", ge=0)
    backstory: str = Field(
        ..., description="Character's background story and motivations"
    )
    appearance: str = Field(..., description="Physical description of the character")
    inventory: List[str] = Field(
        default_factory=list, description="List of items the character carries"
    )
    image_path: Optional[str] = Field(
        None, description="Path to the character's generated portrait"
    )


class PromptType(BaseModel):
    """Represents a prompt type for player interaction."""

    type: str = Field(
        ...,
        description="Type of prompt: 'dialogue', 'action', 'dice_check'",
    )
    dice_type: Optional[str] = Field(
        None,
        description="Type of dice to roll if type is 'dice_check': 'd6' or 'd10'",
    )
    dice_count: Optional[int] = Field(
        None,
        description="Number of dice to roll if type is 'dice_check'",
    )
    target_character: Optional[str] = Field(
        None,
        description="Name of specific character being prompted, or None for entire party",
    )
    prompt_text: str = Field(
        ...,
        description="The actual prompt/question to present to the player(s)",
    )


class Scene(BaseModel):
    """Represents a scene in the game, which can be expanded later."""

    id: int = Field(
        ..., description="Unique integer ID for the scene, ordered by creation"
    )
    text: str = Field(
        ..., description="Text content of the scene, can be markdown formatted"
    )
    prompt: Optional[PromptType] = Field(
        None, description="Structured prompt data for player interaction"
    )
    image_path: Optional[str] = Field(
        None, description="Path to an image representing the scene"
    )
    voiceover_path: Optional[str] = Field(
        None, description="Path to a voiceover file for the scene"
    )


class Game(BaseModel):
    """Represents the entire state of a single game instance."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    players: int = Field(..., description="Number of players in the game", ge=1)
    scenario_name: Optional[str] = None
    characters: List[Character] = []
    scenario_details: Optional[str] = None
    scenes: List[Scene] = []
    player_actions: List[str] = []


class GeneratedScenarios(BaseModel):
    """Represents the list of generated scenarios."""

    scenarios: List[str] = Field(
        ..., description="List of three distinct fantasy adventure scenarios"
    )


class GeneratedScenarioDetails(BaseModel):
    """Structured detailed story sections for a scenario."""

    setting: str = Field(..., description="Description of the world/locale context")
    plot: str = Field(..., description="Overall narrative arc")
    main_quest: str = Field(..., description="Primary objective for the party")
    important_npcs: str = Field(
        ..., description="Key NPCs with short descriptions and relevance"
    )


class GeneratedCharacter(BaseModel):
    """Represents the generated data for a single character."""

    name: str = Field(..., description="The character's full name and title")
    strength: int = Field(..., description="Character's strength attribute (1-20)")
    intelligence: int = Field(
        ..., description="Character's intelligence attribute (1-20)"
    )
    agility: int = Field(..., description="Character's agility attribute (1-20)")
    current_health: int = Field(
        default=100,
        description="Character's current health points (0-100). Update this if character takes damage or is healed.",
    )
    backstory: str = Field(
        ...,
        description="A 2-3 sentence backstory explaining the character's history and motivations",
    )
    appearance: str = Field(
        ...,
        description="A 2-3 sentence physical description including clothing, build, and distinctive features",
    )
    inventory: List[str] = Field(
        ...,
        description="3-5 items the character carries (weapons, tools, magical items, etc.)",
    )


class GeneratedCharacterList(BaseModel):
    """Represents a list of generated characters."""

    characters: List[GeneratedCharacter]


class GeneratedScene(BaseModel):
    """Represents a generated scene with narrative and player prompt."""

    scene_text: str = Field(
        ...,
        description="The narrative text of the scene, describing what happens, what the players see, hear, and experience. Should be vivid and engaging.",
    )
    prompt: PromptType = Field(
        ...,
        description="The prompt for player interaction following the scene",
    )
    updated_characters: List[GeneratedCharacter] = Field(
        default_factory=list,
        description="Full character sheets after this scene. Re-generate all characters with updated stats, health, and inventory based on what happened in the scene. If nothing changed for a character, return them with the same values.",
    )

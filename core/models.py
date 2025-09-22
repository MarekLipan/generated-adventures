"""Pydantic models for the game state."""

import uuid
from pydantic import BaseModel, Field
from typing import List, Optional


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
    image_path: Optional[str] = Field(
        None, description="Path to the character's generated portrait"
    )


class Scene(BaseModel):
    """Represents a scene in the game, which can be expanded later."""

    id: int = Field(
        ..., description="Unique integer ID for the scene, ordered by creation"
    )
    text: str = Field(
        ..., description="Text content of the scene, can be markdown formatted"
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


class GeneratedCharacterList(BaseModel):
    """Represents a list of generated characters."""

    characters: List[GeneratedCharacter]

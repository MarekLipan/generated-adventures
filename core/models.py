"""Pydantic models for the game state."""

import uuid
from pydantic import BaseModel, Field
from typing import List, Optional


class Character(BaseModel):
    """Represents a single character in the game."""

    name: str
    strength: int = Field(
        ..., description="Character's strength attribute", ge=0, example=10
    )
    intelligence: int = Field(
        ..., description="Character's intelligence attribute", ge=0, example=12
    )
    agility: int = Field(
        ..., description="Character's agility attribute", ge=0, example=14
    )
    maximum_health: int = Field(
        ..., description="Character's maximum health", ge=1, example=100
    )
    current_health: int = Field(
        ..., description="Character's current health", ge=0, example=100
    )


class Game(BaseModel):
    """Represents the entire state of a single game instance."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    players: int = Field(
        ..., description="Number of players in the game", ge=1, example=4
    )
    scenario_name: Optional[str] = None
    characters: List[Character] = []
    scenario_details: Optional[str] = None
    # scenes: List[Scene] = [] # A placeholder for future scene management

"""Pydantic models for the game state."""

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Type aliases for strict field validation
GameStatus = Literal["ongoing", "completed", "failed"]
PromptTypeEnum = Literal["dialogue", "action", "dice_check"]
DiceType = Literal["d6", "d10"]


class Asset(BaseModel):
    """Represents an important NPC or object that should have consistent visual representation."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the asset",
    )
    name: str = Field(..., description="Name of the NPC or object")
    type: Literal["npc", "object"] = Field(
        ...,
        description="Type of asset: 'npc' for characters, 'object' for items/places",
    )
    description: str = Field(
        ...,
        description="Physical description of the NPC or object for visual consistency",
    )
    image_path: Optional[str] = Field(
        None, description="Path to the generated asset image"
    )


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
    personality: str = Field(
        ..., description="Character's personality traits, mannerisms, and demeanor"
    )
    skills: List[str] = Field(
        default_factory=list,
        description="List of skills and abilities the character possesses",
    )
    inventory: List[str] = Field(
        default_factory=list, description="List of items the character carries"
    )
    image_path: Optional[str] = Field(
        None, description="Path to the character's generated portrait"
    )


class PromptType(BaseModel):
    """Represents a prompt type for player interaction."""

    type: PromptTypeEnum = Field(
        ...,
        description="Type of prompt: 'dialogue', 'action', 'dice_check'",
    )
    dice_type: Optional[DiceType] = Field(
        None,
        description="Type of dice to roll if type is 'dice_check': 'd6' or 'd10'. Always single die roll.",
    )
    target_character: Optional[str] = Field(
        None,
        description="Name of specific character being prompted. REQUIRED (not None) for single-character dice_check prompts. Can be used for dialogue/action to target a specific character. For multi-character prompts, use target_characters instead and set this to None.",
    )
    target_characters: Optional[List[str]] = Field(
        None,
        description="List of character names for multi-character prompts (dice_check or action) where multiple specific characters act/roll simultaneously. When using this, set target_character to None.",
    )
    prompt_text: str = Field(
        ...,
        description="The actual prompt/question to present to the player(s)",
    )

    @model_validator(mode="after")
    def validate_dice_check_targeting(self) -> "PromptType":
        """Ensure dice_check prompts have proper dice_type and target specified."""
        if self.type == "dice_check":
            # Ensure dice_type is specified for dice checks
            if self.dice_type is None:
                raise ValueError(
                    "dice_check prompts must specify 'dice_type' (either 'd6' or 'd10')"
                )

            # For dice checks, we must have either target_character OR target_characters
            has_single_target = self.target_character is not None
            has_multi_target = (
                self.target_characters is not None and len(self.target_characters) > 0
            )

            if not has_single_target and not has_multi_target:
                raise ValueError(
                    "dice_check prompts must specify either 'target_character' (for single character) "
                    "or 'target_characters' (for multiple characters). Cannot have both as None."
                )

            if has_single_target and has_multi_target:
                raise ValueError(
                    "dice_check prompts cannot specify both 'target_character' and 'target_characters'. "
                    "Use only one: target_character for single character, target_characters for multiple."
                )

        return self


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
    visible_asset_ids: List[str] = Field(
        default_factory=list,
        description="List of asset IDs that are visible in this scene's image",
    )
    game_status: GameStatus = Field(
        default="ongoing",
        description="Status of the game: 'ongoing', 'completed' (victory), or 'failed' (game over)",
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
    assets: dict[str, Asset] = Field(
        default_factory=dict,
        description="Dictionary mapping asset IDs to Asset objects for visual consistency tracking",
    )


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
    personality: str = Field(
        ...,
        description="A 2-3 sentence description of the character's personality, mannerisms, and how they interact with others",
    )
    skills: List[str] = Field(
        ...,
        description="3-5 skills or abilities the character possesses (e.g., 'Lockpicking', 'Persuasion', 'Swordsmanship', 'Arcane Knowledge', 'Tracking')",
    )
    inventory: List[str] = Field(
        ...,
        description="3-5 items the character carries (weapons, tools, magical items, etc.)",
    )


class GeneratedCharacterList(BaseModel):
    """Represents a list of generated characters."""

    characters: List[GeneratedCharacter]


class AssetReference(BaseModel):
    """Reference to an asset (NPC or object) mentioned in the scene."""

    name: str = Field(
        ...,
        description="Name of the NPC or important object. MUST match existing asset name if already introduced, or provide new name for first appearance.",
    )
    type: Literal["npc", "object"] = Field(
        ...,
        description="Type: 'npc' for characters, 'object' for important items/places",
    )
    description: str = Field(
        ...,
        description="Physical description for visual generation. MUST be consistent with existing asset if reusing.",
    )
    is_visible: bool = Field(
        ...,
        description="True if this asset should appear in the scene image, False if only mentioned",
    )


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
    assets_present: List[AssetReference] = Field(
        default_factory=list,
        description="List of important NPCs and objects present in this scene. Include any significant characters or items that should have consistent visual representation. MUST reuse existing asset names when referring to already-introduced NPCs/objects.",
    )
    game_status: GameStatus = Field(
        default="ongoing",
        description="Status of the game: 'ongoing' (continue adventure), 'completed' (main quest fulfilled, party victorious), or 'failed' (all characters dead or quest failed). Set to 'failed' if ANY character reaches 0 health. Set to 'completed' only when the main quest objective is definitively achieved.",
    )

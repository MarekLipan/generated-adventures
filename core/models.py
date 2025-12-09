"""Pydantic models for the game state."""

import uuid
from datetime import datetime
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


class Location(BaseModel):
    """Represents a location where scenes take place, providing rich context for visual consistency."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the location",
    )
    name: str = Field(
        ...,
        description="Name of the location (e.g., 'The Crimson Tavern', 'Dragon's Peak Mountains')",
    )
    location_type: Literal[
        "indoor", "outdoor", "underground", "aerial", "aquatic", "mystical"
    ] = Field(
        ...,
        description="Type of location: indoor, outdoor, underground, aerial, aquatic, mystical",
    )
    description: str = Field(
        ...,
        description="Extensive physical description of the location including architecture, layout, materials, scale",
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="List of distinctive features (e.g., 'massive oak beams', 'crystal chandelier', 'ancient runes')",
    )
    atmosphere: str = Field(
        ...,
        description="General mood and atmosphere (e.g., 'warm and welcoming', 'dark and foreboding', 'ethereal')",
    )
    lighting_default: str = Field(
        ...,
        description="Default lighting conditions (e.g., 'torch-lit', 'moonlit', 'bright daylight', 'bioluminescent')",
    )


class LocationReference(BaseModel):
    """References a location in a specific scene with scene-specific variations.

    These fields provide CONTEXT for the visual_description, not rigid directives.
    The visual_description will incorporate this location context naturally.
    """

    location_id: str = Field(..., description="ID of the Location being referenced")
    time_of_day: Optional[str] = Field(
        None,
        description="Time of day variation (e.g., 'dawn', 'midday', 'dusk', 'night'). Provides context for lighting in visual_description.",
    )
    weather: Optional[str] = Field(
        None,
        description="Weather conditions if outdoor (e.g., 'clear', 'rainy', 'stormy', 'foggy', 'snowing'). Provides context for atmosphere in visual_description.",
    )
    state_changes: List[str] = Field(
        default_factory=list,
        description="Specific changes to location state (e.g., 'tables overturned', 'doors broken', 'fire damage'). Context for visual_description.",
    )
    camera_angle: Optional[str] = Field(
        None,
        description="Suggested camera perspective (e.g., 'wide establishing shot', 'close interior view', 'aerial view'). Suggestion for visual_description composition, not a strict requirement.",
    )
    focus_area: Optional[str] = Field(
        None,
        description="Specific area to emphasize (e.g., 'the bar area', 'the throne', 'the entrance'). Suggestion for visual_description focus, not a strict requirement.",
    )


class ScenarioTemplate(BaseModel):
    """Represents a reusable scenario template that can be played multiple times."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the scenario template",
    )
    name: str = Field(..., description="Name of the scenario")
    one_liner: str = Field(
        ...,
        description="Short enticing description (1-2 sentences) to attract players without spoilers",
    )
    dm_notes: str = Field(
        ...,
        description="Full markdown DM notes including setting, plot, main quest, and important NPCs",
    )
    times_played: int = Field(
        default=0, description="Number of times this scenario has been played", ge=0
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the scenario was generated",
    )
    last_played_at: Optional[datetime] = Field(
        None, description="Timestamp when the scenario was last played"
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
    location_reference: Optional[LocationReference] = Field(
        None,
        description="Reference to the location where this scene takes place, with scene-specific variations",
    )
    game_status: GameStatus = Field(
        default="ongoing",
        description="Status of the game: 'ongoing', 'completed' (victory), or 'failed' (game over)",
    )
    # Character changes that occurred in THIS scene only
    health_changes: List["CharacterHealthChange"] = Field(
        default_factory=list,
        description="Health changes that occurred in this scene",
    )
    inventory_changes: List["CharacterInventoryChange"] = Field(
        default_factory=list,
        description="Inventory changes that occurred in this scene",
    )
    skill_changes: List["CharacterSkillChange"] = Field(
        default_factory=list,
        description="Skill changes that occurred in this scene",
    )
    stat_changes: List["CharacterStatChange"] = Field(
        default_factory=list,
        description="Stat changes that occurred in this scene",
    )


class Game(BaseModel):
    """Represents the entire state of a single game instance."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    players: int = Field(..., description="Number of players in the game", ge=1)
    scenario_id: Optional[str] = Field(
        None, description="ID of the ScenarioTemplate being played"
    )
    characters: List[Character] = []
    scenes: List[Scene] = []
    player_actions: List[str] = []
    assets: dict[str, Asset] = Field(
        default_factory=dict,
        description="Dictionary mapping asset IDs to Asset objects for visual consistency tracking",
    )
    locations: dict[str, Location] = Field(
        default_factory=dict,
        description="Dictionary mapping location IDs to Location objects for visual consistency tracking",
    )


class GeneratedScenarioTemplate(BaseModel):
    """Represents AI-generated scenario template data."""

    name: str = Field(..., description="Name of the scenario")
    one_liner: str = Field(
        ...,
        description="Short enticing description (1-2 sentences) to attract players without spoilers",
    )
    setting: str = Field(
        ..., description="Description of the world/locale context (2-4 paragraphs)"
    )
    plot: str = Field(..., description="Overall narrative arc (2-4 paragraphs)")
    main_quest: str = Field(
        ..., description="Primary objective for the party (2-4 paragraphs)"
    )
    important_npcs: str = Field(
        ...,
        description="Key NPCs with short descriptions and relevance (2-4 paragraphs)",
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


class GeneratedLocation(BaseModel):
    """Represents a generated location for the LLM output."""

    name: str = Field(..., description="Name of the location")
    location_type: Literal[
        "indoor", "outdoor", "underground", "aerial", "aquatic", "mystical"
    ] = Field(..., description="Type of location")
    description: str = Field(
        ..., description="Extensive physical description of the location"
    )
    key_features: List[str] = Field(..., description="List of 3-5 distinctive features")
    atmosphere: str = Field(..., description="General mood and atmosphere")
    lighting_default: str = Field(..., description="Default lighting conditions")


class GeneratedLocationList(BaseModel):
    """Represents a list of generated locations."""

    locations: List[GeneratedLocation]


class CharacterHealthChange(BaseModel):
    """Represents a health change for a character in a scene."""

    character_name: str = Field(
        ..., description="Name of the character whose health changed"
    )
    health_change: int = Field(
        ...,
        description="Amount of health change. Negative for damage (e.g., -15), positive for healing (e.g., +20). Zero if no change.",
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why health changed (e.g., 'goblin attack', 'healing potion', 'fall damage')",
    )


class CharacterInventoryChange(BaseModel):
    """Represents inventory changes for a character in a scene."""

    character_name: str = Field(
        ..., description="Name of the character whose inventory changed"
    )
    items_added: List[str] = Field(
        default_factory=list,
        description="Items gained in this scene (e.g., ['Magic Sword', 'Health Potion'])",
    )
    items_removed: List[str] = Field(
        default_factory=list,
        description="Items lost or consumed in this scene (e.g., ['Rope', 'Torch'])",
    )


class CharacterSkillChange(BaseModel):
    """Represents skill changes for a character in a scene."""

    character_name: str = Field(
        ..., description="Name of the character whose skills changed"
    )
    skills_learned: List[str] = Field(
        default_factory=list,
        description="New skills learned in this scene (rare, e.g., ['Advanced Lockpicking'])",
    )
    skills_lost: List[str] = Field(
        default_factory=list,
        description="Skills lost in this scene (very rare, e.g., due to curse or amnesia)",
    )


class CharacterStatChange(BaseModel):
    """Represents stat changes for a character in a scene."""

    character_name: str = Field(
        ..., description="Name of the character whose stats changed"
    )
    strength_change: int = Field(
        default=0,
        description="Change in strength (-20 to +20). Usually 0. Use only for permanent effects like curses, blessings, or major transformations.",
    )
    intelligence_change: int = Field(
        default=0,
        description="Change in intelligence (-20 to +20). Usually 0. Use only for permanent effects.",
    )
    agility_change: int = Field(
        default=0,
        description="Change in agility (-20 to +20). Usually 0. Use only for permanent effects.",
    )
    reason: str = Field(
        default="",
        description="Explanation for stat changes (required if any stat changed)",
    )


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
    visual_description: str = Field(
        ...,
        description="A detailed visual description of the scene for image generation purposes ONLY (not shown to players). Describe the composition, camera angle, character positions and poses, lighting, atmosphere, environment details, and mood. If location_reference is provided, incorporate the location's description, key features, and any variations (time_of_day, weather, state_changes, camera_angle, focus_area) naturally into this description. Focus on visual elements that would make a compelling scene illustration. 3-5 sentences.",
    )
    prompt: PromptType = Field(
        ...,
        description="The prompt for player interaction following the scene",
    )
    health_changes: List[CharacterHealthChange] = Field(
        default_factory=list,
        description="List of health changes that occurred in THIS scene only. Include an entry for each character whose health changed (damage or healing). Empty list if no health changes.",
    )
    inventory_changes: List[CharacterInventoryChange] = Field(
        default_factory=list,
        description="List of inventory changes that occurred in THIS scene only. Include an entry for each character who gained or lost items. Empty list if no inventory changes.",
    )
    skill_changes: List[CharacterSkillChange] = Field(
        default_factory=list,
        description="List of skill changes that occurred in THIS scene only (rare). Include an entry for each character who learned or lost skills. Empty list if no skill changes.",
    )
    stat_changes: List[CharacterStatChange] = Field(
        default_factory=list,
        description="List of stat changes that occurred in THIS scene only (very rare). Include an entry for each character whose permanent stats changed due to curses, blessings, transformations, etc. Empty list if no stat changes.",
    )
    assets_present: List[AssetReference] = Field(
        default_factory=list,
        description="List of important NPCs and objects present in this scene. Include any significant characters or items that should have consistent visual representation. MUST reuse existing asset names when referring to already-introduced NPCs/objects.",
    )
    location_reference: Optional[LocationReference] = Field(
        None,
        description="Reference to the location where this scene takes place. MUST reuse existing location ID if returning to a known location. The location context (including time_of_day, weather, state_changes, camera_angle, focus_area) provides input for the visual_description - incorporate these elements naturally when writing visual_description. Use variations to add diversity while maintaining location consistency.",
    )
    game_status: GameStatus = Field(
        default="ongoing",
        description="Status of the game: 'ongoing' (continue adventure), 'completed' (main quest fulfilled, party victorious), or 'failed' (all characters dead or quest failed). Set to 'failed' if ANY character reaches 0 health. Set to 'completed' only when the main quest objective is definitively achieved.",
    )

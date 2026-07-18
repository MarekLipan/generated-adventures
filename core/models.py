"""Pydantic models for the game state."""

import uuid
from datetime import datetime
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, model_validator

# Type aliases for strict field validation
GameStatus = Literal["ongoing", "completed", "failed"]
PromptTypeEnum = Literal["dialogue", "action", "dice_check"]
DiceType = Literal["d6", "d10"]


class InventoryItem(BaseModel):
    """A carried item with a plain name and a clear, player-facing purpose.

    The purpose is what tells the player (and the DM) what the item is for and
    when to use it — the missing piece that made bare item names feel cryptic.
    """

    name: str = Field(
        ..., description="Short, plain item name (e.g. 'Clockwork Lockpicks')"
    )
    purpose: str = Field(
        "",
        description="One concise sentence: what the item does and when it helps in play "
        "(e.g. 'Opens mechanical locks and disarms clockwork traps').",
    )


def _coerce_inventory_item(v):
    """Accept a bare string as an item name (older saves / terse LLM output)."""
    if isinstance(v, str):
        return {"name": v, "purpose": ""}
    return v


# Inventory field element: a structured item, but tolerant of legacy string form.
InvItem = Annotated[InventoryItem, BeforeValidator(_coerce_inventory_item)]
ArtStyle = Literal["painterly_hero", "artstation_realism"]


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
    assigned_voice: Optional[str] = Field(
        None,
        description="Kokoro voice id assigned to this NPC for narration, kept stable across scenes",
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
    gender: Literal["male", "female", "nonbinary", "unspecified"] = Field(
        "unspecified",
        description="Character's gender, so narration uses the correct pronouns. "
        "Player-chosen during hero creation.",
    )
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
    inventory: List[InvItem] = Field(
        default_factory=list,
        description="Items the character carries, each with a name and a clear purpose",
    )
    image_path: Optional[str] = Field(
        None, description="Path to the character's generated portrait"
    )
    archetype: Optional[str] = Field(
        None,
        description="Name of the scenario archetype this hero was built from (e.g. 'The Infiltrator')",
    )
    player_photo_path: Optional[str] = Field(
        None,
        description="Path to the player's uploaded profile photo used to shape this hero's likeness, if any",
    )
    assigned_voice: Optional[str] = Field(
        None,
        description="Kokoro voice id assigned to this character for narrated dialogue, kept stable across scenes",
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
    recap_text: Optional[str] = Field(
        None,
        description="Cached AI-generated recap of all scenes up to and including this one",
    )


class Game(BaseModel):
    """Represents the entire state of a single game instance."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    players: int = Field(..., description="Number of players in the game", ge=1)
    scenario_id: Optional[str] = Field(
        None, description="ID of the ScenarioTemplate being played"
    )
    art_style: ArtStyle = Field(
        default="painterly_hero",
        description="Visual art style for all generated imagery in this game",
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


class NarrationSegment(BaseModel):
    """One contiguous chunk of a scene's narration, tagged with who voices it."""

    speaker: str = Field(
        ...,
        description="Who voices this segment: 'Narrator' for descriptive prose, or the exact name of the character/NPC speaking the quoted dialogue.",
    )
    text: str = Field(
        ...,
        description="The text to be read aloud for this segment. For dialogue, the spoken words WITHOUT surrounding quotation marks. Preserve the scene's original wording.",
    )
    gender: Literal["male", "female", "unknown"] = Field(
        default="unknown",
        description="Apparent gender of the speaker (for voice casting). Use 'unknown' for the Narrator or when unclear.",
    )


class NarrationScript(BaseModel):
    """An ordered breakdown of a scene into voiced narration segments."""

    segments: List[NarrationSegment] = Field(
        ...,
        description="The scene split into ordered segments, alternating between Narrator prose and character dialogue, covering the whole scene in reading order.",
    )


class GeneratedArchetype(BaseModel):
    """A scenario-tailored hero archetype offered to a player to pick from.

    Kept deliberately light: this is a cheap text-only option shown as a card.
    The expensive portrait and full lore are only generated after a player
    commits to an archetype.
    """

    name: str = Field(
        ...,
        description="Evocative archetype title fitting the scenario (e.g. 'The Infiltrator', 'Storm-Warden')",
    )
    role: str = Field(
        ...,
        description="Short role/party-function label (e.g. 'Stealth & sabotage', 'Front-line bruiser', 'Arcane support')",
    )
    hook: str = Field(
        ...,
        description="One enticing sentence describing the fantasy of playing this archetype in this scenario",
    )
    concept: str = Field(
        ...,
        description="Visual concept for the portrait: costume, equipment, silhouette, and vibe (2-3 sentences). No facial features — those come from the player's photo when provided.",
    )


class GeneratedArchetypeList(BaseModel):
    """A list of scenario-tailored archetypes for players to choose from."""

    archetypes: List[GeneratedArchetype]


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
    inventory: List[InvItem] = Field(
        ...,
        description="3-4 starting items. Give each a PLAIN, non-cryptic name AND a one-sentence "
        "purpose making clear what it does and when the player would use it. Prefer a few "
        "obviously-useful items (a weapon, a tool tied to a skill, one situational item) over "
        "flavorful-but-opaque trinkets.",
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
    items_added: List[InvItem] = Field(
        default_factory=list,
        description="Items gained in this scene, each with a name AND a one-sentence purpose "
        "(what it does / when to use it).",
    )
    items_removed: List[str] = Field(
        default_factory=list,
        description="Names of items lost or consumed in this scene (e.g., ['Rope', 'Torch'])",
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


class SceneSummary(BaseModel):
    """One-sentence summary of what happened in a specific scene."""

    scene_id: int = Field(
        ...,
        description="The scene ID this summary corresponds to",
    )
    summary: str = Field(
        ...,
        description="A concise one-sentence summary (15-20 words) of the key event or development in this scene. Focus on the main action, discovery, or decision.",
    )


class RecapResponse(BaseModel):
    """Response containing both overall recap narrative and individual scene summaries."""

    recap_text: str = Field(
        ...,
        description="The overall narrative recap (200-300 words) summarizing the adventure so far in an engaging, DM-style narration.",
    )
    scene_summaries: List[SceneSummary] = Field(
        ...,
        description="One-sentence summaries for each scene, in order. Use concise, punchy language that captures the essence of each scene.",
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
    narration_segments: List[NarrationSegment] = Field(
        default_factory=list,
        description="The scene_text broken into ordered voiced-narration segments for a multi-voice audiobook. Descriptive prose -> speaker 'Narrator'; each piece of direct speech -> speaker = the exact name of the character/NPC saying it (dialogue text WITHOUT surrounding quotation marks), with a gender tag. Cover the ENTIRE scene_text in reading order, preserving wording. Do not invent new content.",
    )
    location_reference: Optional[LocationReference] = Field(
        None,
        description="Reference to the location where this scene takes place. MUST reuse existing location ID if returning to a known location. The location context (including time_of_day, weather, state_changes, camera_angle, focus_area) provides input for the visual_description - incorporate these elements naturally when writing visual_description. Use variations to add diversity while maintaining location consistency.",
    )
    game_status: GameStatus = Field(
        default="ongoing",
        description="Status of the game: 'ongoing' (continue adventure), 'completed' (main quest fulfilled, party victorious), or 'failed' (all characters dead or quest failed). Set to 'failed' if ANY character reaches 0 health. Set to 'completed' only when the main quest objective is definitively achieved.",
    )

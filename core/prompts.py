"""Prompt text and steering: scenario themes and scales, scene rules, portrait cues."""

import re

from .models import (
    Asset,
    GeneratedArchetype,
    Location,
)


def setting_only(dm_notes: str | None) -> str:
    """Extract just the non-spoiler '## The Setting' section from DM notes.

    DM notes are assembled as '## The Setting … ## The Plot … ## Main Quest …
    ## Important NPCs …'. Everything after the setting is spoiler material
    (villains, secrets, twists) that must NOT reach player-facing text like the
    archetype cards. If the setting can't be isolated, return "" rather than risk
    leaking the plot.
    """
    if not dm_notes:
        return ""
    m = re.search(r"##\s*The Setting\s*(.*?)(?:\n##\s|\Z)", dm_notes, re.S | re.I)
    return m.group(1).strip() if m else ""


#: Visual gear-tier cue appended to hero portrait prompts, so a "level 1" hero
#: doesn't get rendered in ornate master-tier armor.
#: The hero portrait is the reference every later scene is built from, so the face
#: and hair must be fully visible here — a hood/helm/mask hides the very features
#: that keep the character recognizable, leaving scene renders free to reinvent the
#: head. Characters can still wear hoods in scenes; just not in this reference.
_PORTRAIT_HEAD_RULE = (
    "Head UNCOVERED with the face fully visible: no hood, cowl, helmet, mask, veil "
    "or wide-brimmed hat over the head or shading the face. Hair clearly visible. "
    "Face evenly lit and turned toward the viewer."
)


_PORTRAIT_TIER_CUE = {
    "grounded": (
        " Novice adventurer just starting out: poor, practical, second-hand kit — patched wool and "
        "homespun cloth, worn leather, a plain travel cloak worn back off the head, simple "
        "hand-me-down weapon. No ornate or plate armor, no gilding, jewels, runes, silks or "
        "glowing magic items."
    ),
    "heroic": (
        " Seasoned adventurer: sturdy, well-used practical gear; nothing ornate or artifact-like."
    ),
    "epic": " Legendary hero: storied, signature equipment befitting great renown.",
}


def build_hero_portrait_prompt(
    hero_name: str,
    archetype: GeneratedArchetype,
    has_photo: bool,
    scale: str = "any",
) -> str:
    """Build the portrait prompt for a hero, with or without a player photo."""
    tier = _PORTRAIT_TIER_CUE.get(scale, "")
    if has_photo:
        return (
            f"Reimagine the person in the reference photograph as {hero_name}, "
            f"{archetype.concept}{tier} "
            "Keep their facial features, face shape, skin tone and hairstyle clearly "
            f"recognizable so the hero looks like the same person. {_PORTRAIT_HEAD_RULE}"
        )
    return f"{hero_name}. {archetype.concept}{tier} {_PORTRAIT_HEAD_RULE}"


#: Genre/theme steering for scenario generation. Key -> (UI label, prompt steering).
#: "any" keeps the original maximal-variety behavior (no genre constraint).
SCENARIO_THEMES: dict[str, tuple[str, str]] = {
    "classic_fantasy": (
        "Classic Fantasy",
        "Set this adventure firmly in CLASSIC HEROIC FANTASY — the old Dungeons & Dragons "
        "tradition: a medieval sword-and-sorcery world of kingdoms and untamed wilds, taverns "
        "and torch-lit dungeons, castles and ancient ruins; wizards, clerics, rogues and "
        "knights; goblins, orcs, trolls, undead, cults and dragons; enchanted blades and dusty "
        "spellbooks. Keep an earnest tone of heroic adventure. Do NOT drift into science-fiction, "
        "steampunk/clockwork, modern, post-apocalyptic, or otherwise non-medieval-fantasy settings.",
    ),
    "dark_fantasy": (
        "Dark Fantasy",
        "Set this in GRIM, GOTHIC DARK FANTASY — still a medieval fantasy world, but bleak and "
        "ominous: cursed lands, plague and dread, morally grey choices, monstrous horrors and "
        "desperate stakes. Sword-and-sorcery, never sci-fi or steampunk.",
    ),
    "mystery_intrigue": (
        "Mystery & Intrigue",
        "Set this as a FANTASY MYSTERY or political intrigue in a medieval-fantasy world: "
        "investigation, hidden secrets, scheming factions and betrayal, favoring social and "
        "deductive challenges over constant combat. Keep it medieval fantasy, not sci-fi.",
    ),
    "any": (
        "Surprise Me (any genre)",
        "",
    ),
}


#: Stakes/scale steering for scenarios. Key -> (UI label, prompt steering). Shapes
#: BOTH the scenario's scope/stakes and the hero archetypes offered (aspiring vs
#: legendary). "any" leaves scale unconstrained.
SCENARIO_SCALES: dict[str, tuple[str, str]] = {
    "grounded": (
        "Local & Grounded",
        "Keep the STAKES LOW, SMALL and LOCAL — this is a first adventure for green, unknown "
        "novices, the kind of odd job a village posts on a notice board. The ENTIRE problem must fit "
        "in one settlement and its surroundings and threaten only a handful of ordinary people: a "
        "missing child or livestock, a stolen heirloom, bandits shaking down a mill, a beast in the "
        "sheep pens, a soured well, a neighbours' feud, a debt collector, a haunted barn. "
        "ABSOLUTELY NOT: saving the world/realm/city, ancient prophecies, chosen ones, gods or "
        "cults with cosmic aims, reality/memory/time unravelling, artifacts of great power, "
        "cataclysms, or any 'only you can stop this' framing — even as background. Nobody important "
        "is watching; the reward is modest coin, food, or goodwill. The heroes are green and "
        "unproven, and the danger is human-sized, mundane and personal.",
    ),
    "heroic": (
        "Heroic & Regional",
        "Pitch this at a MID scale. Seasoned but not legendary adventurers face a threat to a town or "
        "region — a cult, a warlord, a spreading blight, a corrupt lord. Real danger and rising "
        "stakes, but NOT apocalyptic and NOT 'only you can save everyone'.",
    ),
    "epic": (
        "Epic & World-Shaping",
        "Pitch this EPIC. Legendary heroes against a world-shaping threat — an awakening god, a "
        "realm-ending cataclysm, a conquering dark power. Grand, high-stakes, save-the-world scope.",
    ),
    "any": (
        "Surprise Me (any scale)",
        "",
    ),
}


#: How the chosen stakes/scale should colour the SCENES themselves (tone, hooks,
#: how NPCs frame requests). Without this the DM writes world-saving hooks even for
#: a deliberately small, local adventure.
_SCENE_SCALE_TONE = {
    "grounded": (
        "STAKES — KEEP THEM SMALL AND LOCAL (this adventure is deliberately low-level):\n"
        "- These are green, no-name adventurers doing an ordinary job. Nobody thinks they are "
        "special, chosen, or the world's last hope.\n"
        "- The problem is small, concrete and personal: a missing person or animal, a theft, a "
        "debt, a spoiled well, a nuisance beast, a family quarrel, a job nobody else will take.\n"
        "- NEVER frame it as saving the world/realm/city, an ancient prophecy, reality unravelling, "
        "or 'only you can do this'. NPCs must NOT plead that everything depends on the party.\n"
        "- NPCs treat them as hired help or passing strangers — practical, sometimes sceptical, "
        "offering modest pay or a favor. Keep the tone everyday, human-sized and grounded.\n"
        "- Any larger mystery stays firmly in the background; it is NOT the party's business."
    ),
    "heroic": (
        "STAKES — REGIONAL, NOT APOCALYPTIC:\n"
        "- Competent but not legendary adventurers. The threat endangers a town or region, not the "
        "world. Avoid prophecy/chosen-one framing and 'only you can save us' pleas."
    ),
    "epic": (
        "STAKES — EPIC:\n"
        "- Legendary heroes against a world-shaping threat. Grand, weighty stakes are appropriate."
    ),
}


def scene_scale_section(scale: str) -> str:
    """Prompt section pinning scene tone/stakes to the scenario's chosen scale."""
    tone = _SCENE_SCALE_TONE.get(scale or "", "")
    return f"\n{tone}\n" if tone else ""


def scenario_scale_options() -> dict[str, str]:
    """{key: label} for the scenario stakes/scale picker."""
    return {key: label for key, (label, _steer) in SCENARIO_SCALES.items()}


def scenario_theme_options() -> dict[str, str]:
    """{key: label} for the scenario genre picker."""
    return {key: label for key, (label, _steer) in SCENARIO_THEMES.items()}


def avoid_names_section(avoid_names: list[str] | None, header: str) -> str:
    """Format an 'already-used names, do not reuse' instruction, or '' if none."""
    names = sorted({n.strip() for n in (avoid_names or []) if n and n.strip()})
    if not names:
        return ""
    shown = ", ".join(names[:60])
    return f"{header} {shown}.\n"


def format_existing_assets(assets: dict[str, Asset]) -> str:
    """Format existing assets for inclusion in prompts.

    Args:
        assets: Dictionary mapping asset IDs to Asset objects

    Returns:
        Formatted string listing existing assets
    """
    if not assets:
        return "No existing assets yet (this is the first scene)."

    # Separate party members from other assets for clarity
    party_assets = []
    other_assets = []

    for asset_id, asset in assets.items():
        asset_str = f"- **{asset.name}** ({asset.type}): {asset.description}"
        if asset_id.startswith("player_"):
            party_assets.append(asset_str)
        else:
            other_assets.append(asset_str)

    result_parts = []

    if party_assets:
        result_parts.append(
            "PARTY MEMBERS (your player characters - ALWAYS include in assets_present when visible):\n"
            + "\n".join(party_assets)
        )

    if other_assets:
        result_parts.append(
            "OTHER EXISTING ASSETS (NPCs and objects already introduced - REUSE THESE NAMES AND DESCRIPTIONS):\n"
            + "\n".join(other_assets)
        )

    return "\n\n".join(result_parts) if result_parts else "No existing assets yet."


def format_existing_locations(locations: dict[str, Location]) -> str:
    """Format existing locations for inclusion in prompts.

    Args:
        locations: Dictionary mapping location IDs to Location objects

    Returns:
        Formatted string listing available locations
    """
    if not locations:
        return "No available locations yet."

    location_lines = []
    for loc_id, location in locations.items():
        features_str = ", ".join(location.key_features[:3])  # First 3 features
        location_lines.append(
            f"- **{location.name}** (ID: `{loc_id}`, Type: {location.location_type})\n"
            f"  Description: {location.description}\n"
            f"  Key Features: {features_str}\n"
            f"  Atmosphere: {location.atmosphere}\n"
            f"  Lighting: {location.lighting_default}"
        )

    return (
        "AVAILABLE LOCATIONS (choose from these for location_reference):\n"
        + "\n\n".join(location_lines)
    )


def build_scene_generation_prompt_rules() -> str:
    """Returns the common rules section for scene generation prompts."""
    return """
After the scene narrative, provide a prompt for player interaction. The prompt should be:
- Either for the entire party or a specific character
- One of three types:
  * dialogue: Character should speak (player enters text in quotes)
  * action: Player describes what they do (for simple, non-risky actions)
  * dice_check: Player must roll dice (for ANY risky action, test of skill, or combat)

CRITICAL RULES FOR PROMPT TYPES - VARY THE PROMPTS:
- DO NOT use dice_check for every scene - mix dialogue, action, dice_check, and party_choice
- Use dialogue when: characters need to talk to NPCs, negotiate, roleplay conversations
- Use action for: simple tasks, exploration without immediate danger, planning, easy skill checks, or coordinated team efforts
- Use dice_check for: combat, risky actions, difficult skill checks, life-or-death situations
- Use party_choice for: a genuine SHARED direction/logistics fork the whole group commits to together
  (which route, accept or refuse an offer, rest or press on, which lead to follow). Provide 2-4
  short, distinct options; the group picks ONE with a single click.
  * party_choice is for DIRECTION, never for expression: do NOT turn roleplay, dialogue, or a
    character's personal action into multiple choice — those stay open-ended (dialogue/action).
  * A party_choice addresses the whole party, so it does not need a single target character.

SPOTLIGHT — GIVE EVERY PLAYER SPACE (avoid one player driving everything):
- For individual prompts (action/dialogue/dice_check), usually spotlight ONE character by name and
  ROTATE who that is from scene to scene, weaving it naturally into the story so different party
  members get their moments. Do NOT default to prompting the whole party open-endedly — that lets a
  single player answer for everyone.
- Only target MULTIPLE characters at once (target_characters) when several genuinely act in the same
  instant (e.g. a combat round where each rolls). Otherwise prefer one rotating spotlight.
- When the decision is truly the group's shared fork, use party_choice (options) instead of an
  open-ended "what does the party do?".

CHOICE vs. ROLL — NEVER MIX THEM (this is critical for a coherent UI):
- If you are offering the player a CHOICE between courses of action ("do you descend OR hesitate?",
  "approach the guard OR sneak past?"), the type MUST be "action" (the player types which they pick).
  A choice is NOT a dice_check — the player has no way to answer an either/or question with a die.
- A "dice_check" is for ONE specific action the player has effectively already committed to, whose
  SUCCESS is uncertain. Its prompt_text must name that single action: "Roll {dX} to <do the specific
  thing>" (e.g. "Roll d6 to disarm the device", "Roll d10 to leap the gap before it closes").
- A dice_check prompt_text must NOT contain an either/or, a "do you…?" question, or the word "or"
  joining two different actions. If you catch yourself offering options, switch the type to "action".
- Do NOT use the die to "decide" a choice for the player (e.g. "roll to decide if you act with
  resolve"). The player makes choices; the die only resolves whether a chosen action succeeds.

DICE CHECK RULES (when using dice_check type):
- ALWAYS single die roll (either d6 or d10)
- Choose dice type based on difficulty:
  * d6: Simple to moderate challenges (lockpicking, climbing, basic combat)
  * d10: Challenging situations (difficult combat, dangerous magic, persuading hostile NPCs)
- Remember: The next scene will interpret the roll with stat modifiers (+2 for 16-20, +1 for 11-15, +0 for 6-10, -1 for 1-5) and skill bonuses
- Each character targeted will roll their own single die

TARGETING RULES (CRITICAL - READ CAREFULLY):
ALWAYS address specific, named PARTY MEMBERS — never a faceless "the party". Every prompt (of any
type) MUST set either target_character (one exact party name) or target_characters (an array of
exact party names). NEVER leave both null, and never write "the party" / "you all" / "everyone" as
if the group were one person.
  * SOLO adventures (only ONE party member): ALWAYS set target_character to that single hero's exact
    name, for EVERY prompt. Address them by name — "Julian, what do you do?" — never "Party, what do
    you do?". There is no group; there is one named character.
  * When ONE character is the natural focus, use target_character with their exact name.
  * When several/all should act at once, use target_characters listing each exact party name; each of
    those players will answer for their OWN character, so phrase prompt_text so every named character
    knows what they are deciding (e.g. "Kael, Bram — how does each of you meet the charge?").
  * prompt_text must NAME who is being addressed and ask what THEIR character does/says/rolls. Use
    the ACTUAL names from PARTY CHARACTER SHEETS. Examples:
    - "Theron, roll d10 to persuade the elder"
    - "Lyra and Doran, both roll d6 for your attacks"
    - "Mira, do you open the vault or guard the door?"

CRITICAL RULES FOR CHARACTER CHANGES (SIMPLIFIED APPROACH):
- The character sheets show CURRENT states as characters ENTER this scene
- DO NOT apply previous scene's changes again! 
- Report ONLY NEW changes that occur in THIS SPECIFIC scene
- Use the change arrays: health_changes, inventory_changes, skill_changes, stat_changes
- Empty arrays mean no changes of that type occurred in THIS scene

HEALTH CHANGES (health_changes array):
  * Report ONLY NEW damage or healing that occurs in THIS SPECIFIC scene
  * NEVER repeat health changes from conversation history - those are already applied
  * Character current_health already reflects past changes - don't subtract them again
  * Use negative values for NEW damage, positive for NEW healing
  * CRITICAL: These are CHANGES (deltas) from what happens NOW, not what happened before
  
  * CORRECT EXAMPLES:
    - Character takes NEW 15 damage in this scene: {"character_name": "Theron", "health_change": -15, "reason": "fell from cliff"}
    - Character gets NEW healing in this scene: {"character_name": "Elara", "health_change": 20, "reason": "healing spell"}
    - Character unharmed in this scene: Don't include them in the array at all
  
  * WRONG EXAMPLES (DON'T DO THIS):
    - ❌ Applying damage from a previous scene again
    - ❌ Using conversation history damage in this scene's health_changes
    - ❌ Seeing "took 10 damage" in history and putting -10 in THIS scene's changes
  
  * Remember: Only report what happens in THIS SPECIFIC scene based on current action

INVENTORY CHANGES (inventory_changes array):
  * Report items gained or lost in THIS scene only, based on what happens NOW
  * Current inventory list shows what they have entering this scene
  * Don't remove items that were used in previous scenes - they're already gone
  * EXAMPLES:
    - Gained NEW items: {"character_name": "Theron", "items_added": [{"name": "Magic Sword", "purpose": "A blade that cuts through enchanted armor — use it against warded foes"}], "items_removed": []}
    - Lost/used item NOW: {"character_name": "Elara", "items_added": [], "items_removed": ["Rope"]}
    - No change: Don't include character in array

SKILL CHANGES (skill_changes array):
  * Report skills learned or lost in THIS scene (rare)
  * EXAMPLES:
    - Learned skill: {"character_name": "Theron", "skills_learned": ["Advanced Combat"], "skills_lost": []}
    - Lost skill: {"character_name": "Elara", "skills_learned": [], "skills_lost": ["Stealth"]} (due to injury/curse)
    - No change: Empty array (most scenes)

STAT CHANGES (stat_changes array):
  * Report permanent stat changes in THIS scene (very rare)
  * Use for curses, blessings, transformations, permanent effects
  * EXAMPLES:
    - Cursed: {"character_name": "Theron", "strength_change": -2, "intelligence_change": 0, "agility_change": 0, "reason": "witch's curse"}
    - Blessed: {"character_name": "Elara", "strength_change": 0, "intelligence_change": 3, "agility_change": 0, "reason": "ancient tome"}
    - No change: Empty array (almost all scenes)

CRITICAL: NARRATE ALL CHARACTER SHEET CHANGES IN SCENE TEXT:
- If a character takes damage, EXPLICITLY state how much health they lose in the narrative
  * Example: "The goblin's blade slashes across your arm, dealing 15 damage. You feel your strength waning."
  * Example: "The fall bruises you badly - you lose 8 health."
- If a character is healed, EXPLICITLY state the healing amount
  * Example: "The healing potion surges through you, restoring 20 health."
  * Example: "The cleric's spell mends your wounds, recovering 12 health."
- If a character gains an item, EXPLICITLY mention it in the narrative
  * Example: "You pick up the enchanted dagger and add it to your pack."
  * Example: "The grateful merchant hands you a Potion of Invisibility."
- If a character loses/uses an item, EXPLICITLY describe it
  * Example: "You consume the Health Potion, feeling its effects immediately."
  * Example: "Your rope snaps under the weight and falls into the chasm below."
- If a character learns a new skill, describe how they acquired it
  * Example: "Through intense practice, you've mastered the art of Stealth."
- Players should NEVER be surprised by character sheet changes - everything must be clear in the story

CRITICAL: SPECIFY EXACT COUNTS IN SCENE TEXT:
- When groups of enemies, NPCs, or creatures appear, ALWAYS mention the exact count in the narrative
  * Example: "Two bandits step out from behind the rocks, weapons drawn."
  * Example: "Three wolves emerge from the shadows, circling your party."
  * Example: "Four guards block the entrance to the throne room."
- DO NOT use vague terms like "some enemies", "several bandits", "a group of wolves"
- The count in scene_text MUST match the count in visual_description and assets_present
- This ensures players understand exactly what they're facing and images match the narrative

CRITICAL RULES FOR GAME STATUS (REQUIRED FIELD):
- ALWAYS include game_status in your response - it is a REQUIRED field
- Set game_status to "ongoing" (default) - continue the adventure normally
- Set game_status to "failed" if ANY character reaches 0 current_health (party death/defeat):
  * This ends the game immediately with a game over
  * The scene should describe their defeat or death dramatically
  * Still provide a prompt field (required by schema), but it won't be shown to players
- Set game_status to "completed" ONLY when the main quest objective is definitively achieved AND its
  immediate consequences have played out — do NOT snap to victory the instant an objective is
  mechanically touched:
  * The primary goal from scenario_details must be fulfilled.
  * If the scenario hinges on a pivotal CHOICE or MORAL DILEMMA (per the DM notes), that decision
    MUST have been presented to the player and resolved by their action before you complete — never
    skip it to reach the ending faster.
  * Play out the fallout of the climactic action first — the cost, the escape, the aftermath. Prefer
    to keep the objective-reaching scene "ongoing" with one more beat, then complete on the FOLLOWING
    scene once the consequences and a brief resolution have been shown, so the ending feels earned.
  * The completing scene should describe both the triumph AND its resolution, not merely state success.
  * Still provide a prompt field (required by schema), but it won't be shown to players.
- Do NOT set completed prematurely — a minor victory, or reaching the objective while a dilemma or
  its consequences remain unplayed, is still "ongoing".
- Do NOT set completed just because things are going well — only when the main quest is truly resolved.

IMAGE / SHOT DIRECTION (frame each scene like a book illustration):
- Each scene has ONE illustration. Direct it like an illustrator choosing the single most evocative
  shot for THIS beat — vary it across scenes: an establishing view of a new location, a lone key
  NPC or the threat, a significant object/discovery, or a tight action moment. Do NOT cram the whole
  cast into every image; most beats are stronger with one clear focus.
- visual_description must describe THAT one shot (its focus, composition, lighting, mood).
- image_subjects: ONLY the figures/objects actually in that shot's frame. For each, give the EXACT
  name (from assets_present or the party roster) AND a short `action` phrase for what THAT subject is
  doing/holding in the image. Use an EMPTY list for a pure environment shot.
  * List them in LEFT-TO-RIGHT order as they appear in the composition. The illustrator places and
    repaints them in exactly this order, so the order + per-subject actions are what keep characters
    from being SWAPPED (e.g. the clerk must be the one holding the papers, not the hero).
  * Each subject's `action` must match what scene_text says THEY are doing — never give one
    character's action to another.
  * Leave out anyone not in this particular image — only these subjects are drawn and referenced.
    This keeps images fitting and fresh AND stops figures copying each other's clothing/look. If the
    hero is not the focus of this shot, do not put them in frame.
  * When two characters MUST share the frame, describe each one's DISTINCT clothing and silhouette so
    they don't blend together.

CRITICAL RULES FOR VISUAL ASSETS (NPCs AND OBJECTS):
- Include assets_present array listing ALL important NPCs and objects in the scene
- Important assets are: Named NPCs, significant objects, key locations that should be visually consistent
- MANDATORY — REGISTER EVERYONE YOU DEPICT (this is what keeps them looking the same):
  * EVERY named NPC or creature that appears in scene_text or visual_description MUST
    have an entry in assets_present — INCLUDING the very first scene they appear in.
  * NEVER describe or draw a named character in a scene without adding them to
    assets_present. Their first entry is what generates their reference portrait; if
    you omit them on first appearance they will look DIFFERENT in every later scene.
  * Use ONE stable canonical name for each character for the whole adventure: no
    leading "The", no honorifics added or dropped between scenes, identical spelling
    and hyphenation every time (e.g. always "Echo-Child", never also "The Echo Child").
  * When you introduce a BRAND-NEW NPC, invent a fresh name — avoid the overused clichés
    (Elara, Kael/Kaelen, Lyra, Seraphina, Thorne, ...) and never reuse a name already used
    by a party member or an existing asset in this game.
- CRITICAL: PARTY MEMBERS ARE ASSETS TOO:
  * ALL party member characters are pre-registered as assets (type: "npc")
  * When party members appear in a scene, ALWAYS include them in assets_present
  * Use their EXACT character name from the party roster
  * Set is_visible=true when they're present in the scene
  * This ensures their visual consistency (reference images) across all scenes
- For each asset, specify:
  * name: The name of the NPC or object
  * type: "npc" for characters (including party members!), "object" for items/places/things
  * description: Physical description for image generation
  * is_visible: MUST be true ONLY if the NPC/object is PHYSICALLY PRESENT in the current scene location
  * is_visible: MUST be false if the NPC/object is only mentioned, talked about, in another location, or not directly visible
- CRITICAL is_visible RULES:
  * Set is_visible=true: NPC is in the room/area with the party, object is visible in the scene
  * Set is_visible=false: NPC is somewhere else, object is mentioned but not present, talked about but not seen
  * Example: If party discusses a villain who is in a distant location, is_visible=false
  * Example: If party sees a door/portal directly in front of them, is_visible=true
  * Example: If an NPC mentions an item in another room that party can't see, is_visible=false
- CRITICAL ASSET REUSE RULES (PREVENT DUPLICATES):
  * ALWAYS check the EXISTING ASSETS list first before creating a new asset
  * If an NPC or object was introduced in a previous scene, you MUST reuse it
  * Use the EXACT SAME name (character-by-character identical, including capitalization, spacing, and punctuation)
  * Use the EXACT SAME physical description (copy it word-for-word from EXISTING ASSETS)
  * DO NOT create variations of the same name (e.g., if existing asset is "Guard Captain", don't create "The Guard Captain" or "Captain of the Guard")
  * DO NOT add adjectives or modifiers to existing names (e.g., if existing is "Ancient Door", don't create "Large Ancient Door")
  * When in doubt, if it could be the same NPC/object, REUSE the existing one
  * Only create a NEW asset if it's genuinely a different NPC/object that has never appeared before
- For genuinely new NPCs/objects appearing for the first time:
  * Choose a clear, unique name that won't conflict with existing assets
  * Create a clear, detailed physical description
  * This description will be used to generate their consistent image
- Examples:
  * NPC present: {"name": "[NPC's actual name]", "type": "npc", "description": "[Physical description of NPC]", "is_visible": true}
  * Object present: {"name": "[Object's actual name]", "type": "object", "description": "[Physical description of object]", "is_visible": true}
  * NPC mentioned/elsewhere: {"name": "[NPC's actual name]", "type": "npc", "description": "[Physical description of NPC]", "is_visible": false}
  * Object mentioned/not visible: {"name": "[Object's actual name]", "type": "object", "description": "[Physical description of object]", "is_visible": false}

CRITICAL: VISUAL DESCRIPTION FOR IMAGE GENERATION:
- Include a visual_description field separate from scene_text
- This description is ONLY for the image generator - players will never see it
- MUST be comprehensive and detailed (8-12 sentences minimum) - treat this as precise direction for a fantasy artist
- REQUIRED elements to describe in detail:
  
  1. CAMERA & FRAMING:
     * Specify exact camera angle (wide shot, medium shot, close-up, bird's eye, low angle, etc.)
     * Aspect ratio reminder: LANDSCAPE/HORIZONTAL (16:9)
     * What is in foreground, midground, background
  
  2. EACH CHARACTER IN SCENE (describe ALL party members present):
     * Name each character explicitly
     * Their exact position in the scene (left/right/center, foreground/background, distance from camera)
     * Body position and stance (standing/crouching/kneeling, facing direction, body angle to camera)
     * What they're doing with their hands/arms (weapon drawn, casting spell, examining object, gesturing)
     * Facial expression and where they're looking
     * Body language conveying emotion/intent
  
  3. IMPORTANT OBJECTS/NPCs (describe ALL visible assets):
     * CRITICAL: If describing groups (enemies, guards, creatures), specify EXACT COUNT (e.g., "two [enemy type]", "three [creature type]", "four [NPC type]")
     * Name each important object or NPC explicitly
     * If there are multiple similar NPCs (enemies, guards), describe each one's position individually
     * Exact placement in the scene for each individual
     * Size relative to characters
     * State/condition (glowing, damaged, active, dormant, etc.)
     * How characters are interacting with them (if applicable)
  
  4. ENVIRONMENT:
     * Specific location type and architectural/natural features
     * Time of day and weather conditions
     * Key environmental details (textures, materials, vegetation, terrain)
     * Spatial layout (how elements are arranged in 3D space)
  
  5. LIGHTING & ATMOSPHERE:
     * Primary light source(s) and direction
     * Quality of light (harsh/soft, warm/cool, colored/white)
     * Shadows and how they fall
     * Atmospheric effects (fog, dust, magic glow, particles, etc.)
     * Overall mood and emotional tone
  
  6. FOCAL POINTS:
     * What draws the viewer's eye first
     * Visual hierarchy of elements
  
- Write as clear, specific directions - avoid vague terms
- CRITICAL for groups: Always specify exact counts (e.g., "two [enemies]", not "[enemies]" or "several [enemies]")
- CRITICAL: Only describe what should actually appear in the image - image generator follows this literally
- Example format with counted NPCs: "Medium wide shot from [angle]. In the foreground left, [Character A] stands with [action/pose], facing right toward two [enemy/NPC type], [expression/stance]. The first [enemy] is in the right foreground, [specific pose/action]. The second [enemy] is [position], [specific pose/action]. Center background, [Character B] [action/pose]. Far right, [Character C] [action/pose]. In the midground, [important object] with [details], [size/placement]. The scene is set in [location type] with [environmental features]. Lighting is [light description] creating [shadow effects]. [Additional atmospheric elements]. Atmosphere is [mood], with [final atmospheric details]."

Return your response in this JSON structure (ALL FIELDS REQUIRED):
{
  "scene_text": "The vivid narrative of the scene...",
  "visual_description": "COMPREHENSIVE visual description for image generation (8-12+ sentences with detailed camera angle, each character's position/pose/expression, all visible objects/NPCs placement, environment details, lighting setup, and atmosphere)...",
  "prompt": {
    "type": "dialogue" | "action" | "dice_check" | "party_choice",
    "dice_type": "d6" | "d10" (only if type is dice_check, always single die roll),
    "target_character": "Character Name" (a concrete party name; REQUIRED for dice_check and the usual case for action/dialogue — rotate who; null only for party_choice) ,
    "target_characters": ["Char1", "Char2"] (only when several characters act in the SAME instant) or null,
    "options": ["Option A", "Option B", "Option C"] (REQUIRED for party_choice: 2-4 short distinct choices; null otherwise),
    "prompt_text": "The question or instruction, addressed to the named character(s) by name"
  },
  "health_changes": [
    {
      "character_name": "Character Name",
      "health_change": -15 (negative for damage, positive for healing),
      "reason": "brief explanation"
    }
  ],
  "inventory_changes": [
    {
      "character_name": "Character Name",
      "items_added": [{"name": "New Item", "purpose": "what it does and when to use it"}],
      "items_removed": ["Used Item"]
    }
  ],
  "skill_changes": [
    {
      "character_name": "Character Name",
      "skills_learned": ["New Skill"],
      "skills_lost": []
    }
  ],
  "stat_changes": [
    {
      "character_name": "Character Name",
      "strength_change": 0,
      "intelligence_change": 2,
      "agility_change": 0,
      "reason": "blessed by ancient tome"
    }
  ],
  "assets_present": [
    {
      "name": "NPC or Object Name",
      "type": "npc" | "object",
      "description": "Physical description...",
      "is_visible": true | false
    }
  ],
  "image_subjects": [{"name": "Exact name of a figure/object in THIS image's frame", "action": "what THIS subject is doing/holding in the image"}] (left-to-right order; subset of assets_present/party; [] for an environment-only establishing shot),
  "narration_segments": [
    {
      "speaker": "Narrator" | "Exact Character/NPC Name",
      "text": "the words to read aloud (dialogue WITHOUT quotation marks)",
      "gender": "male" | "female" | "unknown"
    }
  ],
  "location_reference": {
    "location_id": "UUID of the location from AVAILABLE LOCATIONS list",
    "time_of_day": "dawn" | "midday" | "afternoon" | "dusk" | "night" | null (optional variation),
    "weather": "clear" | "rainy" | "stormy" | "foggy" | "snowing" | null (optional, for outdoor locations),
    "state_changes": ["description of any changes to location", "..."] (optional, e.g., ["tables overturned", "fire damage"]),
    "camera_angle": "wide establishing shot" | "close interior view" | "aerial view" | "low angle" | null (optional),
    "focus_area": "specific area to emphasize" | null (optional, e.g., "the bar area", "the throne", "the entrance")
  } | null,
  "game_status": "ongoing" | "completed" | "failed"
}

NARRATION VOICE & RHYTHM (scene_text is read aloud by a text-to-speech narrator):
- The narrator's phrasing comes ENTIRELY from your punctuation and sentence shape, so shape it on purpose:
  * Vary sentence length — mix short, punchy sentences with longer, flowing ones.
  * Use an em-dash (—) for a dramatic beat or a mid-sentence pause.
  * Use an ellipsis (…) for suspense or a trailing, ominous thought.
  * Lean on short sentences and periods at moments of tension; longer clauses read calmer.
- Avoid run-on sentences strung together with only commas — they read flat and breathless aloud.
- End every sentence with terminal punctuation (. ! ? or …).
- Refer to each PARTY MEMBER with the pronouns matching their stated Gender in the PARTY CHARACTER SHEETS.

PLAYER CHARACTERS SPEAK FOR THEMSELVES (do NOT voice them):
- The party members in PARTY CHARACTER SHEETS are PLAYER characters. Their words belong to the
  players, not to you.
- NEVER write dialogue or quoted speech for a party member in scene_text, and NEVER emit a
  narration_segment whose speaker is a party member. Describe their actions, reactions, expressions
  and presence in Narrator prose instead ("Kaelia raises her blade" — yes; "Kaelia says '...'" — NO).
- NPCs and other non-party characters may speak freely.

MAKE THE PARTY'S ITEMS MATTER:
- Each hero's Inventory in PARTY CHARACTER SHEETS lists items as "Name (purpose)". Read them.
- When a situation fits an item's stated purpose, create a clear opportunity to use it and make it
  legible to the player — reference it in scene_text or the prompt (e.g. present a mechanical lock
  when someone carries lockpicks, a warded door when someone has ward-piercing oil). Don't force an
  item every scene, but let items pay off over the adventure instead of sitting unused.
- When a player's action uses an item as intended, honor it in the outcome; if the item is spent,
  list its name in inventory_changes.items_removed.
- Any NEW item you grant via inventory_changes.items_added MUST include a plain name AND a
  one-sentence purpose — never a cryptic name with no explanation.

CRITICAL RULES FOR NARRATION SEGMENTS (for the voiced audiobook):
- Break scene_text into an ordered list of narration_segments covering the ENTIRE scene in reading order.
- Descriptive prose (anything that is NOT direct speech) -> speaker "Narrator", gender "unknown".
- Each piece of NPC direct speech -> a separate segment whose speaker is the EXACT name of the
  NPC saying it (match existing asset names when it's one of them; for a minor unnamed speaker use a
  short label like "Guard" or "Innkeeper"). Put the spoken words in "text" WITHOUT the surrounding
  quotation marks, and set gender to "male", "female", or "unknown". Party members are NEVER speakers.
- Preserve the original wording; do NOT invent new lines. This is just scene_text split by speaker.
- Merge consecutive same-speaker prose into one segment; keep each distinct speech as its own segment.

CRITICAL RULES FOR LOCATIONS:
- ALWAYS include location_reference for every scene (or null if truly no specific location applies)
- Choose from the AVAILABLE LOCATIONS list provided in the prompt
- REUSE existing locations when narratively appropriate:
  * If party returns to a tavern, reuse the tavern location_id
  * If continuing in same area, reuse the same location_id
  * Locations provide visual consistency across scenes in same place

IMPORTANT - RELATIONSHIP BETWEEN location_reference AND visual_description:
- The location_reference provides CONTEXT that should be incorporated into visual_description
- When writing visual_description, use the location's description, key_features, and variations
- Workflow: 
  1. Set location_reference with location_id and any variations
  2. Write visual_description that naturally incorporates:
     - Location's physical description and key features
     - Time of day (affects lighting: dawn=soft, midday=bright, dusk=golden, night=dark)
     - Weather (affects atmosphere: rainy=wet/gloomy, stormy=dramatic, foggy=mysterious)
     - State changes (damaged furniture, broken doors, scorch marks, etc.)
     - Camera angle suggestion (but feel free to adjust for best composition)
     - Focus area (emphasize that part while maintaining overall scene)
- Example: If location_reference has time_of_day='dusk', camera_angle='wide shot', focus_area='entrance'
  Then visual_description might be: "Wide shot of the tavern entrance bathed in golden dusk light..."
- The visual_description is the complete, unified description used for image generation
- Location variations add diversity while maintaining consistent features

- Use the location variations to add visual diversity:
  * time_of_day: Change time to show passage of time (dawn -> midday -> dusk -> night)
  * weather: For outdoor locations, vary weather conditions
  * state_changes: After battles or events, describe damage/changes to location
  * camera_angle: Vary the view (wide shot, close-up, different angles)
  * focus_area: Emphasize different parts of the same location
- Examples of good location reuse with variation:
  * Same tavern, different time: morning crowd vs late night
  * Same forest, different weather: sunny morning vs stormy afternoon  
  * Same throne room, different focus: wide view of whole room vs close on throne
  * Same location after battle: add state_changes like "broken furniture", "scorch marks"
- This allows visual consistency (same location features) while avoiding repetitive images
- For NEW locations not in the list, choose the CLOSEST match from available locations
  * If moving to new area, pick the most appropriate existing location type
  * Don't request new locations - work with what's available
"""


def build_impossible_action_rules() -> str:
    """Returns rules for handling impossible player actions."""
    return """
CRITICAL RULES FOR HANDLING PLAYER ACTIONS:
- If player attempts an IMPOSSIBLE action (using items they don't have, doing something beyond their character's capabilities, breaking world logic):
  * DO NOT allow the action to succeed
  * In scene_text, explain why it's not possible
  * Prompt them to try something different (use "action" type prompt)
  * Example: "You reach for the magic sword, but you don't have such an item. What do you do instead?"

- If player describes an action that should undergo a test or challenge:
  * DO NOT resolve it immediately
  * Prompt for dice_check to determine success/failure
  * Examples: attacking, dodging, climbing, lockpicking, persuading, searching, casting difficult spells

- COMBAT and FIGHTING scenarios MUST use dice_check prompts:
  * Any attack, defense, or combat maneuver requires dice
  * Multiple characters in combat should each get dice_check prompts
  * Never auto-resolve combat without dice rolls
"""


def build_dice_check_resolution_rules() -> str:
    """Returns rules for resolving dice check results."""
    return """
CRITICAL RULES FOR DICE CHECK RESOLUTION:
When the previous prompt was a dice_check and the player provides their roll result(s):

NOTE: For multi-character dice checks, the player will provide rolls in format: "CharacterName1: 7, CharacterName2: 4"
Parse each character's roll separately and apply their individual modifiers.

1. INTERPRET THE ROLL with character stat modifiers:
   - For PHYSICAL actions (combat, climbing, swimming, breaking): Apply STRENGTH modifier
     * Strength 16-20: +2 to roll
     * Strength 11-15: +1 to roll
     * Strength 6-10: +0 to roll
     * Strength 1-5: -1 to roll
   
   - For MENTAL actions (arcane knowledge, puzzle solving, investigation): Apply INTELLIGENCE modifier
     * Intelligence 16-20: +2 to roll
     * Intelligence 11-15: +1 to roll
     * Intelligence 6-10: +0 to roll
     * Intelligence 1-5: -1 to roll
   
   - For DEXTERITY actions (lockpicking, dodging, stealth, acrobatics): Apply AGILITY modifier
     * Agility 16-20: +2 to roll
     * Agility 11-15: +1 to roll
     * Agility 6-10: +0 to roll
     * Agility 1-5: -1 to roll
   
   - For SOCIAL actions (persuasion, intimidation, deception): Apply INTELLIGENCE modifier (wit/charisma)

2. APPLY SKILL BONUSES:
   - If the character has a relevant skill, add +1 to the final result
   - Examples: "Lockpicking" skill for picking locks, "Swordsmanship" for melee combat, "Persuasion" for social checks

3. DETERMINE SUCCESS:
   - Calculate: Roll + Stat Modifier + Skill Bonus (if applicable)
   - For d6 checks: 4+ is success, 6+ is critical success, 2 or less is critical failure
   - For d10 checks: 6+ is success, 9+ is critical success, 3 or less is critical failure
   - Note: Always single die roll, so modifiers are very impactful

4. NARRATE THE RESULT:
   - ALWAYS mention the dice type that was rolled (d6 or d10) to provide context
   - Show the roll value as a fraction to make success/failure clear to players
   - Explicitly mention the stat modifier and skill bonus in your narrative
   - Example formats:
     * "You rolled 4/6. With your high Strength (+2) and Swordsmanship skill (+1), that's a total of 7 - a solid success!"
     * "You rolled 3/10. Despite your Agility modifier (+1), bringing you to 4, you barely manage..."
     * "A natural 6/6! With your Intelligence bonus (+2), that's a perfect 8 - critical success!"
     * "You rolled 2/10. Even with your skill bonus (+1), you only reach 3 - a critical failure..."
   - The fraction format (roll/max) helps players immediately understand if it was a good or bad roll
   - Make outcomes dramatic and descriptive
   - Critical successes should have extra benefits
   - Critical failures should have interesting consequences

5. UPDATE CHARACTER STATE:
   - Successful combat: Enemy takes damage, may gain items/information
   - Failed combat: Character loses health (appropriate to danger level)
   - Successful skill checks: Progress story, gain items/allies/knowledge
   - Failed skill checks: Setbacks, complications, but not instant death unless extremely dangerous
"""

"""Scenes: the opening and every next scene, with their assets, locations, images and narration."""

import asyncio
import logging

from ..character_updates import apply_character_updates, format_inventory
from ..image_backends import ImageGenerator
from ..llm import retry_on_overload, text_agent
from ..media import (
    ASSET_DIR,
    generate_asset_image_sync,
    generate_scene_image_sync,
    generate_scene_voiceover,
    resolve_image_subject_ids,
    shared_image_generator,
)
from ..models import (
    Asset,
    Character,
    GameStatus,
    GeneratedScene,
    Location,
    LocationReference,
    Scene,
)
from ..names import names_match, npc_names_match
from ..prompts import (
    build_dice_check_resolution_rules,
    build_impossible_action_rules,
    build_scene_generation_prompt_rules,
    format_existing_assets,
    format_existing_locations,
    scene_scale_section,
)

logger = logging.getLogger(__name__)


def _make_scene(
    scene_id: int,
    scene_text: str,
    prompt,
    image_path: str | None = None,
    game_status: GameStatus = "ongoing",
    visible_asset_ids: list[str] = None,
    health_changes: list = None,
    inventory_changes: list = None,
    skill_changes: list = None,
    stat_changes: list = None,
) -> Scene:
    """Helper to create a Scene object from generated data.

    Args:
        scene_id: Unique scene identifier
        scene_text: The narrative text of the scene
        prompt: PromptType object from the generated scene
        image_path: Optional path to the scene image
        game_status: Status of the game ('ongoing', 'completed', 'failed')
        visible_asset_ids: List of asset IDs visible in this scene
        health_changes: List of CharacterHealthChange objects from this scene
        inventory_changes: List of CharacterInventoryChange objects from this scene
        skill_changes: List of CharacterSkillChange objects from this scene
        stat_changes: List of CharacterStatChange objects from this scene
    """
    return Scene(
        id=scene_id,
        text=scene_text,
        prompt=prompt,
        image_path=image_path,
        voiceover_path=None,
        game_status=game_status,
        visible_asset_ids=visible_asset_ids or [],
        health_changes=health_changes or [],
        inventory_changes=inventory_changes or [],
        skill_changes=skill_changes or [],
        stat_changes=stat_changes or [],
    )


def _process_scene_assets(
    game_id: str,
    scene_id: int,
    assets_present: list,
    existing_assets: dict[str, Asset],
    generator: ImageGenerator,
    art_style: str | None = None,
) -> tuple[dict[str, Asset], list[str]]:
    """Process assets from generated scene, create new ones, and generate images.

    Args:
        game_id: Game identifier for storage
        scene_id: Scene identifier
        assets_present: List of AssetReference objects from the generated scene
        existing_assets: Dictionary of existing assets
        generator: Local image backend used to render new asset images
        art_style: Art-style key for the game

    Returns:
        Tuple of (updated assets dictionary, list of visible asset IDs for this scene)
    """
    updated_assets = existing_assets.copy()
    visible_asset_ids = []

    # Create asset directory if it doesn't exist
    asset_dir = ASSET_DIR / game_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    for asset_ref in assets_present:
        # Find if this asset already exists (match by name, case-insensitive)
        existing_asset_id = None
        for asset_id, asset in existing_assets.items():
            if asset.name.lower() == asset_ref.name.lower():
                existing_asset_id = asset_id
                break

        # If no exact match, try a tolerant match. Party portraits carry the
        # player's likeness (photo reference), so match them loosely (subset) to
        # always reuse them even if the hero was named slightly differently. NPCs
        # and objects match on normalized token-set EQUALITY, which reunites
        # naming drift ("The Echo Child" vs "Echo-Child") into ONE asset/portrait
        # while keeping distinct NPCs ("Guard" vs "Guard Captain") apart.
        if not existing_asset_id:
            for asset_id, asset in existing_assets.items():
                is_party = asset_id.startswith("player_")
                matched = (
                    names_match(asset_ref.name, asset.name)
                    if is_party
                    else npc_names_match(asset_ref.name, asset.name)
                )
                if matched:
                    existing_asset_id = asset_id
                    logger.info(
                        f"✓ Matched '{asset_ref.name}' to existing "
                        f"'{asset.name}' (tolerant, {'party' if is_party else 'npc'})"
                    )
                    break

        if existing_asset_id:
            # Asset exists, use existing one
            if asset_ref.is_visible:
                visible_asset_ids.append(existing_asset_id)
            logger.info(
                f"✓ Reusing existing asset: '{asset_ref.name}' (matched with existing '{existing_assets[existing_asset_id].name}')"
            )
        else:
            # New asset - log all existing assets for debugging
            if existing_assets:
                existing_names = [f"'{a.name}'" for a in existing_assets.values()]
                logger.info(
                    f"✗ Creating NEW asset: '{asset_ref.name}' (type: {asset_ref.type}) - Did not match any existing: {', '.join(existing_names)}"
                )
            else:
                logger.info(
                    f"✓ Creating first asset: '{asset_ref.name}' (type: {asset_ref.type})"
                )

            new_asset = Asset(
                name=asset_ref.name,
                type=asset_ref.type,
                description=asset_ref.description,
            )

            # Generate image for the new asset
            try:
                asset_image_path = generate_asset_image_sync(
                    generator,
                    game_id,
                    new_asset.id,
                    new_asset.name,
                    new_asset.description,
                    art_style,
                )
                if asset_image_path:
                    new_asset.image_path = (
                        f"/static/assets/{game_id}/{asset_image_path.name}"
                    )
                logger.info("  → Asset image generated successfully")
            except Exception as e:
                logger.error(
                    f"Failed to generate image for asset {new_asset.name}: {e}"
                )

            updated_assets[new_asset.id] = new_asset

            if asset_ref.is_visible:
                visible_asset_ids.append(new_asset.id)

    return updated_assets, visible_asset_ids


def _process_scene_location(
    location_reference: LocationReference | None,
    existing_locations: dict[str, Location],
) -> tuple[dict[str, Location], LocationReference | None]:
    """Process location from generated scene and create new location if needed.

    Args:
        location_reference: LocationReference from the generated scene (may be None)
        existing_locations: Dictionary of existing locations

    Returns:
        Tuple of (updated locations dictionary, processed location reference)
    """
    if location_reference is None:
        return existing_locations, None

    updated_locations = existing_locations.copy()

    # Check if location_id is provided and exists
    if location_reference.location_id in existing_locations:
        # Reusing existing location
        existing_loc = existing_locations[location_reference.location_id]
        logger.info(
            f"✓ Reusing existing location: '{existing_loc.name}' (ID: {location_reference.location_id})"
        )
        return updated_locations, location_reference

    # Location ID not found - this shouldn't normally happen as LLM should provide valid IDs
    # But if it does, we'll just return without the location rather than creating incomplete data
    logger.warning(
        f"⚠ Location ID {location_reference.location_id} not found in existing locations. Scene will have no location reference."
    )
    return updated_locations, None


async def generate_opening_scene(
    game_id: str,
    scenario_name: str,
    scenario_details: str,
    characters: list[Character],
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
    scene_id: int = 1,
    art_style: str | None = None,
    scale: str = "any",
) -> tuple[Scene, list[Character], dict[str, Asset], dict[str, Location]]:
    """Generate the opening scene for the chosen scenario.

    Args:
        game_id: Unique identifier for the game (for image storage)
        scenario_name: Name of the chosen scenario
        scenario_details: Full markdown details about the scenario
        characters: List of Character objects in the party (with full stats, inventory, backstory)
        existing_assets: Dictionary of existing assets (NPCs/objects) for consistency
        existing_locations: Dictionary of existing locations for consistency
        scene_id: Scene identifier (default 1 for opening)

    Returns:
        Tuple of (Scene object, Updated character list, Updated assets dictionary, Updated locations dictionary)
    """
    agent = text_agent(GeneratedScene)

    logger.info(f"Generating opening scene for: {scenario_name}")

    # Build detailed character information for context
    character_sheets = "\n\n".join(
        [
            f"**{char.name}**\n"
            f"- Gender: {getattr(char, 'gender', 'unspecified')} (use matching pronouns)\n"
            f"- Stats: Strength {char.strength}, Intelligence {char.intelligence}, Agility {char.agility}\n"
            f"- Health: {char.current_health}/{char.maximum_health}\n"
            f"- Backstory: {char.backstory}\n"
            f"- Appearance: {char.appearance}\n"
            f"- Personality: {char.personality}\n"
            f"- Skills: {', '.join(char.skills) if char.skills else 'None'}\n"
            f"- Inventory: {format_inventory(char.inventory)}"
            for char in characters
        ]
    )

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}
{scene_scale_section(scale)}
PARTY CHARACTER SHEETS:
{character_sheets}

{format_existing_assets(existing_assets)}

{format_existing_locations(existing_locations)}

Generate the opening scene for this adventure. Create an engaging, atmospheric introduction that:

CRITICAL FOR OPENING SCENE:
1. **Ground how the party arrived HERE**: Before anything happens, establish how these characters
   came to be at this exact place — their journey in, why they're passing through, or the ordinary
   moment they are in as the scene opens. Use their backstories for natural connections (travelling
   together, hired on, drawn by rumor, chance). 2-4 sentences of grounded on-ramp so the start feels
   lived-in and earned, not dropped in cold.

2. **Ease into the situation — do NOT cold-open with a quest-giver**: Avoid starting with an elder,
   mayor, or official who immediately tells the party to go somewhere and do something — that feels
   forced. Let the place and a natural moment draw them in first: something they notice, overhear,
   stumble into, or are asked in passing. The hook should emerge from lived experience, not an
   exposition dump.

3. **Introduce the goal naturally and at the right pace**: Players don't know the scenario details,
   so reveal the objective gradually through what they see and do — not a briefing. Give a reason to
   care and a direction, but a smaller, closer hook first is good; let any larger stakes surface over
   time rather than all at once.

4. **Set the scene and mood**: Create atmosphere with vivid descriptions
   - Describe the environment, sounds, smells, lighting
   - Build tension or intrigue appropriate to the scenario

5. **Introduce party members naturally**: Reference characters by name, appearance, and personality
   - Show their personalities through brief actions or reactions
   - Consider their backstories when describing how they interact
   - Be aware of their stats, health, skills, and inventory

6. **End with a clear hook**: Give players an immediate decision or action to take

REMEMBER: This is the first thing players experience. They need to understand:
- Who they are (as a group)
- Where they are
- What their goal/quest is
- What they should do next

{build_scene_generation_prompt_rules()}

Make the scene immersive, clear, and exciting!
"""

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Log generated scene details
    logger.info(f"=== Opening Scene Generated (Scene {scene_id}) ===")
    logger.info(
        f"Generated scene model: {generated.model_dump_json(indent=2, exclude={'scene_text'})}"
    )

    # Apply character updates from the scene
    updated_characters = apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread (local backend).
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(shared_image_generator)

    asset_task = asyncio.to_thread(
        _process_scene_assets,
        game_id,
        scene_id,
        generated.assets_present,
        existing_assets,
        generator,
        art_style,
    )

    # Wait for asset processing to complete so we have asset images for scene generation
    updated_assets, visible_asset_ids = await asset_task
    if isinstance(updated_assets, Exception):
        logger.error(f"Asset processing failed: {updated_assets}")
        updated_assets = existing_assets
        visible_asset_ids = []

    # Reference only the art-directed in-frame subjects (reference isolation) so the
    # image fits the beat and other figures don't copy a referenced character's look.
    image_ref_ids, subject_directions = resolve_image_subject_ids(
        generated.image_subjects, updated_assets, visible_asset_ids
    )

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation (generator acquired above)
    image_task = asyncio.to_thread(
        generate_scene_image_sync,
        generator,
        game_id,
        scene_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        image_ref_ids,
        art_style,
        subject_directions,
    )

    voiceover_task = generate_scene_voiceover(
        game_id,
        scene_id,
        generated.scene_text,
        updated_characters,
        updated_assets,
        generated.narration_segments,
    )

    # Wait for both to complete
    image_file_path, voiceover_file_path = await asyncio.gather(
        image_task, voiceover_task, return_exceptions=True
    )

    # Handle exceptions
    if isinstance(image_file_path, BaseException):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, BaseException):
        logger.error(f"Voiceover generation failed: {voiceover_file_path}")
        voiceover_file_path = None

    # Convert to web paths if generated
    image_web_path = None
    if image_file_path:
        image_web_path = f"/static/scenes/{game_id}/{image_file_path.name}"

    voiceover_web_path = None
    if voiceover_file_path:
        voiceover_web_path = f"/static/voiceovers/{game_id}/{voiceover_file_path.name}"

    scene = _make_scene(
        scene_id,
        generated.scene_text,
        generated.prompt,
        image_web_path,
        generated.game_status,
        visible_asset_ids,
        health_changes=generated.health_changes,
        inventory_changes=generated.inventory_changes,
        skill_changes=generated.skill_changes,
        stat_changes=generated.stat_changes,
    )
    scene.voiceover_path = voiceover_web_path
    scene.location_reference = processed_location_ref
    return scene, updated_characters, updated_assets, updated_locations


async def generate_next_scene(
    game_id: str,
    scenario_name: str,
    scenario_details: str,
    characters: list[Character],
    last_scene_id: int,
    player_action: str | None,
    conversation_history: list[dict],
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
    art_style: str | None = None,
    scale: str = "any",
) -> tuple[Scene, list[Character], dict[str, Asset], dict[str, Location]]:
    """Generate the next scene based on the story so far and player action.

    Args:
        game_id: Unique identifier for the game (for image storage)
        scenario_name: Name of the scenario
        scenario_details: Full markdown details about the scenario
        characters: List of Character objects in the party (with full stats, inventory, backstory)
        last_scene_id: ID of the previous scene
        player_action: The player's response to the last prompt
        conversation_history: List of dicts containing scene_text, prompt, and player_action for full context
        existing_assets: Dictionary of existing assets (NPCs/objects) for consistency
        existing_locations: Dictionary of existing locations for consistency

    Returns:
        Tuple of (Scene object, Updated character list, Updated assets dictionary, Updated locations dictionary)
    """
    agent = text_agent(GeneratedScene)

    next_id = last_scene_id + 1
    logger.info(f"Generating scene {next_id} for: {scenario_name}")

    # Build detailed character information for context
    # IMPORTANT: These are CURRENT states as they ENTER this scene, not changes to apply
    character_sheets = "\n\n".join(
        [
            f"**{char.name}** (Current State Entering Scene {next_id})\n"
            f"- Gender: {getattr(char, 'gender', 'unspecified')} (use matching pronouns)\n"
            f"- Stats: Strength {char.strength}, Intelligence {char.intelligence}, Agility {char.agility}\n"
            f"- Health: {char.current_health}/{char.maximum_health} (this is their CURRENT health, don't subtract from it again)\n"
            f"- Backstory: {char.backstory}\n"
            f"- Appearance: {char.appearance}\n"
            f"- Personality: {char.personality}\n"
            f"- Skills: {', '.join(char.skills) if char.skills else 'None'}\n"
            f"- Inventory: {format_inventory(char.inventory)} (current items, don't remove again unless used in THIS scene)"
            for char in characters
        ]
    )

    # Build complete conversation history showing the full exchange between DM and players
    history_context = ""
    if conversation_history:
        history_parts = []
        for entry in conversation_history:
            scene_part = f"**Scene {entry['scene_id']}:**\n{entry['scene_text']}"

            # Add the prompt that was given to the player
            if entry.get("prompt"):
                prompt_obj = entry["prompt"]
                target = (
                    f"{prompt_obj.target_character}"
                    if prompt_obj.target_character
                    else "Party"
                )
                prompt_type_icon = (
                    "🎲"
                    if prompt_obj.type == "dice_check"
                    else "💬"
                    if prompt_obj.type == "dialogue"
                    else "⚔️"
                )
                # Include dice type for dice checks so the AI knows what die was rolled
                prompt_detail = prompt_obj.prompt_text
                if prompt_obj.type == "dice_check" and prompt_obj.dice_type:
                    prompt_detail = (
                        f"{prompt_obj.prompt_text} (roll {prompt_obj.dice_type})"
                    )

                scene_part += (
                    f"\n\n{prompt_type_icon} **DM prompts {target}:** {prompt_detail}"
                )

            # Add the player's response
            if entry.get("player_action"):
                scene_part += f"\n\n**Player responded:** {entry['player_action']}"

            history_parts.append(scene_part)

        history_context = "FULL CONVERSATION HISTORY:\n\n" + "\n\n---\n\n".join(
            history_parts
        )

    # Current action context (this is the response to the last prompt)
    action_context = (
        f"\n\nCURRENT PLAYER ACTION: {player_action}" if player_action else ""
    )

    # Check if the last prompt was a dice_check to include resolution rules
    dice_check_rules = ""
    if conversation_history and len(conversation_history) > 0:
        last_entry = conversation_history[-1]
        if last_entry.get("prompt") and last_entry["prompt"].type == "dice_check":
            dice_check_rules = f"\n\n{build_dice_check_resolution_rules()}"

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}
{scene_scale_section(scale)}
PARTY CHARACTER SHEETS:
{character_sheets}

{format_existing_assets(existing_assets)}

{format_existing_locations(existing_locations)}

{history_context}{action_context}{dice_check_rules}

Generate the next scene in this adventure. The scene should:
1. Respond naturally to the player's action (if provided)
2. Advance the story toward the main quest
3. Introduce new challenges, discoveries, or NPCs as appropriate
4. Maintain tension and engagement
5. Reference party members when relevant (consider their backstories, appearance, personality, skills, and capabilities)
6. Be aware of their CURRENT stats, health, skills, and inventory - check if they actually have items they try to use
7. Adjust challenge difficulty based on party's current health and capabilities
8. Create opportunities for characters to use their unique skills

{build_impossible_action_rules()}

{build_scene_generation_prompt_rules()}

Continue the adventure!
"""

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Log generated scene details
    logger.info(f"=== Next Scene Generated (Scene {next_id}) ===")
    logger.info(
        f"Generated scene model: {generated.model_dump_json(indent=2, exclude={'scene_text'})}"
    )

    # Apply character updates from the scene
    updated_characters = apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread (local backend).
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(shared_image_generator)

    asset_task = asyncio.to_thread(
        _process_scene_assets,
        game_id,
        next_id,
        generated.assets_present,
        existing_assets,
        generator,
        art_style,
    )

    # Wait for asset processing to complete so we have asset images for scene generation
    updated_assets, visible_asset_ids = await asset_task
    if isinstance(updated_assets, Exception):
        logger.error(f"Asset processing failed: {updated_assets}")
        updated_assets = existing_assets
        visible_asset_ids = []

    # Reference only the art-directed in-frame subjects (reference isolation) so the
    # image fits the beat and other figures don't copy a referenced character's look.
    image_ref_ids, subject_directions = resolve_image_subject_ids(
        generated.image_subjects, updated_assets, visible_asset_ids
    )

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation (generator acquired above)
    image_task = asyncio.to_thread(
        generate_scene_image_sync,
        generator,
        game_id,
        next_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        image_ref_ids,
        art_style,
        subject_directions,
    )

    voiceover_task = generate_scene_voiceover(
        game_id,
        next_id,
        generated.scene_text,
        updated_characters,
        updated_assets,
        generated.narration_segments,
    )

    # Wait for both to complete
    image_file_path, voiceover_file_path = await asyncio.gather(
        image_task, voiceover_task, return_exceptions=True
    )

    # Handle exceptions
    if isinstance(image_file_path, BaseException):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, BaseException):
        logger.error(f"Voiceover generation failed: {voiceover_file_path}")
        voiceover_file_path = None

    # Convert to web paths if generated
    image_web_path = None
    if image_file_path:
        image_web_path = f"/static/scenes/{game_id}/{image_file_path.name}"

    voiceover_web_path = None
    if voiceover_file_path:
        voiceover_web_path = f"/static/voiceovers/{game_id}/{voiceover_file_path.name}"

    scene = _make_scene(
        next_id,
        generated.scene_text,
        generated.prompt,
        image_web_path,
        generated.game_status,
        visible_asset_ids,
        health_changes=generated.health_changes,
        inventory_changes=generated.inventory_changes,
        skill_changes=generated.skill_changes,
        stat_changes=generated.stat_changes,
    )
    scene.voiceover_path = voiceover_web_path
    scene.location_reference = processed_location_ref
    return scene, updated_characters, updated_assets, updated_locations

"""
Handles content generation via external APIs.

This file currently contains placeholder deterministic generators. Replace
these with real model calls (Gemini, OpenAI, etc.) later.
"""

import asyncio
import logging
import pathlib
import re
import threading
from typing import Callable, List, Optional

from google.genai.errors import ClientError, ServerError  # type: ignore
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from .config import settings
from .image_backends import ImageGenerator, get_image_generator
from .tts_backends import TTSGenerator, get_tts_generator
from .models import (
    Asset,
    Character,
    GameStatus,
    GeneratedArchetype,
    GeneratedArchetypeList,
    GeneratedCharacter,
    GeneratedCharacterList,
    GeneratedLocationList,
    GeneratedScenarioTemplate,
    GeneratedScene,
    Location,
    LocationReference,
    NarrationSegment,
    RecapResponse,
    ScenarioTemplate,
    Scene,
    SceneSummary,
)
from . import voice_casting

logger = logging.getLogger()  # Use root logger so Uvicorn/NiceGUI picks up logs
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)

# NOTE: google.generativeai high-level configure may not export consistently in current lib version; using provider-based auth only.

IMAGE_DIR = pathlib.Path("webapp/static/characters")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

SCENE_IMAGE_DIR = pathlib.Path("webapp/static/scenes")
SCENE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Lazy initialization of image generator to avoid loading models at import.
# A lock guards the one-time load so concurrent first-callers don't load twice.
_image_generator: Optional[ImageGenerator] = None
_image_generator_lock = threading.Lock()


def _get_image_generator() -> ImageGenerator:
    """Get or initialize the image generator singleton.

    Thread-safe: the model load is expensive and may be triggered from a worker
    thread (asyncio.to_thread) so it never blocks the web server's event loop.
    The lock prevents two concurrent first-callers from loading the model twice
    (which would OOM the GPU).
    """
    global _image_generator
    if _image_generator is None:
        with _image_generator_lock:
            if _image_generator is None:
                _image_generator = get_image_generator()
    return _image_generator


def warm_up_models() -> None:
    """Eagerly load the image + TTS models (blocking). Call from a background
    thread at startup so the first in-game request doesn't pay the load cost on
    the event loop. Safe to call more than once; failures are non-fatal.
    """
    try:
        _get_image_generator()
    except Exception as e:
        logger.warning(f"Image model warmup failed (will retry on first use): {e}")
    try:
        _get_tts_generator()
    except Exception as e:
        logger.warning(f"TTS warmup failed (will retry on first use): {e}")


def _get_text_model():
    """Build the pydantic-ai text model for the configured LLM_PROVIDER.

    - "gemini": Google Gemini (GoogleModel + GoogleProvider), model LLM_MODEL.
    - "ollama"/"openai-compatible": any OpenAI-compatible endpoint via LLM_BASE_URL.

    Centralizing this lets every generation call switch providers from config with
    no code change, and keeps the model id (e.g. gemini-3.1-flash-lite) in one place.
    """
    provider = settings.LLM_PROVIDER.lower()
    if provider == "gemini":
        return GoogleModel(
            settings.LLM_MODEL,
            provider=GoogleProvider(api_key=settings.GOOGLE_API_KEY),
        )

    # OpenAI-compatible local servers (Ollama, llama.cpp, LM Studio).
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIModel(
        settings.LLM_MODEL,
        provider=OpenAIProvider(
            base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY
        ),
    )


def _text_agent(output_type):
    """Build a text Agent using the configured model with NATIVE structured output.

    We use NativeOutput (the provider's native JSON-schema response) instead of
    the default tool-calling output. Gemini 3 requires an encrypted
    `thought_signature` to be echoed back on any function-call turn; pydantic-ai's
    validation-retry re-sends the structured-output tool call without it, which
    Gemini rejects with a 400 ("missing thought_signature in functionCall parts").
    Native JSON output produces no function-call parts, so it sidesteps the issue
    entirely — and works the same way for the local OpenAI-compatible backends.
    """
    return Agent(model=_get_text_model(), output_type=NativeOutput(output_type))


# Lazy TTS generator singleton. `_UNSET` distinguishes "not built yet" from a
# cached None (TTS disabled or failed to initialize — narration is optional, so
# we don't retry every scene).
_UNSET = object()
_tts_generator: object = _UNSET


def _get_tts_generator() -> Optional[TTSGenerator]:
    """Get or initialize the TTS generator singleton (None if unavailable)."""
    global _tts_generator
    if _tts_generator is _UNSET:
        try:
            _tts_generator = get_tts_generator()
        except Exception as e:
            logger.error(f"TTS initialization failed ({e}); narration disabled")
            _tts_generator = None
    return _tts_generator  # type: ignore[return-value]


VOICEOVER_DIR = pathlib.Path("webapp/static/voiceovers")
VOICEOVER_DIR.mkdir(parents=True, exist_ok=True)


# Retry configuration for API calls
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds
RETRY_BACKOFF = 2  # exponential backoff multiplier


async def retry_on_overload(func: Callable, *args, **kwargs) -> any:
    """Retry async function calls when the API is overloaded.

    Retries on 503 (Service Unavailable) and 429 (Too Many Requests) errors
    with exponential backoff.
    """
    last_exception = None
    delay = RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)
        except (ServerError, ClientError) as e:
            last_exception = e
            # Check if it's a retryable error (503, 429, or 500)
            if hasattr(e, "status_code"):
                status_code = e.status_code
            elif hasattr(e, "code"):
                status_code = e.code
            else:
                # Unknown error format, don't retry
                raise

            if status_code in (503, 429, 500):
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"API error {status_code} on attempt {attempt + 1}/{MAX_RETRIES}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    delay *= RETRY_BACKOFF  # exponential backoff
                else:
                    logger.error(
                        f"API error {status_code} after {MAX_RETRIES} attempts. Giving up."
                    )
                    raise
            else:
                # Non-retryable error (e.g., 400 Bad Request)
                raise
        except Exception as e:
            # Unexpected error, don't retry
            logger.error(f"Unexpected error during API call: {type(e).__name__}: {e}")
            raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def _generate_character_image_sync(
    generator: ImageGenerator,
    concept,
    scenario_name: str,
    game_id: str,
    game_dir: pathlib.Path,
    idx: int,
) -> tuple[str, pathlib.Path | None]:
    """Synchronous helper to generate a single character image.

    Returns: (character_name, image_file_path or None)
    """
    # Build detailed prompt using all character information
    prompt_parts = [
        "Generate a high quality fantasy character portrait.",
        f"Character Name: {concept.name}",
        f"Scenario: {scenario_name}",
        f"Appearance: {concept.appearance}",
        f"Personality: {concept.personality}",
        f"Backstory Context: {concept.backstory}",
    ]

    # Add key inventory items that would be visible
    if concept.inventory:
        visible_items = ", ".join(
            concept.inventory[:3]
        )  # First 3 items most likely to be visible
        prompt_parts.append(f"Notable Equipment: {visible_items}")

    # Add character stats (let the LLM interpret physical implications)
    prompt_parts.append(
        f"Character Stats (scale 1-20): "
        f"Strength {concept.strength}, "
        f"Intelligence {concept.intelligence}, "
        f"Agility {concept.agility}"
    )

    # Art style is injected by the image backend (FluxKontextImageGenerator.GAME_ART_STYLE)
    # so it applies uniformly to every image type. Do not add style keywords here.

    prompt_img = " ".join(prompt_parts)
    logger.info(f"Generating image for character: {concept.name}")

    # Prepare output path
    safe_name = (
        "".join(c for c in concept.name if c.isalnum() or c in " _-")
        .rstrip()
        .replace(" ", "_")
    )
    filename = f"{idx:02d}_{safe_name or 'character'}.png"
    output_path = game_dir / filename

    # Generate using backend
    image_file_path = generator.generate_character_image(
        prompt=prompt_img,
        output_path=output_path,
    )

    return concept.name, image_file_path


def _prioritize_scene_references(
    assets: dict[str, Asset],
    visible_asset_ids: List[str],
    max_refs: int,
) -> tuple[List[pathlib.Path], list, list]:
    """Choose which visible assets become scene image references, best-first, capped.

    Klein downsizes ALL references together to fit VRAM, so a crowded reference
    list makes every face a low-res thumbnail. We keep identity-bearing references
    first — party heroes, then NPCs, then objects — and cap the count, so the
    characters that must stay recognizable each get a higher resolution and
    objects/extras are dropped rather than dragging everyone down.

    Returns (reference_files, kept_assets, dropped_assets).
    """

    def priority(aid: str) -> int:
        asset = assets.get(aid)
        if asset is None:
            return 3
        if str(aid).startswith("player_"):  # party hero — most important
            return 0
        if getattr(asset, "type", None) == "npc":
            return 1
        return 2  # object

    candidates = []
    for aid in visible_asset_ids or []:
        asset = assets.get(aid)
        if not asset or not asset.image_path:
            continue
        ref_file = pathlib.Path("webapp") / asset.image_path.lstrip("/")
        if not ref_file.exists():
            logger.warning(f"Reference image not found, skipping: {ref_file}")
            continue
        candidates.append((priority(aid), asset, ref_file))

    # Stable sort keeps the scene's original ordering within each priority band.
    candidates.sort(key=lambda t: t[0])
    kept = candidates[: max(0, max_refs)]
    dropped = candidates[max(0, max_refs) :]
    return (
        [ref_file for _, _, ref_file in kept],
        [asset for _, asset, _ in kept],
        [asset for _, asset, _ in dropped],
    )


def _generate_scene_image_sync(
    generator: ImageGenerator,
    game_id: str,
    scene_id: int,
    scene_text: str,
    visual_description: str,
    characters: List[Character],
    scenario_name: str,
    game_status: GameStatus = "ongoing",
    assets: dict[str, Asset] = None,
    visible_asset_ids: List[str] = None,
    art_style: Optional[str] = None,
) -> pathlib.Path | None:
    """Synchronous helper to generate a scene image.

    Args:
        generator: ImageGenerator backend instance
        game_id: Game identifier for directory structure
        scene_id: Scene number
        scene_text: The narrative text of the scene
        visual_description: Detailed visual description for image composition
        characters: List of characters in the party (for reference images)
        scenario_name: Name of the scenario
        game_status: Status of the game ('ongoing', 'completed', 'failed')
        assets: Dictionary of all assets (NPCs and objects)
        visible_asset_ids: List of asset IDs that should appear in this scene

    Returns:
        Path to saved scene image or None if generation failed
    """
    # Create scene directory for this game
    scene_dir = SCENE_IMAGE_DIR / game_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Build a concise visual prompt from the authoritative visual_description.
    # The structured Gemini-style headers that were here before were ignored by
    # FLUX (CLIP truncated them before reaching the actual scene content) and
    # added no value to Gemini either. Both backends work better with a clean
    # visual description they can actually encode.
    mood = ""
    if game_status == "completed":
        mood = " Triumphant, uplifting atmosphere."
    elif game_status == "failed":
        mood = " Somber, dramatic defeat atmosphere."

    prompt_text = f"{visual_description} {scenario_name} fantasy setting.{mood}"

    # Collect reference images for character/asset consistency — prioritized and
    # capped so each surviving reference renders at a higher resolution (Klein
    # scales all references down together to fit VRAM).
    reference_images: List[pathlib.Path] = []
    if assets and visible_asset_ids:
        reference_images, kept, dropped = _prioritize_scene_references(
            assets, visible_asset_ids, settings.IMAGE_MAX_SCENE_REFERENCES
        )
        for a in kept:
            logger.info(f"✓ Reference [{a.type}]: {a.name}")
        if dropped:
            logger.info(
                f"↓ Dropped {len(dropped)} lower-priority reference(s) to keep "
                f"resolution high: {', '.join(a.name for a in dropped)}"
            )

    logger.info(
        f"Generating scene image for scene {scene_id} with {len(reference_images)} reference images"
    )

    # Prepare output path
    filename = f"scene_{scene_id:03d}.png"
    output_path = scene_dir / filename

    # Generate using backend
    image_file_path = generator.generate_scene_image(
        prompt=prompt_text,
        output_path=output_path,
        reference_images=reference_images if reference_images else None,
        art_style=art_style,
    )

    return image_file_path


def _resolve_segment_voices(
    narration_segments: List["NarrationSegment"],
    characters: List[Character],
    assets: dict[str, Asset],
) -> list:
    """Map narration segments to (voice_id, text), casting/persisting NPC voices.

    Narrator uses the configured default voice. NPCs get a stable voice stored on
    their asset (persisted across scenes); minor unnamed speakers get a deterministic
    voice by name. PLAYER (party) characters are never given a distinct voice — their
    words belong to the players, so any line the model wrongly attributed to a party
    member is read plainly in the narrator's voice.
    """
    narrator = settings.KOKORO_VOICE
    # Only NPC/asset voices are reserved; party members are intentionally unvoiced.
    taken = set(filter(None, [a.assigned_voice for a in assets.values()]))

    def is_party(name: str) -> bool:
        return any(_names_match(name, c.name) for c in characters)

    def match_npc(name: str):
        # NPC assets only — skip the player_-prefixed party image-assets.
        for aid, a in assets.items():
            if aid.startswith("player_"):
                continue
            if _npc_names_match(name, a.name):
                return a
        return None

    segments = []
    for seg in narration_segments:
        key = seg.speaker.strip().lower()
        if key in ("", "narrator", "dm", "the narrator", "narration"):
            segments.append((narrator, seg.text))
            continue

        # Player characters are never voiced — read their (mis-attributed) line as narration.
        if is_party(seg.speaker):
            segments.append((narrator, seg.text))
            continue

        npc = match_npc(seg.speaker)
        if npc is not None:
            if not npc.assigned_voice:
                npc.assigned_voice = voice_casting.cast_voice(
                    npc.name, seg.gender, taken=taken, exclude=narrator
                )
                taken.add(npc.assigned_voice)
            segments.append((npc.assigned_voice, seg.text))
        else:
            # Minor / unknown speaker: deterministic by name, not persisted.
            voice = voice_casting.cast_voice(
                seg.speaker, seg.gender, taken=taken, exclude=narrator
            )
            segments.append((voice, seg.text))
    return segments


async def _generate_scene_voiceover(
    game_id: str,
    scene_id: int,
    scene_text: str,
    characters: List[Character],
    assets: dict[str, Asset],
    narration_segments: Optional[List["NarrationSegment"]] = None,
) -> pathlib.Path | None:
    """Generate scene narration via the configured TTS backend.

    Uses the per-speaker `narration_segments` emitted by the scene model (multi-voice)
    when available on a multi-voice backend; otherwise reads the whole scene in the
    single narrator voice. Narration is optional: returns None (never raises) if
    unavailable or failing.
    """
    tts = _get_tts_generator()
    if tts is None:
        return None

    voiceover_dir = VOICEOVER_DIR / game_id
    voiceover_dir.mkdir(parents=True, exist_ok=True)
    voiceover_file_path = voiceover_dir / f"scene_{scene_id:03d}.wav"

    logger.info(f"Generating voiceover for scene {scene_id} in game {game_id}")

    # Single-voice backend, multi-voice disabled, or no segments: read as narrator.
    if not getattr(tts, "supports_multivoice", False) or not narration_segments:
        return await asyncio.to_thread(tts.synthesize, scene_text, voiceover_file_path)

    segments = _resolve_segment_voices(narration_segments, characters, assets)
    if not segments:
        return await asyncio.to_thread(tts.synthesize, scene_text, voiceover_file_path)
    return await asyncio.to_thread(
        tts.synthesize_segments, segments, voiceover_file_path
    )


async def generate_recap_voiceover(
    game_id: str, scene_id: int, recap_text: str
) -> pathlib.Path | None:
    """Narrate a story recap via the configured TTS backend (single narrator voice).

    Returns None if TTS is unavailable/disabled or synthesis fails.
    """
    tts = _get_tts_generator()
    if tts is None:
        return None
    voiceover_dir = VOICEOVER_DIR / game_id
    voiceover_dir.mkdir(parents=True, exist_ok=True)
    path = voiceover_dir / f"recap_scene_{scene_id}.wav"
    logger.info(f"Generating recap voiceover for scene {scene_id} in game {game_id}")
    return await asyncio.to_thread(tts.synthesize, recap_text, path)


async def generate_characters(
    game_id: str,
    scenario_name: str,
    num_characters: int = 6,
    scenario_details: Optional[str] = None,
) -> List[Character]:
    """Generates a list of characters with stats and portraits for a given scenario.

    Args:
        game_id: Unique identifier for the game (used for image storage)
        scenario_name: Name of the scenario
        num_characters: Number of characters to generate
        scenario_details: Optional markdown details about the scenario (setting, plot, NPCs, etc.)
    """
    # Step 1: Generate character concepts (name, stats)

    logger.info(f"Generating {num_characters} characters for scenario: {scenario_name}")
    agent = _text_agent(GeneratedCharacterList)

    # Build context-aware prompt
    context_section = ""
    if scenario_details:
        context_section = f"""
SCENARIO CONTEXT:
{scenario_details}

Use the above scenario details to inform your character creation. Characters should fit naturally into this world, 
relate to the plot/quest, and potentially interact with the mentioned NPCs and locations.

"""

    prompt = f"""
Generate {num_characters} unique and compelling fantasy characters for a D&D-style adventure named '{scenario_name}'.
Provide a diverse set of classic fantasy archetypes.

{context_section}For each character, provide:
- name: Full name with title/nickname
- strength, intelligence, agility: Stats between 1-20
- backstory: 2-3 sentences about their history and what drives them (reference scenario context when relevant)
- appearance: 2-3 sentences describing their physical look, clothing, and distinctive features
- personality: 2-3 sentences about their personality traits, mannerisms, how they speak, and how they interact with others
- skills: 3-5 specific skills or abilities (e.g., "Swordsmanship", "Lockpicking", "Persuasion", "Arcane Knowledge", "Tracking", "Healing", "Stealth")
- inventory: 3-5 items they carry (weapons, tools, magical items, personal effects)

Make each character distinctive and memorable with rich details that fit the scenario.
Skills should match the character's background, stats, and archetype.
"""
    concept_result = await retry_on_overload(agent.run, prompt)
    generated_concepts = concept_result.output.characters
    logger.info(f"=== {num_characters} Characters Generated ===")
    logger.info(f"Characters: {concept_result.output.model_dump_json(indent=2)}")

    # Step 2: Synchronous image generation using backend abstraction
    # Create game-specific directory for character images
    game_dir = IMAGE_DIR / game_id
    game_dir.mkdir(parents=True, exist_ok=True)

    generator = await asyncio.to_thread(_get_image_generator)

    # Step 3: Generate images asynchronously (in thread pool to avoid blocking event loop)
    final_characters: List[Character] = []

    # Generate all images concurrently using asyncio.to_thread
    image_tasks = [
        asyncio.to_thread(
            _generate_character_image_sync,
            generator,
            concept,
            scenario_name,
            game_id,
            game_dir,
            idx,
        )
        for idx, concept in enumerate(generated_concepts, start=1)
    ]

    logger.info("Starting concurrent image generation for all characters...")
    image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
    logger.info("All image generation tasks completed")

    # Build final character list with image paths
    for concept, (char_name, image_file_path) in zip(generated_concepts, image_results):
        if isinstance(image_file_path, Exception):
            logger.error(
                f"Image generation failed for {concept.name}: {image_file_path}"
            )
            image_file_path = None

        new_char = Character(
            name=concept.name,
            strength=concept.strength,
            intelligence=concept.intelligence,
            agility=concept.agility,
            maximum_health=100,
            current_health=100,
            backstory=concept.backstory,
            appearance=concept.appearance,
            personality=concept.personality,
            skills=concept.skills,
            inventory=concept.inventory,
            image_path=(
                f"/static/characters/{game_id}/{image_file_path.name}"
                if image_file_path
                else None
            ),
        )
        final_characters.append(new_char)

    logger.info(f"Final character list: {[c.name for c in final_characters]}")
    return final_characters


async def generate_archetypes(
    scenario_name: str,
    scenario_details: Optional[str] = None,
    num_archetypes: int = 5,
) -> List[GeneratedArchetype]:
    """Generate scenario-tailored hero archetypes for players to choose from.

    This is a cheap text-only call. Portraits and full lore are only generated
    later, after a player commits to one of these archetypes.

    Args:
        scenario_name: Name of the chosen scenario
        scenario_details: Markdown DM notes (setting/plot/quest/NPCs) for context
        num_archetypes: How many distinct archetypes to offer

    Returns:
        List of GeneratedArchetype options.
    """
    logger.info(
        f"Generating {num_archetypes} archetypes for scenario: {scenario_name}"
    )
    agent = _text_agent(GeneratedArchetypeList)

    context_section = ""
    if scenario_details:
        context_section = f"""
SCENARIO CONTEXT:
{scenario_details}

"""

    prompt = f"""
You are designing playable hero archetypes for a D&D-style adventure named '{scenario_name}'.
{context_section}Generate {num_archetypes} DISTINCT hero archetypes that fit THIS specific scenario's
setting, tone, and quest. Avoid generic filler — each should feel like it belongs in this world and
covers a different party role (e.g. front-line fighter, stealth/skill specialist, arcane/support,
face/social, ranged/scout). Make the set complementary so any pick leads to a fun, viable hero.

For each archetype provide:
- name: An evocative title fitting the scenario (NOT a personal name — a role identity, e.g. 'The Ashen Warden')
- role: A short party-function label (e.g. 'Front-line bruiser', 'Stealth & sabotage', 'Arcane support')
- hook: ONE enticing sentence describing the fantasy of playing this archetype in this scenario
- concept: A vivid VISUAL concept for the portrait — costume, armor/clothing, signature equipment,
  silhouette and overall vibe (2-3 sentences). Describe gear and style ONLY. Do NOT describe facial
  features, age, or ethnicity — the player's own face will be used for the portrait.

Make them diverse, scenario-specific, and exciting.
"""
    result = await retry_on_overload(agent.run, prompt)
    archetypes = result.output.archetypes
    logger.info(f"=== {len(archetypes)} Archetypes Generated ===")
    logger.info(f"Archetypes: {[a.name for a in archetypes]}")
    return archetypes


def _build_hero_portrait_prompt(
    hero_name: str, archetype: GeneratedArchetype, has_photo: bool
) -> str:
    """Build the portrait prompt for a hero, with or without a player photo."""
    if has_photo:
        return (
            f"Reimagine the person in the reference photograph as {hero_name}, "
            f"{archetype.concept} "
            "Keep their facial features, face shape, skin tone and hairstyle clearly "
            "recognizable so the hero looks like the same person."
        )
    return f"{hero_name}. {archetype.concept}"


def _generate_hero_portrait_sync(
    generator: ImageGenerator,
    hero_name: str,
    archetype: GeneratedArchetype,
    art_style: str,
    photo_path: Optional[pathlib.Path],
    game_dir: pathlib.Path,
    idx: int,
) -> pathlib.Path | None:
    """Synchronous helper: render one hero portrait (optionally from a photo)."""
    has_photo = bool(photo_path and pathlib.Path(photo_path).exists())
    prompt_img = _build_hero_portrait_prompt(hero_name, archetype, has_photo)

    safe_name = (
        "".join(c for c in hero_name if c.isalnum() or c in " _-")
        .rstrip()
        .replace(" ", "_")
    )
    filename = f"{idx:02d}_{safe_name or 'hero'}.png"
    output_path = game_dir / filename

    return generator.generate_character_image(
        prompt=prompt_img,
        output_path=output_path,
        reference_images=[pathlib.Path(photo_path)] if has_photo else None,
        art_style=art_style,
    )


async def _generate_hero_lore(
    scenario_name: str,
    scenario_details: Optional[str],
    archetype: GeneratedArchetype,
    has_photo: bool,
    custom_name: Optional[str] = None,
    gender: str = "unspecified",
) -> GeneratedCharacter:
    """Generate stats + lore for a single hero fitted to scenario and archetype."""
    agent = _text_agent(GeneratedCharacter)

    context_section = ""
    if scenario_details:
        context_section = f"""
SCENARIO CONTEXT:
{scenario_details}

"""

    name_rule = (
        f"- name: Use EXACTLY this name for the hero: '{custom_name}'"
        if custom_name
        else "- name: Invent a fitting full name (with an optional title/nickname)"
    )
    appearance_rule = (
        "- appearance: Describe their COSTUME, equipment, and bearing to match the archetype concept. "
        + (
            "Do NOT describe facial features, hair or skin — those come from the player's own photo."
            if has_photo
            else "Include distinctive physical and clothing details."
        )
    )
    gender_line = (
        f"- The hero is {gender}. Use {gender} pronouns and gendered language "
        "throughout the backstory, personality, and appearance.\n"
        if gender and gender != "unspecified"
        else ""
    )

    prompt = f"""
Create ONE fully-realized hero for a D&D-style adventure named '{scenario_name}'.
{context_section}The hero is built from this archetype (make the character a specific, named embodiment of it):
- Archetype: {archetype.name} ({archetype.role})
- Fantasy: {archetype.hook}
- Visual concept: {archetype.concept}
{gender_line}
Provide:
{name_rule}
- strength, intelligence, agility: Stats between 1-20 that reflect this archetype's role
- backstory: 2-3 sentences of history and motivation that tie the hero into THIS scenario
- personality: 2-3 sentences on their traits, mannerisms, and how they interact with others
- skills: 3-5 concrete skills matching the archetype and stats
- inventory: 3-4 starting items. Each item needs a PLAIN, non-cryptic name AND a one-sentence
  purpose (what it does and when the player would use it). Favor a few clearly-useful items
  (a signature weapon, a tool tied to one of their skills, one situational item) over opaque
  flavor trinkets, so the player always knows what each item is for.
{appearance_rule}

Make the hero distinctive, memorable, and a natural fit for both the archetype and the scenario.
"""
    result = await retry_on_overload(agent.run, prompt)
    return result.output


async def generate_hero(
    game_id: str,
    scenario_name: str,
    scenario_details: Optional[str],
    archetype: GeneratedArchetype,
    art_style: str,
    player_index: int,
    photo_path: Optional[pathlib.Path] = None,
    custom_name: Optional[str] = None,
    gender: str = "unspecified",
) -> Character:
    """Generate a single hero (lore + portrait) from a chosen archetype.

    Lore (text) and the portrait (image) are generated concurrently. When a
    player photo is provided, the portrait keeps their likeness while restyling
    them as the archetype hero.

    Args:
        game_id: Game identifier (for image storage)
        scenario_name: Name of the scenario
        scenario_details: Markdown DM notes for context
        archetype: The archetype the player chose
        art_style: Art-style key for imagery ("painterly_hero" / "artstation_realism")
        player_index: 1-based index of the player this hero belongs to
        photo_path: Optional path to the player's uploaded photo
        custom_name: Optional player-chosen hero name

    Returns:
        A fully-populated Character.
    """
    has_photo = bool(photo_path and pathlib.Path(photo_path).exists())
    logger.info(
        f"Generating hero for player {player_index}: archetype '{archetype.name}', "
        f"style '{art_style}', photo={'yes' if has_photo else 'no'}"
    )

    game_dir = IMAGE_DIR / game_id
    game_dir.mkdir(parents=True, exist_ok=True)
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(_get_image_generator)

    # Kick off lore + portrait concurrently.
    lore_task = _generate_hero_lore(
        scenario_name, scenario_details, archetype, has_photo, custom_name, gender
    )
    portrait_task = asyncio.to_thread(
        _generate_hero_portrait_sync,
        generator,
        custom_name or archetype.name,
        archetype,
        art_style,
        pathlib.Path(photo_path) if has_photo else None,
        game_dir,
        player_index,
    )

    lore, image_file_path = await asyncio.gather(
        lore_task, portrait_task, return_exceptions=True
    )

    if isinstance(lore, Exception):
        logger.error(f"Hero lore generation failed: {lore}")
        raise lore
    if isinstance(image_file_path, Exception):
        logger.error(f"Hero portrait generation failed: {image_file_path}")
        image_file_path = None

    hero_name = custom_name or lore.name

    photo_web_path = None
    if has_photo:
        photo_web_path = f"/static/photos/{game_id}/{pathlib.Path(photo_path).name}"

    return Character(
        name=hero_name,
        gender=gender if gender in ("male", "female", "nonbinary") else "unspecified",
        strength=lore.strength,
        intelligence=lore.intelligence,
        agility=lore.agility,
        maximum_health=100,
        current_health=100,
        backstory=lore.backstory,
        appearance=lore.appearance,
        personality=lore.personality,
        skills=lore.skills,
        inventory=lore.inventory,
        image_path=(
            f"/static/characters/{game_id}/{image_file_path.name}"
            if image_file_path
            else None
        ),
        archetype=archetype.name,
        player_photo_path=photo_web_path,
    )


async def generate_scenario_template(
    existing_scenarios: List[ScenarioTemplate],
) -> ScenarioTemplate:
    """Generates a complete new scenario template with contrastive prompting.

    Args:
        existing_scenarios: List of all existing scenario templates (both played and unplayed)
                          to ensure the new scenario is distinct and original

    Returns:
        ScenarioTemplate object with name, one_liner, and full DM notes
    """
    agent = _text_agent(GeneratedScenarioTemplate)

    logger.info(
        f"Generating new scenario template (contrasting with {len(existing_scenarios)} existing scenarios)..."
    )

    # Build contrastive prompt with existing scenarios
    contrastive_context = ""
    if existing_scenarios:
        scenario_summaries = []
        for i, scenario in enumerate(
            existing_scenarios[:10], 1
        ):  # Limit to 10 most recent
            scenario_summaries.append(
                f"**Scenario {i}: {scenario.name}**\n"
                f"Hook: {scenario.one_liner}\n"
                f"Times Played: {scenario.times_played}\n"
                f"DM Notes Preview: {scenario.dm_notes[:300]}..."
            )

        contrastive_context = f"""
EXISTING SCENARIOS (create something DISTINCTLY DIFFERENT):

{chr(10).join(scenario_summaries)}

CRITICAL: Your new scenario MUST differ in at least 3 major aspects:
1. Setting type/era (medieval village, space station, underwater city, floating islands, etc.)
2. Quest structure (rescue mission, mystery investigation, treasure hunt, political intrigue, survival, etc.)
3. Antagonist type (villain, natural disaster, curse, rival faction, internal conflict, etc.)
4. Tone/atmosphere (horror, heroic fantasy, noir mystery, comedy, epic, gritty, whimsical, etc.)
5. Core conflict (combat-focused, social/political, exploration, resource management, moral dilemma, etc.)

Analyze the patterns in existing scenarios and deliberately create something novel.
"""

    prompt = f"""
You are a creative fantasy adventure designer creating scenarios for a D&D-style tabletop RPG.

{contrastive_context}

Generate a COMPLETE scenario with the following components:

1. **name**: A compelling, memorable scenario name (3-7 words)
   - Examples: "The Curse of Ravenmoor", "Heist at the Skyport", "Whispers from the Deep"

2. **one_liner**: A short, enticing hook (1-2 sentences) that attracts players WITHOUT spoiling the plot
   - Should create mystery, excitement, or intrigue
   - NO spoilers about villains, plot twists, or endings
   - Examples: 
     * "A cursed village where children vanish under the blood moon"
     * "Rival treasure hunters race to claim an artifact in ancient sky ruins"
     * "Political intrigue threatens to tear apart the kingdom's fragile peace"

3. **setting**: Detailed description of the world/locale (2-4 rich paragraphs)
   - Geography, climate, culture, history
   - Key locations and landmarks
   - Visual and atmospheric details

4. **plot**: The narrative arc and story structure (2-4 paragraphs)
   - Setup, complications, potential paths
   - Key events and decision points
   - How the story could unfold

5. **main_quest**: The primary objective for the party (2-4 paragraphs)
   - What the party needs to accomplish
   - Why it matters
   - Challenges they'll face
   - Possible approaches and outcomes

6. **important_npcs**: Key non-player characters (2-4 paragraphs)
   - Describe 3-5 important NPCs with names, roles, motivations
   - Include allies, neutrals, and antagonists
   - Make them memorable and useful for roleplay

Create a scenario that is fresh, engaging, and distinctly different from existing ones.
Make it playable, fun, and full of opportunities for player choice and creativity.
"""

    result = await retry_on_overload(agent.run, prompt)
    generated = result.output

    # Log generated scenario
    logger.info("=== Scenario Template Generated ===")
    logger.info(f"Name: {generated.name}")
    logger.info(f"One-liner: {generated.one_liner}")

    # Convert to markdown DM notes
    dm_notes = (
        f"## The Setting\n\n{generated.setting}\n\n"
        f"## The Plot\n\n{generated.plot}\n\n"
        f"## Main Quest\n\n{generated.main_quest}\n\n"
        f"## Important NPCs\n\n{generated.important_npcs}\n"
    )

    # Create ScenarioTemplate object
    scenario_template = ScenarioTemplate(
        name=generated.name,
        one_liner=generated.one_liner,
        dm_notes=dm_notes,
    )

    return scenario_template


async def generate_initial_locations(
    scenario_name: str,
    scenario_details: Optional[str] = None,
) -> dict[str, Location]:
    """Generates initial set of locations for the adventure.

    Args:
        scenario_name: Name of the scenario
        scenario_details: Optional markdown details about the scenario

    Returns:
        Dictionary mapping location IDs to Location objects
    """
    agent = _text_agent(GeneratedLocationList)

    logger.info(f"Generating initial locations for scenario: {scenario_name}")

    # Build context-aware prompt
    context_section = ""
    if scenario_details:
        context_section = f"""
SCENARIO CONTEXT:
{scenario_details}

Use the above scenario details to create locations that fit naturally into this world and support the plot/quest.

"""

    prompt = f"""
Generate 4-6 diverse and memorable locations for a fantasy adventure named '{scenario_name}'.
These locations will serve as settings for scenes throughout the adventure.

{context_section}For each location, provide:
- name: Memorable name (e.g., "The Crimson Tavern", "Shadowfen Swamp", "Dragon's Peak Summit")
- location_type: One of: indoor, outdoor, underground, aerial, aquatic, mystical
- description: Extensive physical description (3-5 sentences) including architecture/terrain, materials, scale, layout
- key_features: 3-5 distinctive features that make this location unique and memorable
- atmosphere: The general mood and feel (1-2 sentences)
- lighting_default: Default lighting conditions (e.g., "torch-lit", "moonlit", "bright daylight", "bioluminescent glow")

Make each location visually distinct and narratively useful. Include a mix of location types (indoor/outdoor/etc).
Ensure locations support different story beats: safe havens, dangerous areas, mysterious places, social hubs, etc.
"""
    result = await retry_on_overload(agent.run, prompt)
    generated_locations = result.output.locations

    logger.info(f"=== {len(generated_locations)} Initial Locations Generated ===")
    logger.info(f"Locations: {result.output.model_dump_json(indent=2)}")

    # Convert to Location objects with IDs
    locations_dict: dict[str, Location] = {}
    for gen_loc in generated_locations:
        location = Location(
            name=gen_loc.name,
            location_type=gen_loc.location_type,
            description=gen_loc.description,
            key_features=gen_loc.key_features,
            atmosphere=gen_loc.atmosphere,
            lighting_default=gen_loc.lighting_default,
        )
        locations_dict[location.id] = location
        logger.info(
            f"Created location '{location.name}' (ID: {location.id}, Type: {location.location_type})"
        )

    return locations_dict


# --- Name matching --------------------------------------------------------------
# The scene model refers to party members and NPCs by name, but doesn't always
# reproduce a long or custom name character-for-character. Exact matching then
# silently drops health/inventory updates (character sheet doesn't change) and —
# worse — makes the asset step create a fresh no-photo portrait instead of reusing
# the player's likeness. These helpers match names tolerantly.

_NAME_STOPWORDS = {"the", "of", "and", "a", "an", "de", "von"}


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _name_tokens(s: str) -> set:
    return {t for t in _norm_name(s).split() if t and t not in _NAME_STOPWORDS}


def _names_match(a: str, b: str) -> bool:
    """True if two names plausibly refer to the same character."""
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return False
    if na == nb or na.replace(" ", "") == nb.replace(" ", ""):
        return True
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return False
    # One name's significant tokens fully contained in the other's, e.g.
    # "Kaelia" vs "Kaelia the Spore-Warden".
    return ta <= tb or tb <= ta


def _npc_names_match(a: str, b: str) -> bool:
    """Tolerant match for NPCs/objects: normalized token-set EQUALITY.

    Reunites naming drift that differs only by articles, case, spacing or
    punctuation ("The Echo Child" / "Echo-Child" / "echo child") so a recurring
    NPC keeps ONE asset and one portrait. Unlike `_names_match` (used for party
    members) it does NOT match on token subset, so genuinely different NPCs like
    "Guard" vs "Guard Captain" are kept apart rather than merged.
    """
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return False
    if na == nb or na.replace(" ", "") == nb.replace(" ", ""):
        return True
    ta, tb = _name_tokens(a), _name_tokens(b)
    return bool(ta) and ta == tb


def _resolve_char(name: str, char_dict: dict) -> Optional[Character]:
    """Find the character a change refers to, tolerating name variations."""
    if name in char_dict:
        return char_dict[name]
    for cname, c in char_dict.items():
        if _names_match(name, cname):
            return c
    return None


def _item_name(item) -> str:
    """Item display name, tolerant of legacy bare-string items."""
    return getattr(item, "name", str(item))


def _format_inventory(items) -> str:
    """Render inventory as 'Name (purpose); ...' for prompts ('None' if empty)."""
    if not items:
        return "None"
    parts = []
    for it in items:
        purpose = getattr(it, "purpose", "") or ""
        parts.append(f"{_item_name(it)} ({purpose})" if purpose else _item_name(it))
    return "; ".join(parts)


def _apply_character_updates(
    characters: List[Character], generated_scene
) -> List[Character]:
    """Apply character state updates from explicit changes in the generated scene.

    Args:
        characters: List of current character objects (with image_path, etc.)
        generated_scene: GeneratedScene object with health_changes, inventory_changes, etc.

    Returns:
        Updated list of characters with modifications applied
    """
    # Create a dictionary for quick character lookup by name
    char_dict = {char.name: char for char in characters}

    # Apply health changes
    for health_change in generated_scene.health_changes:
        char = _resolve_char(health_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Health change for unknown character: {health_change.character_name}. Skipping."
            )
            continue

        old_health = char.current_health
        char.current_health = max(
            0,
            min(char.current_health + health_change.health_change, char.maximum_health),
        )

        change_desc = f"Health: {old_health} -> {char.current_health}"
        if health_change.health_change < 0:
            change_desc += (
                f" ({health_change.health_change} damage: {health_change.reason})"
            )
        elif health_change.health_change > 0:
            change_desc += (
                f" (+{health_change.health_change} healing: {health_change.reason})"
            )

        logger.info(f"Updated {char.name}: {change_desc}")

    # Apply inventory changes
    for inv_change in generated_scene.inventory_changes:
        char = _resolve_char(inv_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Inventory change for unknown character: {inv_change.character_name}. Skipping."
            )
            continue

        changes = []
        have = {_item_name(it).strip().lower() for it in char.inventory}

        # Add new items (structured), skipping ones already carried (by name).
        for item in inv_change.items_added:
            key = _item_name(item).strip().lower()
            if key and key not in have:
                char.inventory.append(item)
                have.add(key)
                changes.append(f"+{_item_name(item)}")

        # Remove items by name, tolerating minor naming variation
        # ("lockpicks" vs "Clockwork Lockpicks").
        for name in inv_change.items_removed:
            key = str(name).strip().lower()
            if not key:
                continue
            match = next(
                (
                    it
                    for it in char.inventory
                    if (n := _item_name(it).strip().lower()) == key
                    or key in n
                    or n in key
                ),
                None,
            )
            if match is not None:
                char.inventory.remove(match)
                changes.append(f"-{_item_name(match)}")

        if changes:
            logger.info(f"Updated {char.name} inventory: {', '.join(changes)}")

    # Apply skill changes
    for skill_change in generated_scene.skill_changes:
        char = _resolve_char(skill_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Skill change for unknown character: {skill_change.character_name}. Skipping."
            )
            continue

        changes = []

        # Learn new skills
        for skill in skill_change.skills_learned:
            if skill not in char.skills:
                char.skills.append(skill)
                changes.append(f"Learned: {skill}")

        # Lose skills
        for skill in skill_change.skills_lost:
            if skill in char.skills:
                char.skills.remove(skill)
                changes.append(f"Lost: {skill}")

        if changes:
            logger.info(f"Updated {char.name} skills: {', '.join(changes)}")

    # Apply stat changes (rare)
    for stat_change in generated_scene.stat_changes:
        char = _resolve_char(stat_change.character_name, char_dict)
        if char is None:
            logger.warning(
                f"Stat change for unknown character: {stat_change.character_name}. Skipping."
            )
            continue

        changes = []

        if stat_change.strength_change != 0:
            old_str = char.strength
            char.strength = max(0, min(20, char.strength + stat_change.strength_change))
            changes.append(f"STR: {old_str} -> {char.strength}")

        if stat_change.intelligence_change != 0:
            old_int = char.intelligence
            char.intelligence = max(
                0, min(20, char.intelligence + stat_change.intelligence_change)
            )
            changes.append(f"INT: {old_int} -> {char.intelligence}")

        if stat_change.agility_change != 0:
            old_agi = char.agility
            char.agility = max(0, min(20, char.agility + stat_change.agility_change))
            changes.append(f"AGI: {old_agi} -> {char.agility}")

        if changes:
            logger.info(
                f"Updated {char.name} stats: {', '.join(changes)} (Reason: {stat_change.reason})"
            )

    return characters


def _make_scene(
    scene_id: int,
    scene_text: str,
    prompt,
    image_path: str | None = None,
    game_status: GameStatus = "ongoing",
    visible_asset_ids: List[str] = None,
    health_changes: List = None,
    inventory_changes: List = None,
    skill_changes: List = None,
    stat_changes: List = None,
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


def _format_existing_assets(assets: dict[str, Asset]) -> str:
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


def _format_existing_locations(locations: dict[str, Location]) -> str:
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


def _process_scene_assets(
    game_id: str,
    scene_id: int,
    assets_present: list,
    existing_assets: dict[str, Asset],
    generator: ImageGenerator,
    art_style: Optional[str] = None,
) -> tuple[dict[str, Asset], List[str]]:
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
    asset_dir = pathlib.Path(f"webapp/static/assets/{game_id}")
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
                    _names_match(asset_ref.name, asset.name)
                    if is_party
                    else _npc_names_match(asset_ref.name, asset.name)
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
                asset_image_path = _generate_asset_image_sync(
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
    location_reference: Optional[LocationReference],
    existing_locations: dict[str, Location],
) -> tuple[dict[str, Location], Optional[LocationReference]]:
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


def _generate_asset_image_sync(
    generator: ImageGenerator,
    game_id: str,
    asset_id: str,
    asset_name: str,
    asset_description: str,
    art_style: Optional[str] = None,
) -> pathlib.Path | None:
    """Generate and save an image for an asset (NPC or object) via the local backend.

    Uses the same local image generator (FLUX) and per-game art style as character
    and scene images, so all imagery is produced on-device and Gemini stays
    text-only. The image is used as a reference for scene consistency.

    Args:
        generator: The configured local image backend
        game_id: Game identifier
        asset_id: Asset identifier
        asset_name: Name of the asset
        asset_description: Physical description
        art_style: Art-style key for the game

    Returns:
        Path to the saved image file, or None if generation failed
    """
    # Create asset directory for this game
    asset_dir = pathlib.Path(f"webapp/static/assets/{game_id}")
    asset_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"{asset_name}. {asset_description} "
        "Single clear, centered subject, minimal background, no text or labels."
    )
    logger.info(f"Generating asset image for: {asset_name}")

    output_path = asset_dir / f"{asset_id}.png"
    return generator.generate_character_image(
        prompt=prompt,
        output_path=output_path,
        art_style=art_style,
    )


def _build_scene_generation_prompt_rules() -> str:
    """Returns the common rules section for scene generation prompts."""
    return """
After the scene narrative, provide a prompt for player interaction. The prompt should be:
- Either for the entire party or a specific character
- One of three types:
  * dialogue: Character should speak (player enters text in quotes)
  * action: Player describes what they do (for simple, non-risky actions)
  * dice_check: Player must roll dice (for ANY risky action, test of skill, or combat)

CRITICAL RULES FOR PROMPT TYPES - VARY THE PROMPTS:
- DO NOT use dice_check for every scene - mix dialogue, action, and dice_check
- Use dialogue when: characters need to talk to NPCs, negotiate, roleplay conversations
- Use action for: simple tasks, exploration without immediate danger, planning, easy skill checks, or coordinated team efforts
- Use dice_check for: combat, risky actions, difficult skill checks, life-or-death situations

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
For dice_check prompts specifically:
  * ALWAYS specify target_character (single name) OR target_characters (array of names)
  * NEVER use target_character: null for dice_check prompts - this causes confusion about who rolls
  * If one character should attempt the check, use target_character with their name
    - Example: "target_character": "Theron the Brave" (use the ACTUAL character name from the party)
  * If multiple specific characters should roll simultaneously, use target_characters array
    - Example: "target_characters": ["ActualName1", "ActualName2"] (use ACTUAL names from the party roster)
  * If the situation is "any character could volunteer," pick the most suitable character based on their skills/stats
    - Example: For persuasion, pick the character with highest Intelligence or relevant skills
  * Make your prompt_text match your targeting:
    - Single target: "[CharacterName], roll d10 to persuade the elder" (use their actual name)
    - Multiple targets: "[Name1] and [Name2], both roll d6 for your attacks" (use their actual names)

For action prompts:
  * Can use target_character for single character actions
  * Can use target_characters for coordinated multi-character actions (e.g., "[Name1] searches left, [Name2] searches right" - use actual party member names)
  * Use target_character: null when the entire party acts together

For dialogue prompts:
  * Use target_character with a name to address a specific character
  * Use target_character: null when the entire party speaks/responds
  
CRITICAL: For ANY dice_check prompt, you MUST set either target_character (string) or target_characters (array). Never null.

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
- Set game_status to "completed" ONLY when the main quest objective is definitively achieved:
  * The primary goal from scenario_details must be fulfilled
  * This ends the game with a victory screen
  * The scene should describe their triumph and resolution
  * Still provide a prompt field (required by schema), but it won't be shown to players
- Do NOT set completed prematurely - minor victories are still "ongoing"
- Do NOT set completed just because things are going well - only when main quest is done

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
    "type": "dialogue" | "action" | "dice_check",
    "dice_type": "d6" | "d10" (only if type is dice_check, always single die roll),
    "target_character": "Character Name" (REQUIRED for single dice_check, optional for action/dialogue) or null,
    "target_characters": ["Char1", "Char2"] (for multi-character dice_check or action) or null,
    "prompt_text": "The question or instruction for the player(s)"
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


def _build_impossible_action_rules() -> str:
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


def _build_dice_check_resolution_rules() -> str:
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


async def generate_opening_scene(
    game_id: str,
    scenario_name: str,
    scenario_details: str,
    characters: List[Character],
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
    scene_id: int = 1,
    art_style: Optional[str] = None,
) -> tuple[Scene, List[Character], dict[str, Asset], dict[str, Location]]:
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
    agent = _text_agent(GeneratedScene)

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
            f"- Inventory: {_format_inventory(char.inventory)}"
            for char in characters
        ]
    )

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}

PARTY CHARACTER SHEETS:
{character_sheets}

{_format_existing_assets(existing_assets)}

{_format_existing_locations(existing_locations)}

Generate the opening scene for this adventure. Create an engaging, atmospheric introduction that:

CRITICAL FOR OPENING SCENE:
1. **Establish how the party came together**: Briefly narrate how these characters met and why they're working together
   - Use their backstories to create natural connections (shared goals, chance meeting, hired together, etc.)
   - Make it believable based on their personalities and backgrounds
   - Keep this part concise (2-3 sentences)

2. **Explain the current situation**: Describe what brings them to this specific moment
   - Where are they? (refer to the setting from scenario details)
   - Why are they here? (connect to the main quest)
   - What immediate situation do they find themselves in?

3. **Introduce the main quest naturally**: Players don't know the scenario details, so reveal the quest/goal through the narrative
   - Don't assume players know anything from the DM notes
   - Make the objective clear through dialogue, events, or circumstances
   - Give them a reason to care and a clear direction

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

{_build_scene_generation_prompt_rules()}

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
    updated_characters = _apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread (local backend).
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(_get_image_generator)

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

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation (generator acquired above)
    image_task = asyncio.to_thread(
        _generate_scene_image_sync,
        generator,
        game_id,
        scene_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        visible_asset_ids,
        art_style,
    )

    voiceover_task = _generate_scene_voiceover(
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
    if isinstance(image_file_path, Exception):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, Exception):
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
    characters: List[Character],
    last_scene_id: int,
    player_action: Optional[str],
    conversation_history: List[dict],
    existing_assets: dict[str, Asset],
    existing_locations: dict[str, Location],
    art_style: Optional[str] = None,
) -> tuple[Scene, List[Character], dict[str, Asset], dict[str, Location]]:
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
    agent = _text_agent(GeneratedScene)

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
            f"- Inventory: {_format_inventory(char.inventory)} (current items, don't remove again unless used in THIS scene)"
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
            dice_check_rules = f"\n\n{_build_dice_check_resolution_rules()}"

    prompt = f"""
You are a creative Dungeon Master for a D&D-style adventure.

SCENARIO: {scenario_name}

SCENARIO DETAILS:
{scenario_details}

PARTY CHARACTER SHEETS:
{character_sheets}

{_format_existing_assets(existing_assets)}

{_format_existing_locations(existing_locations)}

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

{_build_impossible_action_rules()}

{_build_scene_generation_prompt_rules()}

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
    updated_characters = _apply_character_updates(characters, generated)

    # Process locations
    updated_locations, processed_location_ref = _process_scene_location(
        generated.location_reference,
        existing_locations,
    )

    # Process assets and generate images in background thread (local backend).
    # Acquire (and lazily load) the model off the event loop so the web server
    # stays responsive during the one-time model load.
    generator = await asyncio.to_thread(_get_image_generator)

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

    # Generate scene image and voiceover in background threads (concurrently)
    # Now we can include asset images in scene generation (generator acquired above)
    image_task = asyncio.to_thread(
        _generate_scene_image_sync,
        generator,
        game_id,
        next_id,
        generated.scene_text,
        generated.visual_description,
        characters,
        scenario_name,
        generated.game_status,
        updated_assets,
        visible_asset_ids,
        art_style,
    )

    voiceover_task = _generate_scene_voiceover(
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
    if isinstance(image_file_path, Exception):
        logger.error(f"Image generation failed: {image_file_path}")
        image_file_path = None
    if isinstance(voiceover_file_path, Exception):
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


async def generate_scene_recap(
    game_id: str,
    scenario_name: str,
    scenes: List[Scene],
    characters: List[Character],
    current_scene_index: int,
) -> RecapResponse:
    """Generate a narrative recap with per-scene summaries.

    Args:
        game_id: Unique identifier for the game
        scenario_name: Name of the scenario being played
        scenes: List of all scenes in the game
        characters: Current party characters
        current_scene_index: Index of the current scene (0-based)

    Returns:
        RecapResponse containing overall recap text and individual scene summaries
    """
    agent = _text_agent(RecapResponse)

    logger.info(
        f"Generating recap for game {game_id}, scenes 1-{current_scene_index + 1}"
    )

    # Build scene summaries for context
    scene_summaries = []
    for i, scene in enumerate(scenes[: current_scene_index + 1]):
        summary = f"**Scene {scene.id}:**\n{scene.text[:500]}..."
        if scene.prompt:
            summary += f"\n*Player choice was needed here*"
        scene_summaries.append(summary)

    scenes_context = "\n\n".join(scene_summaries)

    # Build character status
    character_status = "\n".join(
        [
            f"- **{char.name}**: HP {char.current_health}/{char.maximum_health}, "
            f"Skills: {', '.join(char.skills[:3]) if char.skills else 'None'}"
            for char in characters
        ]
    )

    prompt = f"""You are a skilled storyteller creating a recap for players returning to their adventure after a break.

SCENARIO: {scenario_name}

CURRENT PARTY STATUS:
{character_status}

SCENES SO FAR:
{scenes_context}

Generate TWO things:

1. **recap_text**: An engaging, concise overall recap (200-300 words) that:
   - Reminds players of the adventure's main goal
   - Summarizes the key events that have happened so far
   - Highlights any important choices the party made
   - Notes any significant character developments (injuries, items gained, etc.)
   - Sets the stage for where the story is now
   - Uses an exciting, narrative tone as if read by a Dungeon Master
   - Written in past tense, like "Previously on [Adventure Name]..."

2. **scene_summaries**: For EACH scene (scenes 1-{current_scene_index + 1}), provide:
   - scene_id: The scene number
   - summary: A concise one-sentence summary (15-20 words) capturing the key event/development
   
   Make each scene summary punchy and informative. Focus on:
   - What significant thing happened (discovery, combat, decision, meeting)
   - Who was involved (specific characters if relevant)
   - The immediate consequence or outcome
   
   Examples of good scene summaries:
   - "The party discovered an ancient temple hidden in the misty mountains."
   - "Theron negotiated with the bandit chief, securing safe passage through the forest."
   - "A fierce battle with goblins left the party wounded but victorious."
   - "The mysterious wizard revealed the location of the legendary artifact."

Return your response in the RecapResponse format with both recap_text and scene_summaries array.
"""

    try:
        result = await retry_on_overload(agent.run, prompt)
        recap_response = result.output
        logger.info(
            f"Successfully generated recap ({len(recap_response.recap_text)} chars, {len(recap_response.scene_summaries)} scene summaries)"
        )
        return recap_response
    except Exception as e:
        logger.error(f"Failed to generate recap: {e}")
        # Return a simple fallback recap
        fallback_summaries = [
            SceneSummary(
                scene_id=scene.id, summary=f"Scene {scene.id}: Adventure continues..."
            )
            for scene in scenes[: current_scene_index + 1]
        ]
        return RecapResponse(
            recap_text=f"""## Previously in {scenario_name}...

Your party has journeyed through {current_scene_index + 1} scene(s) of adventure. 
The story continues with your brave heroes facing new challenges.

Current party status:
{character_status}
""",
            scene_summaries=fallback_summaries,
        )

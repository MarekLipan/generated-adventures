"""Image and narration rendering: shared model instances, media folders, and the glue between generated text and the image/TTS backends."""

import asyncio
import logging
import pathlib
import threading

from . import voice_casting
from .config import settings
from .image_backends import ImageGenerator, get_image_generator
from .models import (
    Asset,
    Character,
    GameStatus,
    GeneratedArchetype,
    NarrationSegment,
)
from .names import names_match, npc_names_match
from .prompts import (
    build_hero_portrait_prompt,
)
from .tts_backends import TTSGenerator, get_tts_generator

logger = logging.getLogger(__name__)


# Generated media lives under DATA_DIR. Saved games store URLs such as
# /static/scenes/<game>/<file>, so the web app serves MEDIA_DIR at that prefix.
MEDIA_DIR = settings.DATA_DIR / "media"
MEDIA_URL_PREFIX = "/static"
ASSET_DIR = MEDIA_DIR / "assets"
PHOTO_DIR = MEDIA_DIR / "photos"

IMAGE_DIR = MEDIA_DIR / "characters"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


SCENE_IMAGE_DIR = MEDIA_DIR / "scenes"
SCENE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def media_path(url: str) -> pathlib.Path:
    """The file behind a media URL such as /static/scenes/<game>/<file>."""
    return MEDIA_DIR / url.removeprefix(MEDIA_URL_PREFIX).lstrip("/")


# Lazy initialization of image generator to avoid loading models at import.
# A lock guards the one-time load so concurrent first-callers don't load twice.
_image_generator: ImageGenerator | None = None


_image_generator_lock = threading.Lock()


def shared_image_generator() -> ImageGenerator:
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
        shared_image_generator()
    except Exception as e:
        logger.warning(f"Image model warmup failed (will retry on first use): {e}")
    try:
        _shared_tts_generator()
    except Exception as e:
        logger.warning(f"TTS warmup failed (will retry on first use): {e}")


# Lazy TTS generator singleton. `_UNSET` distinguishes "not built yet" from a
# cached None (TTS disabled or failed to initialize — narration is optional, so
# we don't retry every scene).
_UNSET = object()


_tts_generator: object = _UNSET


def _shared_tts_generator() -> TTSGenerator | None:
    """Get or initialize the TTS generator singleton (None if unavailable)."""
    global _tts_generator
    if _tts_generator is _UNSET:
        try:
            _tts_generator = get_tts_generator()
        except Exception as e:
            logger.error(f"TTS initialization failed ({e}); narration disabled")
            _tts_generator = None
    return _tts_generator  # type: ignore[return-value]


VOICEOVER_DIR = MEDIA_DIR / "voiceovers"
VOICEOVER_DIR.mkdir(parents=True, exist_ok=True)


def _prioritize_scene_references(
    assets: dict[str, Asset],
    visible_asset_ids: list[str],
    max_refs: int,
) -> tuple[list[pathlib.Path], list, list, list[str]]:
    """Choose which visible assets become scene image references, best-first, capped.

    Klein downsizes ALL references together to fit VRAM, so a crowded reference
    list makes every face a low-res thumbnail. We keep identity-bearing references
    first — party heroes, then NPCs, then objects — and cap the count, so the
    characters that must stay recognizable each get a higher resolution and
    objects/extras are dropped rather than dragging everyone down.

    Returns (reference_files, kept_assets, dropped_assets, kept_asset_ids) — the
    files, assets and ids are parallel, so per-subject art direction can be aligned
    to exactly the references that survived the cap.
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
        ref_file = media_path(asset.image_path)
        if not ref_file.exists():
            logger.warning(f"Reference image not found, skipping: {ref_file}")
            continue
        candidates.append((priority(aid), aid, asset, ref_file))

    # Stable sort keeps the scene's original ordering within each priority band.
    candidates.sort(key=lambda t: t[0])
    kept = candidates[: max(0, max_refs)]
    dropped = candidates[max(0, max_refs) :]
    return (
        [ref_file for _, _, _, ref_file in kept],
        [asset for _, _, asset, _ in kept],
        [asset for _, _, asset, _ in dropped],
        [aid for _, aid, _, _ in kept],
    )


def resolve_image_subject_ids(
    image_subjects: list | None,
    assets: dict[str, Asset],
    visible_asset_ids: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Resolve art-directed in-frame subjects to (asset ids, {asset_id: direction}).

    The direction map lets the renderer bind each reference portrait to what that
    character is doing in frame (which is what stops two characters being swapped),
    and survives the later reference-cap/prioritization step by being keyed on id.

    - image_subjects is None (model didn't art-direct): fall back to all visible
      assets (previous behavior) so we never accidentally drop the hero.
    - image_subjects is [] (explicit environment/establishing shot): no references.
    - otherwise: only the named subjects are referenced (reference isolation),
      matched tolerantly (party subset-match, NPC/object token-equality).
    """
    if image_subjects is None:
        return visible_asset_ids, {}
    ids: list[str] = []
    directions: dict[str, str] = {}
    for subject in image_subjects:
        name = (getattr(subject, "name", None) or str(subject)).strip()
        action = (getattr(subject, "action", "") or "").strip()
        if not name:
            continue
        for aid, asset in assets.items():
            if aid in ids:
                continue
            is_party = aid.startswith("player_")
            matched = asset.name.lower() == name.lower() or (
                names_match(name, asset.name)
                if is_party
                else npc_names_match(name, asset.name)
            )
            if matched:
                ids.append(aid)
                directions[aid] = f"{asset.name}, {action}" if action else asset.name
                break
    return ids, directions


def generate_scene_image_sync(
    generator: ImageGenerator,
    game_id: str,
    scene_id: int,
    scene_text: str,
    visual_description: str,
    characters: list[Character],
    scenario_name: str,
    game_status: GameStatus = "ongoing",
    assets: dict[str, Asset] = None,
    visible_asset_ids: list[str] = None,
    art_style: str | None = None,
    subject_directions: dict[str, str] | None = None,
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
    reference_images: list[pathlib.Path] = []
    aligned_directions: list[str] = []
    if assets and visible_asset_ids:
        reference_images, kept, dropped, kept_ids = _prioritize_scene_references(
            assets, visible_asset_ids, settings.IMAGE_MAX_SCENE_REFERENCES
        )
        for a in kept:
            logger.info(f"✓ Reference [{a.type}]: {a.name}")
        if dropped:
            logger.info(
                f"↓ Dropped {len(dropped)} lower-priority reference(s) to keep "
                f"resolution high: {', '.join(a.name for a in dropped)}"
            )
        # Align each surviving reference with its art direction (who is doing what),
        # in the SAME order the renderer will lay out and repaint them. Keeping these
        # parallel is what stops one character being painted into another's pose.
        if subject_directions:
            aligned_directions = [subject_directions.get(aid, "") for aid in kept_ids]
            if not any(aligned_directions):
                aligned_directions = []

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
        subject_directions=aligned_directions or None,
    )

    return image_file_path


def _resolve_segment_voices(
    narration_segments: list["NarrationSegment"],
    characters: list[Character],
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
        return any(names_match(name, c.name) for c in characters)

    def match_npc(name: str):
        # NPC assets only — skip the player_-prefixed party image-assets.
        for aid, a in assets.items():
            if aid.startswith("player_"):
                continue
            if npc_names_match(name, a.name):
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


async def generate_scene_voiceover(
    game_id: str,
    scene_id: int,
    scene_text: str,
    characters: list[Character],
    assets: dict[str, Asset],
    narration_segments: list["NarrationSegment"] | None = None,
) -> pathlib.Path | None:
    """Generate scene narration via the configured TTS backend.

    Uses the per-speaker `narration_segments` emitted by the scene model (multi-voice)
    when available on a multi-voice backend; otherwise reads the whole scene in the
    single narrator voice. Narration is optional: returns None (never raises) if
    unavailable or failing.
    """
    tts = _shared_tts_generator()
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
    tts = _shared_tts_generator()
    if tts is None:
        return None
    voiceover_dir = VOICEOVER_DIR / game_id
    voiceover_dir.mkdir(parents=True, exist_ok=True)
    path = voiceover_dir / f"recap_scene_{scene_id}.wav"
    logger.info(f"Generating recap voiceover for scene {scene_id} in game {game_id}")
    return await asyncio.to_thread(tts.synthesize, recap_text, path)


def generate_hero_portrait_sync(
    generator: ImageGenerator,
    hero_name: str,
    archetype: GeneratedArchetype,
    art_style: str,
    photo_path: pathlib.Path | None,
    game_dir: pathlib.Path,
    idx: int,
    scale: str = "any",
) -> pathlib.Path | None:
    """Synchronous helper: render one hero portrait (optionally from a photo)."""
    has_photo = bool(photo_path and pathlib.Path(photo_path).exists())
    prompt_img = build_hero_portrait_prompt(hero_name, archetype, has_photo, scale)

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


def generate_asset_image_sync(
    generator: ImageGenerator,
    game_id: str,
    asset_id: str,
    asset_name: str,
    asset_description: str,
    art_style: str | None = None,
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
    asset_dir = ASSET_DIR / game_id
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

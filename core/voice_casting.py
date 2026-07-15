"""Curated Kokoro voice casting for multi-voice scene narration.

The narrator reads descriptive prose; each speaking character/NPC gets a distinct
voice. Voices are drawn from a hand-picked pool of the strongest English Kokoro
voices, split by apparent gender, and assigned so that:

  * the same name always maps to the same voice (stable across scenes/sessions), and
  * main cast members avoid colliding on the same voice.

Only native English voices are used — Kokoro's other languages sound off reading
English. `lang_for_voice` maps a voice to the phonemizer language it expects.
"""

import hashlib
from typing import Iterable, Optional

# Hand-picked English voices, excluding the default narrator (bm_lewis).
# Curated to Kokoro's higher-graded voices only (grade in parens) — auditioning
# showed the low-graded (D) voices sound noticeably flatter, and casting draws
# uniformly across the pool, so duds would surface on real NPCs. US + UK kept for
# accent variety; ranked best-grade first.
MALE_VOICES = [
    "am_michael",  # US, steady (C+)
    "am_fenrir",   # US, deep (C+)
    "am_puck",     # US, lively (C+)
    "bm_george",   # UK, warm (C)
    "bm_fable",    # UK, storybook (C)
]

FEMALE_VOICES = [
    "af_heart",     # US, warm (A, top-rated)
    "af_bella",     # US, rich (A-)
    "af_nicole",    # US, soft (B-)
    "bf_emma",      # UK, measured (B-)
    "af_aoede",     # US, clear (C+)
    "af_sarah",     # US, steady (C+)
    "bf_isabella",  # UK, elegant (C)
]


def lang_for_voice(voice: str) -> str:
    """Phonemizer language for a voice: British voices are en-gb, else en-us."""
    return "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"


def _stable_index(name: str, n: int) -> int:
    """Deterministic index in [0, n) from a name (stable across processes).

    Python's built-in hash() is salted per process, so we use md5 to keep a
    given name mapped to the same voice across runs.
    """
    digest = hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def cast_voice(
    name: str,
    gender: str,
    taken: Optional[Iterable[str]] = None,
    exclude: Optional[str] = None,
) -> str:
    """Pick a stable, collision-avoiding voice for a speaker.

    Args:
        name: Speaker name (drives the deterministic starting choice).
        gender: 'male', 'female', or anything else (falls back to a mixed pool).
        taken: Voices already assigned to other speakers, to avoid duplicates.
        exclude: A voice to never assign (e.g. the narrator's voice).

    Returns:
        A Kokoro voice id.
    """
    g = (gender or "").strip().lower()
    if g == "male":
        pool = MALE_VOICES
    elif g == "female":
        pool = FEMALE_VOICES
    else:
        # Unknown: interleave so alternating unknown speakers still differ.
        pool = [v for pair in zip(MALE_VOICES, FEMALE_VOICES) for v in pair]

    taken_set = set(taken or ())
    if exclude:
        taken_set = taken_set | {exclude}

    start = _stable_index(name, len(pool))
    ordered = pool[start:] + pool[:start]

    for voice in ordered:
        if voice not in taken_set:
            return voice

    # Everything taken (huge cast): fall back to the deterministic first pick,
    # avoiding only the excluded narrator voice if possible.
    for voice in ordered:
        if voice != exclude:
            return voice
    return ordered[0]

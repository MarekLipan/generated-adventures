"""Tolerant matching of character and NPC names.

The scene model refers to party members and NPCs by name, but doesn't always
reproduce a long or custom name character-for-character. Exact matching then
silently drops health/inventory updates (character sheet doesn't change) and,
worse, makes the asset step create a fresh no-photo portrait instead of reusing
the player's likeness.
"""

import re

_NAME_STOPWORDS = {"the", "of", "and", "a", "an", "de", "von"}


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _name_tokens(s: str) -> set:
    return {t for t in _norm_name(s).split() if t and t not in _NAME_STOPWORDS}


def names_match(a: str, b: str) -> bool:
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


def npc_names_match(a: str, b: str) -> bool:
    """Tolerant match for NPCs/objects: normalized token-set EQUALITY.

    Reunites naming drift that differs only by articles, case, spacing or
    punctuation ("The Echo Child" / "Echo-Child" / "echo child") so a recurring
    NPC keeps ONE asset and one portrait. Unlike `names_match` (used for party
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

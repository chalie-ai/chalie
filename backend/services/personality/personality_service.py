"""Personality service — 5-axis slider corpus powering Chalie's voice.

Corpus: voices.jsonl (3,125 rows).  Slider axes (in order):
  warmth, mood, expressiveness, curiosity, humor
Each step ∈ {-2, -1, 0, +1, +2}.  Neutral = (0, 0, 0, 0, 0).

The loaded index is a module-level dict keyed by 5-tuple so lookups are O(1)
after the one-time JSONL scan.  The pattern mirrors _get_lut_conn() in
data_graph_service — double-checked lock, lazy on first call, graceful None
on missing file.
"""

import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

NEUTRAL_TUPLE: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
SLIDER_ORDER: tuple[str, ...] = ('warmth', 'mood', 'expressiveness', 'curiosity', 'humor')

_VOICES_PATH = os.path.join(os.path.dirname(__file__), 'voices.jsonl')

_voices_index: Optional[dict[tuple[int, int, int, int, int], str]] = None
_voices_lock = threading.Lock()
_voices_loaded: bool = False


def _load_voices() -> None:
    """Read voices.jsonl once and populate the module-level index.

    Called inside the double-checked lock — never call directly.
    """
    global _voices_index
    index: dict[tuple[int, int, int, int, int], str] = {}
    try:
        with open(_VOICES_PATH, 'r', encoding='utf-8') as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    tup = tuple(obj['tuple'])
                    if len(tup) != 5:
                        logger.warning("[PERSONALITY] voices.jsonl line %d: tuple length %d (expected 5)", lineno, len(tup))
                        continue
                    index[tup] = obj['voice']
                except Exception as exc:
                    logger.warning("[PERSONALITY] voices.jsonl line %d parse error: %s", lineno, exc)
        logger.info("[PERSONALITY] voices.jsonl loaded: %d entries", len(index))
    except Exception as exc:
        logger.warning("[PERSONALITY] Failed to load voices.jsonl: %s", exc)
    _voices_index = index


def _get_index() -> dict[tuple[int, int, int, int, int], str]:
    """Return the voice index, loading it on the first call."""
    global _voices_loaded
    if _voices_loaded:
        return _voices_index or {}
    with _voices_lock:
        if _voices_loaded:
            return _voices_index or {}
        _load_voices()
        _voices_loaded = True
    return _voices_index or {}


def get_voice(tup: tuple[int, int, int, int, int]) -> str:
    """Return the voice paragraph for *tup*, falling back to neutral.

    The neutral fallback is defensive — the full corpus covers every valid
    combination so a miss should not occur in practice.
    """
    index = _get_index()
    voice = index.get(tup)
    if voice is not None:
        return voice
    logger.warning("[PERSONALITY] Tuple %s not found in index — using neutral fallback", tup)
    return index.get(NEUTRAL_TUPLE, "Engage naturally as a peer.")


def get_current_tuple() -> tuple[int, int, int, int, int]:
    """Read the current personality tuple from settings.

    Returns ``NEUTRAL_TUPLE`` on missing or invalid data.
    """
    try:
        from services.database_service import get_shared_db_service
        from services.settings_service import SettingsService

        db = get_shared_db_service()
        raw = SettingsService(db).get('personality')
        if not raw:
            return NEUTRAL_TUPLE
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or len(parsed) != 5:
            return NEUTRAL_TUPLE
        steps = tuple(int(v) for v in parsed)
        if any(v not in (-2, -1, 0, 1, 2) for v in steps):
            return NEUTRAL_TUPLE
        return steps
    except Exception as exc:
        logger.debug("[PERSONALITY] get_current_tuple error: %s", exc)
        return NEUTRAL_TUPLE


def get_current_voice() -> str:
    """Return the voice paragraph for the currently persisted personality."""
    return get_voice(get_current_tuple())


def set_current_tuple(tup: tuple[int, int, int, int, int]) -> str:
    """Validate, persist, and return the voice paragraph for *tup*.

    Args:
        tup: 5-tuple of ints, each ∈ {-2, -1, 0, +1, +2}.

    Returns:
        The voice paragraph string corresponding to *tup*.

    Raises:
        ValueError: If *tup* is not a 5-tuple or any step is out of range.
    """
    if len(tup) != 5:
        raise ValueError(f"Personality tuple must have exactly 5 elements, got {len(tup)}")
    for i, step in enumerate(tup):
        if step not in (-2, -1, 0, 1, 2):
            raise ValueError(
                f"Step {i} ({SLIDER_ORDER[i]}) = {step!r} is out of range; "
                f"must be one of -2, -1, 0, 1, 2"
            )
    from services.database_service import get_shared_db_service
    from services.settings_service import SettingsService

    db = get_shared_db_service()
    SettingsService(db).set('personality', json.dumps(list(tup)))
    return get_voice(tup)

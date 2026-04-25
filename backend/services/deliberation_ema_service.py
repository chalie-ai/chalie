"""
DeliberationEmaService — per-turn EMA state machine for the deliberation score.

Owns load → update → save on MemoryStore key ``deliberation_score:ema``.
State is a JSON object: {"ema": float, "last_turn_at": ISO-8601 string}.

One instance per turn (same pattern as ModeGateService). State lives in
MemoryStore — survives WebSocket reconnects and process restarts.

Alpha, high_threshold, and med_threshold are read once from the prepackaged
deliberation-score-classifier_meta.json. Chalie does not second-guess the
training-side calibration.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "[DELIBERATION-EMA]"

STATE_KEY = "deliberation_score:ema"

# ── Load thresholds once from classifier_meta.json ────────────────────────────

_meta_cache: Optional[dict] = None


def _load_meta() -> dict:
    """Load deliberation-score classifier_meta.json once per process.

    Falls back to training-calibrated defaults if the file is unreadable.
    Raises RuntimeError if required keys are missing — boot-time contract.
    """
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache

    try:
        import runtime_config
        _default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
        models_dir = runtime_config.get(
            "models_dir",
            os.environ.get("MODELS_DIR", _default),
        )
        meta_path = Path(models_dir) / "deliberation_score" / "deliberation-score-classifier_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as exc:
        logger.warning(
            "%s could not load classifier_meta.json (%s) — using fallback defaults",
            LOG_PREFIX, exc,
        )
        # Fallback: values from the v0.11.0 release (non-authoritative)
        meta = {
            "ema_alpha": 0.6,
            "bucket_thresholds": {"high": 0.67, "medium": 0.46},
        }

    # Validate required keys
    if "ema_alpha" not in meta:
        raise RuntimeError(
            "deliberation-score-classifier_meta.json missing 'ema_alpha' — "
            "boot-time contract violation"
        )
    if "bucket_thresholds" not in meta:
        raise RuntimeError(
            "deliberation-score-classifier_meta.json missing 'bucket_thresholds' — "
            "boot-time contract violation"
        )

    _meta_cache = meta
    return meta


class DeliberationEmaService:
    """Per-turn EMA state machine for the deliberation score.

    One instance per turn. State lives in MemoryStore under ``deliberation_score:ema``.
    """

    STATE_KEY = STATE_KEY

    def __init__(self) -> None:
        meta = _load_meta()
        self._alpha: float = float(meta["ema_alpha"])
        thresholds = meta["bucket_thresholds"]
        self._high_thr: float = float(thresholds["high"])
        self._med_thr: float = float(thresholds["medium"])

    # ── Public API ────────────────────────────────────────────────────────────

    def peek(self) -> 'float | None':
        """Return the current EMA without updating it. None on cold start."""
        state = self._load_state()
        return state

    def update_and_bucket(self, scalar_t: float) -> Tuple[float, str]:
        """Apply EMA update, persist, and return (ema_after, bucket).

        Cold start: ema = scalar_t (no smoothing on first turn).
        Subsequent turns: ema = alpha * prev_ema + (1 - alpha) * scalar_t.

        Bucket assignment:
            ema > high_thr  → 'high'
            ema > med_thr   → 'medium'
            else            → 'low'
        """
        prior = self._load_state()
        if prior is None:
            ema = scalar_t
        else:
            ema = self._alpha * prior + (1.0 - self._alpha) * scalar_t

        self._save_state(ema)

        bucket = self._resolve_bucket(ema)
        return ema, bucket

    def reset(self) -> None:
        """Clear EMA state from MemoryStore. Called by /privacy/delete-all."""
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            store.delete(self.STATE_KEY)
        except Exception as exc:
            logger.warning("%s reset failed: %s", LOG_PREFIX, exc)

    def resolve_bucket(self, ema: float) -> str:
        """Resolve a float EMA to 'high' | 'medium' | 'low'. Exposed for tests."""
        return self._resolve_bucket(ema)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_bucket(self, ema: float) -> str:
        if ema > self._high_thr:
            return "high"
        if ema > self._med_thr:
            return "medium"
        return "low"

    def _load_state(self) -> 'float | None':
        """Read the current EMA float from MemoryStore. None on miss or error."""
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            raw = store.get(self.STATE_KEY)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode()
            parsed = json.loads(raw)
            ema = parsed.get("ema")
            if ema is None:
                return None
            return float(ema)
        except json.JSONDecodeError as exc:
            logger.warning(
                "%s corrupt JSON in state key — treating as cold start: %s",
                LOG_PREFIX, exc,
            )
            return None
        except Exception as exc:
            logger.warning("%s _load_state failed (%s) — treating as cold start", LOG_PREFIX, exc)
            return None

    def _save_state(self, ema: float) -> None:
        """Persist EMA to MemoryStore as JSON. No TTL — survives restart."""
        from services.time_utils import utc_now
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            payload = {
                "ema": round(ema, 6),
                "last_turn_at": utc_now().isoformat(),
            }
            store.set(self.STATE_KEY, json.dumps(payload))
        except Exception as exc:
            logger.warning("%s _save_state failed: %s", LOG_PREFIX, exc)

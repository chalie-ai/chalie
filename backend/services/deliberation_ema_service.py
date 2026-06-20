
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple, cast

logger = logging.getLogger(__name__)

LOG_PREFIX = "[DELIBERATION-EMA]"

STATE_KEY = "deliberation_score:ema"

# ── Load thresholds once from classifier_meta.json ────────────────────────────

_meta_cache: Optional[dict[str, object]] = None


def _load_meta() -> dict[str, object]:
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache

    from services.file_mapper_service import FileMapperService
    meta_path = FileMapperService.get_pretrained_path("deliberation_score", "deliberation-score-classifier_meta.json")
    with open(meta_path) as f:
        meta: dict[str, object] = cast(dict[str, object], json.load(f))

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

    STATE_KEY = STATE_KEY

    def __init__(self) -> None:
        meta = _load_meta()
        self._alpha: float = float(cast(float, meta["ema_alpha"]))
        thresholds = cast(dict[str, float], meta["bucket_thresholds"])
        self._high_thr: float = float(thresholds["high"])
        self._med_thr: float = float(thresholds["medium"])

    # ── Public API ────────────────────────────────────────────────────────────

    def peek(self) -> 'float | None':
        state = self._load_state()
        return state

    def update_and_bucket(self, scalar_t: float) -> Tuple[float, str]:
        prior = self._load_state()
        if prior is None:
            ema = scalar_t
        else:
            ema = self._alpha * prior + (1.0 - self._alpha) * scalar_t

        self._save_state(ema)

        bucket = self._resolve_bucket(ema)
        return ema, bucket

    def reset(self) -> None:
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            store.delete(self.STATE_KEY)
        except Exception as exc:
            logger.warning("%s reset failed: %s", LOG_PREFIX, exc)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_bucket(self, ema: float) -> str:
        if ema > self._high_thr:
            return "high"
        if ema > self._med_thr:
            return "medium"
        return "low"

    def _load_state(self) -> 'float | None':
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

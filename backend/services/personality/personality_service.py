"""Personality service — 5-axis voice corpus singleton.

``personality_service.get_voice()`` is the only call site that matters.
Cached after first read; invalidated on ``set_tuple()``.
"""

import json
import logging
from typing import cast

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

NEUTRAL: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
SLIDER_ORDER = ('warmth', 'mood', 'expressiveness', 'curiosity', 'humor')
_VOICES_PATH = str(FileMapperService.get_backend_path("services", "personality", "voices.jsonl"))


class PersonalityService:

    def __init__(self) -> None:
        self._index = self._load_index()
        self._voice: str | None = None
        self._tuple: tuple[int, int, int, int, int] | None = None

    def get_voice(self) -> str:
        if self._voice is not None:
            return self._voice
        self._tuple = self._read_tuple()
        self._voice = self._index.get(self._tuple) or self._index.get(
            NEUTRAL, "Engage naturally as a peer.",
        )
        return self._voice

    def get_tuple(self) -> tuple[int, int, int, int, int] | None:
        if self._tuple is not None:
            return self._tuple
        self.get_voice()
        return self._tuple

    def set_tuple(self, tup: tuple[int, int, int, int, int]) -> str:
        if len(tup) != 5 or any(v not in range(-2, 3) for v in tup):
            raise ValueError(f"Invalid personality tuple: {tup}")
        from services.database_service import get_shared_db_service
        from services.settings_service import SettingsService
        SettingsService(get_shared_db_service()).set('personality', json.dumps(list(tup)))
        self._tuple = tup
        self._voice = self._index.get(tup) or self._index.get(
            NEUTRAL, "Engage naturally as a peer.",
        )
        return self._voice

    def invalidate(self) -> None:
        self._voice = None
        self._tuple = None

    def voice_for(self, tup: tuple[int, int, int, int, int]) -> str:
        return self._index.get(tup) or self._index.get(
            NEUTRAL, "Engage naturally as a peer.",
        )

    def _read_tuple(self) -> tuple[int, int, int, int, int]:
        try:
            from services.database_service import get_shared_db_service
            from services.settings_service import SettingsService
            raw = SettingsService(get_shared_db_service()).get('personality')
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and len(parsed) == 5:
                    return cast("tuple[int, int, int, int, int]", tuple(int(v) for v in parsed))
        except Exception:
            pass
        return NEUTRAL

    @staticmethod
    def _load_index() -> dict[tuple[int, int, int, int, int], str]:
        index: dict[tuple[int, int, int, int, int], str] = {}
        try:
            with open(_VOICES_PATH, 'r', encoding='utf-8') as fh:
                for raw in fh:
                    raw = raw.strip()
                    if raw:
                        obj = json.loads(raw)
                        tup = tuple(obj['tuple'])
                        if len(tup) == 5:
                            index[tup] = obj['voice']
            logger.info("[PERSONALITY] Loaded %d voices", len(index))
        except Exception as exc:
            logger.warning("[PERSONALITY] Failed to load voices.jsonl: %s", exc)
        return index


personality_service = PersonalityService()

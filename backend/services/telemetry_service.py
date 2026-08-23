"""Telemetry service — JSON-file cache of the last client heartbeat.

The single write path is POST /health (the client heartbeat, every ~5
min). Telemetry is advisory context, so a read degrades to an empty
snapshot when the file is missing or corrupt — the next heartbeat
self-heals the file. ``FileMapperService`` precedent: classmethods only,
no instances are created.
"""

import json
import logging
import os
import threading
import time

from models.telemetry import Telemetry
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class TelemetryService:
    """Classmethod-only access to the telemetry snapshot
    (``data/telemetry.json``).

    A class-level cache mirrors the file for the read hot path; a
    class-level lock serialises writes so the tmp-file + ``os.replace``
    persist and the cache swap never interleave with another write.
    """

    _cache: Telemetry | None = None
    _lock = threading.Lock()

    @classmethod
    def write(cls, ctx: dict[str, object]) -> None:
        """Stamp, persist, and cache the snapshot.

        Persistence is atomic — write a sibling ``.tmp`` file, then
        ``os.replace`` it over the target; never truncate-in-place, so a
        crash mid-write can never leave a torn file. The cache swap
        happens under the same lock, after the file is durable.
        """
        ctx["saved_at"] = time.time()
        path = FileMapperService.get_telemetry_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with cls._lock:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(ctx, fh)
            os.replace(tmp, path)
            cls._cache = Telemetry(ctx)

    @classmethod
    def read(cls) -> Telemetry:
        """The cached snapshot; on cold start, load it from the JSON file.

        A missing file degrades silently to an empty ``Telemetry``; an
        unreadable or invalid file (not a JSON object) logs a warning
        with the path and error, then degrades the same way — advisory
        data must never break a caller.
        """
        cache = cls._cache
        if cache is not None:
            return cache
        path = FileMapperService.get_telemetry_json_path()
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        except (OSError, ValueError) as exc:
            logger.warning(f"[TELEMETRY] Could not read {path}: {exc}")
            raw = {}
        if not isinstance(raw, dict):
            logger.warning(f"[TELEMETRY] {path} is not a JSON object; ignoring it")
            raw = {}
        cls._cache = Telemetry(raw)
        return cls._cache

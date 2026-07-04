"""Heartbeat service — cached client telemetry singleton.

Writer: POST /health (heartbeat.js, every ~5 min).
Readers: WorldState, locale_service, act_dispatcher.
"""

import json
import logging
import time
from typing import cast

logger = logging.getLogger(__name__)


class HeartbeatService:

    def __init__(self) -> None:
        self._ctx: dict[str, object] | None = None

    def read(self) -> dict[str, object]:
        if self._ctx is not None:
            return self._ctx
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                rows = conn.execute("SELECT key, value FROM telemetry").fetchall()
            self._ctx = self._unflatten(rows) if rows else {}
        except Exception:
            self._ctx = {}
        return self._ctx

    def write(self, ctx: dict[str, object]) -> None:
        ctx["saved_at"] = time.time()
        flat = self._flatten(ctx)
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            conn.execute("DELETE FROM telemetry")
            conn.executemany(
                "INSERT INTO telemetry (key, value) VALUES (?, ?)",
                list(flat.items()),
            )
            conn.commit()
        self._ctx = ctx

    @staticmethod
    def _flatten(ctx: dict[str, object], prefix: str = "") -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in ctx.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict) and value:
                out.update(HeartbeatService._flatten(value, prefix=f"{full_key}."))
            else:
                out[full_key] = json.dumps(value)
        return out

    @staticmethod
    def _unflatten(rows: list[tuple[str, str]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for flat_key, raw_value in rows:
            try:
                value = json.loads(raw_value)
            except (TypeError, ValueError):
                value = raw_value
            parts = flat_key.split(".")
            cursor = out
            for part in parts[:-1]:
                if not isinstance(cursor.get(part), dict):
                    cursor[part] = {}
                cursor = cast(dict[str, object], cursor[part])
            cursor[parts[-1]] = value
        return out


heartbeat_service = HeartbeatService()

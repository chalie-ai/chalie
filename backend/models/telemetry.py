"""Telemetry — the client-context snapshot, stored flattened across the
``telemetry`` key/value table.

Active-record row-model (Rule 5 / §4.1) for the ``telemetry`` table. That
table is NOT a collection of independent settings but a SINGLE logical object —
the current client context (device, locale, location, ``saved_at``, …) —
persisted flattened one dotted-key per row. So, like
:class:`~models.setting.Setting` keys its verbs on ``key`` rather than the
base's id-centric ``get``/``save``, this model exposes whole-object verbs keyed
on the snapshot itself: :meth:`curr` reads the current snapshot back as a
nested dict, :meth:`replace` atomically swaps the whole snapshot.

:meth:`replace` is a whole-object SNAPSHOT swap (DELETE-all + re-INSERT inside
one ``Database.transaction()``), NOT a per-key upsert: a key absent from the new
context MUST disappear, or a stale value (a ``now_playing`` after playback
stops) would linger forever. This model is the SOLE home of ``telemetry`` row
SQL; :class:`~services.heartbeat_service.HeartbeatService` holds only the
process cache and delegates every read/write here. Holds no mp, calls no
service beyond the ``Database`` gateway (Rule-3 depth; same carve-out as
:class:`~models.setting.Setting`).
"""

from __future__ import annotations

import json
from typing import ClassVar, cast

from models.model import Model
from services.database import Database


class Telemetry(Model):
    """The ``telemetry`` KV table as one snapshot object: flatten ↔ unflatten
    plus the whole-object ``curr``/``replace`` verbs."""

    __columns__: ClassVar[tuple[str, ...]] = ("key", "value")

    @classmethod
    def get_table(cls) -> str:
        return "telemetry"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    key: str
    value: str

    # ── Whole-object read / replace ───────────────────────────────────────────

    @classmethod
    def curr(cls) -> dict[str, object]:
        """The current client-context snapshot, unflattened from its dotted-key
        rows back into the nested dict callers consume.

        Empty dict when no heartbeat has landed yet — or on any read error:
        telemetry is advisory context, so a failed read must degrade to "no
        context", never break a caller."""
        try:
            rows = Database.conn().execute("SELECT key, value FROM telemetry").fetchall()
        except Exception:
            return {}
        return cls._unflatten(rows) if rows else {}

    @classmethod
    def replace(cls, ctx: dict[str, object]) -> None:
        """Atomically swap the whole snapshot: DELETE every row, then re-INSERT
        the flattened ``ctx``. A key absent from ``ctx`` is REMOVED — this is a
        snapshot swap, not a per-key upsert, so stale keys never survive a beat."""
        flat = cls._flatten(ctx)
        with Database.transaction() as conn:
            conn.execute("DELETE FROM telemetry")
            conn.executemany(
                "INSERT INTO telemetry (key, value) VALUES (?, ?)",
                list(flat.items()),
            )

    # ── Flatten ↔ unflatten (nested object ↔ dotted-key rows) ─────────────────

    @staticmethod
    def _flatten(ctx: dict[str, object], prefix: str = "") -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in ctx.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict) and value:
                out.update(Telemetry._flatten(value, prefix=f"{full_key}."))
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

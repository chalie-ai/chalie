# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TranscriptService — channel-scoped, instance-based transcript access."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from services.database_service import get_shared_db_service

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


class TranscriptService:
    """Transcript access bound to one ``ProcessorConfig`` and one turn.

    Every query an instance runs is automatically constrained to
    ``config.channel`` — the channel where-clause is applied for the caller, who
    never passes (or forgets) it. ``turn_id`` is resolved once at construction
    (verbatim when supplied, else the channel's next turn) and every method reads
    that fixed ``self.turn_id``."""

    _UNSET_TURN_ID: int = -1  # constructor sentinel → allocate the channel's next turn

    def __init__(self, config: ProcessorConfig, turn_id: int = _UNSET_TURN_ID) -> None:
        self.config = config
        # Cross-turn history reads use the read channel when the config splits
        # read/write (None ⇒ read on the write channel — byte-identical to today
        # for every non-split config). Writes and current-turn identity always
        # use ``config.channel`` (see insert_row / _find_or_make_turn_id).
        self._read_channel = config.read_channel or config.channel
        self.db = get_shared_db_service()
        self.turn_id = self._find_or_make_turn_id(turn_id)

    def get_turn_by_id(self, id: int, sort: str = "id DESC",
                       include_post_settled: bool = True) -> list[dict[str, object]]:
        """Every transcript row of ``(read_channel, id)`` — ``id`` a turn_id — ordered by
        ``sort`` (default ``id DESC``). ``include_post_settled=False`` floors the turn
        at settle0 *in the query* — keeps only ``id <=`` its first ``role='assistant'
        AND settled=1`` row (the MAIN-spine view), returning nothing for a turn with no
        settled reply; ``True`` returns the whole turn (the FORK view). The ``?`` flag
        short-circuits the floor to a no-op when the whole turn is wanted."""
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""SELECT * FROM transcript WHERE channel = ? AND turn_id = ?
                    AND (? OR id <= (SELECT MIN(id) FROM transcript s
                                     WHERE s.channel = transcript.channel
                                       AND s.turn_id = transcript.turn_id
                                       AND s.role = 'assistant' AND s.settled = 1))
                    ORDER BY {sort}""",
                (self._read_channel, id, 1 if include_post_settled else 0),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_turn_rows(self, sort: str = "id DESC") -> list[dict[str, object]]:
        """This instance's own turn — :meth:`get_turn_by_id` bound to ``self.turn_id``,
        the whole turn (post-settle0 included)."""
        return self.get_turn_by_id(self.turn_id, sort)

    def get_turns_since(self, since_turn_id: int = 0) -> list[int]:
        """The read channel's distinct turn_ids above ``since_turn_id`` (a turn_id
        watermark), oldest-first — the MAIN spine's turns not yet folded into the
        checkpoint summary. NULL turn_ids fall out naturally."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT turn_id FROM transcript WHERE channel = ? AND turn_id > ? ORDER BY turn_id",
                (self._read_channel, since_turn_id),
            ).fetchall()
        return [int(r[0]) for r in rows]

    def get_channel_rows(self, sort: str = "id DESC", limit: int = 0, offset: int = 0) -> list[dict[str, object]]:
        """The ``limit``/``offset`` page of this instance's channel rows, ordered by
        ``sort`` (default ``id DESC`` — newest first); ``limit`` 0 means no cap.
        Channel-scoped, unlike turn-scoped :meth:`get_turn_rows`; the page is the
        caller's (e.g. the thread feed's ``limit``/``offset``)."""
        sql = f"SELECT * FROM transcript WHERE channel = ? ORDER BY {sort}"
        args: tuple[object, ...] = (self.config.channel,)
        if limit or offset:
            # SQLite OFFSET requires a LIMIT; -1 = "no cap", so offset works alone.
            sql += " LIMIT ? OFFSET ?"
            args = (*args, limit or -1, offset)
        with self.db.connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def count_rows(self, since_id: int = 0) -> int:
        """Number of this channel's transcript rows with ``id > since_id`` (every
        row when ``since_id`` is 0). Mirrors :meth:`get_turn_rows` — the channel
        where-clause is applied automatically — but returns a ``COUNT(*)`` instead
        of the rows. The ``since_id`` marker (e.g. an episode watermark) is supplied
        by the caller, never computed here."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM transcript WHERE channel = ? AND id > ?",
                (self.config.channel, since_id),
            ).fetchone()
        return int(row[0])

    def insert_row(self, text: str, role: str | None = None, settled: int = 0,
                   internal: int = 0, deliberation_score: float = 0.0) -> int:
        """Append one row to this instance's ``(channel, turn_id)`` and return its id.

        ``role`` defaults to the config's role; location is stamped from the places
        service for user-activity channels only (per source profile) — muted channels
        store NULL. ``xml_migrated=1`` marks the row as already wire-format so the boot
        migration never re-processes it."""
        loc = self._location()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transcript (turn_id, role, channel, content, internal,
                                        deliberation_score, location_lat, location_lon,
                                        location_name, settled, xml_migrated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (self.turn_id, role or self.config.role, self.config.channel,
                 text, internal, deliberation_score,
                 loc.get('lat'), loc.get('lon'), loc.get('name'), settled),
            )
            row_id = cursor.lastrowid
            cursor.close()
        return cast("int", row_id)

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _find_or_make_turn_id(self, turn_id: int) -> int:
        """Resolve the turn to bind to: ``turn_id`` verbatim when one was supplied,
        else the channel's next turn — ``MAX(turn_id) + 1`` (``1`` on an empty
        channel). Called once at construction; thereafter callers read ``self.turn_id``."""
        if turn_id != self._UNSET_TURN_ID:
            return turn_id
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_id), 0) + 1 FROM transcript WHERE channel = ?",
                (self.config.channel,),
            ).fetchone()
        return int(row[0])

    def _location(self) -> dict[str, object]:
        """Live location for a new row, but ONLY for channels whose source profile
        permits backfill (``location_backfill`` — user-activity channels). Muted /
        background channels store NULL so their rows never corrupt the geo signal.
        Returns ``{}`` when gated off or when the lookup fails."""
        from services.source_profiles import profile_for
        if not profile_for(self.config.channel).location_backfill:
            return {}
        from services.locale_service import get_location
        try:
            return get_location()
        except Exception as exc:  # noqa: BLE001 — a geo hiccup must not lose the row
            logger.warning("[TranscriptService] location backfill failed for %s: %s",
                           self.config.channel, exc)
            return {}

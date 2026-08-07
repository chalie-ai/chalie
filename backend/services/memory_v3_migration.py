"""Memory v3 one-time migration — backfill Graph + Map from the old memory
stores before cutover.

Deterministic structural translation (no LLM):
  * every live ``data_graph`` fact  -> a ``memory_graph`` row (subject, contents);
  * every episode (leaf + super)     -> a ``memory_map`` row (contents = gist,
    iteration = level + 1, sourced_from from the episode's transcript_ids).

Also backfills legacy NULL ``turn_id`` on transcript rows so the consolidator can
group them (``turn_id = -id`` matches the read path's COALESCE convention).

The deeper distillation (merging related facts, composing timelines) is the
consolidator's ongoing job; an optional consolidator pass over migrated data is a
follow-up. This migration is the backfill the cutover depends on: the new stores
must carry what already exists before the old tables are removed.
"""

from __future__ import annotations

import json
import logging
from typing import cast

from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.database import Database

logger = logging.getLogger(__name__)


class MemoryV3Migration:
    """One-shot backfill of Graph + Map from the legacy memory stores."""

    def run(self) -> dict[str, int]:
        counts = {
            "turns_backfilled": self._backfill_turn_id(),
            "graph_rows": self._migrate_data_graph(),
            "map_rows": self._migrate_episodes(),
        }
        logger.info("[MEMORY V3 MIGRATION] %s", counts)
        return counts

    def _backfill_turn_id(self) -> int:
        cur = Database.conn().execute(
            "UPDATE transcript SET turn_id = -id WHERE turn_id IS NULL"
        )
        return int(cur.rowcount or 0)

    def _migrate_data_graph(self) -> int:
        rows = cast(
            "list[tuple[str, str, str]]",
            Database.conn().execute(
                "SELECT kind, key, value FROM data_graph "
                "WHERE active = 1 AND deleted_at IS NULL"
            ).fetchall(),
        )
        for kind, key, value in rows:
            MemoryGraphRow(
                subject=f"{kind}.{key}" if kind else key,
                contents=value or "",
            ).save()
        return len(rows)

    def _migrate_episodes(self) -> int:
        rows = cast(
            "list[tuple[str, int, str]]",
            Database.conn().execute(
                "SELECT gist, COALESCE(level, 0), COALESCE(transcript_ids, '[]') "
                "FROM episodes WHERE deleted_at IS NULL"
            ).fetchall(),
        )
        n = 0
        for gist, level, tids_json in rows:
            if not gist:
                continue
            raw = json.loads(tids_json or "[]")
            tids = [i for i in raw if isinstance(i, int)] if isinstance(raw, list) else []
            MemoryMapRow(
                contents=gist,
                iteration=int(level) + 1,
                sourced_from=json.dumps(tids),
            ).save()
            n += 1
        return n

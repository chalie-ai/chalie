"""ThreadGistService — persist + retrieve per-thread one-sentence gists.

Stores one current gist per ``(channel, turn_id)`` in ``thread_gist``, with a
companion ``thread_gist_vec`` (sqlite-vec KNN) and ``thread_gist_fts`` (FTS5)
index mirroring the episodes/documents pattern. Cross-thread pollination
surfaces the top-N related gists (excluding the active thread) for context
assembly; ``search`` reuses the same index for the thread-search endpoint.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Optional, cast

from services._fts_delete import fts5_external_delete
from services.database_service import get_shared_db_service
from services.embedding_service import get_embedding_service
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)
LOG_PREFIX = "[THREAD GIST]"


def _l2_dist_to_cosine(distance: float) -> float:
    return max(0.0, 1.0 - (distance ** 2 / 2.0))


class ThreadGistService:
    """Static-style service: upsert + search over the thread_gist index."""

    def upsert(self, channel: str, turn_id: int, summary: str) -> None:
        """Persist (or replace) the gist for one thread, embedding + FTS-syncing it."""
        summary = (summary or "").strip()
        if not summary:
            return
        try:
            embedding = get_embedding_service().generate_embedding(summary)
            blob = pack_embedding(embedding)
            db = get_shared_db_service()
            with db.connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id FROM thread_gist WHERE channel = ? AND turn_id = ?",
                    (channel, turn_id),
                )
                existing = cur.fetchone()
                if existing:
                    gist_id = cast(int, existing[0])
                    self._delete_fts(conn, gist_id, cur)
                    cur.execute(
                        "UPDATE thread_gist SET summary = ?, updated_at = datetime('now') "
                        "WHERE id = ?",
                        (summary, gist_id),
                    )
                else:
                    cur.execute(
                        "INSERT INTO thread_gist (channel, turn_id, summary) VALUES (?, ?, ?)",
                        (channel, turn_id, summary),
                    )
                    gist_id = cast(int, cur.lastrowid)
                if blob is not None:
                    cur.execute(
                        "INSERT OR REPLACE INTO thread_gist_vec(rowid, embedding) VALUES (?, ?)",
                        (gist_id, blob),
                    )
                cur.execute(
                    "INSERT INTO thread_gist_fts(rowid, summary) VALUES (?, ?)",
                    (gist_id, summary),
                )
                cur.close()
        except Exception as exc:
            logger.warning("%s upsert failed for %s turn=%s: %s", LOG_PREFIX, channel, turn_id, exc)

    def _delete_fts(self, conn: sqlite3.Connection, gist_id: int, cur: sqlite3.Cursor) -> None:
        cur.execute("SELECT summary FROM thread_gist WHERE id = ?", (gist_id,))
        row = cur.fetchone()
        if row:
            fts5_external_delete(conn, "thread_gist_fts", gist_id, {"summary": cast(str, row[0])})

    def get(self, channel: str, turn_id: int) -> Optional[str]:
        """The current gist for one thread, or None."""
        try:
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT summary FROM thread_gist WHERE channel = ? AND turn_id = ?",
                    (channel, turn_id),
                ).fetchone()
            return cast(str, row[0]) if row else None
        except Exception as exc:
            logger.debug("%s get failed for %s turn=%s: %s", LOG_PREFIX, channel, turn_id, exc)
            return None

    def pollinate(self, channel: str, active_turn_id: int, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        """Top-N related gists excluding the active thread — cross-thread pollination."""
        try:
            query_emb = get_embedding_service().generate_embedding(query)
            query_blob = pack_embedding(query_emb) if query_emb else None
            candidates: dict[int, dict[str, object]] = {}
            db = get_shared_db_service()
            with db.connection() as conn:
                cur = conn.cursor()
                if query_blob is not None:
                    try:
                        cur.execute(
                            "SELECT rowid, distance FROM thread_gist_vec "
                            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                            (query_blob, limit * 3),
                        )
                        for rowid, dist in cur.fetchall():
                            cos = _l2_dist_to_cosine(dist)
                            candidates.setdefault(rowid, {"cos": 0.0, "fts": 0.0})
                            candidates[rowid]["cos"] = max(cast(float, candidates[rowid]["cos"]), cos)
                    except Exception as exc:
                        logger.debug("%s vec search failed (non-fatal): %s", LOG_PREFIX, exc)
                self._fts_search(cur, candidates, query)
                cur.close()

            if not candidates:
                return []
            results = self._score_and_fetch(channel, active_turn_id, candidates, limit)
            return results
        except Exception as exc:
            logger.warning("%s pollinate failed: %s", LOG_PREFIX, exc)
            return []

    def search(self, query: str, *, channel: Optional[str] = None, limit: int = 10) -> list[dict[str, object]]:
        """Hybrid KNN+FTS search over all thread gists — the thread-search endpoint."""
        try:
            query_emb = get_embedding_service().generate_embedding(query)
            query_blob = pack_embedding(query_emb) if query_emb else None
            candidates: dict[int, dict[str, object]] = {}
            db = get_shared_db_service()
            with db.connection() as conn:
                cur = conn.cursor()
                if query_blob is not None:
                    try:
                        cur.execute(
                            "SELECT rowid, distance FROM thread_gist_vec "
                            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                            (query_blob, limit * 3),
                        )
                        for rowid, dist in cur.fetchall():
                            cos = _l2_dist_to_cosine(dist)
                            candidates.setdefault(rowid, {"cos": 0.0, "fts": 0.0})
                            candidates[rowid]["cos"] = max(cast(float, candidates[rowid]["cos"]), cos)
                    except Exception as exc:
                        logger.debug("%s vec search failed (non-fatal): %s", LOG_PREFIX, exc)
                self._fts_search(cur, candidates, query)
                cur.close()

            if not candidates:
                return []
            return self._score_and_fetch(None, None, candidates, limit)
        except Exception as exc:
            logger.warning("%s search failed: %s", LOG_PREFIX, exc)
            return []

    def _fts_search(self, cur: sqlite3.Cursor, candidates: dict[int, dict[str, object]], query: str) -> None:
        try:
            fts_query = re.sub(r"[^\w\s]", "", query).strip()
            if not fts_query:
                return
            terms = " OR ".join(f'"{w}"*' for w in fts_query.split() if w)
            if not terms:
                return
            cur.execute(
                "SELECT rowid, rank FROM thread_gist_fts "
                "WHERE thread_gist_fts MATCH ? ORDER BY rank LIMIT 30",
                (terms,),
            )
            for rowid, _rank in cur.fetchall():
                candidates.setdefault(rowid, {"cos": 0.0, "fts": 0.0})
                candidates[rowid]["fts"] = 1.0
        except Exception as exc:
            logger.debug("%s FTS search failed (non-fatal): %s", LOG_PREFIX, exc)

    def _score_and_fetch(
        self,
        channel: Optional[str],
        active_turn_id: Optional[int],
        candidates: dict[int, dict[str, object]],
        limit: int,
    ) -> list[dict[str, object]]:
        db = get_shared_db_service()
        scored: list[dict[str, object]] = []
        with db.connection() as conn:
            cur = conn.cursor()
            for rowid, sigs in candidates.items():
                cur.execute(
                    "SELECT id, channel, turn_id, summary, updated_at FROM thread_gist WHERE id = ?",
                    (rowid,),
                )
                row = cur.fetchone()
                if not row:
                    continue
                gist_channel, gist_turn_id = cast(str, row[1]), cast(int, row[2])
                if channel is not None and gist_channel == channel and active_turn_id is not None and gist_turn_id == active_turn_id:
                    continue
                composite = cast(float, sigs["cos"]) + 0.3 * cast(float, sigs["fts"])
                scored.append({
                    "turn_id": gist_turn_id,
                    "channel": gist_channel,
                    "summary": cast(str, row[3]),
                    "updated_at": cast(str, row[4]),
                    "score": composite,
                })
            cur.close()
        scored.sort(key=lambda d: cast(float, d["score"]), reverse=True)
        return scored[:limit]


_instance: Optional[ThreadGistService] = None


def get_thread_gist_service() -> ThreadGistService:
    global _instance
    if _instance is None:
        _instance = ThreadGistService()
    return _instance
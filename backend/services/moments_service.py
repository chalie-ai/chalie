"""Moments service — persistence + search for user-curated assistant-reply pins.

A *moment* is an explicit "remember this" bookmark on one assistant transcript
turn. It lives in the dedicated ``moments`` table (schema.sql) outside the
data_graph knowledge graph, with its own ``moments_fts`` (lexical) and
``moments_vec`` (semantic KNN) indexes. A pin carries no retrieval_weight, no
decay and no janitor on this table — it lives until the user forgets it.

This service is the single home for the moments FTS + vec write/delete sync, so
the index-coherence idiom can never drift across call sites. It mirrors
``DataGraphService``: on store it inserts the row, syncs ``moments_fts``, embeds
the content via the shared ``EmbeddingService`` (synchronous — a pin is a
deliberate user action, not the chat hot path) and writes ``moments_vec``; on
delete it removes the row, the external-content FTS posting and the vec row.

Consumers:
  * ``api/moments.py`` — the pin/list/forget/search HTTP surface.
  * ``services/memory_retrieval.py`` — the explicit-recall moments lane.
"""

import logging
from typing import List, Optional, cast

import sqlite3

from services._fts_delete import fts5_external_delete
from services.database_service import DatabaseService, get_shared_db_service
from services.embedding_utils import pack_embedding
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[MOMENTS]"

# FTS5 external-content column set for moments_fts — order MUST match the schema
# (a single ``content`` column). Shared by the sync and delete idioms.
_FTS_TABLE = "moments_fts"
_VEC_TABLE = "moments_vec"

# How many lexical + semantic candidates to pull per search lane before dedup.
_SEARCH_LIMIT = 10
# KNN depth for the vec lane — a small multiple of the surfaced limit.
_VEC_K = 30

class MomentsService:
    def __init__(self, db_service: Optional[DatabaseService] = None) -> None:
        self.db: DatabaseService = db_service if db_service is not None else get_shared_db_service()

    # ── helpers ───────────────────────────────────────────────────────

    def _row_to_dict(self, row: Optional[sqlite3.Row]) -> Optional[dict[str, object]]:
        return dict(row) if row is not None else None

    def _generate_embedding(self, text: str) -> object:
        try:
            from services.embedding_service import get_embedding_service
            return get_embedding_service().generate_embedding(text)
        except Exception as exc:
            logger.warning("%s embedding generation failed: %s", _LOG_PREFIX, exc)
            return None

    def _sync_fts(self, conn: sqlite3.Connection, rowid: int, content: str) -> None:
        """Insert the content posting into the external-content moments_fts index.

        Failure only logs a warning — this is a soft sync against an external index;
        an individual row missing from FTS does not corrupt data.
        """
        try:
            conn.execute(
                f"INSERT INTO {_FTS_TABLE}(rowid, content) VALUES (?, ?)",
                (rowid, content or ""),
            )
        except Exception as exc:
            logger.warning("%s FTS sync failed for rowid=%s: %s", _LOG_PREFIX, rowid, exc)

    def _store_vec(self, conn: sqlite3.Connection, rowid: int, embedding: object) -> None:
        if embedding is None:
            return
        blob = pack_embedding(embedding)
        if blob is None:
            return
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO {_VEC_TABLE}(rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
        except Exception as exc:
            logger.warning("%s vec write failed for rowid=%s: %s", _LOG_PREFIX, rowid, exc)

    def _delete_fts(self, conn: sqlite3.Connection, rowid: int, content: str) -> None:
        fts5_external_delete(conn, _FTS_TABLE, rowid, {"content": content})

    # ── store / read / delete ──────────────────────────────────────────

    def find_by_transcript(self, transcript_id: int) -> Optional[dict[str, object]]:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM moments WHERE transcript_id = ? LIMIT 1",
                (transcript_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def store(self, transcript_id: int, content: str, *, note: str | None = None) -> dict[str, object] | None:
        """Dedupe on ``transcript_id`` — re-pin returns existing row instead of inserting.

        Embedding failure is non-fatal; the lexical lane still serves the pin.
        """
        existing = self.find_by_transcript(transcript_id)
        if existing is not None:
            return existing

        now_iso = utc_now().isoformat()
        embedding = self._generate_embedding(content)
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO moments (transcript_id, content, note, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(transcript_id) DO NOTHING",
                (transcript_id, content, note, now_iso),
            )
            inserted = cursor.rowcount == 1
            rowid = cast(int, cursor.lastrowid)
            cursor.close()
            if inserted:
                self._sync_fts(conn, rowid, content)
                self._store_vec(conn, rowid, embedding)
                row = conn.execute("SELECT * FROM moments WHERE id = ?", (rowid,)).fetchone()
            else:
                # Concurrent pin won the race — its row (with its own FTS/vec
                # postings) already exists; return it so the caller dedupes.
                row = conn.execute(
                    "SELECT * FROM moments WHERE transcript_id = ? LIMIT 1",
                    (transcript_id,),
                ).fetchone()
        if inserted:
            logger.info("%s pinned moment id=%s transcript_id=%s", _LOG_PREFIX, rowid, transcript_id)
        else:
            logger.info(
                "%s pin lost the dedup race for transcript_id=%s; returning existing row",
                _LOG_PREFIX, transcript_id,
            )
        return self._row_to_dict(row)

    def list_all(self) -> List[dict[str, object] | None]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM moments ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete_by_transcript(self, transcript_id: int) -> bool:
        row = self.find_by_transcript(transcript_id)
        if row is None:
            return False
        rowid = cast(int, row["id"])
        with self.db.connection() as conn:
            self._delete_fts(conn, rowid, cast(str, row.get("content") or ""))
            conn.execute(f"DELETE FROM {_VEC_TABLE} WHERE rowid = ?", (rowid,))
            conn.execute("DELETE FROM moments WHERE id = ?", (rowid,))
        logger.info("%s forgot moment id=%s transcript_id=%s", _LOG_PREFIX, rowid, transcript_id)
        return True

    # ── search ─────────────────────────────────────────────────────────

    def _search_fts(self, conn: sqlite3.Connection, query: str) -> List[int]:
        import re
        terms = re.sub(r"[^\w\s]", " ", query).split()
        if not terms:
            return []
        match = " OR ".join(f'"{t}"*' for t in terms)
        try:
            rows = conn.execute(
                f"SELECT rowid FROM {_FTS_TABLE} WHERE {_FTS_TABLE} MATCH ? "
                "ORDER BY rank LIMIT ?",
                (match, _SEARCH_LIMIT),
            ).fetchall()
        except Exception as exc:
            logger.debug("%s FTS search failed (non-fatal): %s", _LOG_PREFIX, exc)
            return []
        return [r[0] for r in rows]

    def _search_vec(self, conn: sqlite3.Connection, query: str) -> List[int]:
        embedding = self._generate_embedding(query)
        blob = pack_embedding(embedding) if embedding is not None else None
        if blob is None:
            return []
        try:
            rows = conn.execute(
                f"SELECT rowid, distance FROM {_VEC_TABLE} "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (blob, _VEC_K),
            ).fetchall()
        except Exception as exc:
            logger.debug("%s vec search failed (non-fatal): %s", _LOG_PREFIX, exc)
            return []
        return [r[0] for r in rows]

    def search(self, query: str, limit: int = _SEARCH_LIMIT) -> List[dict[str, object] | None]:
        clean = (query or "").strip()
        if not clean:
            return []
        with self.db.connection() as conn:
            ordered_ids: List[int] = []
            seen = set()
            for mid in self._search_fts(conn, clean) + self._search_vec(conn, clean):
                if mid not in seen:
                    seen.add(mid)
                    ordered_ids.append(mid)
            ordered_ids = ordered_ids[:limit]
            rows = []
            for mid in ordered_ids:
                row = conn.execute("SELECT * FROM moments WHERE id = ?", (mid,)).fetchone()
                if row is not None:
                    rows.append(self._row_to_dict(row))
        return rows


# ── Singleton ──────────────────────────────────────────────────────────

_instance: Optional[MomentsService] = None


def get_moments_service() -> MomentsService:
    global _instance
    db = get_shared_db_service()
    if _instance is None or _instance.db is not db:
        _instance = MomentsService(db)
    return _instance

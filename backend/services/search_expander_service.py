


import json
import logging
import sqlite3
import threading
from typing import Optional, cast


from services.database_service import DatabaseService, get_shared_db_service
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_QUEUE_KEY = "ses:queue"          # MemoryStore list key — FIFO via rpush/lpop
_TABLE_DATA_GRAPH = "data_graph"
_VALID_TABLES = frozenset({_TABLE_DATA_GRAPH})


# ── Module-level singleton references ─────────────────────────────────────────

_service_instance: Optional["SearchExpanderService"] = None
_instance_lock = threading.Lock()


def _get_service() -> "SearchExpanderService":
    global _service_instance
    if _service_instance is not None:
        return _service_instance
    with _instance_lock:
        if _service_instance is None:
            _service_instance = SearchExpanderService()
    return _service_instance


def enqueue(table: str, rowid: int) -> None:
    if table not in _VALID_TABLES:
        logger.warning("[SES] enqueue: unknown table '%s', ignoring", table)
        return
    _get_service().enqueue(table, rowid)


# ── Service ───────────────────────────────────────────────────────────────────

class SearchExpanderService:

    def __init__(self, db_service: DatabaseService | None = None) -> None:
        self._db = db_service or get_shared_db_service()
        self._event = threading.Event()

        # MemoryStore queue — lazy import so tests can stub the store before init
        from services.memory_store import get_shared_store
        self._store = get_shared_store()

    # ── Public interface ──────────────────────────────────────────────────────

    def enqueue(self, table: str, rowid: int) -> None:
        item = json.dumps({"table": table, "rowid": rowid})
        self._store.rpush(_QUEUE_KEY, item)
        self._event.set()

    def run(self) -> None:
        logger.info("[SES] Worker starting")
        self._self_heal()
        logger.info("[SES] Self-heal complete — entering event wait loop")

        while True:
            self._event.wait()
            self._event.clear()
            while True:
                item = self._dequeue()
                if item is None:
                    break
                self._process(item)

    # ── Private: queue ────────────────────────────────────────────────────────

    def _dequeue(self) -> Optional[dict[str, object]]:
        raw = self._store.lpop(_QUEUE_KEY)
        if raw is None:
            return None
        try:
            return cast(dict[str, object], json.loads(raw))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("[SES] Malformed queue item — skipping: %s (raw=%r)", e, raw)
            return None

    # ── Private: self-heal ────────────────────────────────────────────────────

    def _self_heal(self) -> None:
        try:
            with self._db.connection() as conn:
                data_graph_ids = [
                    r[0] for r in conn.execute(
                        "SELECT id FROM data_graph "
                        "WHERE search_queries IS NULL AND deleted_at IS NULL AND active=1"
                    ).fetchall()
                ]

            for rowid in data_graph_ids:
                self._store.rpush(_QUEUE_KEY, json.dumps({"table": _TABLE_DATA_GRAPH, "rowid": rowid}))

            total = len(data_graph_ids)
            if total:
                logger.info(
                    "[SES] Self-heal enqueued %d row(s) (data_graph=%d)",
                    total, len(data_graph_ids),
                )
                self._event.set()
        except Exception as e:
            logger.warning("[SES] Self-heal scan failed: %s", e)

    # ── Private: per-item processing ──────────────────────────────────────────

    def _process(self, item: dict[str, object]) -> None:
        table = item.get("table")
        rowid = item.get("rowid")

        if table not in _VALID_TABLES or rowid is None:
            logger.warning("[SES] Invalid item — skipping: %r", item)
            return

        try:
            self._process_data_graph(cast(int, rowid))
        except Exception as e:
            logger.warning("[SES] Processing failed for %s rowid=%s: %s", table, rowid, e)

    def _process_data_graph(self, rowid: int) -> None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT key, value, kind FROM data_graph WHERE id = ?",
                (rowid,)
            ).fetchone()

        if row is None:
            logger.debug("[SES] data_graph rowid=%s gone — skipping", rowid)
            return

        key, value, _ = row[0], row[1], row[2]
        variants = self._generate_variants(key, value)

        with self._db.connection() as conn:
            # Absorbs _schedule_embeddings: populate key_vec/value_vec if missing.
            self._backfill_key_value_vec(conn, rowid, key, value)
            self._write_variants(conn, _TABLE_DATA_GRAPH, rowid, variants)
            self._update_search_queries_data_graph(conn, rowid, variants)

    def _generate_variants(self, key: str, value: str) -> list[str]:
        """Empty return means skip variant writes but still mark search_queries so self-heal doesn't re-enqueue the row."""
        try:
            from services.doc2query_service import get_doc2query_service
            d2q = get_doc2query_service()
            if not d2q.is_available():
                return []
            text = f"{key}: {value}" if value else key
            return d2q.generate_queries(text) or []
        except Exception as e:
            logger.warning("[SES] doc2query failed for key='%s': %s", key, e)
            return []

    def _write_variants(self, conn: sqlite3.Connection, table: str, rowid: int, variants: list[str]) -> None:
        """Write each variant string + embedding into expanded_semantic / expanded_semantic_vec.

        Clears existing variants for this (table, rowid) first so re-processing
        is idempotent. The cascade trigger on expanded_semantic handles vec cleanup.
        """
        conn.execute(
            "DELETE FROM expanded_semantic "
            "WHERE relates_to_table = ? AND related_to_id = ?",
            (table, rowid)
        )

        if not variants:
            return

        try:
            from services.embedding_service import get_embedding_service
            emb_svc = get_embedding_service()
        except Exception as e:
            logger.warning("[SES] EmbeddingService unavailable — skipping variant vecs: %s", e)
            return

        for variant_text in variants:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO expanded_semantic (relates_to_table, related_to_id, str) "
                    "VALUES (?, ?, ?)",
                    (table, rowid, variant_text)
                )
                new_id = cursor.lastrowid
                cursor.close()

                emb = emb_svc.generate_embedding(variant_text)
                blob = pack_embedding(emb)
                if blob:
                    conn.execute(
                        "INSERT OR REPLACE INTO expanded_semantic_vec (rowid, embedding) "
                        "VALUES (?, ?)",
                        (new_id, blob)
                    )
            except Exception as e:
                logger.warning("[SES] Failed to write variant '%s': %s", variant_text[:60], e)

    def _backfill_key_value_vec(self, conn: sqlite3.Connection, rowid: int, key: str, value: str) -> None:
        """Populate data_graph_key_vec / data_graph_value_vec if the row is missing.

        Absorbs the old _schedule_embeddings thread. Noop when both vecs exist.
        """
        try:
            key_exists = conn.execute(
                "SELECT 1 FROM data_graph_key_vec WHERE rowid = ?", (rowid,)
            ).fetchone()
            if not key_exists:
                from services.embedding_service import get_embedding_service
                emb_svc = get_embedding_service()
                key_emb = emb_svc.generate_embedding(key) if key else None
                if key_emb:
                    blob = pack_embedding(key_emb)
                    if blob:
                        conn.execute(
                            "INSERT OR REPLACE INTO data_graph_key_vec (rowid, embedding) "
                            "VALUES (?, ?)",
                            (rowid, blob)
                        )

            value_exists = conn.execute(
                "SELECT 1 FROM data_graph_value_vec WHERE rowid = ?", (rowid,)
            ).fetchone()
            if not value_exists:
                from services.embedding_service import get_embedding_service
                emb_svc = get_embedding_service()
                value_text = value if value else key
                val_emb = emb_svc.generate_embedding(value_text) if value_text else None
                if val_emb:
                    blob = pack_embedding(val_emb)
                    if blob:
                        conn.execute(
                            "INSERT OR REPLACE INTO data_graph_value_vec (rowid, embedding) "
                            "VALUES (?, ?)",
                            (rowid, blob)
                        )
        except Exception as e:
            logger.warning("[SES] backfill_key_value_vec failed for rowid=%s: %s", rowid, e)

    def _update_search_queries_data_graph(
        self, conn: sqlite3.Connection, rowid: int, variants: list[str]
    ) -> None:
        """Persist variant texts in data_graph.search_queries and resync FTS."""
        old = conn.execute(
            "SELECT key, value, kind, search_queries FROM data_graph WHERE id = ?",
            (rowid,)
        ).fetchone()
        if old is None:
            return

        # Delete stale FTS entry before overwriting content
        self._delete_data_graph_fts(conn, rowid, old[0], old[1], old[2], old[3] or '')

        conn.execute(
            "UPDATE data_graph SET search_queries = ? WHERE id = ?",
            (json.dumps(variants), rowid)
        )
        try:
            conn.execute(
                "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) "
                "VALUES (?, ?, ?, ?, ?)",
                (rowid, old[0], old[1] or '', old[2], json.dumps(variants))
            )
        except Exception as e:
            logger.warning("[SES] data_graph FTS sync failed for rowid=%s: %s", rowid, e)

    # ── Private: FTS helpers ──────────────────────────────────────────────────

    def _delete_data_graph_fts(self, conn: sqlite3.Connection, rowid: int, key: str, value: str,
                                kind: str, search_queries: str) -> None:
        """Remove a data_graph FTS entry using the external-content delete command."""
        try:
            conn.execute(
                "INSERT INTO data_graph_fts"
                "(data_graph_fts, rowid, key, value, kind, search_queries) "
                "VALUES('delete', ?, ?, ?, ?, ?)",
                (rowid, key, value or '', kind, search_queries)
            )
        except Exception:
            try:
                conn.execute("DELETE FROM data_graph_fts WHERE rowid = ?", (rowid,))
            except Exception as e:
                logger.warning("[SES] data_graph FTS delete failed for rowid=%s: %s", rowid, e)


# ── Worker entry point ────────────────────────────────────────────────────────

def search_expander_worker() -> None:
    """Entry point registered in run.py via _try_register.

    Creates the singleton and enters the blocking run loop. Registered with
    _try_register so boot continues gracefully when doc2query models are absent.
    """
    service = SearchExpanderService()
    # Share the singleton so module-level enqueue() calls reach the same instance.
    global _service_instance
    with _instance_lock:
        _service_instance = service
    service.run()

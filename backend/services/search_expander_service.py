


import json
import logging
import sqlite3
import threading
from typing import Optional, cast

from contracts.search_config import SearchConfig, config_for_kind, is_searchable
from services.database import Database
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_QUEUE_KEY = "ses:queue"          # MemoryStore list key — FIFO via rpush/lpop
_TABLE_DATA_GRAPH = "data_graph"
_VALID_TABLES = frozenset({_TABLE_DATA_GRAPH})

# ── FTS-indexing policy — the ``Searchable`` trait ──────────────────────────────
# Whether a data_graph row earns a search posting is the model's declared config:
# ``is_searchable(kind)`` reflects the kind→config registry populated by
# ``DataGraphRow.__init_subclass__``, mirroring the save path's
# ``self.__search__ is not None``. Non-searchable kinds (behavioral_pattern,
# machine_state) declare ``__search__ = None`` and are excluded here too, so
# neither enqueue path (save-time or self-heal) indexes them. ``contracts``
# imports nothing from models/services, so this dependency does not cycle.


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

    def __init__(self) -> None:
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
            conn = Database.conn()
            data_graph_ids = [
                r[0] for r in conn.execute(
                    "SELECT id, kind FROM data_graph "
                    "WHERE search_queries IS NULL AND deleted_at IS NULL AND active=1"
                ).fetchall()
                if is_searchable(r[1])
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
        conn = Database.conn()
        row = conn.execute(
            "SELECT key, value, kind FROM data_graph WHERE id = ?",
            (rowid,)
        ).fetchone()

        if row is None:
            logger.debug("[SES] data_graph rowid=%s gone — skipping", rowid)
            return

        key, value, kind = row[0], row[1], row[2]
        config = config_for_kind(kind)
        if config is None:
            logger.warning(
                "[SES] no search config for kind=%s rowid=%s — skipping", kind, rowid
            )
            return
        variants = self._generate_variants(key, value)

        with Database.transaction() as conn:
            # Absorbs _schedule_embeddings: populate the declared vec lanes if missing.
            self._backfill_key_value_vec(conn, rowid, key, value, config)
            self._write_variants(conn, _TABLE_DATA_GRAPH, rowid, variants, config)
            self._update_search_queries_data_graph(conn, rowid, variants, config)

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

    def _write_variants(
        self, conn: sqlite3.Connection, table: str, rowid: int,
        variants: list[str], config: SearchConfig,
    ) -> None:
        """Write each variant string + embedding into the declared variant tables
        (``config.variant_table`` / ``config.variant_vec_table``).

        Clears existing variants for this (table, rowid) first so re-processing
        is idempotent. The cascade trigger on the variant table handles vec
        cleanup. ``table`` is the ``relates_to_table`` value (the base table the
        variant points back at), distinct from the variant storage table.
        """
        conn.execute(
            f"DELETE FROM {config.variant_table} "
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
                    f"INSERT INTO {config.variant_table} (relates_to_table, related_to_id, str) "
                    "VALUES (?, ?, ?)",
                    (table, rowid, variant_text)
                )
                new_id = cursor.lastrowid
                cursor.close()

                emb = emb_svc.generate_embedding(variant_text)
                blob = pack_embedding(emb)
                if blob:
                    conn.execute(
                        f"INSERT OR REPLACE INTO {config.variant_vec_table} (rowid, embedding) "
                        "VALUES (?, ?)",
                        (new_id, blob)
                    )
            except Exception as e:
                logger.warning("[SES] Failed to write variant '%s': %s", variant_text[:60], e)

    def _backfill_key_value_vec(
        self, conn: sqlite3.Connection, rowid: int, key: str, value: str,
        config: SearchConfig,
    ) -> None:
        """Populate each declared vec lane (``config.vec_lanes``) if the row is
        missing it — one embedding per lane, sourced from the base-table column
        the lane names, falling back to ``key`` when that column is empty
        (preserving data_graph's value→key fallback).

        Absorbs the old _schedule_embeddings thread. Noop when every lane's vec
        row already exists.
        """
        sources = {"key": key, "value": value}
        try:
            emb_svc = None
            for lane in config.vec_lanes:
                exists = conn.execute(
                    f"SELECT 1 FROM {lane.table} WHERE rowid = ?", (rowid,)
                ).fetchone()
                if exists:
                    continue
                text = sources.get(lane.source) or key
                if not text:
                    continue
                if emb_svc is None:
                    from services.embedding_service import get_embedding_service
                    emb_svc = get_embedding_service()
                emb = emb_svc.generate_embedding(text)
                if emb:
                    blob = pack_embedding(emb)
                    if blob:
                        conn.execute(
                            f"INSERT OR REPLACE INTO {lane.table} (rowid, embedding) "
                            "VALUES (?, ?)",
                            (rowid, blob)
                        )
        except Exception as e:
            logger.warning("[SES] backfill_key_value_vec failed for rowid=%s: %s", rowid, e)

    def _update_search_queries_data_graph(
        self, conn: sqlite3.Connection, rowid: int, variants: list[str],
        config: SearchConfig,
    ) -> None:
        """Persist variant texts in the declared queries column and resync the
        declared FTS table. Table + column names come from ``config`` so the same
        path serves any searchable model; data_graph's behaviour is unchanged."""
        base_cols = [c for c in config.fts_columns if c != config.queries_column]
        old = conn.execute(
            f"SELECT {', '.join(base_cols)}, {config.queries_column} "
            "FROM data_graph WHERE id = ?",
            (rowid,)
        ).fetchone()
        if old is None:
            return
        row_vals = dict(zip(base_cols, old))
        prior_queries = old[len(base_cols)]
        queries_json = json.dumps(variants)

        # The external-content FTS index is populated ONLY here, in lock-step
        # with the queries column: this method sets it non-NULL exactly when it
        # inserts the posting, and no trigger writes the index. So the queries
        # column IS NULL <=> the row was never indexed, and issuing the FTS5
        # 'delete' command for a posting that was never inserted corrupts the
        # index (delete-before-first-insert). Only remove a prior posting when
        # one exists; a first index goes straight to INSERT.
        if prior_queries is not None:
            self._delete_data_graph_fts(
                conn, rowid, {**row_vals, config.queries_column: prior_queries}, config
            )

        conn.execute(
            f"UPDATE data_graph SET {config.queries_column} = ? WHERE id = ?",
            (queries_json, rowid)
        )
        cols = ", ".join(config.fts_columns)
        placeholders = ", ".join(["?"] * (len(config.fts_columns) + 1))
        fts_values = [
            queries_json if col == config.queries_column else (row_vals.get(col) or '')
            for col in config.fts_columns
        ]
        try:
            conn.execute(
                f"INSERT INTO {config.fts_table}(rowid, {cols}) VALUES ({placeholders})",
                (rowid, *fts_values)
            )
        except Exception as e:
            logger.warning("[SES] data_graph FTS sync failed for rowid=%s: %s", rowid, e)

    # ── Private: FTS helpers ──────────────────────────────────────────────────

    def _delete_data_graph_fts(
        self, conn: sqlite3.Connection, rowid: int,
        indexed: dict[str, object], config: SearchConfig,
    ) -> None:
        """Remove an FTS entry via the external-content 'delete' command. The
        indexed column values (``indexed``, keyed by column name) MUST be listed
        in ``config.fts_columns`` order — the FTS5 external-content requirement —
        with None coerced to ''."""
        cols = ", ".join(config.fts_columns)
        placeholders = ", ".join(["?"] * (len(config.fts_columns) + 1))
        values = [indexed.get(col) or '' for col in config.fts_columns]
        try:
            conn.execute(
                f"INSERT INTO {config.fts_table}"
                f"({config.fts_table}, rowid, {cols}) "
                f"VALUES('delete', {placeholders})",
                (rowid, *values)
            )
        except Exception:
            try:
                conn.execute(f"DELETE FROM {config.fts_table} WHERE rowid = ?", (rowid,))
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

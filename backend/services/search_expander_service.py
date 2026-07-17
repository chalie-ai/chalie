


import json
import logging
import sqlite3
import threading
from typing import Optional, cast

from contracts.search_config import (
    SearchConfig,
    config_for_table,
    is_searchable,
    searchable_tables,
)
from services.database import Database
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_QUEUE_KEY = "ses:queue"          # MemoryStore list key — FIFO via rpush/lpop

# ── FTS-indexing policy — the ``Searchable`` trait ──────────────────────────────
# The engine is table-driven: every sidecar table/column name comes from the
# model's declared ``SearchConfig`` (contracts.search_config), resolved by the
# queued base-table name via ``config_for_table``. A table with no config is
# never indexed. For a table that holds many kinds in one shell (``data_graph``,
# via ``kind_column``), ``is_searchable(kind)`` further gates each row against
# the kind→config registry populated by ``DataGraphRow.__init_subclass__``,
# mirroring the save path's ``self.__search__ is not None`` — so non-searchable
# kinds (behavioral_pattern, machine_state) enter no index from either enqueue
# path (save-time or self-heal). ``contracts`` imports nothing from
# models/services, so this dependency does not cycle.


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
    if config_for_table(table) is None:
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
        """Re-enqueue every searchable row missing its index — a row whose
        ``queries_column`` is still NULL (never indexed) and still live under the
        table's ``heal_where`` predicate. Scans each registered base table; for a
        table with a ``kind_column`` (``data_graph``), each row is further gated
        through ``is_searchable(kind)`` so non-searchable kinds are skipped."""
        try:
            conn = Database.conn()
            total = 0
            for table in searchable_tables():
                config = config_for_table(table)
                if config is None:
                    continue
                select = "rowid" + (f", {config.kind_column}" if config.kind_column else "")
                rows = conn.execute(
                    f"SELECT {select} FROM {config.base_table} "
                    f"WHERE {config.queries_column} IS NULL AND {config.heal_where}"
                ).fetchall()
                enqueued = 0
                for r in rows:
                    if config.kind_column and not is_searchable(r[1]):
                        continue
                    self._store.rpush(_QUEUE_KEY, json.dumps({"table": table, "rowid": r[0]}))
                    enqueued += 1
                if enqueued:
                    logger.info("[SES] Self-heal enqueued %d row(s) (%s)", enqueued, table)
                total += enqueued
            if total:
                self._event.set()
        except Exception as e:
            logger.warning("[SES] Self-heal scan failed: %s", e)

    # ── Private: per-item processing ──────────────────────────────────────────

    def _process(self, item: dict[str, object]) -> None:
        table = item.get("table")
        rowid = item.get("rowid")

        config = config_for_table(table) if isinstance(table, str) else None
        if config is None or not isinstance(table, str) or rowid is None:
            logger.warning("[SES] Invalid item — skipping: %r", item)
            return

        try:
            self._process_row(table, cast(int, rowid), config)
        except Exception as e:
            logger.warning("[SES] Processing failed for %s rowid=%s: %s", table, rowid, e)

    def _process_row(self, table: str, rowid: int, config: SearchConfig) -> None:
        """Index one base-table row through its declared config: backfill the vec
        lanes, write the doc2query variants, and resync the FTS posting — all
        addressed by ``rowid``, all names read off ``config``."""
        conn = Database.conn()

        # The columns the write path reads: doc2query/embedding source text, the
        # vec-lane sources, and the kind discriminator when the table gates on one.
        needed = list(dict.fromkeys(
            [*config.text_columns, *(lane.source for lane in config.vec_lanes)]
            + ([config.kind_column] if config.kind_column else [])
        ))
        row = conn.execute(
            f"SELECT {', '.join(needed)} FROM {config.base_table} WHERE rowid = ?",
            (rowid,)
        ).fetchone()
        if row is None:
            logger.debug("[SES] %s rowid=%s gone — skipping", table, rowid)
            return
        vals = dict(zip(needed, row))

        if config.kind_column and not is_searchable(vals[config.kind_column]):
            logger.warning(
                "[SES] no search config for kind=%s rowid=%s — skipping",
                vals[config.kind_column], rowid,
            )
            return

        variants = self._generate_variants(self._source_text(vals, config))

        with Database.transaction() as conn:
            # Absorbs _schedule_embeddings: populate the declared vec lanes if missing.
            self._backfill_vec(conn, rowid, vals, config)
            self._write_variants(conn, table, rowid, variants, config)
            self._update_search_queries(conn, rowid, variants, config)

    @staticmethod
    def _source_text(vals: dict[str, object], config: SearchConfig) -> str:
        """The doc2query / embedding seed text: the first ``text_columns`` value
        verbatim, then each later column appended as ``": {value}"`` only when
        non-empty. Reproduces data_graph's ``f"{key}: {value}" if value else
        key`` for ``("key", "value")`` and yields the bare column for a single
        text column (episodes' ``gist``)."""
        cols = config.text_columns
        text = str(vals.get(cols[0]) or "")
        for col in cols[1:]:
            v = vals.get(col)
            if v:
                text = f"{text}: {v}"
        return text

    def _generate_variants(self, text: str) -> list[str]:
        """Empty return means skip variant writes but still mark search_queries so self-heal doesn't re-enqueue the row."""
        try:
            from services.doc2query_service import get_doc2query_service
            d2q = get_doc2query_service()
            if not d2q.is_available():
                return []
            return d2q.generate_queries(text) or []
        except Exception as e:
            logger.warning("[SES] doc2query failed for text='%s': %s", text[:60], e)
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

        No-op when the model declares no semantic-variant lane
        (``variant_table is None``) — e.g. episodes, whose recall never reads the
        variant table, so the doc2query expansions live only in the FTS
        ``search_queries`` column."""
        if config.variant_table is None:
            return
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

    def _backfill_vec(
        self, conn: sqlite3.Connection, rowid: int, vals: dict[str, object],
        config: SearchConfig,
    ) -> None:
        """Populate each declared vec lane (``config.vec_lanes``) if the row is
        missing it — one embedding per lane, sourced from the base-table column
        the lane names, falling back to the first ``text_columns`` value when
        that column is empty (preserving data_graph's value→key fallback).

        Absorbs the old _schedule_embeddings thread. Noop when a model declares
        no vec lanes (episodes carry their vectors synchronously) or when every
        lane's vec row already exists.
        """
        fallback = str(vals.get(config.text_columns[0]) or "")
        try:
            emb_svc = None
            for lane in config.vec_lanes:
                exists = conn.execute(
                    f"SELECT 1 FROM {lane.table} WHERE rowid = ?", (rowid,)
                ).fetchone()
                if exists:
                    continue
                text = str(vals.get(lane.source) or "") or fallback
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

    def _update_search_queries(
        self, conn: sqlite3.Connection, rowid: int, variants: list[str],
        config: SearchConfig,
    ) -> None:
        """Persist variant texts in the declared queries column and resync the
        declared FTS table. Table + column names come from ``config`` so the same
        path serves any searchable model; data_graph's behaviour is unchanged."""
        base_cols = [c for c in config.fts_columns if c != config.queries_column]
        old = conn.execute(
            f"SELECT {', '.join(base_cols)}, {config.queries_column} "
            f"FROM {config.base_table} WHERE rowid = ?",
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
            self._delete_fts(
                conn, rowid, {**row_vals, config.queries_column: prior_queries}, config
            )

        conn.execute(
            f"UPDATE {config.base_table} SET {config.queries_column} = ? WHERE rowid = ?",
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
            logger.warning("[SES] %s FTS sync failed for rowid=%s: %s", config.fts_table, rowid, e)

    # ── Private: FTS helpers ──────────────────────────────────────────────────

    def _delete_fts(
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
                logger.warning("[SES] %s FTS delete failed for rowid=%s: %s", config.fts_table, rowid, e)


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

import logging
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, cast

from services._fts_delete import fts5_external_delete
from services.database_service import get_shared_db_service
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService
from services.log_utils import safe
from services.time_utils import utc_now, parse_utc

if TYPE_CHECKING:
    from services.database_service import DatabaseService
# SearchExpanderService: generates doc2query variants + embeds them for KNN recall.
# Replaces _schedule_embeddings and _schedule_doc2query fire-and-forget threads.
import services.search_expander_service as _ses

logger = logging.getLogger(__name__)

KIND_USER_SPECIFIC = 'user_specific'
KIND_SYSTEM = 'system'
KIND_MISC = 'misc'
KIND_DOCUMENT = 'document'
KIND_BEHAVIORAL_PATTERN = 'behavioral_pattern'
KIND_PLACE = 'place'
# Contacts (CardDAV profiles + IMAP sender identities) get their own kind so
# contact reads can be scoped to kind='contact' and never get crowded out of a
# shared top-N recall/fetch window by unrelated user facts / traits. Without
# isolation, a freshly synced card can lose the top-5 recall race against a
# bloated user_specific namespace and the contacts ability reports a false
# not-found (the model then fabricates from memory).
KIND_CONTACT = 'contact'
# 'discovery' is the proactive-research lane: timely findings the subconscious
# research loop turns up about the user's interests. Recallable like any fact (it
# joins generic data_graph recall + the turn-0 flashback) but ephemeral — it
# decays over ~2 weeks so stale news fades on its own.
KIND_DISCOVERY = 'discovery'
VALID_KINDS = frozenset({
    KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC,
    KIND_DOCUMENT, KIND_BEHAVIORAL_PATTERN, KIND_PLACE, KIND_CONTACT, KIND_DISCOVERY,
})

_SELECT_ACTIVE_BY_KIND_KEY_SQL = (
    "SELECT * FROM data_graph WHERE kind=? AND key=? AND active=1 LIMIT 1"
)
_ORDER_FIRST_SEEN_DESC = 'first_seen_at DESC'
_SQL_DELETE_DG_ROW = "DELETE FROM data_graph WHERE rowid=?"
_SQL_DELETE_DG_KEY_VEC = "DELETE FROM data_graph_key_vec WHERE rowid=?"
_SQL_DELETE_DG_VALUE_VEC = "DELETE FROM data_graph_value_vec WHERE rowid=?"

# Superseded facts (valid_to set / active=0) decay fast and are then hard-deleted
# once they have been invalid this long, so an obsolete fact never resurfaces and
# does not accumulate indefinitely.
_SUPERSEDED_DECAY_WEIGHT = 0.01
_SUPERSEDED_DELETE_AFTER_DAYS = 90
# Retrieval-weight haircut applied to a row at the moment it is superseded — it
# is now historical, so it should rank below live facts even before decay runs.
_SUPERSEDE_RW_FACTOR = 0.5
# Below this absolute weight delta a recomputed live-fact weight is left as-is,
# so a settled graph produces zero UPDATEs on a repeat tick.
_RW_DECAY_EPSILON = 0.0001


@dataclass
class _StoreRequest:
    """Groups the four store-time write parameters to satisfy the 5-param ceiling."""

    kind: str
    key: str
    value: str
    source: Optional[str]


_KIND_POLICY: dict[str, dict[str, object]] = {
    KIND_USER_SPECIFIC:      {'ttl_days': 30,    'reinforce': True,  'contradiction': 'lut_canonicalize', 'deletion': 'soft',     'd_base': 0.5,  'salience_floor': 0.2},
    KIND_SYSTEM:             {'ttl_days': None,  'reinforce': True,  'contradiction': 'cosine_supersede', 'deletion': 'explicit', 'd_base': 0.05, 'salience_floor': 0.7},
    KIND_MISC:               {'ttl_days': 2,     'reinforce': False, 'contradiction': None,               'deletion': 'hard',     'd_base': 1.5,  'salience_floor': 0.0},
    KIND_DOCUMENT:           {'ttl_days': None,  'reinforce': False, 'contradiction': None,               'deletion': 'hard',     'd_base': 0.0,  'salience_floor': 0.0},
    # behavioral_pattern: written exclusively by abilities.pattern_match.save_pattern.SavePattern
    # via raw SQL UPSERT (one-active-row-per-(kind, key)). Decay (-0.005/pass with
    # soft-delete at 0) is handled by PatternDecayHook (configs/channels/pattern.py)
    # via PatternConfig.post_turn_hooks — DecayEngine does NOT touch this kind.
    KIND_BEHAVIORAL_PATTERN: {'ttl_days': None,  'reinforce': True,  'contradiction': None,               'deletion': 'soft',     'd_base': 0.1,  'salience_floor': 0.3},
    # place: named locations saved by the user (home, work, gym, etc.). Persistent,
    # reinforced on re-save of same coords, superseded when the user renames/moves a
    # place, only removed on explicit user request. Low decay, moderate salience floor.
    KIND_PLACE:              {'ttl_days': None,  'reinforce': True,  'contradiction': 'cosine_supersede', 'deletion': 'explicit', 'd_base': 0.05, 'salience_floor': 0.5},
    # contact: CardDAV profiles + IMAP sender identities. Persistent, reinforced
    # on re-sync, no contradiction reconciliation (each card is authoritative),
    # soft-deleted so a removed card lingers briefly for its decay window.
    KIND_CONTACT:            {'ttl_days': None,  'reinforce': True,  'contradiction': None,               'deletion': 'soft',     'd_base': 0.1,  'salience_floor': 0.3},
    # discovery: timely research findings from the proactive loop. Each is an
    # independent item (no contradiction reconciliation); reinforced if re-found,
    # decays like a user trait so it fades after ~2 weeks, soft-deleted on forget.
    KIND_DISCOVERY:          {'ttl_days': 14,    'reinforce': True,  'contradiction': None,               'deletion': 'soft',     'd_base': 0.5,  'salience_floor': 0.2},
}

# Concept LUT asset — pre-built sqlite with lut_concepts + lut_embeddings (vec0).
# Regenerate with: cd backend && python -m utils.generate_concept_lut
_CONCEPT_LUT_PATH = str(FileMapperService.get_concept_lut_db_path())

# Cosine threshold for LUT canonical match.
_CONCEPT_LUT_THRESHOLD = 0.80

# Cosine threshold for system_specific key deduplication.
_SYSTEM_KEY_THRESHOLD = 0.80

# KNN depth for LUT lookups — k=1 sufficient for single canonical match.
_LUT_K = 1

# KNN depth for system key cosine deduplication.
_SYSTEM_KEY_K = 3

# Module-level LUT connection, loaded once on first use.
_lut_conn: Optional[sqlite3.Connection] = None
_lut_lock = threading.Lock()
_lut_loaded = False


def _get_lut_conn() -> Optional[sqlite3.Connection]:
    """Return a read-only sqlite connection to the concept LUT, loading it once."""
    global _lut_conn, _lut_loaded
    if _lut_loaded:
        return _lut_conn
    with _lut_lock:
        if _lut_loaded:
            return _lut_conn
        if not os.path.exists(_CONCEPT_LUT_PATH):
            logger.warning("[DATA GRAPH] concept_lut.sqlite not found at %s", _CONCEPT_LUT_PATH)
            _lut_loaded = True
            return None
        try:
            conn = sqlite3.connect(f"file:{_CONCEPT_LUT_PATH}?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception:
                conn.load_extension('vec0')
            count = conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0]
            logger.info("[DATA GRAPH] LUT loaded: %s (concepts=%d)", _CONCEPT_LUT_PATH, count)
            _lut_conn = conn
        except Exception as e:
            logger.warning("[DATA GRAPH] Failed to open concept LUT: %s", e)
            _lut_conn = None
        _lut_loaded = True
        return _lut_conn

_EDGE_TYPE_MULTIPLIER = {
    'causes':        2.0,
    'caused_by':     2.0,
    'contradicts':   1.8,
    'superseded_by': 1.8,
    'supersedes':    1.8,
    'related':       1.0,
}

# ── Stop words (NLTK-backed, lazy-loaded) ─────────────────────────────

_nltk_stop_words = None
_stop_words_lock = threading.Lock()


def _get_stop_words() -> frozenset[str]:
    global _nltk_stop_words
    if _nltk_stop_words is not None:
        return _nltk_stop_words
    with _stop_words_lock:
        if _nltk_stop_words is not None:
            return _nltk_stop_words
        import nltk
        try:
            from nltk.corpus import stopwords
            _nltk_stop_words = frozenset(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            _nltk_stop_words = frozenset(stopwords.words('english'))
    return _nltk_stop_words


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _l2_dist_to_cosine(distance: float) -> float:
    """Convert L2 distance from sqlite-vec to cosine similarity.

    sqlite-vec returns the squared L2 distance for normalized vectors.
    For unit-norm vectors: cos = 1 - dist^2/2.
    """
    return max(0.0, 1.0 - (distance ** 2 / 2.0))


# ── Singleton management ──────────────────────────────────────────────

_instance: Optional["DataGraphService"] = None
_instance_lock = threading.Lock()


def get_data_graph_service() -> "DataGraphService":
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DataGraphService()
    return _instance


# ── Service ───────────────────────────────────────────────────────────

class DataGraphService:

    KIND_USER_SPECIFIC = KIND_USER_SPECIFIC
    KIND_SYSTEM = KIND_SYSTEM
    KIND_MISC = KIND_MISC

    def __init__(self, db_service: Optional["DatabaseService"] = None) -> None:
        if db_service is None:
            db_service = get_shared_db_service()
        self.db = db_service

    # ── Private helpers ───────────────────────────────────────────────

    def _row_to_dict(self, row: object) -> Optional[dict[str, object]]:
        if row is None:
            return None
        return dict(cast(sqlite3.Row, row))

    def _generate_embedding(self, text: str) -> Optional[list[float]]:
        try:
            from services.embedding_service import get_embedding_service
            emb = get_embedding_service().generate_embedding(text)
            if hasattr(emb, 'tolist'):
                return cast(list[float], emb.tolist())
            return list(emb)
        except Exception as e:
            logger.debug("[DATA GRAPH] Embedding generation failed: %s", e)
            return None

    def _store_key_vec(self, conn: sqlite3.Connection, rowid: int, embedding: Optional[list[float]]) -> None:
        if embedding is None or rowid is None:
            return
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return
            conn.execute(
                "INSERT OR REPLACE INTO data_graph_key_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, blob)
            )
        except Exception as e:
            logger.warning("[DATA GRAPH] Failed to store key_vec for rowid=%s: %s", rowid, e)

    def _store_value_vec(self, conn: sqlite3.Connection, rowid: int, embedding: Optional[list[float]]) -> None:
        if embedding is None or rowid is None:
            return
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return
            conn.execute(
                "INSERT OR REPLACE INTO data_graph_value_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, blob)
            )
        except Exception as e:
            logger.warning("[DATA GRAPH] Failed to store value_vec for rowid=%s: %s", rowid, e)

    def _sync_fts(self, conn: sqlite3.Connection, rowid: int, key: Optional[str] = None, value: Optional[str] = None,
                  kind: Optional[str] = None, search_queries: Optional[str] = None) -> None:
        """For external content FTS5 tables, callers must remove old FTS
        entries via _delete_fts BEFORE updating the content table — a
        regular DELETE reads from the content table and corrupts the index
        on mismatch."""
        try:
            cursor = conn.cursor()
            if key is None:
                cursor.execute(
                    "SELECT key, value, kind, search_queries FROM data_graph WHERE rowid = ?",
                    (rowid,)
                )
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return
                key, value, kind, search_queries = row[0], row[1], row[2], row[3]

            cursor.execute(
                "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) "
                "VALUES (?, ?, ?, ?, ?)",
                (rowid, key, value or '', kind, search_queries or '')
            )
            cursor.close()
        except Exception as e:
            logger.warning("[DATA GRAPH] FTS sync failed for rowid=%s: %s", rowid, e)

    def _delete_fts(self, conn: sqlite3.Connection, rowid: int, key: str, value: str, kind: str,
                    search_queries: str = '') -> None:
        """Delegates to the shared FTS5 external-delete idiom in
        services._fts_delete (the single home for this production-safe
        pattern; episodes_fts uses it too). Column order must match the
        data_graph_fts schema."""
        fts5_external_delete(conn, "data_graph_fts", rowid, {
            "key": key,
            "value": value,
            "kind": kind,
            "search_queries": search_queries,
        })

    def _remove_fts(self, conn: sqlite3.Connection, rowid: int) -> None:
        """Remove a row from the FTS index. Must be called BEFORE deleting from data_graph."""
        try:
            row = conn.execute(
                "SELECT key, value, kind, search_queries FROM data_graph WHERE rowid = ?",
                (rowid,)
            ).fetchone()
            if row:
                self._delete_fts(conn, rowid, row[0], row[1], row[2], row[3] or '')
        except Exception as e:
            logger.warning("[DATA GRAPH] FTS removal failed for rowid=%s: %s", rowid, e)

    def _reinforce_row(self, conn: sqlite3.Connection, row_id: int, existing_dict: dict[str, object], now_iso: str) -> None:
        old_evidence = cast(int, existing_dict.get('evidence_count', 1))
        new_evidence = old_evidence + 1
        old_strength = cast(float, existing_dict.get('storage_strength', 0.5))
        boost = 0.05 / math.log2(new_evidence + 1)
        new_strength = min(1.0, old_strength + boost)
        conn.execute("""
            UPDATE data_graph
            SET evidence_count=?, storage_strength=?, retrieval_weight=1.0,
                last_confirmed_at=?, last_accessed_at=?
            WHERE rowid=?
        """, (new_evidence, new_strength, now_iso, now_iso, row_id))

    def _fetch_row_by_id(self, conn: sqlite3.Connection, row_id: int) -> Optional[dict[str, object]]:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM data_graph WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        cursor.close()
        return self._row_to_dict(row)

    # ── LUT helpers ───────────────────────────────────────────────────

    def _lookup_concept_lut(self, key_embedding: Optional[list[float]]) -> Optional[dict[str, object]]:
        """KNN against concept LUT; returns {canonical_key, rule} or None if below threshold.

        Uses the pre-built concept_lut.sqlite (lut_embeddings vec0 table).
        Relies on _get_lut_conn() for lazy open and extension load.
        """
        lut = _get_lut_conn()
        if lut is None:
            return None
        blob = pack_embedding(key_embedding)
        if blob is None:
            return None
        try:
            hits = lut.execute(
                "SELECT rowid, distance FROM lut_embeddings "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (blob, _LUT_K),
            ).fetchall()
        except Exception as e:
            logger.debug("[DATA GRAPH] LUT KNN failed: %s", e)
            return None
        if not hits:
            return None
        rowid, distance = hits[0]
        cos = _l2_dist_to_cosine(distance)
        if cos < _CONCEPT_LUT_THRESHOLD:
            return None
        row = lut.execute(
            "SELECT canonical_key, rule FROM lut_concepts WHERE id = ?", (rowid,)
        ).fetchone()
        if row is None:
            return None
        return {"canonical_key": row[0], "rule": row[1], "cos": cos}

    def _record_lut_miss(self, conn: sqlite3.Connection, kind: str, key: str, value: str, top_cos: float, now_iso: str) -> None:
        """Log a LUT miss and upsert a row into concept_lut_misses for monitoring."""
        logger.info("[DATA GRAPH] LUT miss: kind=%s key='%s' top_cos=%.4f", kind, key, top_cos)
        value_preview = (value or '')[:100]
        try:
            conn.execute(
                "INSERT INTO concept_lut_misses(kind, key, value_preview, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(kind, key) DO UPDATE SET count=count+1, last_seen=excluded.last_seen",
                (kind, key, value_preview, now_iso, now_iso),
            )
        except Exception as e:
            logger.debug("[DATA GRAPH] concept_lut_misses upsert failed: %s", e)

    def _get_lut_miss_top_cos(self, key_embedding: Optional[list[float]]) -> float:
        """Return the top cosine from LUT KNN even when below threshold, for miss logging."""
        lut = _get_lut_conn()
        if lut is None or key_embedding is None:
            return 0.0
        blob = pack_embedding(key_embedding)
        if blob is None:
            return 0.0
        try:
            hits = lut.execute(
                "SELECT rowid, distance FROM lut_embeddings "
                "WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
                (blob,),
            ).fetchall()
            if hits:
                return _l2_dist_to_cosine(hits[0][1])
        except Exception:
            pass
        return 0.0

    def _apply_temporal_supersession(self, conn: sqlite3.Connection, existing_dict: dict[str, object], req: '_StoreRequest', now_iso: str) -> tuple[Optional[dict[str, object]], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        """Bi-temporal invalidation (Graphiti arXiv:2501.13956): the new
        fact's event-time start is also the moment the old fact stopped
        being true, so old.valid_to closes to the same instant. Combined
        with active=0 and the halved retrieval weight, the superseded row
        drops out of every recall lane and enters the fast-decay
        tombstone regime.

        Returns (new_row_dict, schedule_emb_args, schedule_d2q_args).
        """
        row_id = cast(int, existing_dict['id'])
        old_rw = cast(float, existing_dict.get('retrieval_weight', 1.0))
        conn.execute(
            "UPDATE data_graph SET active=0, retrieval_weight=?, valid_to=? WHERE rowid=?",
            (old_rw * _SUPERSEDE_RW_FACTOR, now_iso, row_id),
        )
        conn.execute(
            "INSERT INTO data_graph (kind, key, value, source, first_seen_at, last_confirmed_at, valid_from) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (req.kind, req.key, req.value, req.source, now_iso, now_iso, now_iso),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._add_edge_with_conn(conn, new_id, row_id, 'supersedes')
        self._add_edge_with_conn(conn, row_id, new_id, 'superseded_by')
        self._sync_fts(conn, new_id, req.key, req.value, req.kind)
        logger.info("[DATA GRAPH] temporal supersede: demoted %s, inserted %s for key='%s'", row_id, new_id, req.key)
        return self._fetch_row_by_id(conn, new_id), (new_id, req.key, req.value), (new_id, req.key, req.value)

    def _find_system_key_match(self, conn: sqlite3.Connection, key_embedding: Optional[list[float]], kind: str) -> Optional[dict[str, object]]:
        """KNN on data_graph_key_vec for system_specific kind; returns best matching row or None."""
        if key_embedding is None:
            return None
        blob = pack_embedding(key_embedding)
        if blob is None:
            return None
        try:
            hits = conn.execute(
                "SELECT rowid, distance FROM data_graph_key_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (blob, _SYSTEM_KEY_K),
            ).fetchall()
        except Exception as e:
            logger.debug("[DATA GRAPH] system key_vec KNN failed: %s", e)
            return None

        for rowid, distance in hits:
            cos = _l2_dist_to_cosine(distance)
            if cos < _SYSTEM_KEY_THRESHOLD:
                break
            row = conn.execute(
                "SELECT * FROM data_graph "
                "WHERE id=? AND kind=? AND active=1 AND deleted_at IS NULL LIMIT 1",
                (rowid, kind),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    # ── fact pipeline write path ──────────────────────────────────────

    def upsert_fact(self, key: str, value: str, *, source: Optional[str] = None) -> Optional[dict[str, object]]:
        """Exact-key ADD/UPDATE for the worker fact pipeline ().
        The reconciliation decision was already made by the LLM against the
        *actual* neighbour keys the worker showed it, so this path writes
        the chosen key verbatim and does NOT re-run concept-LUT
        canonicalization (which would fight the model's choice and break
        exact-key supersession targeting). The chat model's ``memory.store``
        keeps the LUT path; this is the router's path.

        Semantics on the ``user_specific`` kind, keyed exactly on ``key``:
        no live row → insert new (status ``created``); same value → reinforce
        (status ``reinforced``); contradicting value → bi-temporal
        supersession (status ``superseded``): old ``active=0`` +
        ``valid_to = now``, new row ``valid_from = now``.

        Returns the structured store result, or ``None`` on DB failure.
        """
        kind = KIND_USER_SPECIFIC
        _schedule_emb_args = None
        _schedule_d2q_args = None
        result = None
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(_SELECT_ACTIVE_BY_KIND_KEY_SQL, (kind, key))
                existing = cursor.fetchone()
                cursor.close()

                now_iso = utc_now().isoformat()
                req = _StoreRequest(kind, key, value, source)

                if existing is None:
                    row, _schedule_emb_args, _schedule_d2q_args = self._insert_new_row(
                        conn, req, now_iso
                    )
                    result = self._make_store_result(
                        "created", key, key, None, value, None, None, None, row,
                    )
                else:
                    existing_dict = self._row_to_dict(existing)
                    old_value = cast(str, cast(dict[str, object], existing_dict).get('value') or '')
                    existing_date = self._row_date(cast(dict[str, object], existing_dict))
                    if (value or '').lower().strip() == old_value.lower().strip():
                        self._reinforce_row(conn, cast(int, cast(dict[str, object], existing_dict)['id']), cast(dict[str, object], existing_dict), now_iso)
                        row = self._fetch_row_by_id(conn, cast(int, cast(dict[str, object], existing_dict)['id']))
                        result = self._make_store_result(
                            "reinforced", key, key, None, value, None, None, existing_date, row,
                        )
                    else:
                        row, _schedule_emb_args, _schedule_d2q_args = cast(
                            tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]],
                            self._apply_temporal_supersession(conn, cast(dict[str, object], existing_dict), req, now_iso)
                        )
                        result = self._make_store_result(
                            "superseded", key, key, None, value, old_value, None, existing_date, row,
                        )

            if _schedule_emb_args or _schedule_d2q_args:
                rowid = cast(tuple[int, str, str], _schedule_emb_args or _schedule_d2q_args)[0]
                _ses.enqueue("data_graph", rowid)
            return result
        except Exception:
            logger.exception("[DATA GRAPH] upsert_fact failed for key='%s'", key)
            return None

    # ── store() ───────────────────────────────────────────────────────

    def store(self, kind: str, key: str, value: str, *, source: Optional[str] = None) -> Optional[dict[str, object]]:
        if kind not in VALID_KINDS:
            logger.warning("[DATA GRAPH] Invalid kind '%s'", kind)
            return None

        policy = _KIND_POLICY[kind]
        _schedule_emb_args = None
        _schedule_d2q_args = None
        result = None

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(_SELECT_ACTIVE_BY_KIND_KEY_SQL, (kind, key))
                existing = cursor.fetchone()
                cursor.close()

                now_iso = utc_now().isoformat()

                if existing:
                    existing_dict = cast(dict[str, object], self._row_to_dict(existing))
                    result, _schedule_emb_args, _schedule_d2q_args = self._store_existing_row(
                        conn, existing_dict, kind, key, value, source, now_iso, policy,
                    )
                else:
                    result, _schedule_emb_args, _schedule_d2q_args = self._store_new_row(
                        conn, kind, key, value, source, now_iso, policy,
                    )

            # Enqueue for SearchExpanderService AFTER connection exits (row committed).
            # SES absorbs both _schedule_embeddings (key_vec/value_vec backfill) and
            # _schedule_doc2query (variant generation + FTS sync + expanded_semantic writes).
            # Either arg being set means a new or superseded row was written.
            if _schedule_emb_args or _schedule_d2q_args:
                rowid = cast(tuple[int, str, str], _schedule_emb_args or _schedule_d2q_args)[0]
                _ses.enqueue("data_graph", rowid)

            return result

        except Exception as e:
            logger.error("[DATA GRAPH] store failed for kind=%s key='%s': %s", kind, key, e)
            return None

    def _store_existing_row(
        self, conn: sqlite3.Connection, existing_dict: dict[str, object],
        kind: str, key: str, value: str, source: Optional[str], now_iso: str,
        policy: dict[str, object],
    ) -> tuple[Optional[dict[str, object]], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        row_id = cast(int, existing_dict['id'])
        old_value = cast(str, existing_dict.get('value') or '')
        existing_date = self._row_date(existing_dict)
        emb_args: Optional[tuple[int, str, str]] = None
        d2q_args: Optional[tuple[int, str, str]] = None

        if (value or '').lower().strip() == old_value.lower().strip():
            if policy['reinforce']:
                self._reinforce_row(conn, row_id, existing_dict, now_iso)
            row = self._fetch_row_by_id(conn, row_id)
            return self._make_store_result("reinforced", key, key, None, value, None, None, existing_date, row), None, None

        contradiction_mode = policy.get('contradiction')
        req = _StoreRequest(kind, key, value, source)

        if contradiction_mode == 'lut_canonicalize':
            return self._store_existing_lut(conn, existing_dict, kind, key, value, old_value, existing_date, req, now_iso)
        if contradiction_mode == 'cosine_supersede':
            row, emb_args, d2q_args = cast(
                tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]],
                self._apply_temporal_supersession(conn, existing_dict, req, now_iso),
            )
            return self._make_store_result("superseded", key, key, None, value, old_value, None, existing_date, row), emb_args, d2q_args

        row, emb_args, d2q_args = self._insert_new_row(conn, req, now_iso)
        return self._make_store_result("created", key, key, None, value, None, None, None, row), emb_args, d2q_args

    def _store_existing_lut(
        self, conn: sqlite3.Connection, existing_dict: dict[str, object],
        kind: str, key: str, value: str, old_value: str, existing_date: Optional[str],
        req: '_StoreRequest', now_iso: str,
    ) -> tuple[Optional[dict[str, object]], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        key_emb = self._generate_embedding(key)
        lut_hit = self._lookup_concept_lut(key_emb) if key_emb else None
        rule = cast(Optional[str], lut_hit['rule']) if lut_hit else None
        emb_args: Optional[tuple[int, str, str]] = None
        d2q_args: Optional[tuple[int, str, str]] = None

        if lut_hit and rule == 'coexist':
            row, emb_args, d2q_args = self._insert_new_row(conn, req, now_iso)
            all_vals = self._fetch_coexist_values(conn, kind, key)
            return self._make_store_result("appended", key, key, rule, value, None, all_vals, existing_date, row), emb_args, d2q_args
        if lut_hit and rule == 'immutable':
            self._log_immutable_conflict(key)
            return self._make_store_result("conflict", key, key, rule, value, old_value, None, existing_date, existing_dict), None, None

        row, emb_args, d2q_args = cast(
            tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]],
            self._apply_temporal_supersession(conn, existing_dict, req, now_iso),
        )
        return self._make_store_result("superseded", key, key, rule, value, old_value, None, existing_date, row), emb_args, d2q_args

    def _store_new_row(
        self, conn: sqlite3.Connection, kind: str, key: str, value: str, source: Optional[str],
        now_iso: str, policy: dict[str, object],
    ) -> tuple[Optional[dict[str, object]], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        contradiction_mode = policy.get('contradiction')
        req = _StoreRequest(kind, key, value, source)

        if contradiction_mode == 'lut_canonicalize':
            return self._store_user_specific_new(conn, req, now_iso)
        if contradiction_mode == 'cosine_supersede':
            return self._store_system_new(conn, req, now_iso)

        row, emb_args, d2q_args = self._insert_new_row(conn, req, now_iso)
        return self._make_store_result("created", key, key, None, value, None, None, None, row), emb_args, d2q_args

    def _make_store_result(
        self,
        status: str,
        provided_key: str,
        canonical_key: str,
        rule: Optional[str],
        value: str,
        old_value: Optional[str],
        all_values: Optional[list[str]],
        date: Optional[str],
        row: Optional[dict[str, object]],
    ) -> dict[str, object]:
        """Construct the structured store result dict.

        Merges row fields at the base so callers can access row-level fields
        (id, kind, evidence_count, etc.) directly. Structured fields overlay.
        """
        result = {}
        if row and isinstance(row, dict):
            result.update(row)
        result.update({
            "action": "store",
            "status": status,
            "canonical_key": canonical_key,
            "provided_key": provided_key,
            "rule": rule,
            "value": value,
            "old_value": old_value,
            "all_values": all_values,
            "date": date,
            "row": row,
        })
        return result

    def _fetch_coexist_values(self, conn: sqlite3.Connection, kind: str, key: str) -> list[str]:
        """Return all active values for a coexist key."""
        try:
            rows = conn.execute(
                "SELECT value FROM data_graph WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL",
                (kind, key),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def _log_immutable_conflict(self, key: str) -> None:
        logger.info("[DATA GRAPH] IMMUTABLE conflict on '%s'", key)

    def _store_user_specific_new(self, conn: sqlite3.Connection, req: '_StoreRequest', now_iso: str) -> tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        """Handle store() for user_specific kind with no existing row at the given key.

        Embeds the key, checks the concept LUT for a canonical form, then dispatches
        to a rule-specific helper (temporal/coexist/immutable). Falls back to plain
        insert on LUT miss or unknown rule.
        Returns (structured_result_dict, emb_args, d2q_args).
        """
        key_emb = self._generate_embedding(req.key)
        lut_hit = self._lookup_concept_lut(key_emb) if key_emb else None

        if lut_hit is None:
            top_cos = self._get_lut_miss_top_cos(key_emb) if key_emb else 0.0
            self._record_lut_miss(conn, req.kind, req.key, req.value, top_cos, now_iso)
            raw_row, emb_args, d2q_args = self._insert_new_row(conn, req, now_iso)
            return self._make_store_result(
                "lut_miss_created", req.key, req.key, None, req.value, None, None, None, raw_row,
            ), emb_args, d2q_args

        canonical_key = cast(str, lut_hit['canonical_key'])
        rule = cast(str, lut_hit['rule'])

        if rule == 'temporal':
            return self._store_user_specific_temporal(conn, req, canonical_key, rule, now_iso)
        if rule == 'coexist':
            return self._store_user_specific_coexist(conn, req, canonical_key, rule, now_iso)
        if rule == 'immutable':
            return self._store_user_specific_immutable(conn, req, canonical_key, rule, now_iso)

        # Unknown rule — insert as-is
        canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
        raw_row, emb_args, d2q_args = self._insert_new_row(conn, canon_req, now_iso)
        return self._make_store_result(
            "created", req.key, canonical_key, rule, req.value, None, None, None, raw_row,
        ), emb_args, d2q_args

    def _store_user_specific_temporal(self, conn: sqlite3.Connection, req: _StoreRequest, canonical_key: str, rule: str, now_iso: str) -> tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        existing_canon = conn.execute(
            _SELECT_ACTIVE_BY_KIND_KEY_SQL,
            (req.kind, canonical_key),
        ).fetchone()
        if existing_canon is None:
            canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
            raw_row, emb_args, d2q_args = self._insert_new_row(conn, canon_req, now_iso)
            return self._make_store_result(
                "created", req.key, canonical_key, rule, req.value, None, None, None, raw_row,
            ), emb_args, d2q_args

        existing_dict = self._row_to_dict(existing_canon)
        old_val = cast(str, cast(dict[str, object], existing_dict).get('value') or '')
        existing_date = cast(str, (
            cast(dict[str, object], existing_dict).get("last_confirmed_at") or cast(dict[str, object], existing_dict).get("first_seen_at") or ""
        ))[:10] or None
        if req.value.lower().strip() == old_val.lower().strip():
            self._reinforce_row(conn, cast(int, cast(dict[str, object], existing_dict)['id']), cast(dict[str, object], existing_dict), now_iso)
            row = self._fetch_row_by_id(conn, cast(int, cast(dict[str, object], existing_dict)['id']))
            return self._make_store_result(
                "reinforced", req.key, canonical_key, rule, req.value, None, None, existing_date, row,
            ), None, None
        canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
        row, emb_args, d2q_args = cast(
            tuple[dict[str, object], tuple[int, str, str], tuple[int, str, str]],
            self._apply_temporal_supersession(conn, cast(dict[str, object], existing_dict), canon_req, now_iso)
        )
        return self._make_store_result(
            "superseded", req.key, canonical_key, rule, req.value, old_val, None, existing_date, row,
        ), emb_args, d2q_args

    def _store_user_specific_coexist(self, conn: sqlite3.Connection, req: _StoreRequest, canonical_key: str, rule: str, now_iso: str) -> tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        existing_exact = conn.execute(
            "SELECT * FROM data_graph "
            "WHERE kind=? AND key=? AND active=1 AND LOWER(TRIM(value))=LOWER(TRIM(?)) LIMIT 1",
            (req.kind, canonical_key, req.value),
        ).fetchone()
        if existing_exact is not None:
            existing_dict = self._row_to_dict(existing_exact)
            existing_date = cast(str, (
                cast(dict[str, object], existing_dict).get("last_confirmed_at") or cast(dict[str, object], existing_dict).get("first_seen_at") or ""
            ))[:10] or None
            self._reinforce_row(conn, cast(int, cast(dict[str, object], existing_dict)['id']), cast(dict[str, object], existing_dict), now_iso)
            row = self._fetch_row_by_id(conn, cast(int, cast(dict[str, object], existing_dict)['id']))
            return self._make_store_result(
                "reinforced", req.key, canonical_key, rule, req.value, None, None, existing_date, row,
            ), None, None
        any_existing = conn.execute(
            "SELECT last_confirmed_at, first_seen_at FROM data_graph "
            "WHERE kind=? AND key=? AND active=1 LIMIT 1",
            (req.kind, canonical_key),
        ).fetchone()
        existing_date = (any_existing[0] or any_existing[1] or "")[:10] or None if any_existing else None
        canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
        raw_row, emb_args, d2q_args = self._insert_new_row(conn, canon_req, now_iso)
        all_vals = self._fetch_coexist_values(conn, req.kind, canonical_key)
        status = "appended" if any_existing else "created"
        return self._make_store_result(
            status, req.key, canonical_key, rule, req.value, None, all_vals, existing_date, raw_row,
        ), emb_args, d2q_args

    def _store_user_specific_immutable(self, conn: sqlite3.Connection, req: _StoreRequest, canonical_key: str, rule: str, now_iso: str) -> tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        existing_canon = conn.execute(
            _SELECT_ACTIVE_BY_KIND_KEY_SQL,
            (req.kind, canonical_key),
        ).fetchone()

        if existing_canon is None:
            canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
            raw_row, emb_args, d2q_args = self._insert_new_row(conn, canon_req, now_iso)
            return self._make_store_result(
                "created", req.key, canonical_key, rule, req.value, None, None, None, raw_row,
            ), emb_args, d2q_args

        existing_dict = self._row_to_dict(existing_canon)
        old_val = cast(str, cast(dict[str, object], existing_dict).get('value') or '')
        existing_date = cast(str, (
            cast(dict[str, object], existing_dict).get("last_confirmed_at") or cast(dict[str, object], existing_dict).get("first_seen_at") or ""
        ))[:10] or None

        if req.value.lower().strip() == old_val.lower().strip():
            self._reinforce_row(conn, cast(int, cast(dict[str, object], existing_dict)['id']), cast(dict[str, object], existing_dict), now_iso)
            row = self._fetch_row_by_id(conn, cast(int, cast(dict[str, object], existing_dict)['id']))
            return self._make_store_result(
                "reinforced", req.key, canonical_key, rule, req.value, None, None, existing_date, row,
            ), None, None

        self._log_immutable_conflict(canonical_key)
        return self._make_store_result(
            "conflict", req.key, canonical_key, rule, req.value, old_val, None, existing_date, cast(dict[str, object], existing_dict),
        ), None, None

    def _store_system_new(self, conn: sqlite3.Connection, req: _StoreRequest, now_iso: str) -> tuple[dict[str, object], Optional[tuple[int, str, str]], Optional[tuple[int, str, str]]]:
        """Handle store() for system_specific kind with no existing row at the given key.

        Embeds the key and runs KNN against data_graph_key_vec to find a semantically
        close existing system key. Above threshold → temporal supersession. Below → plain insert.
        Returns (structured_result_dict, emb_args, d2q_args).
        """
        key_emb = self._generate_embedding(req.key)
        match = self._find_system_key_match(conn, key_emb, req.kind)

        if match is None:
            raw_row, emb_args, d2q_args = self._insert_new_row(conn, req, now_iso)
            return self._make_store_result(
                "created", req.key, req.key, None, req.value, None, None, None, raw_row,
            ), emb_args, d2q_args

        canonical_key = cast(str, match.get('key', req.key))
        old_value = cast(str, match.get('value') or '')
        existing_date = cast(str, (
            match.get("last_confirmed_at") or match.get("first_seen_at") or ""
        ))[:10] or None

        if req.value.lower().strip() == old_value.lower().strip():
            self._reinforce_row(conn, cast(int, match['id']), match, now_iso)
            refreshed = self._fetch_row_by_id(conn, cast(int, match['id']))
            return self._make_store_result(
                "reinforced", req.key, canonical_key, None, req.value, None, None, existing_date, refreshed,
            ), None, None

        canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
        row, emb_args, d2q_args = cast(
            tuple[dict[str, object], tuple[int, str, str], tuple[int, str, str]],
            self._apply_temporal_supersession(conn, match, canon_req, now_iso)
        )
        return self._make_store_result(
            "superseded", req.key, canonical_key, None, req.value, old_value, None, existing_date, row,
        ), emb_args, d2q_args

    def _insert_new_row(self, conn: sqlite3.Connection, req: _StoreRequest, now_iso: str) -> tuple[Optional[dict[str, object]], tuple[int, str, str], tuple[int, str, str]]:
        """Insert a brand-new data_graph row; sync FTS; return (row_dict, emb_args, d2q_args)."""
        conn.execute(
            "INSERT INTO data_graph (kind, key, value, source, first_seen_at, last_confirmed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (req.kind, req.key, req.value, req.source, now_iso, now_iso),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._sync_fts(conn, new_id, req.key, req.value, req.kind)
        logger.info("[DATA GRAPH] Stored new %s '%s'='%s' (source=%s)", req.kind, req.key, (req.value or '')[:60], req.source)
        return self._fetch_row_by_id(conn, new_id), (new_id, req.key, req.value), (new_id, req.key, req.value)

    # ── backfill ──────────────────────────────────────────────────────

    def _backfill_missing_embeddings(self) -> None:
        """Embed and FTS-index rows that were inserted outside DataGraphService.store().

        Finds active, non-deleted rows whose id is absent from either vec table
        or the FTS index, then generates embeddings and syncs FTS for each.
        One row failure never aborts the rest.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, key, value, kind
                    FROM data_graph
                    WHERE active = 1 AND deleted_at IS NULL
                      AND (
                          id NOT IN (SELECT rowid FROM data_graph_key_vec)
                          OR id NOT IN (SELECT rowid FROM data_graph_value_vec)
                          OR id NOT IN (SELECT rowid FROM data_graph_fts)
                      )
                """)
                rows = cursor.fetchall()
                cursor.close()

            if not rows:
                return

            logger.info("[DATA GRAPH] Backfilling embeddings for %d row(s)", len(rows))

            for row in rows:
                row_id, key, value, kind = row[0], row[1], row[2], row[3]
                try:
                    key_emb = self._generate_embedding(key) if key else None
                    value_emb = self._generate_embedding(value) if value else None
                    with self.db.connection() as conn:
                        self._store_key_vec(conn, row_id, key_emb)
                        self._store_value_vec(conn, row_id, value_emb)
                        self._sync_fts(conn, row_id, key, value, kind)
                except Exception as e:
                    logger.warning(
                        "[DATA GRAPH] Backfill failed for rowid=%s: %s", row_id, e
                    )
        except Exception as e:
            logger.warning("[DATA GRAPH] _backfill_missing_embeddings error: %s", e)

    # ── recall() ─────────────────────────────────────────────────────

    def _recall_vec_search(self, cursor: sqlite3.Cursor, candidates: dict[int, dict[str, object]], query_blob: Optional[bytes], k: int) -> None:
        """Run key-vec and value-vec KNN searches, populating candidates in-place."""
        if not query_blob:
            return
        try:
            cursor.execute(
                "SELECT rowid, distance FROM data_graph_key_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (query_blob, k),
            )
            for rowid, dist in cursor.fetchall():
                cos = _l2_dist_to_cosine(dist)
                candidates.setdefault(rowid, {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0})
                candidates[rowid]['key_cos'] = max(cast(float, candidates[rowid]['key_cos']), cos)
        except Exception as e:
            logger.debug("[DATA GRAPH] key_vec search failed (non-fatal): %s", e)
        try:
            cursor.execute(
                "SELECT rowid, distance FROM data_graph_value_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (query_blob, k),
            )
            for rowid, dist in cursor.fetchall():
                cos = _l2_dist_to_cosine(dist)
                candidates.setdefault(rowid, {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0})
                candidates[rowid]['value_cos'] = max(cast(float, candidates[rowid]['value_cos']), cos)
        except Exception as e:
            logger.debug("[DATA GRAPH] value_vec search failed (non-fatal): %s", e)

    def _recall_fts_search(self, cursor: sqlite3.Cursor, candidates: dict[int, dict[str, object]], query: str) -> None:
        """Run FTS search, setting fts_bonus=1.0 for matching rows."""
        try:
            fts_query = re.sub(r'[^\w\s]', '', query).strip()
            if not fts_query:
                return
            fts_words = [w for w in fts_query.split() if w and w.lower() not in _get_stop_words()]
            fts_terms = ' OR '.join(f'"{w}"*' for w in fts_words if w)
            if not fts_terms:
                return
            cursor.execute(
                "SELECT rowid, rank FROM data_graph_fts "
                "WHERE data_graph_fts MATCH ? ORDER BY rank LIMIT 30",
                (fts_terms,),
            )
            for rowid, _rank in cursor.fetchall():
                candidates.setdefault(rowid, {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0})
                candidates[rowid]['fts_bonus'] = 1.0
        except Exception as e:
            logger.debug("[DATA GRAPH] FTS search failed (non-fatal): %s", e)

    def _recall_variant_search(self, cursor: sqlite3.Cursor, candidates: dict[int, dict[str, object]], query_blob: Optional[bytes], k: int) -> None:
        """Run expanded-semantic (doc2query variant) vector search."""
        if not query_blob:
            return
        try:
            variant_k = min(k, 50)
            cursor.execute("""
                SELECT CAST(es.related_to_id AS INTEGER) AS source_id, v.distance
                FROM expanded_semantic_vec v
                JOIN expanded_semantic es ON es.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                  AND es.relates_to_table = 'data_graph'
                ORDER BY v.distance
            """, (query_blob, variant_k))
            for source_id, dist in cursor.fetchall():
                cos = _l2_dist_to_cosine(dist)
                candidates.setdefault(source_id, {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0, 'variant_cos': 0.0})
                candidates[source_id]['variant_cos'] = max(
                    cast(float, candidates[source_id].get('variant_cos', 0.0)), cos
                )
        except Exception as e:
            logger.debug("[DATA GRAPH] Variant vec search failed (non-fatal): %s", e)

    def _recall_score_candidates(self, cursor: sqlite3.Cursor, candidates: dict[int, dict[str, object]],
                                  filter_clause: str, filter_params: list[object]) -> list[dict[str, object]]:
        """Fetch full rows for candidates, compute composite scores, return sorted list."""
        now_ts = utc_now().timestamp()
        scored: list[dict[str, object]] = []
        for rowid, sigs in candidates.items():
            cursor.execute(
                f"SELECT * FROM data_graph WHERE id=? AND {filter_clause}",
                [rowid] + filter_params,
            )
            row = cursor.fetchone()
            if not row:
                continue
            d = self._row_to_dict(row)

            base_score = (
                2.0 * cast(float, sigs['key_cos'])
                + 1.0 * cast(float, sigs['value_cos'])
                + 0.3 * cast(float, sigs['fts_bonus'])
                + 0.8 * cast(float, sigs.get('variant_cos', 0.0))
            )
            ref_ts_str = cast(dict[str, object], d).get('last_accessed_at') or cast(dict[str, object], d).get('last_confirmed_at')
            try:
                ref_ts = parse_utc(cast(str, ref_ts_str)).timestamp() if ref_ts_str else now_ts - 3600
            except Exception:
                ref_ts = now_ts - 3600
            age_seconds = max(1, now_ts - ref_ts)
            evidence = max(1, cast(int, cast(dict[str, object], d).get('evidence_count', 1)))
            actr_boost = math.log(evidence) - 0.5 * math.log(age_seconds)
            composite = base_score * cast(float, cast(dict[str, object], d).get('retrieval_weight', 1.0)) * (1 + 0.3 * _sigmoid(actr_boost))

            cast(dict[str, object], d)['composite_score'] = composite
            cast(dict[str, object], d)['cos_score'] = max(cast(float, sigs.get('key_cos', 0.0)), cast(float, sigs.get('value_cos', 0.0)))
            scored.append(cast(dict[str, object], d))

        scored.sort(key=lambda x: cast(float, x['composite_score']), reverse=True)
        return scored

    def _recall_expand_graph(self, cursor: sqlite3.Cursor, top_k: list[dict[str, object]], filter_clause: str,
                              filter_params: list[object], limit: int) -> list[dict[str, object]]:
        """1-hop graph expansion: add high-scoring neighbours not already in top_k."""
        expansion: dict[int, dict[str, object]] = {}
        top_k_ids = {d['id'] for d in top_k}

        for seed in top_k:
            seed_id = seed.get('id')
            if not seed_id:
                continue
            cursor.execute(
                "SELECT e.to_id, e.edge_type, e.strength, e.from_id "
                "FROM data_graph_edges e WHERE e.from_id = ?",
                (seed_id,),
            )
            edges = cursor.fetchall()
            out_degree = len(edges)

            for to_id, edge_type, strength, _ in edges:
                if to_id in top_k_ids:
                    continue
                cursor.execute(
                    f"SELECT * FROM data_graph WHERE id=? AND {filter_clause}",
                    [to_id] + filter_params,
                )
                n_row = cursor.fetchone()
                if not n_row:
                    continue
                self._expand_seed_neighbour(seed, n_row, edge_type, strength, out_degree, expansion)

        all_candidates: dict[int, dict[str, object]] = {cast(int, d['id']): d for d in top_k}
        for nid, nd in expansion.items():
            if nid not in all_candidates:
                all_candidates[nid] = nd
        return sorted(all_candidates.values(), key=lambda x: cast(float, x['composite_score']), reverse=True)[:limit]

    def _expand_seed_neighbour(
        self, seed: dict[str, object], n_row: object, edge_type: object, strength: object,
        out_degree: int, expansion: dict[int, dict[str, object]],
    ) -> None:
        n_dict = cast(dict[str, object], self._row_to_dict(n_row))
        n_score = cast(float, seed['composite_score']) * cast(float, strength) * _EDGE_TYPE_MULTIPLIER.get(cast(str, edge_type), 1.0)
        if n_dict.get('kind') != seed.get('kind'):
            n_score *= 1.2
        if out_degree > 10:
            n_score /= math.sqrt(out_degree)
        neighbour_id = cast(int, n_dict.get('id'))
        if neighbour_id not in expansion or cast(float, expansion[neighbour_id]['composite_score']) < n_score:
            n_dict['composite_score'] = n_score
            n_dict['cos_score'] = cast(float, seed.get('cos_score', 0.0)) / 2.0
            expansion[neighbour_id] = n_dict

    @staticmethod
    def _build_kind_filter(kinds: Optional[list[str]]) -> tuple[str, list[object]]:
        """Return (filter_clause, params) for recall over LIVE rows only.

        Live = ``deleted_at IS NULL AND active=1 AND valid_to IS NULL``: an
        invalidated (superseded / bi-temporally closed) fact must never surface
        in recall, even while it lingers for its fast-decay deletion window.
        """
        filters = ["deleted_at IS NULL", "active=1", "valid_to IS NULL"]
        params: list[object] = []
        if kinds:
            placeholders = ','.join('?' for _ in kinds)
            filters.append(f"kind IN ({placeholders})")
            params.extend(kinds)
        return " AND ".join(filters), params

    def recall(self, query: str, *, kinds: Optional[list[str]] = None, limit: int = 10, expand_graph: bool = True) -> list[dict[str, object]]:
        try:
            self._backfill_missing_embeddings()
            query_emb = self._generate_embedding(query)
            query_blob = pack_embedding(query_emb) if query_emb else None

            candidates: dict[int, dict[str, object]] = {}
            k = min(limit * 3, 50)

            with self.db.connection() as conn:
                cursor = conn.cursor()

                filter_clause, filter_params = self._build_kind_filter(kinds)

                self._recall_vec_search(cursor, candidates, query_blob, k)
                self._recall_fts_search(cursor, candidates, query)
                self._recall_variant_search(cursor, candidates, query_blob, k)

                for sigs in candidates.values():
                    sigs.setdefault('variant_cos', 0.0)

                # No absolute cosine floor / FTS floor-bypass: a hard threshold
                # on gte-modernbert muted relevant rows and the FTS-bypass let
                # keyword-only noise win. Ranking + the limit truncation govern;
                # the relative score floor lives in the episodic lane.
                scored = self._recall_score_candidates(cursor, candidates, filter_clause, filter_params)
                top_k = scored[:limit]

                if expand_graph and top_k:
                    top_k = self._recall_expand_graph(cursor, top_k, filter_clause, filter_params, limit)

                # Recall is a pure read: relevance advances only on
                # write-relevant events (store / confirm / supersede), never on
                # access, so the read no longer bumps retrieval_weight.

                cursor.close()

                return [
                    {
                        'id': d.get('id'), 'kind': d.get('kind'), 'key': d.get('key'),
                        'value': d.get('value'), 'source': d.get('source'),
                        'retrieval_weight': d.get('retrieval_weight'),
                        'evidence_count': d.get('evidence_count'),
                        'composite_score': d.get('composite_score'),
                        'cos_score': d.get('cos_score', 0.0),
                    }
                    for d in top_k
                ]

        except Exception as e:
            logger.error("[DATA GRAPH] recall failed: %s", e)
            return []

    # ── fetch() ───────────────────────────────────────────────────────

    _VALID_ORDER_BY = frozenset({
        _ORDER_FIRST_SEEN_DESC, 'last_confirmed_at DESC',
        'retrieval_weight DESC', 'evidence_count DESC',
        'first_seen_at ASC', 'last_confirmed_at ASC',
        'key ASC',
    })

    def fetch(self, *, kinds: Optional[list[str]] = None, limit: Optional[int] = None, order_by: str = _ORDER_FIRST_SEEN_DESC,
              include_inactive: bool = False, include_deleted: bool = False) -> list[Optional[dict[str, object]]]:
        try:
            if order_by not in self._VALID_ORDER_BY:
                logger.warning("[DATA GRAPH] Invalid order_by '%s', using default", order_by)
                order_by = _ORDER_FIRST_SEEN_DESC

            filters: list[str] = []
            params: list[object] = []

            if not include_deleted:
                filters.append("deleted_at IS NULL")
            if not include_inactive:
                filters.append("active=1")
            if kinds:
                placeholders = ','.join('?' for _ in kinds)
                filters.append(f"kind IN ({placeholders})")
                params.extend(kinds)

            where = f"WHERE {' AND '.join(filters)}" if filters else ""
            limit_clause = f"LIMIT {int(limit)}" if limit else ""

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT * FROM data_graph {where} ORDER BY {order_by} {limit_clause}",
                    params
                )
                rows = cursor.fetchall()
                cursor.close()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error("[DATA GRAPH] fetch failed: %s", e)
            return []

    # ── Edge operations ───────────────────────────────────────────────

    def _add_edge_with_conn(self, conn: sqlite3.Connection, from_id: int, to_id: int, edge_type: str = 'related', strength: float = 1.0) -> int:
        now_iso = utc_now().isoformat()
        conn.execute("""
            INSERT OR IGNORE INTO data_graph_edges
                (from_id, to_id, edge_type, strength, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (from_id, to_id, edge_type, strength, now_iso))
        row = conn.execute(
            "SELECT id FROM data_graph_edges WHERE from_id=? AND to_id=? AND edge_type=?",
            (from_id, to_id, edge_type)
        ).fetchone()
        return row[0] if row else 0

    # ── Deletion operations ───────────────────────────────────────────

    def soft_delete_by_id(self, row_id: int) -> bool:
        try:
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE data_graph SET deleted_at=? WHERE rowid=?",
                    (utc_now().isoformat(), row_id)
                )
                self._remove_fts(conn, row_id)
            return True
        except Exception as e:
            logger.warning("[DATA GRAPH] soft_delete_by_id failed for rowid=%s: %s", row_id, e)
            return False

    def hard_delete_by_source_prefix(self, source_prefix: str) -> int:
        """Hard-delete all rows whose source starts with the given prefix.

        Used by document delete cascade to remove all artifacts for a document.
        Returns count of deleted rows.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT rowid FROM data_graph WHERE source LIKE ?",
                    (source_prefix + '%',)
                )
                rowids = [r[0] for r in cursor.fetchall()]
                cursor.close()

                for rid in rowids:
                    self._remove_fts(conn, rid)
                    conn.execute(_SQL_DELETE_DG_ROW, (rid,))
                    conn.execute(_SQL_DELETE_DG_KEY_VEC, (rid,))
                    conn.execute(_SQL_DELETE_DG_VALUE_VEC, (rid,))

                return len(rowids)
        except Exception as e:
            logger.warning("[DATA GRAPH] hard_delete_by_source_prefix failed for '%s': %s", safe(source_prefix), e)
            return 0

    def _make_forget_result(
        self,
        status: str,
        provided_key: str,
        canonical_key: str,
        rule: Optional[str],
        value: Optional[str],
        *,
        old_value: Optional[str] = None,
        remaining_values: Optional[list[str]] = None,
        versions_removed: Optional[int] = None,
        date: Optional[str] = None,
    ) -> dict[str, object]:
        """Construct the structured forget result dict with explicit parameters; no closure captures."""
        base: dict[str, object] = {
            "action": "forget",
            "status": status,
            "canonical_key": canonical_key,
            "provided_key": provided_key,
            "rule": rule,
            "value": value,
            "old_value": old_value,
            "remaining_values": remaining_values,
            "versions_removed": versions_removed,
            "date": date,
        }
        return base

    def _hard_delete_row(self, conn: sqlite3.Connection, row_id: int) -> None:
        """Hard-delete a data_graph row and all associated index/edge entries.

        Removes FTS, main row, edges, and both vec tables. Vec table failures
        are non-fatal and logged at debug level.
        """
        self._remove_fts(conn, row_id)
        conn.execute(_SQL_DELETE_DG_ROW, (row_id,))
        conn.execute("DELETE FROM data_graph_edges WHERE from_id=? OR to_id=?", (row_id, row_id))
        try:
            conn.execute(_SQL_DELETE_DG_KEY_VEC, (row_id,))
        except Exception as e:
            logger.debug("vec table delete failed for id=%s: %s", row_id, e)
        try:
            conn.execute(_SQL_DELETE_DG_VALUE_VEC, (row_id,))
        except Exception as e:
            logger.debug("vec table delete failed for id=%s: %s", row_id, e)

    # ── forget() ──────────────────────────────────────────────────────

    @staticmethod
    def _row_date(d: dict[str, object]) -> Optional[str]:
        """Extract a YYYY-MM-DD date string from a row dict, or None."""
        raw = d.get('last_confirmed_at') or d.get('first_seen_at') or ""
        return cast(str, raw)[:10] or None

    def _resolve_lut_key(self, kind: str, key: str) -> tuple[str, Optional[str]]:
        """Return (canonical_key, rule) after LUT lookup, or (key, None) on miss."""
        policy = _KIND_POLICY[kind]
        if policy.get('contradiction') != 'lut_canonicalize':
            return key, None
        key_emb = self._generate_embedding(key)
        lut_hit = self._lookup_concept_lut(key_emb) if key_emb else None
        if lut_hit:
            return cast(str, lut_hit['canonical_key']), cast(Optional[str], lut_hit['rule'])
        return key, None

    def _forget_immutable(self, conn: sqlite3.Connection, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict[str, object]:
        row = conn.execute(
            "SELECT * FROM data_graph WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL LIMIT 1",
            (kind, canonical_key),
        ).fetchone()
        if row is None:
            return self._make_forget_result("not_found", provided_key, canonical_key, rule, value)
        d = self._row_to_dict(row)
        self._hard_delete_row(conn, cast(int, cast(dict[str, object], d)['id']))
        return self._make_forget_result("forgotten", provided_key, canonical_key, rule, value, old_value=cast(Optional[str], cast(dict[str, object], d).get('value')), date=self._row_date(cast(dict[str, object], d)))

    def _forget_temporal(self, conn: sqlite3.Connection, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict[str, object]:
        rows = conn.execute(
            "SELECT * FROM data_graph WHERE kind=? AND key=? AND deleted_at IS NULL",
            (kind, canonical_key),
        ).fetchall()
        if not rows:
            return self._make_forget_result("not_found", provided_key, canonical_key, rule, value)
        for r in rows:
            self._hard_delete_row(conn, cast(int, cast(dict[str, object], self._row_to_dict(r))['id']))
        return self._make_forget_result("forgotten_all", provided_key, canonical_key, rule, value, versions_removed=len(rows))

    def _forget_coexist(self, conn: sqlite3.Connection, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict[str, object]:
        if value is None:
            return {"action": "forget", "status": "error", "message": "value required for coexist key"}
        exact = conn.execute(
            "SELECT * FROM data_graph "
            "WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL "
            "AND LOWER(TRIM(value))=LOWER(TRIM(?)) LIMIT 1",
            (kind, canonical_key, value),
        ).fetchone()
        if exact is None:
            remaining = self._fetch_coexist_values(conn, kind, canonical_key)
            return self._make_forget_result("value_not_found", provided_key, canonical_key, rule, value, remaining_values=remaining)
        d = self._row_to_dict(exact)
        self._hard_delete_row(conn, cast(int, cast(dict[str, object], d)['id']))
        remaining = self._fetch_coexist_values(conn, kind, canonical_key)
        status = "forgotten_empty" if not remaining else "forgotten"
        return self._make_forget_result(status, provided_key, canonical_key, rule, value, date=self._row_date(cast(dict[str, object], d)), remaining_values=remaining or None)

    def _forget_raw(self, conn: sqlite3.Connection, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict[str, object]:
        """LUT-miss path: raw key lookup, optional value filter."""
        rows = conn.execute(
            "SELECT * FROM data_graph WHERE kind=? AND key=? AND deleted_at IS NULL",
            (kind, canonical_key),
        ).fetchall()
        if not rows:
            return self._make_forget_result("not_found", provided_key, canonical_key, rule, value)
        if value is not None:
            exact = conn.execute(
                "SELECT * FROM data_graph "
                "WHERE kind=? AND key=? AND deleted_at IS NULL "
                "AND LOWER(TRIM(value))=LOWER(TRIM(?)) LIMIT 1",
                (kind, canonical_key, value),
            ).fetchone()
            if exact is None:
                return self._make_forget_result("not_found", provided_key, canonical_key, rule, value)
            d = self._row_to_dict(exact)
            self._hard_delete_row(conn, cast(int, cast(dict[str, object], d)['id']))
            return self._make_forget_result("forgotten", provided_key, canonical_key, rule, value, old_value=cast(Optional[str], cast(dict[str, object], d).get('value')), date=self._row_date(cast(dict[str, object], d)))
        for r in rows:
            self._hard_delete_row(conn, cast(int, cast(dict[str, object], self._row_to_dict(r))['id']))
        if len(rows) == 1:
            d = self._row_to_dict(rows[0])
            return self._make_forget_result("forgotten", provided_key, canonical_key, rule, value, old_value=cast(Optional[str], cast(dict[str, object], d).get('value')), date=self._row_date(cast(dict[str, object], d)))
        return self._make_forget_result("forgotten_all", provided_key, canonical_key, rule, value, versions_removed=len(rows))

    def forget(self, kind: str, key: str, value: Optional[str] = None) -> Optional[dict[str, object]]:
        """Hard-delete memory rows by key (and optionally value).

        Rule-aware: temporal deletes all versions, coexist deletes the specific
        value row (value param required), immutable hard-deletes the single row.
        LUT miss path falls back to raw key lookup.
        """
        if kind not in VALID_KINDS:
            logger.warning("[DATA GRAPH] forget: invalid kind '%s'", kind)
            return None

        _RULE_HANDLERS = {
            'immutable': self._forget_immutable,
            'temporal': self._forget_temporal,
            'coexist': self._forget_coexist,
        }

        try:
            canonical_key, rule = self._resolve_lut_key(kind, key)
            handler = _RULE_HANDLERS.get(cast(str, rule), self._forget_raw)
            with self.db.connection() as conn:
                return handler(conn, kind, canonical_key, key, rule, value)
        except Exception as e:
            logger.error("[DATA GRAPH] forget failed for kind=%s key='%s': %s", kind, key, e)
            return None

    def invalidate(self, kind: str, key: str) -> int:
        """Bi-temporally invalidate every live row for the EXACT ``(kind, key)``.

        The DELETE op of the worker fact pipeline: a fact the user has
        revoked is not erased (that would lose the audit trail and the
        bi-temporal history) but closed — ``active=0``, ``valid_to`` set
        to now, retrieval weight halved. Recall lanes filter
        ``valid_to IS NULL`` (), so the row vanishes from retrieval
        immediately, then the fast-decay tombstone regime ()
        hard-deletes it after the superseded window.

        The key is matched VERBATIM (no concept-LUT canonicalization): the
        LLM chose it from the neighbour facts the worker showed it, so
        re-mapping it would target the wrong row. Mirrors ``upsert_fact``'s
        exact-key contract.

        Returns the number of rows invalidated (0 when no live row
        matches). Pure-failure: a DB error bubbles up to the caller, which
        counts it.
        """
        if kind not in VALID_KINDS:
            logger.warning("[DATA GRAPH] invalidate: invalid kind '%s'", kind)
            return 0

        now_iso = utc_now().isoformat()
        with self.db.connection() as conn:
            cursor = conn.execute(
                "UPDATE data_graph "
                "SET active=0, valid_to=?, "
                "    retrieval_weight=retrieval_weight*? "
                "WHERE kind=? AND key=? AND active=1 "
                "  AND deleted_at IS NULL AND valid_to IS NULL",
                (now_iso, _SUPERSEDE_RW_FACTOR, kind, key),
            )
            count: int = cursor.rowcount
        if count:
            logger.info(
                "[DATA GRAPH] invalidated %d row(s) for kind=%s key='%s'",
                count, kind, key,
            )
        return count

    # ── Decay cycle ───────────────────────────────────────────────────

    def decay_cycle(self) -> int:
        """Run one decay/deletion pass over the data graph.

        Three branches: power-law decay of live facts (per-kind ``d_base``),
        fast-decay-then-delete of superseded facts (``valid_to`` set / inactive),
        and the short fixed-TTL purge of misc scratch rows.
        """
        total_updated = 0
        try:
            now = utc_now()
            one_hour_ago = (now - timedelta(hours=1)).isoformat()

            with self.db.connection() as conn:
                cursor = conn.cursor()
                total_updated += self._decay_live_facts(cursor, now, one_hour_ago)
                total_updated += self._decay_superseded_facts(cursor)
                cursor.close()

                total_updated += self._delete_superseded_facts(conn, now)
                total_updated += self._delete_expired_misc(conn, now)

            if total_updated > 0:
                logger.info("[DATA GRAPH] Decay cycle updated %d rows", total_updated)
            return total_updated

        except Exception as e:
            logger.error("[DATA GRAPH] decay_cycle failed: %s", e)
            return 0

    def _decay_live_facts(self, cursor: sqlite3.Cursor, now: datetime, one_hour_ago: str) -> int:
        """Absolute power-law decay of live (active) facts with a TTL policy."""
        updated = 0
        now_ts = now.timestamp()
        for kind, policy in _KIND_POLICY.items():
            if policy['ttl_days'] is None:
                continue
            d_base = cast(float, policy['d_base'])
            salience_floor = cast(float, policy['salience_floor'])

            cursor.execute("""
                SELECT rowid, retrieval_weight, last_confirmed_at
                FROM data_graph
                WHERE kind=?
                  AND deleted_at IS NULL
                  AND active=1
                  AND last_confirmed_at < ?
            """, (kind, one_hour_ago))

            for rowid, rw, confirmed_at_str in cursor.fetchall():
                new_rw = self._compute_decayed_rw(confirmed_at_str, now_ts, d_base, salience_floor)
                if new_rw is not None and abs(new_rw - rw) > _RW_DECAY_EPSILON:
                    cursor.execute(
                        "UPDATE data_graph SET retrieval_weight=? WHERE rowid=?",
                        (new_rw, rowid),
                    )
                    updated += 1
        return updated

    @staticmethod
    def _compute_decayed_rw(confirmed_at_str: Optional[str], now_ts: float, d_base: float, salience_floor: float) -> Optional[float]:
        if not confirmed_at_str:
            return None
        try:
            confirmed_ts = parse_utc(confirmed_at_str).timestamp()
        except Exception:
            return None
        age_days = (now_ts - confirmed_ts) / 86400.0
        if age_days <= 0:
            return None
        # Power-law absolute level: rw = max(1, age)^(-d_base) — not a
        # multiplier; it directly sets the weight from how old the fact
        # is since last confirmation, so the cycle is idempotent.
        return cast(float, max(salience_floor, max(1.0, age_days) ** (-d_base)))

    def _decay_superseded_facts(self, cursor: sqlite3.Cursor) -> int:
        """Drop superseded facts to a near-zero weight so they stop resurfacing.

        A superseded fact has been invalidated by a newer one (``valid_to`` set
        or ``active=0``); it is kept briefly for bi-temporal history but must not
        compete with live facts in recall.
        """
        cursor.execute(
            """
            SELECT rowid FROM data_graph
            WHERE deleted_at IS NULL
              AND (active=0 OR valid_to IS NOT NULL)
              AND retrieval_weight > ?
            """,
            (_SUPERSEDED_DECAY_WEIGHT,),
        )
        rows = [r[0] for r in cursor.fetchall()]
        for rowid in rows:
            cursor.execute(
                "UPDATE data_graph SET retrieval_weight=? WHERE rowid=?",
                (_SUPERSEDED_DECAY_WEIGHT, rowid),
            )
        return len(rows)

    def _delete_superseded_facts(self, conn: sqlite3.Connection, now: datetime) -> int:
        """Hard-delete superseded facts that have been invalid past the window.

        The invalidation clock is ``valid_to`` when present, falling back to
        ``last_confirmed_at`` for rows superseded before the bi-temporal columns
        existed.  Vec/FTS shadow rows are cleared alongside the base row.
        """
        cutoff = (now - timedelta(days=_SUPERSEDED_DELETE_AFTER_DAYS)).isoformat()
        rows = conn.execute(
            """
            SELECT rowid FROM data_graph
            WHERE deleted_at IS NULL
              AND (active=0 OR valid_to IS NOT NULL)
              AND julianday(COALESCE(valid_to, last_confirmed_at)) < julianday(?)
            """,
            (cutoff,),
        ).fetchall()
        for (rowid,) in rows:
            self._purge_dg_row(conn, rowid)
        return len(rows)

    def _delete_expired_misc(self, conn: sqlite3.Connection, now: datetime) -> int:
        """Hard-delete misc scratch rows older than their short fixed TTL.

        The TTL is sourced from the single ``_KIND_POLICY`` table so the misc
        expiry window has one home and cannot drift from the decay policy.
        """
        cutoff = (now - timedelta(days=cast(float, _KIND_POLICY[KIND_MISC]['ttl_days']))).isoformat()
        rows = conn.execute(
            """
            SELECT rowid FROM data_graph
            WHERE kind=?
              AND deleted_at IS NULL
              AND julianday(last_confirmed_at) < julianday(?)
            """,
            (KIND_MISC, cutoff),
        ).fetchall()
        for (rowid,) in rows:
            self._purge_dg_row(conn, rowid)
        return len(rows)

    def _purge_dg_row(self, conn: sqlite3.Connection, rowid: int) -> None:
        """Hard-delete one data_graph row and its FTS / vector shadow rows."""
        self._remove_fts(conn, rowid)
        conn.execute(_SQL_DELETE_DG_ROW, (rowid,))
        conn.execute(_SQL_DELETE_DG_KEY_VEC, (rowid,))
        conn.execute(_SQL_DELETE_DG_VALUE_VEC, (rowid,))

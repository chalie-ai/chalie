import logging
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Optional

from services.database_service import get_shared_db_service
from services.embedding_utils import pack_embedding
from services.log_utils import safe
from services.time_utils import utc_now, parse_utc
# SearchExpanderService: generates doc2query variants + embeds them for KNN recall.
# Replaces _schedule_embeddings and _schedule_doc2query fire-and-forget threads.
import services.search_expander_service as _ses

logger = logging.getLogger(__name__)

KIND_USER_SPECIFIC = 'user_specific'
KIND_SYSTEM = 'system'
KIND_MISC = 'misc'
KIND_MOMENT = 'moment'
KIND_DOCUMENT = 'document'
KIND_BEHAVIORAL_PATTERN = 'behavioral_pattern'
VALID_KINDS = frozenset({
    KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC,
    KIND_MOMENT, KIND_DOCUMENT, KIND_BEHAVIORAL_PATTERN,
})

_SELECT_ACTIVE_BY_KIND_KEY_SQL = (
    "SELECT * FROM data_graph WHERE kind=? AND key=? AND active=1 LIMIT 1"
)
_ORDER_FIRST_SEEN_DESC = 'first_seen_at DESC'
_SQL_DELETE_DG_ROW = "DELETE FROM data_graph WHERE rowid=?"
_SQL_DELETE_DG_KEY_VEC = "DELETE FROM data_graph_key_vec WHERE rowid=?"
_SQL_DELETE_DG_VALUE_VEC = "DELETE FROM data_graph_value_vec WHERE rowid=?"


@dataclass
class _StoreRequest:
    """Groups the four store-time write parameters to satisfy the 5-param ceiling."""

    kind: str
    key: str
    value: str
    source: Optional[str]


_KIND_POLICY = {
    KIND_USER_SPECIFIC:      {'ttl_days': 30,    'reinforce': True,  'contradiction': 'lut_canonicalize', 'deletion': 'soft',     'd_base': 0.5,  'salience_floor': 0.2},
    KIND_SYSTEM:             {'ttl_days': None,  'reinforce': True,  'contradiction': 'cosine_supersede', 'deletion': 'explicit', 'd_base': 0.05, 'salience_floor': 0.7},
    KIND_MISC:               {'ttl_days': 2,     'reinforce': False, 'contradiction': None,               'deletion': 'hard',     'd_base': 1.5,  'salience_floor': 0.0},
    KIND_MOMENT:             {'ttl_days': None,  'reinforce': False, 'contradiction': None,               'deletion': 'soft',     'd_base': 0.3,  'salience_floor': 0.0},
    KIND_DOCUMENT:           {'ttl_days': None,  'reinforce': False, 'contradiction': None,               'deletion': 'hard',     'd_base': 0.0,  'salience_floor': 0.0},
    # behavioral_pattern: written exclusively by abilities.pattern_match.save_pattern.SavePattern
    # via raw SQL UPSERT (one-active-row-per-(kind, key)). Decay (-0.005/pass with
    # soft-delete at 0) is handled by PatternMatchProcessor.post_turn() — DecayEngine
    # does NOT touch this kind.
    KIND_BEHAVIORAL_PATTERN: {'ttl_days': None,  'reinforce': True,  'contradiction': None,               'deletion': 'soft',     'd_base': 0.1,  'salience_floor': 0.3},
}

# Concept LUT asset — pre-built sqlite with lut_concepts + lut_embeddings (vec0).
# Regenerate with: cd backend && python -m utils.generate_concept_lut
_CONCEPT_LUT_PATH = os.path.join(
    os.path.dirname(__file__), 'data_graph', 'assets', 'concept_lut.sqlite'
)

# Cosine threshold for LUT canonical match.
_CONCEPT_LUT_THRESHOLD = 0.80

# Cosine threshold for system_specific key deduplication.
_SYSTEM_KEY_THRESHOLD = 0.80

# KNN depth for LUT lookups — k=1 sufficient for single canonical match.
_LUT_K = 1

# KNN depth for system key cosine deduplication.
_SYSTEM_KEY_K = 3

# Minimum cosine score for a recall candidate to pass the relevance floor.
_RECALL_COSINE_FLOOR = 0.42

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


def _get_stop_words() -> frozenset:
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
    KIND_MOMENT = KIND_MOMENT

    def __init__(self, db_service=None):
        if db_service is None:
            db_service = get_shared_db_service()
        self.db = db_service

    # ── Private helpers ───────────────────────────────────────────────

    def _row_to_dict(self, row) -> Optional[dict]:
        if row is None:
            return None
        return dict(row)

    def _generate_embedding(self, text: str):
        try:
            from services.embedding_service import get_embedding_service
            emb = get_embedding_service().generate_embedding(text)
            if hasattr(emb, 'tolist'):
                return emb.tolist()
            return list(emb)
        except Exception as e:
            logger.debug("[DATA GRAPH] Embedding generation failed: %s", e)
            return None

    def _store_key_vec(self, conn, rowid: int, embedding):
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

    def _store_value_vec(self, conn, rowid: int, embedding):
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

    def _sync_fts(self, conn, rowid: int, key: str = None, value: str = None,
                  kind: str = None, search_queries: str = None):
        """Insert a row into the FTS index. Reads from data_graph if values not provided.

        For external content FTS5 tables, callers must remove old FTS entries
        via _delete_fts BEFORE updating the content table — regular DELETE
        reads from the content table and corrupts the index on mismatch.
        """
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

    def _delete_fts(self, conn, rowid: int, key: str, value: str, kind: str,
                    search_queries: str = ''):
        """Remove a row from the FTS index.

        Tries the FTS5 'delete' command first (required for external content
        tables in production). Falls back to regular DELETE for standalone FTS
        tables (used in tests).
        """
        try:
            conn.execute(
                "INSERT INTO data_graph_fts(data_graph_fts, rowid, key, value, kind, search_queries) "
                "VALUES('delete', ?, ?, ?, ?, ?)",
                (rowid, key, value or '', kind, search_queries or '')
            )
        except Exception:
            try:
                conn.execute("DELETE FROM data_graph_fts WHERE rowid = ?", (rowid,))
            except Exception as e:
                logger.warning("[DATA GRAPH] FTS delete failed for rowid=%s: %s", rowid, e)

    def _remove_fts(self, conn, rowid: int):
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

    def _reinforce_row(self, conn, row_id: int, existing_dict: dict, now_iso: str):
        old_evidence = existing_dict.get('evidence_count', 1)
        new_evidence = old_evidence + 1
        old_strength = existing_dict.get('storage_strength', 0.5)
        boost = 0.05 / math.log2(new_evidence + 1)
        new_strength = min(1.0, old_strength + boost)
        conn.execute("""
            UPDATE data_graph
            SET evidence_count=?, storage_strength=?, retrieval_weight=1.0,
                last_confirmed_at=?, last_accessed_at=?
            WHERE rowid=?
        """, (new_evidence, new_strength, now_iso, now_iso, row_id))

    def _fetch_row_by_id(self, conn, row_id: int) -> dict:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM data_graph WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        cursor.close()
        return self._row_to_dict(row)

    # ── LUT helpers ───────────────────────────────────────────────────

    def _lookup_concept_lut(self, key_embedding) -> Optional[dict]:
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

    def _record_lut_miss(self, conn, kind: str, key: str, value: str, top_cos: float, now_iso: str) -> None:
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

    def _get_lut_miss_top_cos(self, key_embedding) -> float:
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

    def _apply_temporal_supersession(self, conn, existing_dict: dict, req: '_StoreRequest', now_iso: str) -> tuple[dict, Optional[tuple], Optional[tuple]]:
        """Demote old row, insert new, add supersedes/superseded_by edges.

        Returns (new_row_dict, schedule_emb_args, schedule_d2q_args).
        """
        row_id = existing_dict['id']
        old_rw = existing_dict.get('retrieval_weight', 1.0)
        conn.execute(
            "UPDATE data_graph SET active=0, retrieval_weight=? WHERE rowid=?",
            (old_rw * 0.5, row_id),
        )
        conn.execute(
            "INSERT INTO data_graph (kind, key, value, source, first_seen_at, last_confirmed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (req.kind, req.key, req.value, req.source, now_iso, now_iso),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._add_edge_with_conn(conn, new_id, row_id, 'supersedes')
        self._add_edge_with_conn(conn, row_id, new_id, 'superseded_by')
        self._sync_fts(conn, new_id, req.key, req.value, req.kind)
        logger.info("[DATA GRAPH] temporal supersede: demoted %s, inserted %s for key='%s'", row_id, new_id, req.key)
        return self._fetch_row_by_id(conn, new_id), (new_id, req.key, req.value), (new_id, req.key, req.value)

    def _find_system_key_match(self, conn, key_embedding, kind: str) -> Optional[dict]:
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

    # ── store() ───────────────────────────────────────────────────────

    def store(self, kind: str, key: str, value: str, *, source=None) -> Optional[dict]:
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
                cursor.execute(
                    _SELECT_ACTIVE_BY_KIND_KEY_SQL,
                    (kind, key)
                )
                existing = cursor.fetchone()
                cursor.close()

                now_iso = utc_now().isoformat()

                if existing:
                    existing_dict = self._row_to_dict(existing)
                    row_id = existing_dict['id']
                    old_value = existing_dict.get('value') or ''
                    new_value = value or ''
                    existing_date = (
                        existing_dict.get("last_confirmed_at")
                        or existing_dict.get("first_seen_at")
                        or ""
                    )[:10] or None

                    if new_value.lower().strip() == old_value.lower().strip():
                        if policy['reinforce']:
                            self._reinforce_row(conn, row_id, existing_dict, now_iso)
                        row = self._fetch_row_by_id(conn, row_id)
                        result = self._make_store_result(
                            "reinforced", key, key, None, value, None,
                            None, existing_date, row,
                        )
                    else:
                        contradiction_mode = policy.get('contradiction')
                        existing_req = _StoreRequest(kind, key, value, source)

                        if contradiction_mode == 'lut_canonicalize':
                            # Exact-key match: existing row found → apply rule based on LUT lookup.
                            # LUT only consulted to determine the rule for this key.
                            key_emb = self._generate_embedding(key)
                            lut_hit = self._lookup_concept_lut(key_emb) if key_emb else None
                            rule = lut_hit['rule'] if lut_hit else None

                            if lut_hit and rule == 'coexist':
                                # Coexist with existing same key — insert additive (different value)
                                row, _schedule_emb_args, _schedule_d2q_args = self._insert_new_row(
                                    conn, existing_req, now_iso
                                )
                                all_vals = self._fetch_coexist_values(conn, kind, key)
                                result = self._make_store_result(
                                    "appended", key, key, rule, value, None,
                                    all_vals, existing_date, row,
                                )
                            elif lut_hit and rule == 'immutable':
                                result = self._make_store_result(
                                    "conflict", key, key, rule, value, old_value,
                                    None, existing_date, existing_dict,
                                )
                                self._log_immutable_conflict(key)
                            else:
                                # No LUT hit for this key or embedding unavailable — temporal default.
                                # Miss was already recorded on the first write (new-row path).
                                row, _schedule_emb_args, _schedule_d2q_args = self._apply_temporal_supersession(
                                    conn, existing_dict, existing_req, now_iso
                                )
                                result = self._make_store_result(
                                    "superseded", key, key, rule, value, old_value,
                                    None, existing_date, row,
                                )

                        elif contradiction_mode == 'cosine_supersede':
                            # System kind: exact-key match with different value → temporal supersession
                            row, _schedule_emb_args, _schedule_d2q_args = self._apply_temporal_supersession(
                                conn, existing_dict, existing_req, now_iso
                            )
                            result = self._make_store_result(
                                "superseded", key, key, None, value, old_value,
                                None, existing_date, row,
                            )

                        else:
                            # None policy — insert directly (additive)
                            row, _schedule_emb_args, _schedule_d2q_args = self._insert_new_row(
                                conn, existing_req, now_iso
                            )
                            result = self._make_store_result(
                                "created", key, key, None, value, None,
                                None, None, row,
                            )
                else:
                    # No existing row with this exact key — run canonicalization paths
                    contradiction_mode = policy.get('contradiction')
                    store_req = _StoreRequest(kind, key, value, source)

                    if contradiction_mode == 'lut_canonicalize':
                        result, _schedule_emb_args, _schedule_d2q_args = self._store_user_specific_new(
                            conn, store_req, now_iso
                        )

                    elif contradiction_mode == 'cosine_supersede':
                        result, _schedule_emb_args, _schedule_d2q_args = self._store_system_new(
                            conn, store_req, now_iso
                        )

                    else:
                        row, _schedule_emb_args, _schedule_d2q_args = self._insert_new_row(
                            conn, store_req, now_iso
                        )
                        result = self._make_store_result(
                            "created", key, key, None, value, None, None, None, row,
                        )

            # Enqueue for SearchExpanderService AFTER connection exits (row committed).
            # SES absorbs both _schedule_embeddings (key_vec/value_vec backfill) and
            # _schedule_doc2query (variant generation + FTS sync + expanded_semantic writes).
            # Either arg being set means a new or superseded row was written.
            if _schedule_emb_args or _schedule_d2q_args:
                rowid = (_schedule_emb_args or _schedule_d2q_args)[0]
                _ses.enqueue("data_graph", rowid)

            return result

        except Exception as e:
            logger.error("[DATA GRAPH] store failed for kind=%s key='%s': %s", kind, key, e)
            return None

    def _make_store_result(
        self,
        status: str,
        provided_key: str,
        canonical_key: str,
        rule: Optional[str],
        value: str,
        old_value: Optional[str],
        all_values: Optional[list],
        date: Optional[str],
        row: Optional[dict],
    ) -> dict:
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

    def _fetch_coexist_values(self, conn, kind: str, key: str) -> list:
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

    def _store_user_specific_new(self, conn, req: '_StoreRequest', now_iso: str) -> tuple:
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

        canonical_key = lut_hit['canonical_key']
        rule = lut_hit['rule']

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

    def _store_user_specific_temporal(self, conn, req, canonical_key, rule, now_iso) -> tuple:
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
        old_val = existing_dict.get('value') or ''
        existing_date = (
            existing_dict.get("last_confirmed_at") or existing_dict.get("first_seen_at") or ""
        )[:10] or None
        if req.value.lower().strip() == old_val.lower().strip():
            self._reinforce_row(conn, existing_dict['id'], existing_dict, now_iso)
            row = self._fetch_row_by_id(conn, existing_dict['id'])
            return self._make_store_result(
                "reinforced", req.key, canonical_key, rule, req.value, None, None, existing_date, row,
            ), None, None
        canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
        row, emb_args, d2q_args = self._apply_temporal_supersession(
            conn, existing_dict, canon_req, now_iso
        )
        return self._make_store_result(
            "superseded", req.key, canonical_key, rule, req.value, old_val, None, existing_date, row,
        ), emb_args, d2q_args

    def _store_user_specific_coexist(self, conn, req, canonical_key, rule, now_iso) -> tuple:
        existing_exact = conn.execute(
            "SELECT * FROM data_graph "
            "WHERE kind=? AND key=? AND active=1 AND LOWER(TRIM(value))=LOWER(TRIM(?)) LIMIT 1",
            (req.kind, canonical_key, req.value),
        ).fetchone()
        if existing_exact is not None:
            existing_dict = self._row_to_dict(existing_exact)
            existing_date = (
                existing_dict.get("last_confirmed_at") or existing_dict.get("first_seen_at") or ""
            )[:10] or None
            self._reinforce_row(conn, existing_dict['id'], existing_dict, now_iso)
            row = self._fetch_row_by_id(conn, existing_dict['id'])
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

    def _store_user_specific_immutable(self, conn, req, canonical_key, rule, now_iso) -> tuple:
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
        old_val = existing_dict.get('value') or ''
        existing_date = (
            existing_dict.get("last_confirmed_at") or existing_dict.get("first_seen_at") or ""
        )[:10] or None

        if req.value.lower().strip() == old_val.lower().strip():
            self._reinforce_row(conn, existing_dict['id'], existing_dict, now_iso)
            row = self._fetch_row_by_id(conn, existing_dict['id'])
            return self._make_store_result(
                "reinforced", req.key, canonical_key, rule, req.value, None, None, existing_date, row,
            ), None, None

        self._log_immutable_conflict(canonical_key)
        return self._make_store_result(
            "conflict", req.key, canonical_key, rule, req.value, old_val, None, existing_date, existing_dict,
        ), None, None

    def _store_system_new(self, conn, req: _StoreRequest, now_iso: str) -> tuple:
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

        canonical_key = match.get('key', req.key)
        old_value = match.get('value') or ''
        existing_date = (
            match.get("last_confirmed_at") or match.get("first_seen_at") or ""
        )[:10] or None

        if req.value.lower().strip() == old_value.lower().strip():
            self._reinforce_row(conn, match['id'], match, now_iso)
            refreshed = self._fetch_row_by_id(conn, match['id'])
            return self._make_store_result(
                "reinforced", req.key, canonical_key, None, req.value, None, None, existing_date, refreshed,
            ), None, None

        canon_req = _StoreRequest(req.kind, canonical_key, req.value, req.source)
        row, emb_args, d2q_args = self._apply_temporal_supersession(
            conn, match, canon_req, now_iso
        )
        return self._make_store_result(
            "superseded", req.key, canonical_key, None, req.value, old_value, None, existing_date, row,
        ), emb_args, d2q_args

    def _insert_new_row(self, conn, req: _StoreRequest, now_iso: str) -> tuple:
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

    def recall(self, query: str, *, kinds=None, limit: int = 10, expand_graph: bool = True) -> list:
        try:
            self._backfill_missing_embeddings()
            query_emb = self._generate_embedding(query)
            query_blob = pack_embedding(query_emb) if query_emb else None

            candidates = {}  # rowid -> {'row': dict, 'key_cos': float, 'value_cos': float, 'fts_bonus': float}

            k = min(limit * 3, 50)

            with self.db.connection() as conn:
                cursor = conn.cursor()

                def _build_filters():
                    filters = ["deleted_at IS NULL", "active=1"]
                    params = []
                    if kinds:
                        placeholders = ','.join('?' for _ in kinds)
                        filters.append(f"kind IN ({placeholders})")
                        params.extend(kinds)
                    return " AND ".join(filters), params

                filter_clause, filter_params = _build_filters()

                # ── Key vec search ──────────────────────────────────
                if query_blob:
                    try:
                        cursor.execute("""
                            SELECT rowid, distance
                            FROM data_graph_key_vec
                            WHERE embedding MATCH ? AND k = ?
                            ORDER BY distance
                        """, (query_blob, k))
                        for rowid, dist in cursor.fetchall():
                            cos = _l2_dist_to_cosine(dist)
                            if rowid not in candidates:
                                candidates[rowid] = {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0}
                            candidates[rowid]['key_cos'] = max(candidates[rowid]['key_cos'], cos)
                    except Exception as e:
                        logger.debug("[DATA GRAPH] key_vec search failed (non-fatal): %s", e)

                # ── Value vec search ────────────────────────────────
                if query_blob:
                    try:
                        cursor.execute("""
                            SELECT rowid, distance
                            FROM data_graph_value_vec
                            WHERE embedding MATCH ? AND k = ?
                            ORDER BY distance
                        """, (query_blob, k))
                        for rowid, dist in cursor.fetchall():
                            cos = _l2_dist_to_cosine(dist)
                            if rowid not in candidates:
                                candidates[rowid] = {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0}
                            candidates[rowid]['value_cos'] = max(candidates[rowid]['value_cos'], cos)
                    except Exception as e:
                        logger.debug("[DATA GRAPH] value_vec search failed (non-fatal): %s", e)

                # ── FTS search ──────────────────────────────────────
                try:
                    fts_query = re.sub(r'[^\w\s]', '', query).strip()
                    if fts_query:
                        fts_words = [
                            w for w in fts_query.split()
                            if w and w.lower() not in _get_stop_words()
                        ]
                        fts_terms = ' OR '.join(f'"{w}"*' for w in fts_words if w)
                        if fts_terms:
                            cursor.execute("""
                                SELECT rowid, rank
                                FROM data_graph_fts
                                WHERE data_graph_fts MATCH ?
                                ORDER BY rank
                                LIMIT 30
                            """, (fts_terms,))
                            for rowid, _rank in cursor.fetchall():
                                if rowid not in candidates:
                                    candidates[rowid] = {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0}
                                candidates[rowid]['fts_bonus'] = 1.0
                except Exception as e:
                    logger.debug("[DATA GRAPH] FTS search failed (non-fatal): %s", e)

                # ── Variant vec search (expanded_semantic_vec) ──────────
                # doc2query variants written by SearchExpanderService.  Hits
                # join back to data_graph via expanded_semantic.related_to_id.
                if query_blob:
                    try:
                        variant_k = min(k, 50)
                        cursor.execute("""
                            SELECT CAST(es.related_to_id AS INTEGER) AS source_id,
                                   v.distance
                            FROM expanded_semantic_vec v
                            JOIN expanded_semantic es ON es.id = v.rowid
                            WHERE v.embedding MATCH ? AND k = ?
                              AND es.relates_to_table = 'data_graph'
                            ORDER BY v.distance
                        """, (query_blob, variant_k))
                        for source_id, dist in cursor.fetchall():
                            cos = _l2_dist_to_cosine(dist)
                            if source_id not in candidates:
                                candidates[source_id] = {'key_cos': 0.0, 'value_cos': 0.0, 'fts_bonus': 0.0, 'variant_cos': 0.0}
                            candidates[source_id]['variant_cos'] = max(
                                candidates[source_id].get('variant_cos', 0.0), cos
                            )
                    except Exception as e:
                        logger.debug("[DATA GRAPH] Variant vec search failed (non-fatal): %s", e)

                # Ensure all existing candidates have the variant_cos key.
                for sigs in candidates.values():
                    sigs.setdefault('variant_cos', 0.0)

                # ── Relevance floor — drop candidates with no strong signal ──
                candidates = {
                    rid: sigs for rid, sigs in candidates.items()
                    if sigs['key_cos'] >= _RECALL_COSINE_FLOOR
                    or sigs['value_cos'] >= _RECALL_COSINE_FLOOR
                    or sigs.get('variant_cos', 0.0) >= _RECALL_COSINE_FLOOR
                    or sigs['fts_bonus'] > 0
                }

                # ── Fetch full rows ─────────────────────────────────
                scored = []
                now_ts = utc_now().timestamp()

                for rowid, sigs in candidates.items():
                    cursor.execute(
                        f"SELECT * FROM data_graph WHERE id=? AND {filter_clause}",
                        [rowid] + filter_params
                    )
                    row = cursor.fetchone()
                    if not row:
                        continue
                    d = self._row_to_dict(row)

                    base_score = (
                        2.0 * sigs['key_cos']
                        + 1.0 * sigs['value_cos']
                        + 0.3 * sigs['fts_bonus']
                        + 0.8 * sigs.get('variant_cos', 0.0)
                    )

                    ref_ts_str = d.get('last_accessed_at') or d.get('last_confirmed_at')
                    if ref_ts_str:
                        try:
                            ref_ts = parse_utc(ref_ts_str).timestamp()
                        except Exception:
                            ref_ts = now_ts - 3600
                    else:
                        ref_ts = now_ts - 3600

                    age_seconds = max(1, now_ts - ref_ts)
                    evidence = max(1, d.get('evidence_count', 1))
                    actr_boost = math.log(evidence) - 0.5 * math.log(age_seconds)
                    composite = base_score * d.get('retrieval_weight', 1.0) * (1 + 0.3 * _sigmoid(actr_boost))

                    d['composite_score'] = composite
                    d['cos_score'] = max(sigs.get('key_cos', 0.0), sigs.get('value_cos', 0.0))
                    scored.append(d)

                scored.sort(key=lambda x: x['composite_score'], reverse=True)
                top_k = scored[:limit]

                # ── 1-hop graph expansion ───────────────────────────
                if expand_graph and top_k:
                    expansion = {}

                    for seed in top_k:
                        seed_id = seed.get('id')
                        if not seed_id:
                            continue

                        cursor.execute("""
                            SELECT e.to_id, e.edge_type, e.strength, e.from_id
                            FROM data_graph_edges e
                            WHERE e.from_id = ?
                        """, (seed_id,))
                        edges = cursor.fetchall()
                        out_degree = len(edges)

                        for to_id, edge_type, strength, _ in edges:
                            if to_id in {d['id'] for d in top_k}:
                                continue

                            cursor.execute(
                                f"SELECT * FROM data_graph WHERE id=? AND {filter_clause}",
                                [to_id] + filter_params
                            )
                            n_row = cursor.fetchone()
                            if not n_row:
                                continue
                            n_dict = self._row_to_dict(n_row)

                            multiplier = _EDGE_TYPE_MULTIPLIER.get(edge_type, 1.0)
                            n_score = seed['composite_score'] * strength * multiplier

                            if n_dict.get('kind') != seed.get('kind'):
                                n_score *= 1.2

                            if out_degree > 10:
                                n_score /= math.sqrt(out_degree)

                            neighbour_id = n_dict.get('id')
                            if neighbour_id not in expansion or expansion[neighbour_id]['composite_score'] < n_score:
                                n_dict['composite_score'] = n_score
                                n_dict['cos_score'] = seed.get('cos_score', 0.0) / 2.0
                                expansion[neighbour_id] = n_dict

                    all_candidates = {d['id']: d for d in top_k}
                    for nid, nd in expansion.items():
                        if nid not in all_candidates:
                            all_candidates[nid] = nd
                    top_k = sorted(all_candidates.values(), key=lambda x: x['composite_score'], reverse=True)[:limit]

                # ── Touch accessed ──────────────────────────────────
                if top_k:
                    now_iso = utc_now().isoformat()
                    for d in top_k:
                        rid = d.get('id')
                        if rid:
                            old_rw = d.get('retrieval_weight', 1.0)
                            new_rw = min(1.0, old_rw + 0.1)
                            cursor.execute("""
                                UPDATE data_graph
                                SET last_accessed_at=?, retrieval_weight=?
                                WHERE rowid=?
                            """, (now_iso, new_rw, rid))
                            d['retrieval_weight'] = new_rw

                cursor.close()

                return [
                    {
                        'id': d.get('id'),
                        'kind': d.get('kind'),
                        'key': d.get('key'),
                        'value': d.get('value'),
                        'source': d.get('source'),
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

    def fetch(self, *, kinds=None, limit=None, order_by=_ORDER_FIRST_SEEN_DESC,
              include_inactive=False, include_deleted=False) -> list:
        try:
            if order_by not in self._VALID_ORDER_BY:
                logger.warning("[DATA GRAPH] Invalid order_by '%s', using default", order_by)
                order_by = _ORDER_FIRST_SEEN_DESC

            filters = []
            params = []

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

    def _add_edge_with_conn(self, conn, from_id: int, to_id: int, edge_type: str = 'related', strength: float = 1.0) -> int:
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
        remaining_values: Optional[list] = None,
        versions_removed: Optional[int] = None,
        date: Optional[str] = None,
    ) -> dict:
        """Construct the structured forget result dict with explicit parameters; no closure captures."""
        base = {
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

    def _hard_delete_row(self, conn, row_id: int) -> None:
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
    def _row_date(d: dict) -> Optional[str]:
        """Extract a YYYY-MM-DD date string from a row dict, or None."""
        raw = d.get('last_confirmed_at') or d.get('first_seen_at') or ""
        return raw[:10] or None

    def _resolve_lut_key(self, kind: str, key: str) -> tuple:
        """Return (canonical_key, rule) after LUT lookup, or (key, None) on miss."""
        policy = _KIND_POLICY[kind]
        if policy.get('contradiction') != 'lut_canonicalize':
            return key, None
        key_emb = self._generate_embedding(key)
        lut_hit = self._lookup_concept_lut(key_emb) if key_emb else None
        if lut_hit:
            return lut_hit['canonical_key'], lut_hit['rule']
        return key, None

    def _forget_immutable(self, conn, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict:
        row = conn.execute(
            "SELECT * FROM data_graph WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL LIMIT 1",
            (kind, canonical_key),
        ).fetchone()
        if row is None:
            return self._make_forget_result("not_found", provided_key, canonical_key, rule, value)
        d = self._row_to_dict(row)
        self._hard_delete_row(conn, d['id'])
        return self._make_forget_result("forgotten", provided_key, canonical_key, rule, value, old_value=d.get('value'), date=self._row_date(d))

    def _forget_temporal(self, conn, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict:
        rows = conn.execute(
            "SELECT * FROM data_graph WHERE kind=? AND key=? AND deleted_at IS NULL",
            (kind, canonical_key),
        ).fetchall()
        if not rows:
            return self._make_forget_result("not_found", provided_key, canonical_key, rule, value)
        for r in rows:
            self._hard_delete_row(conn, self._row_to_dict(r)['id'])
        return self._make_forget_result("forgotten_all", provided_key, canonical_key, rule, value, versions_removed=len(rows))

    def _forget_coexist(self, conn, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict:
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
        self._hard_delete_row(conn, d['id'])
        remaining = self._fetch_coexist_values(conn, kind, canonical_key)
        status = "forgotten_empty" if not remaining else "forgotten"
        return self._make_forget_result(status, provided_key, canonical_key, rule, value, date=self._row_date(d), remaining_values=remaining or None)

    def _forget_raw(self, conn, kind: str, canonical_key: str, provided_key: str, rule: Optional[str], value: Optional[str]) -> dict:
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
            self._hard_delete_row(conn, d['id'])
            return self._make_forget_result("forgotten", provided_key, canonical_key, rule, value, old_value=d.get('value'), date=self._row_date(d))
        for r in rows:
            self._hard_delete_row(conn, self._row_to_dict(r)['id'])
        if len(rows) == 1:
            d = self._row_to_dict(rows[0])
            return self._make_forget_result("forgotten", provided_key, canonical_key, rule, value, old_value=d.get('value'), date=self._row_date(d))
        return self._make_forget_result("forgotten_all", provided_key, canonical_key, rule, value, versions_removed=len(rows))

    def forget(self, kind: str, key: str, value: str = None) -> Optional[dict]:
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
            handler = _RULE_HANDLERS.get(rule, self._forget_raw)
            with self.db.connection() as conn:
                return handler(conn, kind, canonical_key, key, rule, value)
        except Exception as e:
            logger.error("[DATA GRAPH] forget failed for kind=%s key='%s': %s", kind, key, e)
            return None

    # ── Decay cycle ───────────────────────────────────────────────────

    def decay_cycle(self) -> int:
        total_updated = 0
        try:
            from datetime import timedelta
            now = utc_now()
            one_hour_ago = (now - timedelta(hours=1)).isoformat()
            two_days_ago = (now - timedelta(days=2)).isoformat()

            with self.db.connection() as conn:
                cursor = conn.cursor()

                for kind, policy in _KIND_POLICY.items():
                    if policy['ttl_days'] is None:
                        continue

                    d_base = policy['d_base']
                    salience_floor = policy['salience_floor']

                    cursor.execute("""
                        SELECT rowid, retrieval_weight, last_confirmed_at
                        FROM data_graph
                        WHERE kind=?
                          AND deleted_at IS NULL
                          AND active=1
                          AND last_confirmed_at < ?
                    """, (kind, one_hour_ago))
                    rows = cursor.fetchall()

                    for rowid, rw, confirmed_at_str in rows:
                        if confirmed_at_str:
                            try:
                                confirmed_ts = parse_utc(confirmed_at_str).timestamp()
                            except Exception:
                                continue
                        else:
                            continue

                        age_days = (now.timestamp() - confirmed_ts) / 86400.0
                        if age_days <= 0:
                            continue

                        # Power-law absolute level: rw = max(1, age)^(-d_base)
                        # Not a multiplier — directly sets the retrieval_weight
                        # based on how old the fact is since last confirmation.
                        new_rw = max(salience_floor, max(1.0, age_days) ** (-d_base))

                        if abs(new_rw - rw) > 0.0001:
                            cursor.execute(
                                "UPDATE data_graph SET retrieval_weight=? WHERE rowid=?",
                                (new_rw, rowid)
                            )
                            total_updated += 1

                # Hard-delete expired misc rows past TTL (2 days)
                cursor.execute("""
                    SELECT rowid FROM data_graph
                    WHERE kind='misc'
                      AND deleted_at IS NULL
                      AND last_confirmed_at < ?
                """, (two_days_ago,))
                expired_misc = [r[0] for r in cursor.fetchall()]
                cursor.close()

                for rowid in expired_misc:
                    self._remove_fts(conn, rowid)
                    conn.execute(_SQL_DELETE_DG_ROW, (rowid,))
                    conn.execute(_SQL_DELETE_DG_KEY_VEC, (rowid,))
                    conn.execute(_SQL_DELETE_DG_VALUE_VEC, (rowid,))
                    total_updated += 1

            if total_updated > 0:
                logger.info("[DATA GRAPH] Decay cycle updated %d rows", total_updated)
            return total_updated

        except Exception as e:
            logger.error("[DATA GRAPH] decay_cycle failed: %s", e)
            return 0

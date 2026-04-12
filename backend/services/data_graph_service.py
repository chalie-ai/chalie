import json
import logging
import math
import re
import threading
from typing import Optional

from services.database_service import get_shared_db_service
from services.embedding_utils import pack_embedding
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

KIND_USER_SPECIFIC = 'user_specific'
KIND_SYSTEM = 'system'
KIND_MISC = 'misc'
KIND_MOMENT = 'moment'
VALID_KINDS = frozenset({KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC, KIND_MOMENT})

_KIND_POLICY = {
    KIND_USER_SPECIFIC: {'ttl_days': 30,   'reinforce': True,  'contradiction': 'classify',     'deletion': 'soft',     'd_base': 0.5,  'salience_floor': 0.2},
    KIND_SYSTEM:        {'ttl_days': None,  'reinforce': True,  'contradiction': 'newest_wins',  'deletion': 'explicit', 'd_base': 0.05, 'salience_floor': 0.7},
    KIND_MISC:          {'ttl_days': 2,     'reinforce': False, 'contradiction': None,           'deletion': 'hard',     'd_base': 1.5,  'salience_floor': 0.0},
    KIND_MOMENT:        {'ttl_days': None,  'reinforce': False, 'contradiction': None,           'deletion': 'soft',     'd_base': 0.3,  'salience_floor': 0.0},
}

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


# ── Seed function (module-level) ──────────────────────────────────────

_USER_SPECIFIC_KEYS = {
    'name', 'user_name', 'birthday', 'age', 'email', 'phone',
    'address', 'location', 'timezone', 'language', 'spouse',
    'children_names', 'job', 'occupation', 'pronouns',
}


def seed_from_legacy_knowledge(db_service):
    """
    One-time idempotent migration: copy rows from knowledge → data_graph.

    Uses entity-based heuristics per plan D10. Skips procedure and moment_context.
    Skips ambiguous rows. Direct INSERT for speed; schedules batch embeddings
    in a background thread.
    """
    try:
        with db_service.connection() as conn:
            if conn.execute("SELECT 1 FROM data_graph LIMIT 1").fetchone():
                return

            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity, kind, key, value, source, evidence_count, created_at
                FROM knowledge
                WHERE deleted_at IS NULL
                  AND kind NOT IN ('procedure', 'moment_context')
            """)
            rows = cursor.fetchall()

            now_iso = utc_now().isoformat()
            svc = DataGraphService(db_service)
            inserted_ids = []

            for row in rows:
                entity = (row['entity'] or '').lower().strip()
                kind_src = row['kind']
                key = row['key']
                value = row['value']
                source = row['source']
                evidence_count = row['evidence_count'] or 1
                created_at = row['created_at'] or now_iso

                if entity in ('', 'dylan', 'user') or any(
                    target in key.lower() for target in _USER_SPECIFIC_KEYS
                ):
                    target_kind = KIND_USER_SPECIFIC
                elif entity == 'chalie' and kind_src == 'rule':
                    target_kind = KIND_SYSTEM
                else:
                    continue

                seed_source = f"seed:from_knowledge:{source or 'unknown'}"
                cursor.execute("""
                    INSERT INTO data_graph
                        (kind, key, value, source, evidence_count,
                         first_seen_at, last_confirmed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (target_kind, key, value, seed_source, evidence_count,
                      created_at, now_iso))
                rid = cursor.lastrowid
                svc._sync_fts(conn, rid, key, value, target_kind)
                inserted_ids.append((rid, key, value))

            cursor.close()
            logger.info("[DATA GRAPH] Seed complete — %d rows migrated from knowledge", len(inserted_ids))

            # Batch-schedule embeddings in background (non-blocking boot)
            if inserted_ids:
                def _batch_embed():
                    for rid, k, v in inserted_ids:
                        svc._schedule_embeddings(rid, k, v)
                threading.Thread(target=_batch_embed, daemon=True).start()

    except Exception as e:
        logger.warning("[DATA GRAPH] seed_from_legacy_knowledge failed: %s", e)


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

    def _row_to_dict(self, row) -> dict:
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

            cursor.execute("DELETE FROM data_graph_fts WHERE rowid = ?", (rowid,))
            cursor.execute(
                "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) "
                "VALUES (?, ?, ?, ?, ?)",
                (rowid, key, value or '', kind, search_queries or '')
            )
            cursor.close()
        except Exception as e:
            logger.warning("[DATA GRAPH] FTS sync failed for rowid=%s: %s", rowid, e)

    def _remove_fts(self, conn, rowid: int):
        try:
            conn.execute("DELETE FROM data_graph_fts WHERE rowid = ?", (rowid,))
        except Exception as e:
            logger.warning("[DATA GRAPH] FTS removal failed for rowid=%s: %s", rowid, e)

    def _schedule_embeddings(self, rowid: int, key: str, value: str):
        def _run():
            try:
                key_emb = self._generate_embedding(key)
                value_emb = self._generate_embedding(value or key)
                with self.db.connection() as conn:
                    if key_emb:
                        self._store_key_vec(conn, rowid, key_emb)
                    if value_emb:
                        self._store_value_vec(conn, rowid, value_emb)
            except Exception as e:
                logger.warning("[DATA GRAPH] Embedding generation failed for rowid=%s: %s", rowid, e)
        threading.Thread(target=_run, daemon=True).start()

    def _schedule_doc2query(self, rowid: int, key: str, value: str):
        def _run():
            try:
                from services.doc2query_service import get_doc2query_service
                d2q = get_doc2query_service()
                if not d2q.is_available():
                    return
                queries = d2q.generate_queries(f"{key}: {value}")
                if not queries:
                    return
                with self.db.connection() as conn:
                    conn.execute(
                        "UPDATE data_graph SET search_queries = ? WHERE rowid = ?",
                        (json.dumps(queries), rowid)
                    )
                    self._sync_fts(conn, rowid)
            except Exception as e:
                logger.warning("[DATA GRAPH] doc2query failed for rowid=%s: %s", rowid, e)
        threading.Thread(target=_run, daemon=True).start()

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

    # ── store() ───────────────────────────────────────────────────────

    def store(self, kind: str, key: str, value: str, *, source=None) -> dict:
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
                    "SELECT * FROM data_graph WHERE kind=? AND key=? AND active=1 LIMIT 1",
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

                    if new_value.lower().strip() == old_value.lower().strip():
                        if policy['reinforce']:
                            self._reinforce_row(conn, row_id, existing_dict, now_iso)
                        result = self._fetch_row_by_id(conn, row_id)
                    else:
                        contradiction_mode = policy.get('contradiction')

                        if contradiction_mode == 'classify':
                            from services.contradiction_classifier_service import ContradictionClassifierService
                            cls_result = ContradictionClassifierService().check_new_trait(
                                new_value, old_value, source='chat'
                            )
                            if cls_result is None:
                                # compatible — reinforce
                                self._reinforce_row(conn, row_id, existing_dict, now_iso)
                                result = self._fetch_row_by_id(conn, row_id)
                            elif cls_result['classification'] == 'temporal_change':
                                # Demote old, insert new
                                old_rw = existing_dict.get('retrieval_weight', 1.0)
                                conn.execute("""
                                    UPDATE data_graph
                                    SET active=0, retrieval_weight=?
                                    WHERE rowid=?
                                """, (old_rw * 0.5, row_id))
                                conn.execute("""
                                    INSERT INTO data_graph
                                        (kind, key, value, source, first_seen_at, last_confirmed_at)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (kind, key, value, source, now_iso, now_iso))
                                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                                self._add_edge_with_conn(conn, new_id, row_id, 'supersedes')
                                self._add_edge_with_conn(conn, row_id, new_id, 'superseded_by')
                                self._sync_fts(conn, new_id, key, value, kind)
                                _schedule_emb_args = (new_id, key, value)
                                _schedule_d2q_args = (new_id, key, value)
                                result = self._fetch_row_by_id(conn, new_id)
                                logger.info("[DATA GRAPH] temporal_change: demoted %s, inserted %s for key='%s'", row_id, new_id, key)
                            else:
                                # true_contradiction or ambiguous — don't store
                                result = {
                                    'conflict': True,
                                    'classification': cls_result['classification'],
                                    'existing': existing_dict,
                                    'proposed_key': key,
                                    'proposed_value': value,
                                    'reasoning': cls_result.get('reasoning', ''),
                                }

                        elif contradiction_mode == 'newest_wins':
                            old_rw = existing_dict.get('retrieval_weight', 1.0)
                            conn.execute("""
                                UPDATE data_graph
                                SET active=0, retrieval_weight=?
                                WHERE rowid=?
                            """, (old_rw * 0.5, row_id))
                            conn.execute("""
                                INSERT INTO data_graph
                                    (kind, key, value, source, first_seen_at, last_confirmed_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (kind, key, value, source, now_iso, now_iso))
                            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                            self._add_edge_with_conn(conn, new_id, row_id, 'supersedes')
                            self._add_edge_with_conn(conn, row_id, new_id, 'superseded_by')
                            self._sync_fts(conn, new_id, key, value, kind)
                            _schedule_emb_args = (new_id, key, value)
                            _schedule_d2q_args = (new_id, key, value)
                            result = self._fetch_row_by_id(conn, new_id)

                        else:
                            # None policy — insert directly
                            conn.execute("""
                                INSERT INTO data_graph
                                    (kind, key, value, source, first_seen_at, last_confirmed_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (kind, key, value, source, now_iso, now_iso))
                            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                            self._sync_fts(conn, new_id, key, value, kind)
                            _schedule_emb_args = (new_id, key, value)
                            _schedule_d2q_args = (new_id, key, value)
                            result = self._fetch_row_by_id(conn, new_id)
                else:
                    # No existing row — insert new
                    conn.execute("""
                        INSERT INTO data_graph
                            (kind, key, value, source, first_seen_at, last_confirmed_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (kind, key, value, source, now_iso, now_iso))
                    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    self._sync_fts(conn, new_id, key, value, kind)
                    _schedule_emb_args = (new_id, key, value)
                    _schedule_d2q_args = (new_id, key, value)
                    result = self._fetch_row_by_id(conn, new_id)
                    logger.info("[DATA GRAPH] Stored new %s '%s'='%s' (source=%s)",
                                kind, key, (value or '')[:60], source)

            if _schedule_emb_args:
                self._schedule_embeddings(*_schedule_emb_args)
            if _schedule_d2q_args:
                self._schedule_doc2query(*_schedule_d2q_args)

            return result

        except Exception as e:
            logger.error("[DATA GRAPH] store failed for kind=%s key='%s': %s", kind, key, e)
            return None

    # ── recall() ─────────────────────────────────────────────────────

    def recall(self, query: str, *, kinds=None, limit: int = 10, expand_graph: bool = True) -> list:
        try:
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
                            cos = 1.0 - (dist ** 2 / 2.0)
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
                            cos = 1.0 - (dist ** 2 / 2.0)
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

                cursor.close()

                return [
                    {
                        'id': d.get('id'),
                        'kind': d.get('kind'),
                        'key': d.get('key'),
                        'value': d.get('value'),
                        'retrieval_weight': d.get('retrieval_weight'),
                        'evidence_count': d.get('evidence_count'),
                        'composite_score': d.get('composite_score'),
                    }
                    for d in top_k
                ]

        except Exception as e:
            logger.error("[DATA GRAPH] recall failed: %s", e)
            return []

    # ── fetch() ───────────────────────────────────────────────────────

    _VALID_ORDER_BY = frozenset({
        'first_seen_at DESC', 'last_confirmed_at DESC',
        'retrieval_weight DESC', 'evidence_count DESC',
        'first_seen_at ASC', 'last_confirmed_at ASC',
        'key ASC',
    })

    def fetch(self, *, kinds=None, limit=None, order_by='first_seen_at DESC',
              include_inactive=False, include_deleted=False) -> list:
        try:
            if order_by not in self._VALID_ORDER_BY:
                logger.warning("[DATA GRAPH] Invalid order_by '%s', using default", order_by)
                order_by = 'first_seen_at DESC'

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

    # ── find_similar_by_kind() ────────────────────────────────────────

    def find_similar_by_kind(self, embedding, kind: str, exclude_id: int, limit: int = 3) -> list:
        try:
            blob = pack_embedding(embedding) if embedding else None
            if blob is None:
                return []
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT v.rowid, v.distance
                    FROM data_graph_value_vec v
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                """, (blob, limit * 5))
                vec_results = cursor.fetchall()

                results = []
                for rowid, dist in vec_results:
                    if dist >= 0.25:
                        continue
                    cursor.execute(
                        "SELECT id, key, value, retrieval_weight, kind "
                        "FROM data_graph WHERE id=? AND kind=? AND id!=? AND deleted_at IS NULL AND active=1",
                        (rowid, kind, exclude_id)
                    )
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            'id': row[0],
                            'key': row[1],
                            'value': row[2],
                            'retrieval_weight': row[3],
                            'kind': row[4],
                        })
                    if len(results) >= limit:
                        break

                cursor.close()
                return results
        except Exception as e:
            logger.error("[DATA GRAPH] find_similar_by_kind failed: %s", e)
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

    def add_edge(self, from_id: int, to_id: int, *, edge_type: str = 'related', strength: float = 1.0) -> int:
        try:
            with self.db.connection() as conn:
                return self._add_edge_with_conn(conn, from_id, to_id, edge_type, strength)
        except Exception as e:
            logger.warning("[DATA GRAPH] add_edge failed from=%s to=%s type=%s: %s", from_id, to_id, edge_type, e)
            return 0

    def expand_edges(self, seed_ids: list, *, edge_types=None) -> list:
        if not seed_ids:
            return []
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' for _ in seed_ids)
                params = list(seed_ids)
                type_clause = ""
                if edge_types:
                    type_placeholders = ','.join('?' for _ in edge_types)
                    type_clause = f" AND edge_type IN ({type_placeholders})"
                    params.extend(edge_types)
                cursor.execute(
                    f"SELECT * FROM data_graph_edges WHERE from_id IN ({placeholders}){type_clause}",
                    params
                )
                rows = cursor.fetchall()
                cursor.close()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error("[DATA GRAPH] expand_edges failed: %s", e)
            return []

    def touch_edge(self, from_id: int, to_id: int, edge_type: str) -> None:
        try:
            with self.db.connection() as conn:
                conn.execute("""
                    UPDATE data_graph_edges
                    SET last_accessed_at=?
                    WHERE from_id=? AND to_id=? AND edge_type=?
                """, (utc_now().isoformat(), from_id, to_id, edge_type))
        except Exception as e:
            logger.warning("[DATA GRAPH] touch_edge failed: %s", e)

    # ── Reinforcement / demotion ──────────────────────────────────────

    def reinforce(self, row_id: int) -> None:
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT evidence_count, storage_strength FROM data_graph WHERE rowid=?",
                    (row_id,)
                )
                row = cursor.fetchone()
                cursor.close()
                if not row:
                    return
                old_evidence, old_strength = row
                new_evidence = old_evidence + 1
                boost = 0.05 / math.log2(new_evidence + 1)
                new_strength = min(1.0, old_strength + boost)
                now_iso = utc_now().isoformat()
                conn.execute("""
                    UPDATE data_graph
                    SET evidence_count=?, storage_strength=?, retrieval_weight=1.0,
                        last_confirmed_at=?, last_accessed_at=?
                    WHERE rowid=?
                """, (new_evidence, new_strength, now_iso, now_iso, row_id))
        except Exception as e:
            logger.warning("[DATA GRAPH] reinforce failed for rowid=%s: %s", row_id, e)

    def demote(self, row_id: int, factor: float = 0.5) -> None:
        try:
            with self.db.connection() as conn:
                conn.execute("""
                    UPDATE data_graph
                    SET retrieval_weight = retrieval_weight * ?
                    WHERE rowid=?
                """, (factor, row_id))
        except Exception as e:
            logger.warning("[DATA GRAPH] demote failed for rowid=%s: %s", row_id, e)

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

    def hard_delete_by_id(self, row_id: int) -> bool:
        try:
            with self.db.connection() as conn:
                conn.execute("DELETE FROM data_graph WHERE rowid=?", (row_id,))
                conn.execute("DELETE FROM data_graph_key_vec WHERE rowid=?", (row_id,))
                conn.execute("DELETE FROM data_graph_value_vec WHERE rowid=?", (row_id,))
                self._remove_fts(conn, row_id)
            return True
        except Exception as e:
            logger.warning("[DATA GRAPH] hard_delete_by_id failed for rowid=%s: %s", row_id, e)
            return False

    def set_active(self, row_id: int, active: int) -> None:
        try:
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE data_graph SET active=? WHERE rowid=?",
                    (int(active), row_id)
                )
        except Exception as e:
            logger.warning("[DATA GRAPH] set_active failed for rowid=%s: %s", row_id, e)

    # ── Decay cycle ───────────────────────────────────────────────────

    def decay_cycle(self) -> int:
        total_updated = 0
        try:
            now = utc_now()

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
                          AND last_confirmed_at < datetime('now', '-1 hour')
                    """, (kind,))
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
                      AND last_confirmed_at < datetime('now', '-2 days')
                """)
                expired_misc = [r[0] for r in cursor.fetchall()]
                cursor.close()

                for rowid in expired_misc:
                    conn.execute("DELETE FROM data_graph WHERE rowid=?", (rowid,))
                    conn.execute("DELETE FROM data_graph_key_vec WHERE rowid=?", (rowid,))
                    conn.execute("DELETE FROM data_graph_value_vec WHERE rowid=?", (rowid,))
                    conn.execute("DELETE FROM data_graph_fts WHERE rowid=?", (rowid,))
                    total_updated += 1

            if total_updated > 0:
                logger.info("[DATA GRAPH] Decay cycle updated %d rows", total_updated)
            return total_updated

        except Exception as e:
            logger.error("[DATA GRAPH] decay_cycle failed: %s", e)
            return 0

"""FactService — the one gateway to the ``user_specific`` (FACTS) lane of
``data_graph``. A thin wrapper over :class:`~models.fact.FactRow` (the sole home
of the *data_graph* SQL): it adds the typed status envelope the memory tool /
worker / save tool render, layers the concept-LUT canonicalizer over the store
path, and registers as the off-turn FACTS :class:`~orchestrators.decayable.Decayable`.
mp-optional (T2): store/forget take their ``source`` provenance explicitly, and
decay is off-turn and mp-less, so no method needs ``mp``.

The concept LUT is a SEPARATE preshipped database (``concept_lut.sqlite``, a
vec0 KNN index over 768-dim key embeddings), not part of the Model/Database
layer — it lives here on its own read-only connection. It maps a raw fact key
(``birthday``) to a canonical key (``birth_date``) plus a contradiction rule
(``temporal`` / ``coexist`` / ``immutable``), which selects the FactRow store
primitive to dispatch to."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import TYPE_CHECKING, cast

from models.fact import FactRow
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


# ── Concept LUT engine (a separate vec0 database, not the Model/DB layer) ─────

_CONCEPT_LUT_PATH = str(FileMapperService.get_concept_lut_db_path())
_CONCEPT_LUT_THRESHOLD = 0.80
_LUT_K = 1

_lut_conn: sqlite3.Connection | None = None
_lut_lock = threading.Lock()
_lut_loaded = False


def _l2_dist_to_cosine(distance: float) -> float:
    """sqlite-vec returns squared L2 for unit-norm vectors: cos = 1 - dist^2/2."""
    return max(0.0, 1.0 - (distance ** 2 / 2.0))


def _get_lut_conn() -> sqlite3.Connection | None:
    """Read-only sqlite connection to the concept LUT, loaded once (double-checked
    lock). Returns None when the asset is absent or fails to open — the store path
    then treats every key as a LUT miss rather than crashing."""
    global _lut_conn, _lut_loaded
    if _lut_loaded:
        return _lut_conn
    with _lut_lock:
        if _lut_loaded:
            return _lut_conn
        if not os.path.exists(_CONCEPT_LUT_PATH):
            logger.warning("[FACTS] concept_lut.sqlite not found at %s", _CONCEPT_LUT_PATH)
            _lut_loaded = True
            return None
        try:
            conn = sqlite3.connect(
                f"file:{_CONCEPT_LUT_PATH}?mode=ro", uri=True, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception:
                conn.load_extension("vec0")
            count = conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0]
            logger.info("[FACTS] LUT loaded: %s (concepts=%d)", _CONCEPT_LUT_PATH, count)
            _lut_conn = conn
        except Exception as exc:
            logger.warning("[FACTS] Failed to open concept LUT: %s", exc)
            _lut_conn = None
        _lut_loaded = True
        return _lut_conn


def _lookup_concept_lut(key_embedding: list[float] | None) -> dict[str, object] | None:
    """KNN a key embedding against the concept LUT; return
    ``{canonical_key, rule, cos}`` for the nearest concept, or None when the LUT
    is unavailable, the embedding is unusable, or the top match is below the
    cosine threshold (a LUT miss)."""
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
    except Exception as exc:
        logger.debug("[FACTS] LUT KNN failed: %s", exc)
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


def _get_lut_miss_top_cos(key_embedding: list[float] | None) -> float:
    """Top cosine from the LUT KNN even when below threshold — the miss-logging
    signal (how close the miss was to a canonical concept)."""
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


def _record_lut_miss(kind: str, key: str, value: str, top_cos: float) -> None:
    """Log a LUT miss and upsert a monitoring row into ``concept_lut_misses``
    (repeat misses bump ``count``). Best-effort: a write failure is logged at
    debug and never breaks the store."""
    logger.info("[FACTS] LUT miss: kind=%s key='%s' top_cos=%.4f", kind, key, top_cos)
    now_iso = utc_now().isoformat()
    value_preview = (value or "")[:100]
    try:
        from services.database import Database
        Database.conn().execute(
            "INSERT INTO concept_lut_misses(kind, key, value_preview, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET count=count+1, last_seen=excluded.last_seen",
            (kind, key, value_preview, now_iso, now_iso),
        )
    except Exception as exc:
        logger.debug("[FACTS] concept_lut_misses upsert failed: %s", exc)


class FactService:
    """Store, forget, read and decay the FACTS (user_specific) lane."""

    def __init__(self, mp: MessageProcessor | None = None) -> None:
        self.mp = mp

    def store(self, key: str, value: str, source: str | None = None,
              canonicalize: bool = True) -> dict[str, object]:
        """Upsert one fact; return the typed status envelope.

        ``canonicalize=True`` (default — the LLM memory tool and ``save_graph``)
        runs the concept LUT: the raw key is embedded, mapped to a canonical key
        plus a contradiction rule, and dispatched at the canonical key —
        ``temporal`` (created/reinforced/superseded), ``coexist`` (reinforced/
        appended/created) or ``immutable`` (created/reinforced/conflict). A LUT
        miss records a monitoring row and stores at the RAW key as
        ``lut_miss_created``. An embedding failure propagates (fail loud): storing
        under a non-canonical key when canonicalization was intended is silent
        key-space corruption.

        ``canonicalize=False`` (the fact-extraction worker, which already shows
        the extraction model the real neighbour keys) short-circuits to the
        exact-key temporal store with NO LUT — re-canonicalizing there would
        fight the model's choice and break exact-key supersession targeting."""
        if not canonicalize:
            row, status, old_value = FactRow.store(key, value, source=source)
            return self._store_envelope(status, key, key, value, old_value, None, None, row)

        from services.embedding_service import get_embedding_service
        key_embedding = get_embedding_service().generate_embedding(key)  # RAISES → fail loud
        hit = _lookup_concept_lut(key_embedding)

        if hit is None:
            top_cos = _get_lut_miss_top_cos(key_embedding)
            _record_lut_miss(FactRow.KIND, key, value, top_cos)
            row = FactRow.store(key, value, source=source)[0]
            return self._store_envelope("lut_miss_created", key, key, value, None, None, None, row)

        canonical_key = cast(str, hit["canonical_key"])
        rule = cast(str, hit["rule"])
        if rule == "coexist":
            row, status, all_values = FactRow.store_coexist(canonical_key, value, source=source)
            return self._store_envelope(status, key, canonical_key, value, None, rule, all_values, row)
        if rule == "immutable":
            row, status, old_value = FactRow.store_immutable(canonical_key, value, source=source)
            return self._store_envelope(status, key, canonical_key, value, old_value, rule, None, row)
        # temporal — and any unknown rule falls back to temporal supersession.
        row, status, old_value = FactRow.store(canonical_key, value, source=source)
        return self._store_envelope(status, key, canonical_key, value, old_value, rule, None, row)

    def forget(self, key: str, value: str | None = None) -> dict[str, object]:
        """Invalidate a fact by exact key (optionally a specific value); return the
        typed forget envelope (status ∈ {forgotten, value_not_found, not_found})."""
        closed = FactRow.forget(key, value)
        if closed:
            return {"action": "forget", "status": "forgotten", "canonical_key": key,
                    "provided_key": key, "value": value, "old_value": value, "versions_removed": closed}
        status = "value_not_found" if value is not None else "not_found"
        return {"action": "forget", "status": status, "canonical_key": key,
                "provided_key": key, "value": value, "remaining_values": []}

    def traits(self) -> list[FactRow]:
        """Live FACTS, most-reinforced first — the user-summary channel's facts
        section. (E9 repoints the prompt's traits read here from the gateway.)"""
        return FactRow.live().order_by("retrieval_weight DESC").get()

    def decay(self) -> int:
        """Off-turn Decayable entry point: power-law rw-decay of live FACTS."""
        return FactRow.decay()

    @staticmethod
    def _store_envelope(status: str, provided_key: str, canonical_key: str, value: str,
                        old_value: str | None, rule: str | None,
                        all_values: list[str] | None, row: FactRow) -> dict[str, object]:
        """The store status envelope the memory tool / save_graph / worker read —
        ports ``_make_store_result``'s shape. ``provided_key`` is the raw key the
        caller passed; ``canonical_key`` is the LUT canonical key (the raw key on a
        miss or a non-canonicalising store). ``rule`` is the concept's
        contradiction rule (temporal/coexist/immutable) or None off the LUT;
        ``all_values`` is the live value set for a coexist ``appended``. ``date``
        is the row's ``last_confirmed_at`` (the untouched existing row's, for an
        immutable ``conflict``)."""
        return {"action": "store", "status": status,
                "canonical_key": canonical_key, "provided_key": provided_key,
                "value": value, "old_value": old_value,
                "rule": rule, "all_values": all_values,
                "date": row.last_confirmed_at, "row": row}

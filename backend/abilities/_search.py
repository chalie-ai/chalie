"""Shared discovery-search base for find_tools and find_skills.

Both abilities take a ``query`` ARRAY of intents — one tool/skill name or one
described action per entry — and run every entry through ONE shared
precise→broad escalation cascade (:meth:`SearchableAbility._discover`):

1. strip stopwords;
2. **exact name match** on the whole normalised row → pinned (outside the global
   cap); a bare name owned by more than one source is refused and surfaced;
3. else **bm25 on the NAME/title only**, gated by name-segment alignment — NOT a
   bm25 score floor (the segment gate is the sole discriminator);
4. else **vector search on the full prose**, kept only at/under ``VEC_CEILING``.

The first rung that yields wins for that row; only the vector terminus failing
leaves a row empty. The two indexes are single-purpose: names/titles → bm25
(FTS5 trigram), full prose (summary/examples or use_for) → vec0 KNN cosine.

Each concrete ability supplies only what differs — which DB/tables it queries
and how it resolves an exact name (find_tools adds MCP and an ambiguity guard;
find_skills has neither). Everything else — the cascade, the segment gate, the
ceiling, per-name dedup, the embedding, the caps — lives here, shared verbatim.
"""

import logging
import re
import sqlite3
from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability

logger = logging.getLogger(__name__)

KNN_DEPTH = 30  # k passed to the vec0 MATCH; per-name dedup + cap trim the list

# Vector terminus ceiling for the discovery cascade — a row escalates to the
# vector rung only when bm25-name missed, and a vector hit counts only when its
# cosine distance is at or under this ceiling. FINE-TUNED 2026-06-23 over a
# 36-query sweep against the real abilities.sqlite (Charlie-typo 1.008 KEEP vs
# donut-junk 1.033 DROP, margin 0.025). DO NOT retune without re-running that sweep.
VEC_CEILING = 1.02

# Trigram indexes 3-char windows, so a term shorter than 3 chars matches nothing
# in FTS — it is dropped from the MATCH string but still feeds the vector path.
_MIN_FTS_LEN = 3

# A small static stopword set keeps the pre-pass hermetic (no NLTK download).
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its me my of on or "
    "that the their them then there these they this to was what when where which "
    "who will with you your".split()
)

# Strip everything but letters, digits and underscores so a term matches the
# same trigrams the index stored ("code." → "code", "list_tickets" preserved).
_TERM_RE = re.compile(r"[^a-z0-9_]")


def search_terms(raw: str) -> list[str]:
    """Stopword-stripped, punctuation-normalised terms for one cascade query row."""
    return [
        term
        for token in raw.split()
        if (term := _TERM_RE.sub("", token.lower())) and term not in _STOPWORDS
    ]


def segment_match(term: str, name: str) -> bool:
    """True when ``term`` aligns to a name SEGMENT — it equals a full segment or
    strongly prefixes one (≥ ``_MIN_FTS_LEN`` chars). The name is split on any
    non-alphanumeric run, so it spans both underscored tool names (``web_search``
    → {web, search}) and spaced skill titles (``Competitive Teardown`` →
    {competitive, teardown}).

    This gate, NOT a bm25 score floor, is the sole discriminator for the name
    rung: it accepts ``search`` ⊂ ``web_search`` / ``docs`` ⊂ ``chalie_docs`` while
    rejecting incidental substrings like ``all`` ⊂ ``review_tool_calls`` that raw
    trigram matching would wrongly admit. Fine-tuned 2026-06-23 — do not add a floor."""
    return any(
        term == seg or (len(term) >= _MIN_FTS_LEN and seg.startswith(term))
        for seg in re.split(r"[^a-z0-9]+", name.lower())
    )


class SearchableAbility(Ability, ABC):
    _DB_PATH: ClassVar[Path]
    _LOG_PREFIX: ClassVar[str] = ""

    # Per-entry cap: at most this many discovery results from the winning rung.
    _RESULT_CAP: ClassVar[int] = 3
    # Global backstop on discovery results across the whole query array (tunable,
    # NOT a fine-tuned floor). Exact-name pins sit OUTSIDE this cap.
    _GLOBAL_CAP: ClassVar[int] = 6

    @staticmethod
    def _norm(text: str) -> str:
        """Lowercase + collapse non-alphanumeric runs to single spaces — the shared
        normal form for exact-name resolution across underscored tool names and
        spaced skill titles (``web_search`` and ``Web Search`` both → ``web search``)."""
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @staticmethod
    def _fts_or(terms: list[str]) -> str:
        """OR-join the FTS-indexable terms (≥ trigram minimum), each quoted so
        punctuation can never raise a MATCH syntax error. Empty when nothing
        survives — callers skip the lexical rung."""
        return " OR ".join(f'"{t}"' for t in terms if len(t) >= _MIN_FTS_LEN)

    def _embed(self, terms: list[str]) -> "bytes | None":
        """Pack an embedding for the vector rung; ``None`` (logged) on any failure
        so the rung degrades to empty rather than raising."""
        try:
            from services.embedding_service import EmbeddingService
            from services.embedding_utils import pack_embedding
            return cast(
                "bytes",
                pack_embedding(EmbeddingService().generate_embedding(" ".join(terms), mp=self.mp)),
            )
        except Exception as exc:
            logger.warning("%s embedding failed (vector rung skipped): %s", self._LOG_PREFIX, exc)
            return None

    def _discover(
        self,
        rows: "list[object]",
        exact_fn: Callable[[str], "tuple[object | None, bool]"],
        bm25_fn: Callable[[list[str]], "list[object]"],
        vector_fn: Callable[[list[str]], "list[object]"],
    ) -> "tuple[list[object], list[object], list[str]]":
        """The shared precise→broad discovery cascade — identical for find_tools and
        find_skills. Per query-array row: stopword-strip → exact name (pinned,
        outside the global cap; ambiguous bare name refused + surfaced) → bm25 on
        the NAME only (segment-gated) → vector on the full prose (``VEC_CEILING``).
        The first rung that yields wins; only the vector terminus failing leaves a
        row empty. Returns ``(pinned_keys, discovered_keys_capped, not_found_rows)``."""
        pinned: list[object] = []
        discovered: list[object] = []
        not_found: list[str] = []
        for raw in rows:
            terms = search_terms(str(raw))
            if not terms:
                continue
            key, ambiguous = exact_fn(self._norm(" ".join(terms)))
            if ambiguous:
                not_found.append(
                    f"{' '.join(terms)} (ambiguous — name owned by more than one "
                    "source; use the prefixed name)"
                )
                continue
            if key is not None:
                pinned.append(key)
                continue
            hits = bm25_fn(terms) or vector_fn(terms)
            if hits:
                discovered.extend(hits)
            else:
                not_found.append(str(raw))
        pins = list(dict.fromkeys(pinned))
        disc = [k for k in dict.fromkeys(discovered) if k not in pins][: self._GLOBAL_CAP]
        return pins, disc, not_found

    def _gate(self, rows: "list[tuple[object, object, float]]", terms: list[str]) -> list[object]:
        """Rung-2 post-filter: keep bm25 rows whose NAME (``row[1]``) segment-aligns
        to a query term, per-key dedup (``row[0]``), return the top RESULT_CAP keys.
        The segment gate — NOT a score floor — is the discriminator (fine-tuned
        2026-06-23, do not add a floor)."""
        gated = [r for r in rows if any(segment_match(t, str(r[1])) for t in terms)]
        return [r[0] for r in self._best_per_key(gated)[: self._RESULT_CAP]]

    def _ceiling(self, rows: "list[tuple[object, object, float]]") -> list[object]:
        """Rung-3 post-filter: per-name dedup, keep only rows at/under ``VEC_CEILING``,
        return the top RESULT_CAP keys (fine-tuned 2026-06-23, do not retune)."""
        return [r[0] for r in self._best_per_key(rows) if r[2] <= VEC_CEILING][: self._RESULT_CAP]

    @staticmethod
    def _load_vec(conn: sqlite3.Connection) -> None:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception as exc:
            logger.debug(f"sqlite_vec module load failed, trying vec0: {exc}")
            conn.load_extension("vec0")

    def _vec_search(
        self,
        blob: "bytes | None",
        vec_sql: str,
        vec_params: "tuple[object, ...]",
        db_path: "Path | None" = None,
    ) -> "list[tuple[object, object, float]]":
        """KNN rows as ``(key, name, distance)``; ``[]`` when the embedding is
        absent, the vec table is missing, or the blob is invalid — never raises."""
        target = db_path if db_path is not None else self._DB_PATH
        if blob is None or not target.exists():
            return []
        try:
            conn = sqlite3.connect(str(target))
            try:
                self._load_vec(conn)
                return cast("list[tuple[object, object, float]]", conn.execute(vec_sql, vec_params).fetchall())
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(f"{self._LOG_PREFIX} vec search failed (skipping): {exc}")
            return []

    def _fts_search(
        self,
        fts_match: str,
        fts_sql: str,
        fts_params: "tuple[object, ...]",
        db_path: "Path | None" = None,
    ) -> "list[tuple[object, object, float]]":
        """FTS rows as ``(key, name, bm25)``; ``[]`` for an empty match string or
        any sqlite error — never raises."""
        target = db_path if db_path is not None else self._DB_PATH
        if not fts_match or not target.exists():
            return []
        try:
            conn = sqlite3.connect(str(target))
            try:
                return cast("list[tuple[object, object, float]]", conn.execute(fts_sql, fts_params).fetchall())
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(f"{self._LOG_PREFIX} FTS search failed (skipping): {exc}")
            return []

    @staticmethod
    def _best_per_key(rows: "list[tuple[object, object, float]]") -> "list[tuple[object, object, float]]":
        """Collapse repeat entries of one item (e.g. its name and a prose row both
        matched) to its single best-ranked row, ordered best-first (rank asc)."""
        best: "dict[object, tuple[object, object, float]]" = {}
        for key, label, rank in rows:
            if key not in best or rank < best[key][2]:
                best[key] = (key, label, rank)
        return sorted(best.values(), key=lambda r: r[2])

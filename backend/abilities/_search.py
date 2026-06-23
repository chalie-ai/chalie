"""Shared keyword + vector search base for find_tools and find_skills.

The ``query`` both abilities take is a keyword grammar, not prose:

* ``+term`` — required (the result must contain ``term`` as a substring)
* ``-term`` — excluded (the result must NOT contain ``term``)
* bare ``term`` — optional (boosts, at least one matches when no required term)

``build_keyword_query`` is the single chokepoint that turns that grammar into
(a) an FTS5 MATCH string over the trigram index — ``+term`` → ``"term"`` AND,
``-term`` → ``NOT "term"``, bare → an OR group — with every term quoted so
apostrophes and punctuation can never raise a syntax error, and (b) the plain
text fed to the embedding model (positive terms only, sigils stripped).

Keyword and vector search then run INDEPENDENTLY: the top results of each are
deduped and returned together (keyword first), so a lexical miss is covered by
semantics and vice-versa. There is no score fusion and no relevance floor — a
result is surfaced iff one of the two signals ranked it.
"""

import logging
import re
import sqlite3
from abc import ABC
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability

logger = logging.getLogger(__name__)

KNN_DEPTH = 30  # k passed to the vec0 MATCH; the dedup cap trims the final list

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


def build_keyword_query(raw: str) -> tuple[str, str]:
    """Return ``(fts_match, embed_text)`` from the keyword grammar.

    ``fts_match`` is empty when no term survives for the lexical path (e.g. every
    term was a stopword or shorter than the trigram minimum); callers skip FTS in
    that case. ``embed_text`` carries the positive terms for the vector path.

    A ``+``/bare term shorter than ``_MIN_FTS_LEN`` cannot be indexed by the
    trigram tokenizer, so it is dropped from the lexical clause and survives only
    as a vector hint — a required term that short is demoted, not enforced.
    """
    must: list[str] = []
    should: list[str] = []
    exclude: list[str] = []
    embed: list[str] = []

    for token in raw.split():
        sign = ""
        if token[0] in "+-":
            sign, token = token[0], token[1:]
        term = _TERM_RE.sub("", token.lower())
        if not term or term in _STOPWORDS:
            continue
        if sign == "-":
            exclude.append(term)
            continue
        embed.append(term)  # positive terms drive the semantic embedding
        if len(term) < _MIN_FTS_LEN:
            continue  # trigram cannot index it — vector path only
        (must if sign == "+" else should).append(term)

    clauses: list[str] = []
    if must:
        clauses.append(" AND ".join(f'"{t}"' for t in must))
    if should:
        clauses.append("(" + " OR ".join(f'"{t}"' for t in should) + ")")
    match = " AND ".join(clauses)
    if match:  # NOT needs a left operand — excludes are meaningless without one
        for term in exclude:
            if len(term) >= _MIN_FTS_LEN:
                match += f' NOT "{term}"'

    return match, " ".join(embed)


class SearchableAbility(Ability, ABC):
    _DB_PATH: ClassVar[Path]
    _LOG_PREFIX: ClassVar[str] = ""

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
        """KNN rows as ``(key, label, distance)``; ``[]`` when the embedding is
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
        """FTS rows as ``(key, label, bm25)``; ``[]`` for an empty match string or
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
        """Collapse repeat entries of one item (e.g. its name and summary both
        matched) to its single best-ranked row, ordered best-first (rank asc)."""
        best: "dict[object, tuple[object, object, float]]" = {}
        for key, label, rank in rows:
            if key not in best or rank < best[key][2]:
                best[key] = (key, label, rank)
        return sorted(best.values(), key=lambda r: r[2])

    @classmethod
    def _combine(
        cls,
        vec_rows: "list[tuple[object, object, float]]",
        fts_rows: "list[tuple[object, object, float]]",
        cap: int,
    ) -> "list[dict[str, object]]":
        """Top ``cap`` keyword + top ``cap`` vector results, deduped, keyword
        first. Each row is ``{"key", "label", "source"}``."""
        out: "dict[object, dict[str, object]]" = {}
        for key, label, _ in cls._best_per_key(fts_rows)[:cap]:
            out.setdefault(key, {"key": key, "label": label, "source": "keyword"})
        for key, label, _ in cls._best_per_key(vec_rows)[:cap]:
            out.setdefault(key, {"key": key, "label": label, "source": "vector"})
        return list(out.values())

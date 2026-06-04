"""Shared vec+FTS5 RRF search base for find_tools and find_skills."""

import logging
import sqlite3
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability

logger = logging.getLogger(__name__)

RRF_K = 15
KNN_DEPTH = 30


class SearchableAbility(Ability, ABC):
    """Ability that searches a vec+FTS5 sqlite database via RRF fusion."""

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

    @staticmethod
    def rrf_merge(vec_rows: list, fts_rows: list, k: int) -> list[dict]:
        label_by_key: dict = {}
        vec_best: dict = {}
        for key, label, distance in vec_rows:
            label_by_key[key] = label
            if key not in vec_best or distance < vec_best[key]:
                vec_best[key] = distance

        fts_best: dict = {}
        for key, label, score in fts_rows:
            label_by_key.setdefault(key, label)
            if key not in fts_best or score < fts_best[key]:
                fts_best[key] = score

        rrf_scores: dict = defaultdict(float)
        for rank, (key, _) in enumerate(sorted(vec_best.items(), key=lambda x: x[1]), start=1):
            rrf_scores[key] += 1.0 / (RRF_K + rank)
        for rank, (key, _) in enumerate(sorted(fts_best.items(), key=lambda x: x[1]), start=1):
            rrf_scores[key] += 1.0 / (RRF_K + rank)

        if not rrf_scores:
            return []

        merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return [
            {"key": key, "label": label_by_key.get(key, ""), "score": score}
            for key, score in merged[:k]
        ]

    def _hybrid_search(
        self,
        query: str,
        blob: bytes,
        k: int,
        vec_sql: str,
        fts_sql: str,
        vec_params: tuple,
        fts_params: tuple,
    ) -> list[dict]:
        if not self._DB_PATH.exists():
            logger.warning(f"{self._LOG_PREFIX} DB not found at {self._DB_PATH}")
            return []
        try:
            conn = sqlite3.connect(str(self._DB_PATH))
            try:
                self._load_vec(conn)
                vec_rows = conn.execute(vec_sql, vec_params).fetchall()
                try:
                    fts_rows = conn.execute(fts_sql, fts_params).fetchall()
                except sqlite3.OperationalError as exc:
                    logger.warning(f"{self._LOG_PREFIX} FTS5 query failed: {exc}")
                    fts_rows = []
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(f"{self._LOG_PREFIX} DB query failed: {exc}")
            return []

        return self.rrf_merge(vec_rows, fts_rows, k)

    def _fts_only_search(self, fts_sql: str, fts_params: tuple) -> list:
        if not self._DB_PATH.exists():
            return []
        try:
            conn = sqlite3.connect(str(self._DB_PATH))
            try:
                return conn.execute(fts_sql, fts_params).fetchall()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(f"{self._LOG_PREFIX} FTS fallback failed: {exc}")
            return []

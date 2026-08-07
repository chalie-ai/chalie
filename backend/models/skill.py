"""Skill — one ``skills`` row: a curated or user-authored playbook served by
the Brain Skills tab (``api/skills.py``) and consumed by the skill-discovery
cascade (``abilities/find_skills.py``).

Active-record row-model (Rule 5 / §4.1), bound to the dedicated
``skills.sqlite`` file rather than the default ``chalie.db``.
:meth:`_bound_connection` is overridden to reach that file directly through
``Database.conn(FileMapperService.get_skills_db_path())`` instead of the
per-thread getter :meth:`~models.model.Model.bind` installs onto the base at
boot (owner ruling: skills.sqlite is a separate db file the base's
chalie.db-bound getter can never resolve). ``Database``/``FileMapperService``
are the ONE sanctioned service import on this model and its sibling
:class:`~models.skill_association.SkillAssociation` — every other model
reaches the DB purely through the bound getter.

Holds no ``mp``, calls no other service. The ``skill_search_entries`` /
``skill_search_vec`` / ``skill_search_fts`` side-tables (embedding index) are
WRITTEN through ``utils.build_skills_db.SkillsDbBuilder.index_skill`` /
``utils.skills_io.remove_search_entries`` on a raw connection, not through
this model — but this model DOES own the read side of that index: the
``find_skills`` discovery cascade's bm25/vector search queries
(:meth:`search_by_title_bm25`, :meth:`search_by_vector`) and its
index-health probe (:meth:`probe_search_fts`) live here as named
classmethods, since the query builder cannot express a JOIN or an FTS5/vec0
``MATCH``.
"""

from __future__ import annotations

import sqlite3
from typing import ClassVar, Self, cast

from models.model import Model
from services.database import Database
from services.file_mapper_service import FileMapperService


class Skill(Model):
    """One ``skills`` row — curated (shipped) or user-authored playbook."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id",
        "title",
        "use_for",
        "content",
        "tags",
        "version",
        "source",
        "enabled",
        "based_on",
    )

    title: str
    use_for: str
    content: str
    tags: str | None
    version: int
    source: str
    enabled: int
    based_on: int | None

    @classmethod
    def get_table(cls) -> str:
        return "skills"

    @classmethod
    def _bound_connection(cls) -> sqlite3.Connection:
        """Reach skills.sqlite directly rather than the chalie.db-bound
        getter ``Database.bind()`` installs on the base — the sanctioned
        exception for this table's dedicated db file (owner ruling)."""
        return Database.conn(str(FileMapperService.get_skills_db_path()))

    @classmethod
    def find_by_title_ci(cls, title: str, source: str = "user") -> Self | None:
        """One skill matching ``title`` case-insensitively within ``source``
        — the create/copy uniqueness check (ported from ``api/skills.py``);
        also reused by ``skill_builder``'s create/edit/delete lookups (same
        predicate shape: ``source = 'user'`` + case-insensitive title).

        Raw SQL: the query builder's ``filter`` binds a bare column against a
        ``?`` value, it cannot wrap the column in ``lower(...)``."""
        row = cls._bound_connection().execute(
            "SELECT * FROM skills WHERE source = ? AND lower(title) = lower(?)",
            (source, title),
        ).fetchone()
        return cls.hydrate(row) if row is not None else None

    @classmethod
    def find_by_title_ci_prefer_user(cls, title: str) -> Self | None:
        """One skill matching ``title`` case-insensitively across ANY source,
        preferring the user copy when a user skill shadows a curated one of
        the same title — ``skill_builder``'s ``action=read`` lookup (ported
        from ``abilities/skill_builder.py``'s ``_handle_read``).

        Raw SQL: a ``lower(title) = lower(?)`` predicate the builder cannot
        wrap a column in, plus an expression ``ORDER BY (source = 'user')
        DESC`` the builder cannot express."""
        row = cls._bound_connection().execute(
            "SELECT * FROM skills WHERE lower(title) = lower(?) "
            "ORDER BY (source = 'user') DESC",
            (title,),
        ).fetchone()
        return cls.hydrate(row) if row is not None else None

    @classmethod
    def probe_search_fts(cls) -> None:
        """MATCH-probe ``skill_search_fts`` for corruption, discarding any
        hit — the ``find_skills`` index-health guardrail (ported from
        ``abilities/find_skills.py``'s ``_probe_index``). A ``sqlite3.Error``
        is left to propagate to the caller's guard, never swallowed here.

        Raw SQL: FTS5 ``MATCH`` syntax the query builder cannot express."""
        cls._bound_connection().execute(
            "SELECT 1 FROM skill_search_fts WHERE skill_search_fts MATCH 'probe' LIMIT 1"
        ).fetchone()

    @classmethod
    def search_by_title_bm25(cls, fts_query: str) -> list[tuple[object, object, float]]:
        """Rung 2 of the ``find_skills`` discovery cascade: bm25-ranked rows
        as ``(skill_id, title, score)`` over the skill TITLE only (ported
        from ``abilities/find_skills.py``'s ``_bm25_name``). Callers gate the
        result on title-segment alignment, not a score floor.

        Raw SQL: a JOIN across ``skill_search_fts`` / ``skill_search_entries``
        / ``skills`` with an FTS5 ``MATCH`` and ``bm25()`` ranking — none of
        which the query builder can express (no JOIN, no FTS support)."""
        return cast(
            "list[tuple[object, object, float]]",
            cls._bound_connection().execute(
                """
                    SELECT s.id, s.title, bm25(skill_search_fts) AS score
                    FROM skill_search_fts
                    JOIN skill_search_entries e ON e.id = skill_search_fts.rowid
                    JOIN skills s ON s.id = e.skill_id
                    WHERE skill_search_fts MATCH ? AND e.kind = 'title' AND s.enabled = 1
                    ORDER BY score ASC
                """,
                (fts_query,),
            ).fetchall(),
        )

    @classmethod
    def search_by_vector(cls, embedding: bytes, k: int) -> list[tuple[object, object, float]]:
        """Rung 3 of the ``find_skills`` discovery cascade: vector-KNN rows
        as ``(skill_id, title, distance)`` over the full prose (ported from
        ``abilities/find_skills.py``'s ``_vector_name``). Callers keep only
        rows at/under the cascade's distance ceiling.

        Raw SQL: a JOIN across ``skill_search_vec`` / ``skill_search_entries``
        / ``skills`` with a vec0 ``MATCH``, which the query builder can't
        express."""
        return cast(
            "list[tuple[object, object, float]]",
            cls._bound_connection().execute(
                """
                    SELECT s.id, s.title, v.distance
                    FROM skill_search_vec v
                    JOIN skill_search_entries e ON e.id = v.rowid
                    JOIN skills s ON s.id = e.skill_id
                    WHERE v.embedding MATCH ? AND k = ? AND s.enabled = 1
                    ORDER BY v.distance ASC
                """,
                (embedding, k),
            ).fetchall(),
        )

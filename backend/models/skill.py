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
``skill_search_vec`` / ``skill_search_fts`` side-tables (embedding index) stay
out of scope here — they are written through
``utils.build_skills_db.index_skill`` / ``utils.skills_io.remove_search_entries``
on a raw connection, not through this model.
"""

from __future__ import annotations

import sqlite3
from typing import ClassVar, Self

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
        — the create/copy uniqueness check (ported from ``api/skills.py``).

        Raw SQL: the query builder's ``filter`` binds a bare column against a
        ``?`` value, it cannot wrap the column in ``lower(...)``."""
        row = cls._bound_connection().execute(
            "SELECT * FROM skills WHERE source = ? AND lower(title) = lower(?)",
            (source, title),
        ).fetchone()
        return cls.hydrate(row) if row is not None else None

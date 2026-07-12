"""SkillAssociation — one ``skill_associations`` row: a personalisation rule
binding a behavioural pattern to a skill, written by
:class:`~services.skill_association_service.SkillAssociationService` and read
by the Brain Skills tab (``api/skills.py``).

Active-record row-model (Rule 5 / §4.1), bound to the same dedicated
``skills.sqlite`` file as :class:`~models.skill.Skill` — see that model's
:meth:`_bound_connection` docstring for the sanctioned
``Database``/``FileMapperService`` import (owner ruling).

Holds no ``mp``, calls no other service.
"""

from __future__ import annotations

import sqlite3
from typing import ClassVar

from models.model import Model
from services.database import Database
from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now


class SkillAssociation(Model):
    """One ``skill_associations`` row: a ``pattern_name -> skill`` rule."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id",
        "skill_id",
        "pattern_name",
        "rule",
        "created_at",
    )

    skill_id: int
    pattern_name: str
    rule: str
    created_at: str

    @classmethod
    def get_table(cls) -> str:
        return "skill_associations"

    @classmethod
    def _bound_connection(cls) -> sqlite3.Connection:
        """Reach skills.sqlite directly — see
        :meth:`models.skill.Skill._bound_connection`."""
        return Database.conn(str(FileMapperService.get_skills_db_path()))

    @classmethod
    def upsert(cls, skill_id: int, pattern_name: str, rule: str) -> None:
        """Insert a fresh rule, or replace the existing one for this
        ``(skill_id, pattern_name)`` pair.

        ``skill_associations`` carries a ``UNIQUE(skill_id, pattern_name)``
        constraint (see ``utils/build_skills_db.py``'s DDL); ``INSERT OR
        REPLACE`` against it is an upsert shape the query builder cannot
        express, so it stays raw SQL here (I6). Ported from
        ``SkillAssociationService._write_associations``."""
        cls._bound_connection().execute(
            "INSERT OR REPLACE INTO skill_associations "
            "(skill_id, pattern_name, rule, created_at) VALUES (?, ?, ?, ?)",
            (skill_id, pattern_name, rule, utc_now().isoformat()),
        )

    @classmethod
    def with_skill_titles(cls) -> list[dict[str, object]]:
        """Every association joined to its skill's title, newest first — the
        Brain Skills tab's co-loaded read (ported from ``api/skills.py``'s
        ``_load_associations``). A join the query builder cannot express (no
        JOIN support), so it stays raw SQL here (I6)."""
        rows = cls._bound_connection().execute(
            "SELECT sa.skill_id, s.title AS skill_title, sa.pattern_name, "
            "sa.rule, sa.created_at "
            "FROM skill_associations sa "
            "JOIN skills s ON s.id = sa.skill_id "
            "ORDER BY sa.created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

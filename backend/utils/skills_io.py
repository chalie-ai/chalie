"""Shared I/O helpers for skill YAML files and skills.sqlite access."""

import logging
import re
import sqlite3
from pathlib import Path

import yaml

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

DEFAULT_VERSION = 1
SLUG_MAX_LENGTH = 64


def slugify_title(title: str) -> str:
    """Convert a skill title to a safe filename slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:SLUG_MAX_LENGTH]


def skill_yaml_path(title: str) -> Path:
    """Return the YAML file path for a user skill."""
    return FileMapperService.get_user_skills_path(f"{slugify_title(title)}.yaml")


def ensure_user_skills_dir() -> None:
    """Create the user skills directory if it doesn't exist."""
    FileMapperService.get_user_skills_path().mkdir(parents=True, exist_ok=True)


def write_skill_file(path: Path, meta: dict[str, str | int]) -> None:
    """Write a skill metadata dict to a YAML frontmatter file."""
    if not path.resolve().is_relative_to(FileMapperService.get_user_skills_path().resolve()):
        raise ValueError("Path outside user skills directory")
    frontmatter = {
        "title": meta["title"],
        "use_for": meta["use_for"],
        "tags": meta.get("tags", ""),
        "version": meta.get("version", DEFAULT_VERSION),
    }
    body = meta.get("content", "")
    content = (
        "---\n"
        + yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        + "---\n\n"
        + body
        + "\n"
    )
    path.write_text(content, encoding="utf-8")


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into a connection."""
    conn.enable_load_extension(True)
    try:
        import sqlite_vec

        sqlite_vec.load(conn)
    except Exception:
        try:
            conn.load_extension("vec0")
        except Exception:
            logger.warning("[SKILLS_IO] Failed to load sqlite-vec extension")


def open_skills_db(*, row_factory: bool = False) -> sqlite3.Connection:
    """Open a connection to skills.sqlite with vec loaded."""
    conn = sqlite3.connect(str(FileMapperService.get_skills_db_path()))
    conn.execute("PRAGMA foreign_keys = ON")
    if row_factory:
        conn.row_factory = sqlite3.Row
    load_sqlite_vec(conn)
    return conn


def remove_search_entries(conn: sqlite3.Connection, skill_id: int) -> None:
    """Remove all search index entries for a skill.

    Uses the FTS5 'delete' command for external-content tables so the
    shadow index stays consistent.
    """
    rows = conn.execute(
        "SELECT id, text FROM skill_search_entries WHERE skill_id = ?",
        (skill_id,),
    ).fetchall()
    for entry_id, text in rows:
        conn.execute(
            "DELETE FROM skill_search_vec WHERE rowid = ?", (entry_id,)
        )
        conn.execute(
            "INSERT INTO skill_search_fts(skill_search_fts, rowid, text) "
            "VALUES('delete', ?, ?)",
            (entry_id, text),
        )
    conn.execute(
        "DELETE FROM skill_search_entries WHERE skill_id = ?", (skill_id,)
    )

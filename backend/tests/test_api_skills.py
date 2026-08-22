# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the skills CRUD/copy API — real skills.sqlite copy, real
YAML write-back, real ONNX embeddings, zero mocks.

Create and copy both build a new row from a bare constructor and let SQL
defaults fill in ``enabled``/``based_on`` — those two columns are never
attributes on the just-saved instance. A response builder that reads them
off that instance rather than a fresh row crashes; one that re-reads the row
first does not. These tests assert the wire response actually mirrors what
landed on disk (row, search index, YAML file, and listing), not just that a
status code came back. Copying also disables the curated skill it
supersedes, and deleting the copy has to hand that back — both directions of
that toggle are covered, including the case where there is nothing to hand
back.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import TypeAlias, cast

import pytest
from flask.testing import FlaskClient

from services.file_mapper_service import FileMapperService
from services.memory_store import MemoryStore
from utils.skills_io import slugify_title

pytestmark = pytest.mark.unit

_REAL_SKILLS_DB = FileMapperService.get_skills_db_path()

_AuthedClient: TypeAlias = tuple[FlaskClient, sqlite3.Connection, MemoryStore]


@pytest.fixture
def skills_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real skills.sqlite COPY under tmp_path, with the path authority
    ``FileMapperService.get_skills_db_path`` redirected to it — the endpoint
    and the Skill model both resolve it at call time. The checked-in
    skills.sqlite is only ever opened here, to read its bytes for the copy."""
    dest = tmp_path / "skills.sqlite"
    shutil.copy2(str(_REAL_SKILLS_DB), str(dest))
    monkeypatch.setattr(FileMapperService, "get_skills_db_path", lambda *_: dest)
    return dest


@pytest.fixture
def user_skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the user-skill YAML directory to tmp_path so create/copy/
    delete write-back never touches the real data/skills/user/ tree."""
    user_dir = tmp_path / "user_skills"
    monkeypatch.setattr(FileMapperService, "get_user_skills_path", lambda *parts: user_dir.joinpath(*parts))
    return user_dir


def _unwrap_success(body: dict[str, object]) -> object:
    """Assert the success envelope shape and return the bare result payload."""
    assert body.get("success") is True
    assert "error" not in body
    return body["result"]


def _unwrap_error(body: dict[str, object]) -> str:
    """Assert the error envelope shape and return the error message."""
    assert body.get("success") is False
    assert body.get("result") == []
    return cast("str", body["error"])


def _pick_curated(dest: Path) -> tuple[int, str]:
    """One real enabled curated skill (id, title) from the seeded copy — the
    fixture's curated set can change, so no id or title is ever hardcoded."""
    conn = sqlite3.connect(str(dest))
    try:
        row: tuple[int, str] | None = conn.execute(
            "SELECT id, title FROM skills WHERE source = 'curated' AND enabled = 1 "
            "ORDER BY id LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "fixture skills.sqlite has no enabled curated skill to copy"
    return row


def _row(dest: Path, skill_id: int) -> sqlite3.Row:
    """The live ``skills`` row for ``skill_id``, read by column name over a
    fresh connection — proves what actually committed, independent of
    whatever the app's own connection currently holds in memory."""
    conn = sqlite3.connect(str(dest))
    conn.row_factory = sqlite3.Row
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT * FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"skill id={skill_id} not found"
    return row


def _row_exists(dest: Path, skill_id: int) -> bool:
    conn = sqlite3.connect(str(dest))
    try:
        row: tuple[object] | None = conn.execute(
            "SELECT 1 FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _search_entry_count(dest: Path, skill_id: int) -> int:
    conn = sqlite3.connect(str(dest))
    try:
        row: tuple[int] | None = conn.execute(
            "SELECT COUNT(*) FROM skill_search_entries WHERE skill_id = ?", (skill_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return row[0]


def _curated_enabled_snapshot(dest: Path) -> list[tuple[int, int]]:
    conn = sqlite3.connect(str(dest))
    try:
        rows: list[tuple[int, int]] = conn.execute(
            "SELECT id, enabled FROM skills WHERE source = 'curated' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return rows


# ─── POST /api/skills/-1 ──────────────────────────────────────────────────

class TestCreateSkill:
    """Creating a user skill through the API."""

    def test_create_persists_enabled_row_indexed_and_listed(
        self, authed_client: _AuthedClient, skills_db: Path, user_skills_dir: Path
    ) -> None:
        client, _, _ = authed_client

        resp = client.post(
            "/api/skills/-1",
            json={
                "title": "Zephyr Test Skill Alpha",
                "use_for": "Testing the skill create endpoint end to end.",
                "content": "Step one. Step two. Step three.",
                "tags": "a, b",
            },
        )

        assert resp.status_code == 201
        result = _unwrap_success(resp.get_json())
        assert isinstance(result, dict)
        assert result["title"] == "Zephyr Test Skill Alpha"
        assert result["use_for"] == "Testing the skill create endpoint end to end."
        assert result["content"] == "Step one. Step two. Step three."
        assert result["tags"] == "a, b"
        assert result["version"] == 1
        assert result["source"] == "user"
        assert result["enabled"] is True
        assert result["based_on"] is None

        # The bug this pins: enabled/based_on are SQL defaults, never set on
        # the instance the endpoint just saved. A response built from that
        # instance would already have failed the asserts above with a 500;
        # these prove the SAME facts are also true in the row that landed.
        skill_id = result["id"]
        row = _row(skills_db, skill_id)
        assert row["enabled"] == 1
        assert row["based_on"] is None
        assert row["source"] == "user"
        assert _search_entry_count(skills_db, skill_id) > 0

        yaml_path = user_skills_dir / f"{slugify_title('Zephyr Test Skill Alpha')}.yaml"
        assert yaml_path.exists()

        listing = client.get("/api/skills/all?page=1&limit=500")
        items = _unwrap_success(listing.get_json())
        assert isinstance(items, list)
        match = next((item for item in items if item["id"] == skill_id), None)
        assert match is not None
        assert match["enabled"] is True

    def test_duplicate_title_is_rejected(
        self, authed_client: _AuthedClient, skills_db: Path, user_skills_dir: Path
    ) -> None:
        client, _, _ = authed_client
        payload = {
            "title": "Zephyr Duplicate Skill",
            "use_for": "First one wins.",
            "content": "Body.",
        }
        first = client.post("/api/skills/-1", json=payload)
        assert first.status_code == 201

        # Case-insensitive: the dedupe check must catch this as the same title.
        second = client.post("/api/skills/-1", json={**payload, "title": "zephyr duplicate skill"})

        assert second.status_code == 409
        assert _unwrap_error(second.get_json()) == "A user skill named 'zephyr duplicate skill' already exists"


# ─── POST /api/skills/copy/<id> ───────────────────────────────────────────

class TestCopySkill:
    """Copying a curated skill into an editable user skill."""

    def test_copy_disables_original_and_creates_enabled_copy(
        self, authed_client: _AuthedClient, skills_db: Path, user_skills_dir: Path
    ) -> None:
        client, _, _ = authed_client
        curated_id, curated_title = _pick_curated(skills_db)

        resp = client.post(f"/api/skills/copy/{curated_id}")

        assert resp.status_code == 201
        result = _unwrap_success(resp.get_json())
        assert isinstance(result, dict)
        expected_title = f"{curated_title} (Custom)"
        assert result["title"] == expected_title
        assert result["source"] == "user"
        assert result["enabled"] is True
        assert result["based_on"] == curated_id
        assert result["version"] == 1

        # Same bug as create, plus the disable side-effect on the original —
        # both rows have to be right, not just the copy's response.
        copy_id = result["id"]
        assert _row(skills_db, curated_id)["enabled"] == 0
        copy_row = _row(skills_db, copy_id)
        assert copy_row["enabled"] == 1
        assert copy_row["based_on"] == curated_id

        listing = client.get("/api/skills/all?limit=500")
        items = _unwrap_success(listing.get_json())
        assert isinstance(items, list)
        by_id = {item["id"]: item for item in items}
        assert by_id[curated_id]["enabled"] is False
        assert by_id[copy_id]["enabled"] is True

        second = client.post(f"/api/skills/copy/{curated_id}")

        assert second.status_code == 409
        assert _unwrap_error(second.get_json()) == f"A user copy named '{expected_title}' already exists"

    def test_cannot_copy_user_skill(
        self, authed_client: _AuthedClient, skills_db: Path, user_skills_dir: Path
    ) -> None:
        client, _, _ = authed_client
        create_resp = client.post(
            "/api/skills/-1",
            json={"title": "Zephyr Guard Skill", "use_for": "x", "content": "y"},
        )
        assert create_resp.status_code == 201
        created = _unwrap_success(create_resp.get_json())
        assert isinstance(created, dict)
        user_skill_id = created["id"]

        resp = client.post(f"/api/skills/copy/{user_skill_id}")

        assert resp.status_code == 422
        assert _unwrap_error(resp.get_json()) == "Only curated skills can be copied"


# ─── DELETE /api/skills/<id> ──────────────────────────────────────────────

class TestDeleteSkill:
    """Deleting a user skill, and what it does or doesn't hand back."""

    def test_delete_copy_reenables_curated_original(
        self, authed_client: _AuthedClient, skills_db: Path, user_skills_dir: Path
    ) -> None:
        client, _, _ = authed_client
        curated_id, _curated_title = _pick_curated(skills_db)

        copy_resp = client.post(f"/api/skills/copy/{curated_id}")
        assert copy_resp.status_code == 201
        copy_result = _unwrap_success(copy_resp.get_json())
        assert isinstance(copy_result, dict)
        copy_id = copy_result["id"]
        copy_title = copy_result["title"]
        assert _row(skills_db, curated_id)["enabled"] == 0

        del_resp = client.delete(f"/api/skills/{copy_id}")

        assert del_resp.status_code == 204
        assert del_resp.data == b""
        assert not _row_exists(skills_db, copy_id)
        assert _search_entry_count(skills_db, copy_id) == 0

        yaml_path = user_skills_dir / f"{slugify_title(copy_title)}.yaml"
        assert not yaml_path.exists()

        assert _row(skills_db, curated_id)["enabled"] == 1

        listing = client.get("/api/skills/all?limit=500")
        items = _unwrap_success(listing.get_json())
        assert isinstance(items, list)
        curated_item = next((item for item in items if item["id"] == curated_id), None)
        assert curated_item is not None
        assert curated_item["enabled"] is True

    def test_delete_plain_user_skill_leaves_curated_rows_untouched(
        self, authed_client: _AuthedClient, skills_db: Path, user_skills_dir: Path
    ) -> None:
        # based_on is NULL for a hand-made skill — the re-enable branch in
        # delete() is conditional on it, so no curated row should move. This
        # is the other side of test_delete_copy_reenables_curated_original:
        # the branch must be conditional, not blanket.
        client, _, _ = authed_client
        before = _curated_enabled_snapshot(skills_db)

        create_resp = client.post(
            "/api/skills/-1",
            json={
                "title": "Zephyr Standalone Skill",
                "use_for": "Prove a based_on=NULL delete leaves curated rows alone.",
                "content": "Body.",
            },
        )
        assert create_resp.status_code == 201
        created = _unwrap_success(create_resp.get_json())
        assert isinstance(created, dict)
        skill_id = created["id"]

        del_resp = client.delete(f"/api/skills/{skill_id}")

        assert del_resp.status_code == 204
        assert _curated_enabled_snapshot(skills_db) == before

    def test_cannot_delete_curated_skill(
        self, authed_client: _AuthedClient, skills_db: Path, user_skills_dir: Path
    ) -> None:
        client, _, _ = authed_client
        curated_id, _curated_title = _pick_curated(skills_db)

        resp = client.delete(f"/api/skills/{curated_id}")

        assert resp.status_code == 403
        assert _unwrap_error(resp.get_json()) == "Only user-created skills can be deleted"
        assert _row_exists(skills_db, curated_id)

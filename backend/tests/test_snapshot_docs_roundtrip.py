# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: whole-instance snapshot round-trip of the documents tree.

``services/snapshot_service.py::SnapshotService`` wires the documents tree both
directions through ``FileMapperService.get_documents_path()`` — export stages
every file under it into the zip (``_stage_trees``, ~L199), restore moves the
staged tree back into the live destination (``_swap_in`` → ``_swap_artifact``,
~L405-464). This drives the REAL ``SnapshotService`` end to end: write real
files → export → delete/corrupt the live files → ``stage_import`` →
``apply_pending`` → assert the tree is back byte-identical.

Every ``FileMapperService`` path the round trip can touch is redirected under
``tmp_path`` before a single line of ``SnapshotService`` runs — the
conftest-blessed ``_DOCUMENTS_DIR`` redirection (``test_document_api.py``,
``test_seed_parallel_image_upload.py``) extended to every sibling constant the
snapshot walk also reaches:

* ``_DATA_DIR`` — so ``.pending-restore`` / ``.snapshot-staging`` /
  ``mcp_tools.sqlite`` land under ``tmp_path``, never the real repo ``data/``.
* ``_DOCUMENTS_DIR`` / ``_USER_SKILLS_DIR`` — the two tree KINDs the export
  walks; neither is derived from ``_DATA_DIR`` at call time (both are
  separate class attributes resolved once at class-load time), so each needs
  its own redirect.
* ``get_version_path`` / ``get_skills_db_path`` — overridden to a path that
  does not exist. Neither is derived from ``_DATA_DIR`` either: VERSION lives
  at the repo ROOT and ``skills.sqlite`` is a real checked-in asset under
  ``abilities/assets/``. Left unredirected, export would stage them (harmless,
  read-only) but restore's ``_swap_in`` would then move the REAL repo VERSION
  file and the REAL checked-in ``skills.sqlite`` into an aside directory and
  back — mutating tracked repo files mid-test. Pointing both at a
  non-existent path makes export's ``if not src.exists(): continue/return``
  skip them cleanly instead.

``get_db_path`` is already redirected by the ``db`` fixture to the per-test
SQLite file, independently of ``_DATA_DIR`` — the ``chalie_db`` artifact
``stage_import``'s schema-downgrade guard hard-requires
(``SnapshotError("Snapshot has no chalie.db to restore")`` otherwise) is
satisfied by that same converged test database, so no extra wiring is needed
for it here.
"""

import sqlite3
from pathlib import Path

import pytest

from services.file_mapper_service import FileMapperService
from services.snapshot_service import SnapshotService

pytestmark = pytest.mark.unit


def _redirect_snapshot_paths(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    """Point every directory/file constant the snapshot round trip touches at
    *data_dir* (or a path under it that does not exist) — see module
    docstring for why each one is needed."""
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", data_dir)
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", data_dir / "docs")
    monkeypatch.setattr(FileMapperService, "_USER_SKILLS_DIR", data_dir / "skills" / "user")
    # Neither is computed from _DATA_DIR — override directly (same style the
    # `db` fixture uses for get_db_path in conftest.py) to a path that does
    # not exist, so both KINDs are cleanly skipped rather than staged from
    # (and later swapped over) the real repo files.
    monkeypatch.setattr(FileMapperService, "get_version_path", lambda *_: data_dir / "VERSION")
    monkeypatch.setattr(FileMapperService, "get_skills_db_path", lambda *_: data_dir / "unused-skills.sqlite")


def test_documents_tree_round_trips_byte_identical_through_export_and_restore(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real files (one nested in a subdirectory) survive a full
    export → delete/corrupt → stage_import → apply_pending cycle byte-for-byte:
    a deleted top-level file is recreated, a corrupted nested file is reverted."""
    assert db is not None
    data_dir = tmp_path / "data"
    _redirect_snapshot_paths(monkeypatch, data_dir)

    docs_dir = FileMapperService.get_documents_path()
    docs_dir.mkdir(parents=True, exist_ok=True)
    top_level = docs_dir / "notes.txt"
    top_level.write_text("Top-level document content.\n", encoding="utf-8")
    nested = docs_dir / "reports" / "quarterly.txt"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("Nested document content, safely tucked away.\n", encoding="utf-8")

    svc = SnapshotService()
    zip_path = svc.export(password=None)
    try:
        # Simulate loss: the top-level file is deleted outright, the nested
        # file is corrupted in place.
        top_level.unlink()
        nested.write_text("CORRUPTED", encoding="utf-8")

        svc.stage_import(zip_path, password=None)
        SnapshotService.apply_pending()

        assert top_level.read_text(encoding="utf-8") == "Top-level document content.\n"
        assert nested.read_text(encoding="utf-8") == "Nested document content, safely tucked away.\n"
    finally:
        zip_path.unlink(missing_ok=True)

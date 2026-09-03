"""Feature test: every attachment lands, and every landed file gets indexed.

Attachments cross two real production methods, in this order:

* ``MessageProcessor._land_attachments`` — synchronous inside ``begin``, after
  its write transaction commits and before the ``working`` frame goes out. A
  serial loop: copy each staged file flat into ``<documents>/uploads/`` via
  ``FileParserService.place`` (staging prefix ``chalie_<8hex>_`` stripped),
  write the ``transcript_files`` link row, delete the tmp original.
* ``MessageProcessor._seed_turn_zero`` — on the drive thread, fans the placed
  paths out across a thread pool that extracts and indexes each one, joining
  at the pool barrier before it returns.

Both are driven here for real against the real test DB and real files. ZERO
mocks — the ``_DOCUMENTS_DIR`` and ``get_file_index_db_path`` patches are the
same conftest-blessed path redirections every docs-dir test uses (both are
needed BEFORE either method runs: FileIndexService's db_path resolves at call
time).
"""

import os
import uuid
from pathlib import Path

import pytest

from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor
from services.file_index_service import FileIndexService
from services.file_mapper_service import FileMapperService
from services.tmp_storage import new_tmp_path

pytestmark = pytest.mark.unit


def _write_attachment(label: str) -> str:
    """Stage exactly the way ``ThreadsEndpoint._stage_uploads`` does: paths
    must be under ``TMP_PATH_PREFIX`` (so the sandbox guard passes) with an
    8-hex collision prefix on the basename. The body carries the label as a
    searchable token, so the index assertion cannot pass on filename alone."""
    path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_seed_landing_{label}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Attachment body mentioning {label.upper()}shire.")
    return path


def _build_parent(attachments: "list[str]") -> MessageProcessor:
    """Build the REAL MessageProcessor via its actual public constructor — no
    private-field poking. The constructor is "constructed inert": it wires every
    coordinating service (``dispatch_service`` included) eagerly but performs no
    DB writes and never assigns ``self.uid`` outside ``begin()``'s transaction.
    Both methods under test tolerate that: ``_land_attachments`` skips only the
    link row when ``uid`` is None (the ``skip_input_row`` path) and still places
    the file, and ``_seed_turn_zero`` dispatches through the real, fully-wired
    ``dispatch_service`` regardless."""
    return MessageProcessor(
        UserConfig(), -1, "Here are some files.", {"attachments": attachments},
    )


def _redirect_docs_and_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hermetic redirection every docs-dir test uses (blessed pattern): both
    patches land BEFORE any ingest runs, so the real data/ dirs and the real
    data/file_index.sqlite are never touched."""
    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(
        FileMapperService, "get_file_index_db_path", lambda *_: tmp_path / "file_index.sqlite"
    )
    return docs_dir


def test_land_attachments_stores_every_attachment_and_clears_staging(
    db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every attachment lands flat in <documents>/uploads/ with its staging
    prefix stripped — none lost, none duplicated, staging left clean — and it
    all happens synchronously, before ``begin()`` opens the turn."""
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    labels = ["alpha", "bravo", "charlie"]
    attachments = [_write_attachment(label) for label in labels]

    parent = _build_parent(attachments)
    parent._land_attachments()  # the REAL production method

    uploads_dir = docs_dir / "uploads"
    landed = sorted(p.name for p in uploads_dir.iterdir())
    assert landed == sorted(f"seed_landing_{label}.txt" for label in labels), landed
    # Nothing else landed directly under the documents root.
    assert [p.name for p in docs_dir.iterdir()] == ["uploads"]
    # The staged tmp files are gone from staging — copied, then deleted.
    assert not any(os.path.exists(p) for p in attachments), attachments


def test_seed_turn_zero_joins_every_index_before_returning(
    db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pool barrier holds: once ``_seed_turn_zero`` returns, EVERY placed
    attachment is searchable by its own content — not just the first, and not
    a subset that happened to finish early."""
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    labels = ["alpha", "bravo", "charlie"]
    parent = _build_parent([_write_attachment(label) for label in labels])
    parent._land_attachments()
    parent._seed_turn_zero()  # the REAL production method — pool __exit__ joins

    index = FileIndexService()
    for label in labels:
        hits = index.search(f"{label.upper()}shire")
        assert str(docs_dir / "uploads" / f"seed_landing_{label}.txt") in hits, (label, hits)


def test_land_attachments_skips_unreadable_attachment_without_aborting_others(
    db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the per-file guard doesn't take down the whole loop: the
    attachment outside the tmp sandbox is refused, the good one still lands
    in <documents>/uploads/."""
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    good = _write_attachment("delta")
    bad = "/nonexistent/not_under_tmp_prefix.png"  # fails the realpath guard

    parent = _build_parent([good, bad])
    parent._land_attachments()

    uploads_dir = docs_dir / "uploads"
    assert [p.name for p in uploads_dir.iterdir()] == ["seed_landing_delta.txt"]

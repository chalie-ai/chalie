"""Feature test: turn-0 attachments fan out in parallel at a barrier.

Drives the REAL ``MessageProcessor._seed_turn_zero`` — the SAME production
method that fires once before iteration 0 — with N real image attachments on a
real ``UserConfig`` channel against the real test DB and real files. ZERO
mocks — the ``_DOCUMENTS_DIR`` and ``get_file_index_db_path`` patches are the
same conftest-blessed path redirections every docs-dir test uses (both are
needed BEFORE ingest runs: FileIndexService's db_path resolves at call time
inside ``FileParserService.ingest``, so both patches must be in place first).

Attachments are ingested via ``FileParserService.ingest(..., subdir="uploads")``,
so they land flat under ``<documents>/uploads/``, not the documents root — the
staging prefix (``chalie_<8hex>_``) is stripped from the basename before
landing.
"""

import io
import os
import uuid
from pathlib import Path

import pytest

from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor
from services.file_mapper_service import FileMapperService
from services.tmp_storage import new_tmp_path

pytestmark = pytest.mark.unit


def _png_with_text(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    font: "ImageFont.FreeTypeFont | ImageFont.ImageFont | None" = None
    for candidate in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, 72)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((20, 40), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_attachment(label: str) -> str:
    """Stage exactly the way ``ThreadsEndpoint._stage_uploads`` does: paths
    must be under ``TMP_PATH_PREFIX`` (so the sandbox guard passes) with an
    8-hex collision prefix on the basename."""
    path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_seed_parallel_{label}.png")
    with open(path, "wb") as fh:
        fh.write(_png_with_text(label.upper()))
    return path


def _build_parent(attachments: "list[str]") -> MessageProcessor:
    """Build the REAL MessageProcessor via its actual public constructor — no
    private-field poking. The constructor is "constructed inert": it wires every
    coordinating service (``dispatch_service`` included) eagerly but performs no
    DB writes and never assigns ``self.uid`` outside ``begin()``'s transaction.
    ``_seed_turn_zero`` -> ``_seed_upload_attachment`` tolerates that: it moves
    the file and dispatches ``read`` through the real, fully-wired
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


def test_seed_moves_all_attachments_in_parallel(
    db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the barrier joins all ingests before _seed_turn_zero returns: every
    attachment lands flat in <documents>/uploads/ with its staging prefix
    stripped — none lost, none duplicated, staging left clean."""
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    labels = ["alpha", "bravo", "charlie"]
    attachments = [_write_attachment(label) for label in labels]

    parent = _build_parent(attachments)
    # The REAL production method — the barrier (pool __exit__) joins all ingests.
    parent._seed_turn_zero()

    uploads_dir = docs_dir / "uploads"
    landed = sorted(p.name for p in uploads_dir.iterdir())
    assert landed == sorted(f"seed_parallel_{label}.png" for label in labels), landed
    # Nothing else landed directly under the documents root.
    assert [p.name for p in docs_dir.iterdir()] == ["uploads"]
    # The staged tmp files are gone from staging — copied-then-deleted on
    # ingest success, moved on the extraction-failure fallback; either way
    # nothing is left behind at the original tmp path.
    assert not any(os.path.exists(p) for p in attachments), attachments


def test_seed_skips_unreadable_attachment_without_aborting_others(
    db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the per-task guard doesn't take down the whole barrier: the
    attachment outside the tmp sandbox is refused, the good one still lands
    in <documents>/uploads/."""
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    good = _write_attachment("delta")
    bad = "/nonexistent/not_under_tmp_prefix.png"  # fails the realpath guard

    parent = _build_parent([good, bad])
    parent._seed_turn_zero()

    uploads_dir = docs_dir / "uploads"
    assert [p.name for p in uploads_dir.iterdir()] == ["seed_parallel_delta.png"]

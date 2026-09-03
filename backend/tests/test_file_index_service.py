"""Feature tests for FileIndexService and the watchdog wiring in
workers/file_index_worker.py — real SQLite FTS5, real filesystem, real
watchdog Observer. Zero mocks.

should_index() rejects every path under /tmp, /var, /private, /System,
/Volumes (see file_index_service._SKIP_SUBTREES) and any path with a
dot-leading component. tempfile.mkdtemp()/pytest's tmp_path resolve under
/var/folders (-> /private/var/folders) on macOS, so a scan root built there
would make the live watchdog path silently pass should_index=False for
every file and prove nothing. The `scan_root` fixture below builds a real,
non-dot directory under the user's home instead — verified against
should_index directly before this suite was written — so the live
create/edit/delete path is genuinely exercised, not a no-op that happens to
return an empty search list.

index_file()/reconcile() are called directly in a couple of tests where the
ticket doesn't require should_index gating (PDF extraction, restart
catch-up) — index_file() itself never consults should_index, only the live
handler and reconcile()'s directory walk do.

Live-observer assertions poll the real service.search() output (see
_poll_until below) rather than counting/waiting on raw watchdog events: a
single write reliably produces more than one raw event on macOS, at
unpredictable offsets, so synchronising on event count is itself a race.
Polling the actual observable behaviour sidesteps that entirely.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from services.file_index_service import (
    FileIndexService,
    _is_chalie_main_db,
    _is_in_skip_subtree,
)
from services.file_mapper_service import FileMapperService
from workers.file_index_worker import _FileIndexHandler

pytestmark = pytest.mark.unit

_POLL_TIMEOUT = 10.0
_POLL_INTERVAL = 0.05


def _poll_until(predicate: Callable[[], bool], timeout: float = _POLL_TIMEOUT) -> bool:
    """Retry *predicate* until it's true or *timeout* elapses — never a blind
    fixed sleep. Needed because a single filesystem write reliably produces
    MORE than one raw watchdog event on macOS (a create is followed by a
    directory-modified event and often a duplicate/coalesced file-modified
    event too, arriving asynchronously at an unpredictable offset — confirmed
    empirically while writing this suite). Waiting on event *count* races
    against that duplication; polling the actual observable output
    (``service.search()``) doesn't — it only cares that the real state
    converges, however many raw events it took."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_INTERVAL)


# ── Inline minimal-PDF builder (no new dependency; pdfplumber is already a
# hard dependency of services/text_extractor.py and is what index_file's
# extraction path uses) ────────────────────────────────────────────────────

def _minimal_pdf_bytes(text: str) -> bytes:
    """Hand-build the smallest single-page PDF pdfplumber can extract text from:
    a Catalog, one Page with a Helvetica font resource, and a content stream
    that draws *text* with a single Tj operator. No compression, no images —
    just enough structure (objects, xref table, trailer) to be a valid PDF."""
    content = f"BT /F1 24 Tf 10 50 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 200 100] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
        + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii")
        out += body
        out += b"\nendobj\n"

    xref_offset = len(out)
    count = len(objects) + 1
    out += f"xref\n0 {count}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += b"trailer\n"
    out += f"<< /Size {count} /Root 1 0 R >>\n".encode("ascii")
    out += b"startxref\n"
    out += f"{xref_offset}\n".encode("ascii")
    out += b"%%EOF"
    return bytes(out)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def scan_root() -> Iterator[Path]:
    """A real directory FileIndexService.should_index actually accepts.

    Built under the home directory (no skipped subtree, no dot-leading
    component, no skipped dir name) so the live watchdog path is genuinely
    gated the same way it is in production, not vacuously true.
    """
    root = Path.home() / f"chalie_test_file_index_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def index_db_path(tmp_path: Path) -> str:
    """Throwaway sqlite file for the FTS5 index. Only should_index-gated
    *scan targets* need to dodge the skip subtrees — the db file itself is
    never walked or should_index-checked, so tmp_path is fine here."""
    return str(tmp_path / "file_index_test.sqlite")


def _start_observer(handler: _FileIndexHandler, root: Path) -> BaseObserver:
    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()
    return observer


def _stop_observer(observer: BaseObserver) -> None:
    observer.stop()
    observer.join(timeout=5)


# ── Tests ─────────────────────────────────────────────────────────────────

def test_live_observer_tracks_create_edit_delete(scan_root: Path, index_db_path: str) -> None:
    """A file dropped into the watched root becomes searchable within seconds
    of creation; editing it makes the new content searchable and the old
    content stop matching; deleting it removes it from search entirely.

    Drives a REAL watchdog Observer + the REAL _FileIndexHandler scheduled
    exactly as workers/file_index_worker.py wires them (minus the 30s startup
    delay and hourly reconcile loop, neither of which affects event dispatch).
    """
    service = FileIndexService(scan_root=str(scan_root), db_path=index_db_path)
    target = scan_root / "notes.md"
    observer = _start_observer(_FileIndexHandler(service), scan_root)

    try:
        # Create -> searchable by a word in its content.
        target.write_text("The giraffe wandered through the savanna at dusk.")
        assert _poll_until(lambda: str(target) in service.search("giraffe")), (
            "created file never became searchable"
        )

        # Edit -> new content searchable, old-only content stops matching.
        target.write_text("The narwhal surfaced beside the icy floe at dawn.")
        assert _poll_until(lambda: str(target) in service.search("narwhal")), (
            "edited content never became searchable"
        )
        assert str(target) not in service.search("giraffe")

        # Delete -> vanishes from search.
        target.unlink()
        assert _poll_until(lambda: str(target) not in service.search("narwhal")), (
            "deleted file never left the index"
        )
    finally:
        _stop_observer(observer)


def test_index_file_extracts_pdf_text_directly(tmp_path: Path) -> None:
    """A PDF's body text is searchable after index_file() — proves the real
    TextReader -> pdfplumber extraction path, not just the plain-text branch.

    index_file() never consults should_index (only the live handler and
    reconcile()'s walk do), so a plain tmp_path is fine as the target here.
    """
    service = FileIndexService(scan_root=str(tmp_path), db_path=str(tmp_path / "index.sqlite"))
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(_minimal_pdf_bytes("ChalieProbeXyzzy"))

    service.index_file(str(pdf_path))

    assert str(pdf_path) in service.search("ChalieProbeXyzzy")


def test_reconcile_catches_up_after_restart_with_no_observer(
    scan_root: Path, index_db_path: str
) -> None:
    """Mutate the filesystem with NO observer running (create, edit, delete),
    then prove a FRESH FileIndexService pointed at the same db converges via
    reconcile() alone — the startup catch-up path a real process restart
    relies on when it comes back up after being down.
    """
    changed = scan_root / "changed.md"
    deleted = scan_root / "deleted.md"
    changed.write_text("alpha " * 5)
    deleted.write_text("beta " * 5)

    setup_service = FileIndexService(scan_root=str(scan_root), db_path=index_db_path)
    setup_service.index_file(str(changed))
    setup_service.index_file(str(deleted))
    assert str(changed) in setup_service.search("alpha")
    assert str(deleted) in setup_service.search("beta")

    # Mutate the filesystem directly — no observer, no service call at all.
    changed.write_text("gamma " * 50)  # size deliberately far from the old version
    deleted.unlink()
    created = scan_root / "created.md"
    created.write_text("delta " * 5)

    fresh_service = FileIndexService(scan_root=str(scan_root), db_path=index_db_path)
    fresh_service.reconcile()

    assert str(changed) in fresh_service.search("gamma")
    assert str(changed) not in fresh_service.search("alpha")
    assert str(deleted) not in fresh_service.search("beta")
    assert str(created) in fresh_service.search("delta")


def test_concurrent_observer_and_reconcile_do_not_corrupt_the_index(
    scan_root: Path, index_db_path: str
) -> None:
    """The flagged risk: a live watchdog Observer indexing on its own thread
    while the main thread hammers reconcile() against the SAME sqlite file at
    the same time. Proves the thread-local-connection + BEGIN IMMEDIATE +
    busy_timeout design (services/database.py) holds under real contention —
    no exception escapes either thread, and the index converges to the true
    final on-disk state.
    """
    service = FileIndexService(scan_root=str(scan_root), db_path=index_db_path)
    observer = _start_observer(_FileIndexHandler(service), scan_root)

    watched = scan_root / "storm.md"
    stable = scan_root / "stable.md"
    stable.write_text("kept " * 5)

    writer_errors: list[BaseException] = []

    def _hammer_filesystem() -> None:
        try:
            for i in range(30):
                watched.write_text(f"iteration-{i} " * (i + 1))
            watched.unlink()
        except BaseException as exc:  # captured, never swallowed — asserted below
            writer_errors.append(exc)

    writer = threading.Thread(target=_hammer_filesystem)
    writer.start()
    # Hammer reconcile() from the main thread while the writer thread is
    # actively mutating the same scan root — real overlap, not simulated.
    for _ in range(50):
        service.reconcile()
    writer.join(timeout=10)

    assert not writer.is_alive(), "background filesystem writer never finished"
    assert writer_errors == []
    assert observer.is_alive()

    _stop_observer(observer)
    service.reconcile()  # settle: the documented truth path once both quiesce

    assert str(stable) in service.search("kept")
    assert service.search("iteration") == []
    assert not os.path.exists(str(watched))


def test_reconcile_keeps_parser_owned_rows_until_the_file_is_gone(
    scan_root: Path, index_db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row FileParserService creates for a non-indexable extension under the
    documents root (e.g. a screenshot .png, whose content is a vision/OCR
    description upserted at ingest — not walk-extracted text) survives
    reconcile()'s directory walk while the file still exists on disk, and is
    removed once the file is actually gone. Proves the `_is_parser_owned`
    carve-out end to end, not just should_index gating (which the walk
    already covers for ordinary indexable files).
    """
    documents_dir = scan_root / "documents"
    documents_dir.mkdir()
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", documents_dir)

    png = documents_dir / "screenshot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n0000")

    service = FileIndexService(scan_root=str(scan_root), db_path=index_db_path)
    service.upsert_content(str(png), "invoice screenshot description")
    assert str(png) in service.search("invoice")

    # File still exists: the walk skips it (unsupported extension) but the
    # deletion pass must not treat that as "gone" for a parser-owned row.
    service.reconcile()
    assert str(png) in service.search("invoice")

    png.unlink()
    service.reconcile()
    assert service.search("invoice") == []


def test_the_walk_never_descends_into_etc(scan_root: Path) -> None:
    """/etc is pruned, so the walk never reaches the indexable files in it.

    TextReader has always refused /etc, but _SKIP_SUBTREES did not list it, so
    every reconcile walked in, offered each .txt/.py it found to the reader and
    took a raise back — a verdict that can never change, re-earned hourly.

    Asserted through _is_in_skip_subtree, which is what reconcile()'s
    directory-pruning filter calls, rather than through should_index:
    should_index bottoms out in os.path.getsize, so on any host lacking those
    exact /etc files it returns False for "does not exist" and proves nothing.
    """
    assert _is_in_skip_subtree("/etc")
    assert _is_in_skip_subtree("/etc/python3.13/sitecustomize.py")
    assert _is_in_skip_subtree("/etc/X11/rgb.txt")
    assert not _is_in_skip_subtree(str(scan_root / "notes.md"))


def test_should_index_refuses_a_symlink_resolving_onto_a_refused_prefix(
    scan_root: Path, index_db_path: str
) -> None:
    """A link sitting outside every skipped subtree, pointing into one, is refused.

    /usr/lib/python3.13/sitecustomize.py is an ordinary path that no prefix
    rule rejects, yet it points into /etc — and TextReader tests the resolved
    path, so it refused the read while the walk kept queueing it. Pruning /etc
    alone does not close this leg; the walk has to resolve too.

    The target is /dev/null: it exists (so os.path.getsize succeeds and the
    assertion is not vacuously true for "missing file"), it sits on a refused
    prefix, and unlike /etc it resolves to itself on macOS as well as Linux.
    """
    service = FileIndexService(scan_root=str(scan_root), db_path=index_db_path)

    link = scan_root / "probe.txt"
    link.symlink_to("/dev/null")
    assert not service.should_index(str(link))

    # Targeted, not a blanket refusal of symlinks: one pointing at ordinary
    # content is still indexed.
    real = scan_root / "real_notes.md"
    real.write_text("quarterly figures")
    alias = scan_root / "alias_notes.md"
    alias.symlink_to(real)
    assert service.should_index(str(alias))


def test_skip_rule_covers_every_main_db_release_and_sidecars(
    scan_root: Path, index_db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main database is versioned per release (data/chalie-<version>.sqlite),
    so a skip list frozen at import time can only name the running build's own
    file: a previous release's database would slip through and get indexed.
    The rule is evaluated per path instead — the legacy chalie.db, ANY
    chalie-<version>.sqlite whenever it appears, and the -wal/-shm sidecars of
    all of them are skipped, while an ordinary file in the same data directory
    is still indexed.

    Asserted through _is_chalie_main_db — the exact predicate _is_chalie_path
    (and thus should_index and reconcile's directory pruning) consults —
    rather than through should_index: a .sqlite/.db path and an extensionless
    sidecar are already refused by the extension whitelist on their own, so a
    should_index assertion on them would be vacuous (same reasoning as
    test_the_walk_never_descends_into_etc).
    """
    data_dir = scan_root / "data"
    data_dir.mkdir()
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", data_dir)

    # Every main-database file in the data dir is skipped — including a
    # release that is not the running build (the rule must not depend on
    # which version is running, which is exactly the frozen-list failure).
    skipped = (
        data_dir / "chalie-9.9.9.sqlite",  # a release's versioned database
        data_dir / "chalie-1.3.0-beta.sqlite",  # dotted/hyphenated version
        data_dir / "chalie.db",  # the pre-versioning name
        data_dir / "chalie-9.9.9.sqlite-wal",
        data_dir / "chalie-9.9.9.sqlite-shm",
        data_dir / "chalie.db-wal",
        data_dir / "chalie.db-shm",
    )
    for db_file in skipped:
        assert _is_chalie_main_db(str(db_file)), f"{db_file.name} was not skipped"

    # The rule is about the data dir's own databases, not the name anywhere:
    # a same-named file the user owns outside data/ is not Chalie's.
    assert not _is_chalie_main_db(str(scan_root / "chalie-9.9.9.sqlite"))

    # ...and it must not widen into "skip the whole data dir", which would
    # strand every user file under data/ (documents, downloads, ...).
    ordinary = data_dir / "notes.txt"
    ordinary.write_text("quarterly figures")
    assert not _is_chalie_main_db(str(ordinary))

    service = FileIndexService(scan_root=str(scan_root), db_path=index_db_path)
    assert service.should_index(str(ordinary)), "an ordinary data-dir file must still be indexed"
    # End to end: the walk descends into the data dir and indexes the ordinary
    # file while no main-database file ever reaches the index.
    service.reconcile()
    assert str(ordinary) in service.search("quarterly")
    for db_file in skipped:
        assert str(db_file) not in service.search("chalie", limit=100)

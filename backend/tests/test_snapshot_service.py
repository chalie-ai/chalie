"""Feature tests for the whole-instance snapshot Time-Machine.

These drive the REAL production stack — no mocks of production code. The only
``patch()`` calls inherited from ``authed_client`` sit at the auth/session
boundary. ``FileMapperService`` class attributes are *redirected* to a temp
data root — configuration redirection, not mocking (same idiom as
``test_vault_fix.py`` and ``test_mcp_client_service.py``).

Boot ordering note: the import HTTP endpoint calls ``request_restart()`` whose
daemon thread runs ``os._exit(42)`` two seconds later — it bypasses pytest. The
roundtrip tests drive the SAME production engine method directly
(``SnapshotService.stage_import`` → ``apply_pending``). The HTTP endpoint is
exercised only on paths that fail BEFORE ``request_restart`` is reached (wrong
password, >50 MB junk zip).

Methods: ``export()``  ·  ``stage_import()``  ·  ``apply_pending()`` boot  ·
POST /api/snapshot/export  ·  POST /api/snapshot/import
"""

import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import services.database_service as _db_mod
import services.vault_service as _vault_mod
from services.database_service import DatabaseService
from services.file_mapper_service import FileMapperService
from services.vault_service import _vault_state

if TYPE_CHECKING:
    from flask.testing import FlaskClient


# ── Shared production-path helpers (no re-implementation of prod logic) ───────

def _seed_transcript(channel: str, role: str, content: str) -> int:
    """Seed via the SAME entry point production uses to persist a turn."""
    from services.transcript_service import Transcript
    rowid = Transcript.append(channel=channel, role=role, content=content)
    assert rowid is not None, "production transcript append must persist a row"
    return rowid


def _recent_contents(channel: str) -> list[str]:
    from services.transcript_service import Transcript
    return cast(list[str], [r['content'] for r in Transcript.get_recent(channel, limit=50)])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_vault_singletons() -> Iterator[None]:
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None
    yield
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None


@pytest.fixture
def instance(_db_template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    import shutil

    root = tmp_path / "instance"
    data_dir = root / "data"
    secure_dir = data_dir / "secure"
    docs_dir = data_dir / "documents"
    user_skills_dir = data_dir / "skills" / "user"
    backend_dir = root / "backend"
    abilities_assets = backend_dir / "abilities" / "assets"
    for d in (data_dir, secure_dir, docs_dir, user_skills_dir, abilities_assets):
        d.mkdir(parents=True, exist_ok=True)

    # Redirect the path chokepoint at the class-attribute level. Each helper
    # (get_db_path/get_secure_dir/get_data_path/...) reads these at call time.
    monkeypatch.setattr(FileMapperService, "_CHALIE_ROOT", root)
    monkeypatch.setattr(FileMapperService, "_BACKEND_DIR", backend_dir)
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", data_dir)
    monkeypatch.setattr(FileMapperService, "_SECURE_DIR", secure_dir)
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(FileMapperService, "_USER_SKILLS_DIR", user_skills_dir)
    monkeypatch.setattr(FileMapperService, "_ABILITIES_DIR", backend_dir / "abilities")

    # Real converged DB copied to the redirected chalie.db path.
    db_path = str(FileMapperService.get_db_path())
    shutil.copy2(_db_template, db_path)

    # Copy the real VERSION + schema.sql so the manifest + downgrade guard read
    # genuine build artifacts (not fabricated constants). The originals live at
    # the *real* repo root, captured before any redirect (see _RealRoots).
    _orig_root = _RealRoots.chalie_root
    shutil.copy2(_orig_root / "VERSION", FileMapperService.get_version_path())
    shutil.copy2(
        _orig_root / "backend" / "schema.sql", FileMapperService.get_schema_path()
    )
    # A real skills.sqlite + mcp_tools.sqlite so the WAL-fold step has genuine DBs.
    _make_min_sqlite(str(FileMapperService.get_skills_db_path()))
    _make_min_sqlite(str(FileMapperService.get_mcp_tools_db_path()))

    # Inject a real DatabaseService bound to the redirected path as the singleton.
    db_service = DatabaseService(db_path)
    _db_mod._local.conn = None
    _db_mod._local.db_path = None
    original = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service

    import services.data_graph_service as _dgs_mod
    original_dgs = _dgs_mod._instance
    _dgs_mod._instance = None

    try:
        yield root
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        _dgs_mod._instance = original_dgs


class _RealRoots:
    chalie_root = FileMapperService.get_chalie_root()


def _make_min_sqlite(path: str) -> None:
    """Create a tiny but real SQLite DB with one row for WAL-fold backup."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS marker (k TEXT, v TEXT)")
        conn.execute("INSERT INTO marker (k, v) VALUES ('seeded', '1')")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def client(instance: Path) -> "Iterator[FlaskClient]":
    """Real Flask app via ``create_app``, bound to temp instance.
    Reuses auth/session/store seam from conftest's ``authed_client``."""
    from unittest.mock import patch
    from api import create_app
    from services.memory_store import MemoryStore

    real_store = MemoryStore()
    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.memory_store.get_shared_store', return_value=real_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=real_store):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c


def _pre_restore_asides() -> list[Path]:
    return sorted(FileMapperService.get_data_path().glob(".pre-restore-*"))


@pytest.mark.unit
class TestSnapshotRoundtrip:

    def test_plain_roundtrip_restores_seeded_rows_and_keeps_pre_restore_aside(self, instance: Path) -> None:
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "ALPHA-snapshot-marker")
        assert "ALPHA-snapshot-marker" in _recent_contents("user")

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        assert zipfile.is_zipfile(str(zip_path)), "export must yield a real zip"

        # Mutate live state AFTER the snapshot — restore must roll this back.
        _seed_transcript("user", "user", "BRAVO-post-snapshot")
        assert "BRAVO-post-snapshot" in _recent_contents("user")

        svc.stage_import(zip_path, None)
        assert FileMapperService.get_data_path(".pending-restore").exists(), \
            "stage_import must write the .pending-restore marker"

        SnapshotService.apply_pending()

        # Reopen the DB through the production singleton path (fresh connection).
        _db_mod._local.conn = None
        contents = _recent_contents("user")
        assert "ALPHA-snapshot-marker" in contents, "pre-export row must be restored"
        assert "BRAVO-post-snapshot" not in contents, "post-export row must be wiped"

        asides = _pre_restore_asides()
        assert len(asides) == 1, "exactly one .pre-restore-<ts> aside expected"
        assert any(asides[0].rglob("chalie.db")), "aside must hold the prior chalie.db"
        assert not FileMapperService.get_data_path(".pending-restore").exists(), \
            "apply_pending must clear the marker"

    def test_encrypted_roundtrip_zip_is_real_aes_and_restores_with_password(self, instance: Path) -> None:
        """Password export produces a genuine AES zip (reading WITHOUT the
        password raises RuntimeError per pyzipper) and a password import →
        apply_pending restores the pre-export state."""
        from pyzipper import AESZipFile
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "ENCRYPTED-ALPHA")

        svc = SnapshotService()
        password = "correct horse battery staple"
        zip_path = svc.export(password=password)

        # Genuinely encrypted: a member read without the password must raise.
        with AESZipFile(str(zip_path), 'r') as zf:
            member = next(n for n in zf.namelist() if not n.endswith('/'))
            with pytest.raises(RuntimeError):
                zf.read(member)

        _seed_transcript("user", "user", "ENCRYPTED-BRAVO-post")

        svc.stage_import(zip_path, password)
        SnapshotService.apply_pending()

        _db_mod._local.conn = None
        contents = _recent_contents("user")
        assert "ENCRYPTED-ALPHA" in contents
        assert "ENCRYPTED-BRAVO-post" not in contents


@pytest.mark.unit
class TestSnapshotRejections:

    def test_wrong_password_is_rejected_loudly_and_stages_nothing(self, instance: Path, client: "FlaskClient") -> None:
        """Importing an AES snapshot with the WRONG password fails loudly (4xx
        at the HTTP boundary, typed error from stage_import), stages NOTHING,
        leaves the live DB untouched, and triggers no restart (no .pending
        marker → boot apply is a no-op)."""
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "LIVE-UNTOUCHED-MARKER")

        svc = SnapshotService()
        zip_path = svc.export(password="the-right-password")

        # Engine layer: a typed error bubbles, nothing staged.
        with pytest.raises(Exception):
            svc.stage_import(zip_path, "the-WRONG-password")
        assert not FileMapperService.get_data_path(".pending-restore").exists(), \
            "a failed stage_import must leave NO .pending-restore marker"
        assert _pre_restore_asides() == [], "no aside before any apply"

        # HTTP layer: the same wrong-password path surfaces a 4xx (not swallowed),
        # and because staging failed, request_restart() is never reached.
        with open(str(zip_path), 'rb') as fh:
            resp = client.post(
                '/api/snapshot/import',
                data={'file': (fh, 'snapshot.zip'), 'password': 'the-WRONG-password'},
                content_type='multipart/form-data',
            )
        assert 400 <= resp.status_code < 500, \
            f"wrong password must be a 4xx, got {resp.status_code}"

        # Live instance untouched.
        _db_mod._local.conn = None
        assert "LIVE-UNTOUCHED-MARKER" in _recent_contents("user")
        assert not FileMapperService.get_data_path(".pending-restore").exists()

    def test_schema_downgrade_snapshot_is_blocked_and_stages_nothing(self, instance: Path) -> None:
        """A snapshot whose chalie.db carries a column the running build's
        schema.sql does NOT declare would force convergence to DROP user data on
        restore. stage_import must block it with a loud typed error and stage
        nothing. The stray column is added to the snapshot's OWN chalie.db
        via real sqlite3 — no production test hook."""
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "DOWNGRADE-GUARD-PROBE")

        svc = SnapshotService()
        zip_path = svc.export(password=None)

        # Rewrite the snapshot zip, adding a stray column to the snapshot's
        # chalie.db so its schema is a strict superset of the build's.
        tampered = str(instance / "tampered-snapshot.zip")
        _add_stray_column_to_zipped_db(str(zip_path), tampered, "transcript", "tkt949_ghost_col")

        with pytest.raises(Exception):
            svc.stage_import(cast(Path, tampered), None)
        assert not FileMapperService.get_data_path(".pending-restore").exists(), \
            "schema-downgrade must block staging entirely"
        assert _pre_restore_asides() == []


def _add_stray_column_to_zipped_db(src_zip: str, dst_zip: str, table: str, column: str) -> None:
    """Open a plain snapshot zip, ALTER the embedded chalie.db to add a stray
    column, and rewrite the zip with the mutated DB in place. Uses real sqlite3;
    no production code involved in building the adversarial fixture."""
    import os
    import tempfile

    with zipfile.ZipFile(src_zip, 'r') as zin:
        names = zin.namelist()
        db_member = next(n for n in names if n.endswith('chalie.db'))
        payloads = {n: zin.read(n) for n in names}

    tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp_db.write(payloads[db_member])
    tmp_db.close()
    try:
        conn = sqlite3.connect(tmp_db.name)
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
            conn.commit()
        finally:
            conn.close()
        payloads[db_member] = open(tmp_db.name, 'rb').read()
    finally:
        os.unlink(tmp_db.name)

    with zipfile.ZipFile(dst_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, payloads[n])


@pytest.mark.unit
class TestSnapshotRollbackAndLimits:

    def test_apply_pending_rolls_back_on_real_swap_failure(self, instance: Path) -> None:
        """A REAL failure mid-apply (a staged artifact path made unreadable so
        the swap raises) must restore the ``.pre-restore-<ts>`` aside and leave
        the live artifacts intact (locked decision #4). No test-only branch is
        added to production — a genuine filesystem failure is induced."""
        import os
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "ROLLBACK-LIVE-STATE")

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        svc.stage_import(zip_path, None)
        assert FileMapperService.get_data_path(".pending-restore").exists()

        # Induce a real failure: make one staged artifact unreadable so the
        # targeted swap raises mid-flight. The staging dir lives under the
        # pending-restore path; find a staged file and strip its perms.
        staged_root = FileMapperService.get_data_path(".pending-restore")
        staged_files = [p for p in staged_root.rglob("*") if p.is_file()]
        assert staged_files, "stage_import must have extracted artifacts to stage"
        victim = next((p for p in staged_files if p.name.endswith('.db')), staged_files[0])
        os.chmod(victim, 0o000)

        try:
            # apply_pending owns its try/except; it must NOT propagate but must
            # restore the aside. Either way the live DB must remain intact.
            try:
                SnapshotService.apply_pending()
            except Exception:
                pass

            _db_mod._local.conn = None
            assert "ROLLBACK-LIVE-STATE" in _recent_contents("user"), \
                "live DB must survive a failed apply_pending (rollback)"
            assert FileMapperService.get_db_path().exists(), \
                "live chalie.db must still be present after rollback"
        finally:
            os.chmod(victim, 0o600)

    def test_import_over_50mb_reaches_snapshot_logic_not_413(self, instance: Path, client: "FlaskClient") -> None:
        """A ~60 MB upload must NOT be rejected with HTTP 413 by the global cap —
        proving the per-route MAX_CONTENT_LENGTH bypass. The junk zip has
        no valid manifest, so a WORKING endpoint then fails the manifest/checksum
        check LOUDLY with a snapshot-level 4xx error envelope. The two assertions
        together are RED until the endpoint exists: status must be 'not 413' AND
        the route must actually exist (a missing route 404/405 fails the second
        assertion, so this cannot pass vacuously against absent production code)."""
        import os

        # Build a ~60 MB junk zip (one incompressible stored member) so the
        # upload exceeds the global 50 MB cap.
        big_zip = instance / "big-junk-snapshot.zip"
        with zipfile.ZipFile(str(big_zip), 'w', zipfile.ZIP_STORED) as zf:
            zf.writestr("junk.bin", os.urandom(60 * 1024 * 1024))
        assert big_zip.stat().st_size > 50 * 1024 * 1024, "fixture must exceed 50 MB"

        with open(str(big_zip), 'rb') as fh:
            resp = client.post(
                '/api/snapshot/import',
                data={'file': (fh, 'big-junk-snapshot.zip')},
                content_type='multipart/form-data',
            )

        assert resp.status_code != 413, \
            "large upload must bypass the global 50 MB cap (got 413)"
        # The route must exist AND have run the snapshot logic far enough to
        # reject the manifest-less junk zip loudly — i.e. a snapshot-level 4xx,
        # not a framework 404/405 for a missing route. This is the assertion that
        # keeps the test RED until /api/snapshot/import is actually built.
        assert resp.status_code not in (404, 405), \
            "the import route must exist (no catch-all 404/405)"
        assert 400 <= resp.status_code < 500, \
            f"manifest-less junk must fail loudly with a 4xx, got {resp.status_code}"
        assert (resp.get_json() or {}).get('error'), \
            "a loud snapshot-level error envelope must be surfaced (not swallowed)"

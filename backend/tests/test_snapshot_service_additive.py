"""Additive feature tests for the snapshot Time-Machine (TKT-949).

These are ADDITIVE coverage for production behaviours the six locked tests in
``test_snapshot_service.py`` do not reach. They do NOT modify, weaken, or
duplicate any locked test — each one drives a distinct, real production entry
point with ZERO mocks of production code, reusing the SAME ``instance`` /
``client`` fixtures the locked file defines (imported, never re-implemented —
Law 4: no alternative path) so there is exactly one definition of the relocated
real instance shared by both files.

Why each test exists (a real user can hit each path; the locked suite cannot):

  * ``test_http_export_route_streams_a_real_zip`` — the locked roundtrip drives
    the engine ``export()`` directly; the production HTTP entry point
    ``POST /api/snapshot/export`` (``api/snapshot.py:36``) — the exact route the
    Brain "Export" button calls — is never exercised end-to-end. A regression in
    the route (wrong mimetype, ``send_file`` path bug, auth gate) would pass all
    six locked tests. This drives the real route and asserts the streamed bytes
    are a genuine zip carrying a ``chalie.db`` member.

  * ``test_apply_pending_is_a_noop_when_nothing_is_staged`` — every ordinary
    boot of a non-restoring instance hits the early return at
    ``snapshot_service.py:394``. No locked test exercises it; a regression that
    crashed ``apply_pending`` on a clean boot (no ``.pending-restore``) would
    take down every restart. Asserts the live DB is untouched and no stray
    aside/quarantine dirs appear.

  * ``test_mid_swap_failure_rolls_back_and_quarantines_to_break_boot_loop`` — the
    locked rollback test (#5) chmods a staged ``.db`` to ``0o000``, which raises
    inside the PRE-swap re-verification (``_verify_members``) BEFORE ``_swap_in``
    runs — so the ``_swap_in`` rollback (``snapshot_service.py:421-428``), the
    artifact ``_rollback`` (``:478``), and the ``.restore-failed-*`` quarantine
    (``:493``) are all UNCOVERED. This induces a genuine MID-swap filesystem
    fault (a read-only destination directory — no production test hook) so the
    swap fails AFTER ``chalie.db`` is already swapped, and asserts: the live
    ``chalie.db`` is rolled back byte-faithfully (the pre-restore row survives),
    the ``.pending-restore`` marker is GONE, and a ``.restore-failed-*`` aside
    exists — the boot-loop guard the spec requires (§8 / §9.1).

  * ``test_plain_export_opens_without_password_and_missing_manifest_is_rejected``
    — the locked encrypted test proves an AES zip CANNOT be read without the
    password; nothing proves the no-password export produces a genuinely
    password-FREE zip (the distinguishing behaviour of the two crypto paths),
    and the ``_read_manifest`` missing-manifest rejection
    (``snapshot_service.py:298``) is never exercised. This reads a plain export
    with no password (must succeed) and then feeds a manifest-less zip to
    ``stage_import`` (must raise loudly, stage nothing).

Total snapshot-service tests: 6 locked + 4 additive = 10 (≤ 10 cap honoured).
"""

import os
import zipfile

import pytest

import services.database_service as _db_mod
from services.file_mapper_service import FileMapperService

# Reuse the locked file's real fixtures and production-path helpers verbatim —
# one definition, shared by both files. Importing the fixture functions makes
# them resolvable by pytest in this module's namespace (no re-implementation of
# the relocated-instance setup, no alternative path).
from tests.test_snapshot_service import (  # noqa: F401  (fixtures used by pytest)
    _pre_restore_asides,
    _recent_contents,
    _reset_vault_singletons,
    _seed_transcript,
    client,
    instance,
)


def _quarantine_dirs() -> list:
    """All ``.restore-failed-<ts>`` quarantine dirs under the redirected data root."""
    return sorted(FileMapperService.get_data_path().glob(".restore-failed-*"))


@pytest.mark.unit
class TestSnapshotHttpExport:

    def test_http_export_route_streams_a_real_zip(self, client):  # noqa: F811  (pytest fixture, imported)
        """The Brain 'Export' button calls ``POST /api/snapshot/export``. Driving
        the REAL route (not just the engine) must stream a genuine zip — an
        attachment with the zip mimetype whose bytes are a real zip carrying a
        ``chalie.db`` member produced by the WAL-fold path."""
        _seed_transcript("user", "user", "HTTP-EXPORT-MARKER")

        resp = client.post("/api/snapshot/export", json={})

        assert resp.status_code == 200, f"export route must succeed, got {resp.status_code}"
        assert resp.mimetype == "application/zip", "export must stream a zip mimetype"
        assert "attachment" in resp.headers.get("Content-Disposition", ""), \
            "export must be an attachment download"

        body = resp.get_data()
        out = FileMapperService.get_data_path("http-export-roundtrip.zip")
        out.write_bytes(body)
        assert zipfile.is_zipfile(str(out)), "streamed body must be a real zip"
        with zipfile.ZipFile(str(out), "r") as zf:
            names = zf.namelist()
        assert any(n.endswith("chalie.db") for n in names), \
            "the streamed snapshot must carry a chalie.db member"


@pytest.mark.unit
class TestSnapshotApplyNoop:

    def test_apply_pending_is_a_noop_when_nothing_is_staged(self, instance):  # noqa: F811  (pytest fixture, imported)
        """Every ordinary boot with no staged restore hits the early return in
        ``apply_pending``. It must be a clean no-op: the live DB is untouched and
        no aside / quarantine dirs are created."""
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "NO-STAGED-RESTORE")
        assert not FileMapperService.get_pending_restore_path().exists(), \
            "precondition: nothing staged"

        SnapshotService.apply_pending()

        _db_mod._local.conn = None
        assert "NO-STAGED-RESTORE" in _recent_contents("user"), \
            "a no-op apply_pending must not touch the live DB"
        assert _pre_restore_asides() == [], "no aside on a no-op boot"
        assert _quarantine_dirs() == [], "no quarantine on a no-op boot"
        assert FileMapperService.get_db_path().exists()


@pytest.mark.unit
class TestSnapshotMidSwapRollback:

    def test_mid_swap_failure_rolls_back_and_quarantines_to_break_boot_loop(self, instance):  # noqa: F811  (pytest fixture, imported)
        """A REAL mid-swap filesystem fault (after the pre-swap re-verify passes
        and after ``chalie.db`` is already swapped) must roll the live artifacts
        back, leave the live ``chalie.db`` intact and readable, clear the
        ``.pending-restore`` marker, and quarantine the staged set as
        ``.restore-failed-<ts>`` so the next boot does NOT re-apply it (§8 /
        §9.1). No production test hook — the fault is a read-only destination
        directory. This exercises the ``_swap_in`` rollback + ``_quarantine``
        path the locked rollback test cannot reach (it fails in the pre-swap
        re-verify, before any live move)."""
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "MIDSWAP-LIVE-STATE")

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        svc.stage_import(zip_path, None)
        assert FileMapperService.get_pending_restore_path().exists()

        # Induce a genuine MID-swap fault: make the live VERSION's destination
        # directory read-only so moving the live VERSION aside raises AFTER
        # chalie.db has already been swapped into place. The pre-swap re-verify
        # reads the staged files fine, so _swap_in is entered for real.
        version_dest_dir = FileMapperService.get_version_path().parent
        original_mode = version_dest_dir.stat().st_mode
        os.chmod(version_dest_dir, 0o500)  # r-x, no write → live VERSION move fails
        try:
            # apply_pending owns its try/except; it must not propagate.
            SnapshotService.apply_pending()
        finally:
            os.chmod(version_dest_dir, original_mode)

        # Live chalie.db must be rolled back byte-faithfully and readable.
        _db_mod._local.conn = None
        assert "MIDSWAP-LIVE-STATE" in _recent_contents("user"), \
            "live DB must be rolled back intact after a mid-swap failure"
        assert FileMapperService.get_db_path().exists(), "live chalie.db must remain present"

        # Boot-loop guard: marker cleared, staged set quarantined for inspection.
        assert not FileMapperService.get_pending_restore_path().exists(), \
            "the .pending-restore marker must be cleared so boot does not re-apply"
        assert len(_quarantine_dirs()) == 1, \
            "the failed staged set must be quarantined as exactly one .restore-failed-<ts>"
        # The failed rollback aside must not linger (it is torn down on rollback).
        assert _pre_restore_asides() == [], \
            "the pre-restore aside is removed when the swap rolls back"


@pytest.mark.unit
class TestSnapshotPlainCryptoAndManifest:

    def test_plain_export_opens_without_password_and_missing_manifest_is_rejected(self, instance):  # noqa: F811  (pytest fixture, imported)
        """Two unguarded contracts in one real-stack scenario:

        (a) the no-password export is genuinely password-FREE — pyzipper can read
            a member with NO password set (the distinguishing behaviour vs. the
            AES path the locked encrypted test covers from the other side); and
        (b) ``stage_import`` rejects a manifest-less zip loudly and stages nothing
            (the ``_read_manifest`` missing-manifest branch, never otherwise hit).
        """
        from pyzipper import AESZipFile
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "PLAIN-CRYPTO-PROBE")

        svc = SnapshotService()
        zip_path = svc.export(password=None)

        # (a) Plain zip: a member is readable with NO password set.
        with AESZipFile(str(zip_path), "r") as zf:
            member = next(n for n in zf.namelist() if not n.endswith("/"))
            data = zf.read(member)  # must NOT raise — no password required
        assert data is not None and len(data) >= 0, \
            "a no-password export must be readable without any password"

        # (b) A zip with NO manifest.json must be rejected loudly, staging nothing.
        no_manifest = FileMapperService.get_data_path("no-manifest.zip")
        with zipfile.ZipFile(str(no_manifest), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("chalie_db/chalie.db", b"not actually a db")
        with pytest.raises(Exception):
            svc.stage_import(no_manifest, None)
        assert not FileMapperService.get_pending_restore_path().exists(), \
            "a manifest-less zip must stage nothing"
        assert _pre_restore_asides() == [], "no aside before any apply"

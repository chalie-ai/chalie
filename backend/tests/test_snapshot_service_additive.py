"""Additive feature tests for the snapshot Time-Machine (TKT-949).

Covers production paths the six locked tests in ``test_snapshot_service.py``
do not reach: the HTTP export route, the no-staged-restore early return, the
mid-swap rollback + quarantine path, the plain-crypto / missing-manifest
branch, and cross-version artifact skipping.
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
from tests.test_snapshot_service import (  # noqa: F401
    _pre_restore_asides,
    _recent_contents,
    _reset_vault_singletons,
    _seed_transcript,
    client,
    instance,
)


def _quarantine_dirs() -> list:
    return sorted(FileMapperService.get_data_path().glob(".restore-failed-*"))


@pytest.mark.unit
class TestSnapshotHttpExport:

    def test_http_export_route_streams_a_real_zip(self, client):  # noqa: F811
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

    def test_apply_pending_is_a_noop_when_nothing_is_staged(self, instance):  # noqa: F811
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

    def test_mid_swap_failure_rolls_back_and_quarantines_to_break_boot_loop(self, instance):  # noqa: F811
        """A mid-swap filesystem fault (read-only destination dir, AFTER chalie.db
        is already swapped) must roll back live artifacts, clear
        ``.pending-restore``, and quarantine the staged set as
        ``.restore-failed-<ts>`` (boot-loop guard). Exercises the ``_swap_in``
        rollback + ``_quarantine`` path the locked rollback test cannot reach."""
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

    def test_plain_export_opens_without_password_and_missing_manifest_is_rejected(self, instance):  # noqa: F811
        """(a) Plain export is readable with no password (distinct from AES path).
        (b) A manifest-less zip raises from ``stage_import`` and stages nothing
        (exercises ``_read_manifest`` missing-manifest branch)."""
        from pyzipper import AESZipFile
        from services.snapshot_service import SnapshotService

        _seed_transcript("user", "user", "PLAIN-CRYPTO-PROBE")

        svc = SnapshotService()
        zip_path = svc.export(password=None)

        # (a) Plain zip: a member is readable with NO password set.
        with AESZipFile(str(zip_path), "r") as zf:
            member = next(n for n in zf.namelist() if not n.endswith("/"))
            data = zf.read(member)  # must NOT raise — no password required
        assert data is not None and len(data) > 0, \
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


@pytest.mark.unit
class TestSnapshotCrossVersionRestore:

    def test_restore_skips_unknown_artifact_kind_from_an_older_build(self, instance):  # noqa: F811
        """A snapshot carrying a dropped artifact kind (``session_secret``) must
        skip it and complete the restore - not ``KeyError`` and quarantine."""
        import json
        import zipfile as _zip

        from services.snapshot_service import _MANIFEST_NAME, SnapshotService

        _seed_transcript("user", "user", "XVERSION-RESTORE-MARKER")
        svc = SnapshotService()
        real_zip = svc.export(password=None)

        # Hash one legacy session_secret payload with the production hasher, then
        # rebuild the zip exactly as a pre-removal build would have: every real
        # member, plus a single-file 'session_secret' artifact this build no
        # longer knows how to route.
        secret_bytes = b"legacy-session-secret-bytes"
        secret_src = FileMapperService.get_data_path("legacy-secret.bin")
        secret_src.write_bytes(secret_bytes)
        legacy_arcname = "session_secret/session_secret"

        with _zip.ZipFile(str(real_zip), "r") as zin:
            manifest = json.loads(zin.read(_MANIFEST_NAME))
            members = {n: zin.read(n) for n in zin.namelist() if n != _MANIFEST_NAME}
        manifest["artifacts"].append({
            "kind": "session_secret",
            "arcname": legacy_arcname,
            "sha256": SnapshotService._sha256(secret_src),
        })
        members[legacy_arcname] = secret_bytes

        legacy_zip = FileMapperService.get_data_path("legacy-xversion.zip")
        with _zip.ZipFile(str(legacy_zip), "w", _zip.ZIP_DEFLATED) as zout:
            zout.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2))
            for name, data in members.items():
                zout.writestr(name, data)

        svc.stage_import(legacy_zip, None)
        assert FileMapperService.get_pending_restore_path().exists(), \
            "the cross-version snapshot must stage like any other"

        SnapshotService.apply_pending()

        # The restore must COMPLETE, not roll back: marker cleared, no quarantine.
        assert not FileMapperService.get_pending_restore_path().exists(), \
            "restore must complete and clear .pending-restore, not abort on the legacy artifact"
        assert _quarantine_dirs() == [], \
            "a skippable legacy artifact must not crash the swap into a .restore-failed-* quarantine"
        # The known artifacts were applied: the restored chalie.db is live + readable.
        _db_mod._local.conn = None
        assert "XVERSION-RESTORE-MARKER" in _recent_contents("user"), \
            "the known artifacts (chalie.db) must be restored while the unknown one is skipped"

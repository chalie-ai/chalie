# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
App Update Service — in-place update system for installed Chalie instances.

Detects deployment mode, checks GitHub for new releases, downloads and validates
tarballs, and performs atomic rename-swap upgrades with automatic database backup
and rollback on failure. Docker and dev environments receive mode-appropriate
guidance instead of in-place mutation.
"""

import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from services.memory_client import MemoryClientService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

# App root is three levels up from this file:
# backend/services/app_update_service.py -> backend/services -> backend -> <root>
APP_ROOT = Path(__file__).parent.parent.parent

GITHUB_API_URL = "https://api.github.com/repos/chalie-ai/chalie/releases/latest"
GITHUB_TARBALL_URL = "https://github.com/chalie-ai/chalie/archive/refs/tags/{tag}.tar.gz"

CACHE_KEY = "app_update:info"
CACHE_TTL = 6 * 60 * 60  # 6 hours in seconds
IN_PROGRESS_KEY = "app_update:in_progress"


class AppUpdateService:
    """Manages in-place application updates for installed Chalie instances."""

    # ── Deployment Detection ─────────────────────────────────────────────

    @staticmethod
    def detect_deployment_mode() -> str:
        """Detect how Chalie was deployed.

        Returns:
            ``"docker"`` if running inside a Docker container,
            ``"dev"`` if a ``.git/`` directory exists at the app root,
            ``"installed"`` otherwise.
        """
        if os.path.exists("/.dockerenv"):
            return "docker"
        if (APP_ROOT / ".git").is_dir():
            return "dev"
        return "installed"

    # ── Version Handling ─────────────────────────────────────────────────

    @staticmethod
    def get_current_version() -> str:
        """Read the current version from the ``VERSION`` file at the app root.

        Returns:
            Version string (e.g. ``"0.2.0"``), or ``"0.0.0"`` if the file
            is missing or unreadable.
        """
        version_file = APP_ROOT / "VERSION"
        try:
            return version_file.read_text().strip()
        except (OSError, IOError):
            logger.warning("VERSION file not found at %s, falling back to 0.0.0", version_file)
            return "0.0.0"

    @staticmethod
    def parse_version(tag: str) -> tuple:
        """Parse a version tag into a comparable integer tuple.

        Strips a leading ``v`` if present, splits on ``.``, and converts
        each segment to an integer.  Non-numeric segments become ``0``.

        Args:
            tag: Version string such as ``"v1.0.1"`` or ``"0.2.0"``.

        Returns:
            Tuple of ints, e.g. ``(1, 0, 1)``.
        """
        tag = tag.strip().lstrip("v")
        parts = []
        for segment in tag.split("."):
            try:
                parts.append(int(segment))
            except (ValueError, TypeError):
                parts.append(0)
        return tuple(parts) if parts else (0, 0, 0)

    # ── Update Check ─────────────────────────────────────────────────────

    def check_for_update(self) -> dict:
        """Check GitHub for a newer release and return update info.

        Results are cached in MemoryStore for 6 hours.  On network failure
        the cached result is returned if available; otherwise a safe default
        with ``update_available: False`` is returned.

        Returns:
            Dict with keys: ``current_version``, ``latest_version``,
            ``latest_tag``, ``update_available``, ``release_notes``,
            ``release_url``, ``deployment_mode``, ``checked_at``.
        """
        store = MemoryClientService.create_connection()
        current = self.get_current_version()
        mode = self.detect_deployment_mode()

        # Return cached result if available (avoids hammering GitHub on manual endpoint calls)
        cached = store.get(CACHE_KEY)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                pass

        try:
            req = Request(
                GITHUB_API_URL,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "Chalie/1.0",
                },
            )
            with urlopen(req, timeout=5) as resp:
                release = json.loads(resp.read())

            latest_tag = release.get("tag_name", "v0.0.0")
            latest_version = latest_tag.lstrip("v")
            update_available = self.parse_version(latest_tag) > self.parse_version(current)

            result = {
                "current_version": current,
                "latest_version": latest_version,
                "latest_tag": latest_tag,
                "update_available": update_available,
                "release_notes": release.get("body", ""),
                "release_url": release.get("html_url", ""),
                "deployment_mode": mode,
                "checked_at": utc_now().isoformat(),
            }

            store.setex(CACHE_KEY, CACHE_TTL, json.dumps(result))
            return result

        except (URLError, OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to check for updates: %s", exc)

            cached = store.get(CACHE_KEY)
            if cached:
                try:
                    return json.loads(cached)
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "current_version": current,
                "latest_version": current,
                "latest_tag": f"v{current}",
                "update_available": False,
                "release_notes": "",
                "release_url": "",
                "deployment_mode": mode,
                "checked_at": utc_now().isoformat(),
            }

    # ── Database Backup ──────────────────────────────────────────────────

    @staticmethod
    def backup_database() -> Path:
        """Back up the SQLite database before an update.

        Creates a copy at ``backend/data/chalie.db.pre-{version}`` and
        removes any older ``.pre-*`` backups so only the latest remains.

        Returns:
            Path to the new backup file.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        db_path = APP_ROOT / "backend" / "data" / "chalie.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found at {db_path}")

        version = AppUpdateService.get_current_version()
        backup_path = db_path.parent / f"chalie.db.pre-{version}"

        # Remove older .pre-* backups (keep only the one we are about to create)
        for old_backup in db_path.parent.glob("chalie.db.pre-*"):
            if old_backup != backup_path:
                try:
                    old_backup.unlink()
                    logger.info("Removed old database backup: %s", old_backup.name)
                except OSError as exc:
                    logger.warning("Could not remove old backup %s: %s", old_backup, exc)

        shutil.copy2(str(db_path), str(backup_path))
        logger.info("Database backed up to %s", backup_path)
        return backup_path

    # ── Download & Validate ──────────────────────────────────────────────

    @staticmethod
    def download_and_validate(tag: str) -> Path:
        """Download a release tarball and validate its contents.

        Downloads from GitHub, extracts with path-traversal protection via
        ``_safe_tar_extract``, and verifies that the archive contains the
        required files (``backend/run.py``, ``backend/schema.sql``,
        ``VERSION``).

        Args:
            tag: Git tag to download (e.g. ``"v1.0.1"``).

        Returns:
            Path to the extracted top-level directory.

        Raises:
            RuntimeError: If download fails, extraction is unsafe, or
                required files are missing.
        """
        from run import _safe_tar_extract

        tarball_url = GITHUB_TARBALL_URL.format(tag=tag)
        logger.info("Downloading release %s from %s", tag, tarball_url)

        tmp_dir = Path(tempfile.mkdtemp(prefix="chalie_update_"))
        tarball_path = tmp_dir / "release.tar.gz"

        try:
            with urlopen(tarball_url, timeout=30) as resp:
                tarball_path.write_bytes(resp.read())
        except (URLError, OSError) as exc:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise RuntimeError(f"Failed to download release {tag}: {exc}") from exc

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()

        try:
            with tarfile.open(str(tarball_path)) as tf:
                _safe_tar_extract(tf, extract_dir)
        except (tarfile.TarError, RuntimeError) as exc:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise RuntimeError(f"Failed to extract release {tag}: {exc}") from exc

        # GitHub tarballs contain a single top-level directory (e.g. chalie-v1.0.1/)
        children = list(extract_dir.iterdir())
        if len(children) == 1 and children[0].is_dir():
            source_dir = children[0]
        else:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise RuntimeError(f"Unexpected tarball structure for {tag}: expected single directory")

        # Validate required files
        required = ["backend/run.py", "backend/schema.sql", "VERSION"]
        for rel_path in required:
            if not (source_dir / rel_path).exists():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
                raise RuntimeError(f"Release {tag} is missing required file: {rel_path}")

        logger.info("Release %s downloaded and validated at %s", tag, source_dir)
        return source_dir

    # ── Apply Update (Rename-Swap) ───────────────────────────────────────

    def apply_update(self, tag: str) -> dict:
        """Orchestrate the full in-place update.

        For Docker or dev deployments, returns guidance instead of mutating
        the filesystem.  For installed deployments, performs:

        1. Database backup
        2. Download and validate the release
        3. Rename-swap (``backend/`` and ``frontend/``)
        4. Copy preserved data (``data/``, ``tools/``)
        5. Stamp deletion (``.deps-installed``)
        6. Cleanup

        On any failure during the swap phase, renames are reversed to
        restore the previous state.

        Args:
            tag: Git tag to apply (e.g. ``"v1.0.1"``).

        Returns:
            Dict with ``ok`` (bool), ``message`` (str), and additional
            context fields.
        """
        mode = self.detect_deployment_mode()

        if mode == "docker":
            return {
                "ok": False,
                "deployment_mode": mode,
                "message": (
                    "Docker deployments update by pulling a new image. "
                    "Run: docker pull ghcr.io/chalie-ai/chalie:{tag} && "
                    "docker compose up -d".format(tag=tag)
                ),
            }

        if mode == "dev":
            return {
                "ok": False,
                "deployment_mode": mode,
                "message": (
                    "Development installs update via git. "
                    "Run: git fetch origin && git checkout {tag}".format(tag=tag)
                ),
            }

        store = MemoryClientService.create_connection()

        # Block concurrent updates
        if store.get(IN_PROGRESS_KEY):
            return {
                "ok": False,
                "deployment_mode": mode,
                "message": "An update is already in progress.",
            }

        store.set(IN_PROGRESS_KEY, "1")

        backend_dir = APP_ROOT / "backend"
        frontend_dir = APP_ROOT / "frontend"
        backend_old = APP_ROOT / f"backend.pre-{tag}"
        frontend_old = APP_ROOT / f"frontend.pre-{tag}"

        # Track what we have renamed so we can roll back precisely
        renamed_backend = False
        renamed_frontend = False

        try:
            # Step 1: Backup database
            logger.info("Starting update to %s — backing up database", tag)
            self.backup_database()

            # Step 2: Download and validate
            logger.info("Downloading release %s", tag)
            source_dir = self.download_and_validate(tag)

            # Step 3: Rename-swap
            logger.info("Performing rename-swap for %s", tag)

            if backend_dir.exists():
                backend_dir.rename(backend_old)
                renamed_backend = True

            if frontend_dir.exists():
                frontend_dir.rename(frontend_old)
                renamed_frontend = True

            # Move new directories into place
            new_backend = source_dir / "backend"
            new_frontend = source_dir / "frontend"

            if new_backend.exists():
                shutil.move(str(new_backend), str(backend_dir))
            else:
                raise RuntimeError(f"Release {tag} has no backend/ directory")

            if new_frontend.exists():
                shutil.move(str(new_frontend), str(frontend_dir))
            else:
                raise RuntimeError(f"Release {tag} has no frontend/ directory")

            # Step 4: Preserve data and tools from old backend
            if renamed_backend:
                old_data = backend_old / "data"
                if old_data.exists():
                    new_data = backend_dir / "data"
                    if new_data.exists():
                        shutil.rmtree(str(new_data))
                    shutil.copytree(str(old_data), str(new_data))
                    logger.info("Preserved data/ directory")

                old_tools = backend_old / "tools"
                if old_tools.exists():
                    new_tools = backend_dir / "tools"
                    if new_tools.exists():
                        shutil.rmtree(str(new_tools))
                    shutil.copytree(str(old_tools), str(new_tools))
                    logger.info("Preserved tools/ directory")

            # Step 5: Copy root-level files from the release
            for filename in ("run.sh", "VERSION"):
                src = source_dir / filename
                if src.exists():
                    shutil.copy2(str(src), str(APP_ROOT / filename))

            for req_file in source_dir.glob("requirements*.txt"):
                shutil.copy2(str(req_file), str(APP_ROOT / req_file.name))

            # Step 6: Delete .deps-installed stamp so dependencies are re-synced
            deps_stamp = APP_ROOT / ".deps-installed"
            if deps_stamp.exists():
                deps_stamp.unlink()
                logger.info("Removed .deps-installed stamp")

        except Exception as exc:
            logger.error("Update to %s failed: %s — rolling back", tag, exc)

            # Rollback: reverse renames
            try:
                if renamed_backend:
                    # Remove partially-placed new backend if it exists
                    if backend_dir.exists():
                        shutil.rmtree(str(backend_dir))
                    backend_old.rename(backend_dir)
                    logger.info("Rolled back backend/")

                if renamed_frontend:
                    if frontend_dir.exists():
                        shutil.rmtree(str(frontend_dir))
                    frontend_old.rename(frontend_dir)
                    logger.info("Rolled back frontend/")
            except Exception as rollback_exc:
                logger.critical(
                    "Rollback FAILED: %s — manual intervention required", rollback_exc
                )

            store.delete(IN_PROGRESS_KEY)
            return {
                "ok": False,
                "deployment_mode": mode,
                "message": f"Update to {tag} failed: {exc}",
            }

        # Success — clean up old directories and temp files
        for old_dir in (backend_old, frontend_old):
            if old_dir.exists():
                try:
                    shutil.rmtree(str(old_dir))
                    logger.info("Cleaned up %s", old_dir.name)
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", old_dir, exc)

        # Clean up the temp download directory (parent of source_dir)
        tmp_root = source_dir.parent
        if tmp_root.exists():
            shutil.rmtree(str(tmp_root), ignore_errors=True)

        store.delete(IN_PROGRESS_KEY)
        logger.info("Update to %s completed successfully", tag)

        return {
            "ok": True,
            "deployment_mode": mode,
            "message": f"Update to {tag} applied successfully. Restart to activate.",
        }

    # ── Restart ──────────────────────────────────────────────────────────

    @staticmethod
    def request_restart():
        """Request a process restart by exiting with code 42.

        Spawns a daemon thread that waits 2 seconds (allowing the HTTP
        response to flush) then calls ``os._exit(42)``.  The exit code 42
        signals the ``run.sh`` wrapper to restart the process.
        """
        def _deferred_exit():
            import time
            time.sleep(2)
            logger.info("Restarting Chalie (exit code 42)")
            os._exit(42)

        thread = threading.Thread(target=_deferred_exit, daemon=True)
        thread.start()
        logger.info("Restart requested — exiting in 2 seconds")

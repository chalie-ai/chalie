# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""In-place update system for installed Chalie instances.

Detects deployment mode, checks GitHub for new releases, downloads and
validates tarballs, performs atomic rename-swap upgrades with rollback on
failure. Dev environments get mode-appropriate guidance instead of
in-place mutation.
"""

import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from services.file_mapper_service import FileMapperService
from services.log_utils import safe
from services.memory_client import MemoryClientService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

APP_ROOT = FileMapperService.get_chalie_root()

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
        """``"dev"`` if ``.git/`` exists at the app root, else ``"installed"``."""
        if (APP_ROOT / ".git").is_dir():
            return "dev"
        return "installed"

    # ── Version Handling ─────────────────────────────────────────────────

    @staticmethod
    def get_current_version() -> str:
        """Returns ``"0.0.0"`` if the file is missing or unreadable."""
        version_file = APP_ROOT / "VERSION"
        try:
            return version_file.read_text().strip()
        except (OSError, IOError):
            logger.warning("VERSION file not found at %s, falling back to 0.0.0", version_file)
            return "0.0.0"

    @staticmethod
    def parse_version(tag: str) -> tuple:
        """Strips a leading ``v`` if present, splits on ``.``, and converts
        each segment to an int. Non-numeric segments become 0."""
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
        """Results are cached in MemoryStore for 6 hours. On network failure
        the cached result is returned if available; otherwise a safe default
        with ``update_available: False`` is returned."""
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

        except (OSError, json.JSONDecodeError, KeyError) as exc:
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

    # ── Download & Validate ──────────────────────────────────────────────

    @staticmethod
    def download_and_validate(tag: str) -> Path:
        """Downloads from GitHub, extracts with path-traversal protection
        via ``_safe_tar_extract``, and verifies the archive contains the
        required files (``backend/run.py``, ``backend/schema.sql``,
        ``VERSION``). Raises ``RuntimeError`` on download/extract/missing
        file failures."""
        from run import _safe_tar_extract

        tarball_url = GITHUB_TARBALL_URL.format(tag=tag)
        logger.info("Downloading release %s from %s", safe(tag), tarball_url)

        tmp_dir = Path(tempfile.mkdtemp(prefix="chalie_update_"))
        tarball_path = tmp_dir / "release.tar.gz"

        try:
            with urlopen(tarball_url, timeout=30) as resp:
                tarball_path.write_bytes(resp.read())
        except OSError as exc:
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

        logger.info("Release %s downloaded and validated at %s", safe(tag), source_dir)
        return source_dir

    # ── Apply Update (Rename-Swap) ───────────────────────────────────────

    def _rename_swap(self, source_dir: Path, tag: str) -> tuple:
        """Move current backend/frontend aside and put new ones in place.

        Returns (backend_old, frontend_old, renamed_backend, renamed_frontend).
        Raises RuntimeError if the new release is missing required directories.
        """
        backend_dir = APP_ROOT / "backend"
        frontend_dir = APP_ROOT / "frontend"
        backend_old = APP_ROOT / f"backend.pre-{tag}"
        frontend_old = APP_ROOT / f"frontend.pre-{tag}"

        renamed_backend = False
        renamed_frontend = False

        logger.info("Performing rename-swap for %s", safe(tag))

        if backend_dir.exists():
            backend_dir.rename(backend_old)
            renamed_backend = True

        if frontend_dir.exists():
            frontend_dir.rename(frontend_old)
            renamed_frontend = True

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

        return backend_old, frontend_old, renamed_backend, renamed_frontend

    @staticmethod
    def _preserve_tools(backend_old: Path) -> None:
        """Copy user-installed tools from the old backend into the new one."""
        old_tools = backend_old / "tools"
        if not old_tools.exists():
            return
        new_tools = APP_ROOT / "backend" / "tools"
        if new_tools.exists():
            shutil.rmtree(str(new_tools))
        shutil.copytree(str(old_tools), str(new_tools))
        logger.info("Preserved tools/ directory")

    @staticmethod
    def _rollback_swap(backend_dir: Path, frontend_dir: Path,
                       backend_old: Path, frontend_old: Path,
                       renamed_backend: bool, renamed_frontend: bool) -> None:
        """Reverse a rename-swap on failure."""
        try:
            if renamed_backend:
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
            logger.critical("Rollback FAILED: %s — manual intervention required", rollback_exc)

    def _apply_release_files(self, tag: str) -> tuple:
        """Download, validate, swap, and copy release files.

        Returns (backend_old, frontend_old, renamed_backend, renamed_frontend, source_dir).
        Raises on any failure.
        """
        logger.info("Starting update to %s — downloading release", safe(tag))
        source_dir = self.download_and_validate(tag)

        backend_old, frontend_old, renamed_backend, renamed_frontend = self._rename_swap(source_dir, tag)

        if renamed_backend:
            self._preserve_tools(backend_old)

        for filename in ("run.sh", "VERSION"):
            src = source_dir / filename
            if src.exists():
                shutil.copy2(str(src), str(APP_ROOT / filename))

        pyproject = source_dir / "pyproject.toml"
        if pyproject.exists():
            shutil.copy2(str(pyproject), str(APP_ROOT / "pyproject.toml"))

        deps_stamp = APP_ROOT / ".deps-installed"
        if deps_stamp.exists():
            deps_stamp.unlink()
            logger.info("Removed .deps-installed stamp")

        return backend_old, frontend_old, renamed_backend, renamed_frontend, source_dir

    def _cleanup_after_update(self, backend_old, frontend_old, source_dir: Path) -> None:
        """Remove old backup directories and the temporary release directory."""
        for old_dir in filter(None, (backend_old, frontend_old)):
            if old_dir.exists():
                try:
                    shutil.rmtree(str(old_dir))
                    logger.info("Cleaned up %s", old_dir.name)
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", old_dir, exc)

        tmp_root = source_dir.parent
        if tmp_root.exists():
            shutil.rmtree(str(tmp_root), ignore_errors=True)

    def apply_update(self, tag: str) -> dict:
        """For dev deployments, returns guidance instead of mutating the
        filesystem. For installed deployments: backup DB → download+validate
        → rename-swap (backend/, frontend/) → copy preserved data → stamp
        deletion (``.deps-installed``) → cleanup. On any failure during
        the swap phase, renames are reversed to restore the previous
        state."""
        mode = self.detect_deployment_mode()

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

        if store.get(IN_PROGRESS_KEY):
            return {
                "ok": False,
                "deployment_mode": mode,
                "message": "An update is already in progress.",
            }

        store.set(IN_PROGRESS_KEY, "1", ex=3600)

        backend_dir = APP_ROOT / "backend"
        frontend_dir = APP_ROOT / "frontend"
        backend_old = None
        frontend_old = None
        renamed_backend = False
        renamed_frontend = False
        source_dir = None  # set on success by _apply_release_files

        try:
            backend_old, frontend_old, renamed_backend, renamed_frontend, source_dir = (
                self._apply_release_files(tag)
            )
        except Exception as exc:
            logger.error("Update to %s failed: %s — rolling back", safe(tag), exc)
            if backend_old is not None or frontend_old is not None:
                self._rollback_swap(
                    backend_dir, frontend_dir,
                    backend_old or APP_ROOT / f"backend.pre-{tag}",
                    frontend_old or APP_ROOT / f"frontend.pre-{tag}",
                    renamed_backend, renamed_frontend,
                )
            store.delete(IN_PROGRESS_KEY)
            return {
                "ok": False,
                "deployment_mode": mode,
                "message": f"Update to {tag} failed: {exc}",
            }

        if source_dir is not None:
            self._cleanup_after_update(backend_old, frontend_old, source_dir)

        store.delete(IN_PROGRESS_KEY)
        logger.info("Update to %s completed successfully", safe(tag))

        return {
            "ok": True,
            "deployment_mode": mode,
            "message": f"Update to {tag} applied successfully. Restart to activate.",
        }

    # ── Restart ──────────────────────────────────────────────────────────

    @staticmethod
    def request_restart():
        """Daemon thread waits 2 s (HTTP response to flush) then
        ``os._exit(42)``. Exit code 42 signals ``run.sh`` to restart the
        process."""
        def _deferred_exit():
            import time
            time.sleep(2)
            logger.info("Restarting Chalie (exit code 42)")
            os._exit(42)

        thread = threading.Thread(target=_deferred_exit, daemon=True)
        thread.start()
        logger.info("Restart requested — exiting in 2 seconds")

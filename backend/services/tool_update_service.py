"""
Tool Update Service — Periodic check for newer git tags on installed embodiments.

Runs every 6 hours. For each tool with _source_url and _installed_tag, runs
`git ls-remote --tags` to find the latest tag. If it differs from the installed
tag, writes _latest_tag so the brain UI can surface an update button.

This service is purely informational — it never modifies tool directories.
"""

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 6 * 3600  # 6 hours
INITIAL_DELAY_SECONDS = 120        # Wait for startup builds to settle


def _get_latest_tag(repo_url: str) -> str | None:
    """
    Query a remote git repo for its latest version tag.

    Uses `git ls-remote --tags --sort=-v:refname` so tags are returned in
    descending version order. Annotated tag dereferences (^{}) are skipped.

    Returns the tag name (e.g. "v1.3.0"), or None on error or if no tags exist.
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", "--sort=-v:refname", repo_url],
            timeout=30,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.strip().split("\n"):
            if not line or "^{}" in line or "\t" not in line:
                continue
            tag = line.split("\t")[1].replace("refs/tags/", "").strip()
            if tag:
                return tag
    except Exception:
        pass
    return None


class ToolUpdateService:
    """Checks installed embodiment tools for newer upstream git tags.

    Scans every subdirectory of ``backend/tools/`` that carries ``_source_url``
    and ``_installed_tag`` metadata, queries the remote repo for its latest tag,
    and writes ``_latest_tag`` via :class:`ToolConfigService` when an update is
    available.  The service is purely informational — it never modifies the tool
    directory or triggers an install.
    """

    def __init__(self):
        """Initialise the service with shared config and database dependencies.

        Resolves the tools directory relative to this file so the service works
        regardless of the working directory when the process was started.
        """
        from services.tool_config_service import ToolConfigService
        from services.database_service import get_shared_db_service
        self._config_svc = ToolConfigService(get_shared_db_service())
        self._tools_dir = Path(__file__).parent.parent / "tools"

    def check_all_updates(self):
        """Scan all installed tools and write _latest_tag when a newer version exists."""
        if not self._tools_dir.exists():
            return

        for tool_dir in sorted(self._tools_dir.iterdir()):
            if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
                continue
            try:
                self._check_tool(tool_dir.name)
            except Exception as e:
                logger.debug(f"[TOOL UPDATE] Check failed for '{tool_dir.name}': {e}")

    def _check_tool(self, tool_name: str):
        """Check a single tool for upstream updates."""
        meta = self._config_svc.get_source_metadata(tool_name)
        source_url = meta.get("_source_url")
        installed_tag = meta.get("_installed_tag")

        if not source_url or not installed_tag:
            return  # Not a git-installed tool or no version info

        latest_tag = _get_latest_tag(source_url)
        if not latest_tag:
            return

        if latest_tag != installed_tag:
            logger.info(f"[TOOL UPDATE] Update available for '{tool_name}': {installed_tag} → {latest_tag}")
            self._config_svc._set_latest_tag(tool_name, latest_tag)
        else:
            # Clear any stale _latest_tag if versions now match (e.g. after manual update)
            if meta.get("_latest_tag"):
                self._config_svc._clear_latest_tag(tool_name)


def tool_update_worker(shared_state=None):
    """
    Entry point for run.py service registration.

    Runs a 6-hour cycle checking for new upstream tags on installed embodiments.
    An initial delay gives the system time to complete startup builds before
    the first check fires.
    """
    logger.info("[TOOL UPDATE WORKER] Starting — first check in 2 minutes.")
    time.sleep(INITIAL_DELAY_SECONDS)

    while True:
        try:
            logger.info("[TOOL UPDATE WORKER] Running update check cycle.")
            svc = ToolUpdateService()
            svc.check_all_updates()
            logger.info("[TOOL UPDATE WORKER] Cycle complete.")
        except Exception as e:
            logger.error(f"[TOOL UPDATE WORKER] Cycle failed: {e}", exc_info=True)
        time.sleep(CHECK_INTERVAL_SECONDS)

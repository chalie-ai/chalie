"""
App Update Service — Handles version checking and in-place updates.

Detects deployment mode, checks GitHub releases for newer versions,
and performs tarball overlay updates for installed deployments.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


class AppUpdateService:
    """Service for managing application version detection and updates."""

    def __init__(self):
        """Initialize the update service with logger and MemoryStore connection.

        Sets up:
        - Logger for debugging and status messages
        - MemoryStore connection for caching update check results
        """
        self.logger = logger
        from .memory_client import MemoryClientService
        self.store = MemoryClientService.create_connection()

    def get_deployment_mode(self) -> str:
        """Detect the current deployment mode.

        Returns one of:
            - 'docker': If IS_DOCKER environment variable is set
            - 'dev': If .git directory exists in project root
            - 'installed': Default fallback for production installs

        Returns:
            str: The detected deployment mode ('docker', 'dev', or 'installed')
        """
        # Check for Docker deployment via environment variable
        if os.environ.get("IS_DOCKER"):
            return "docker"

        # Check for development mode by looking for .git directory
        project_root = Path(__file__).resolve().parent.parent.parent
        git_dir = project_root / ".git"
        if git_dir.exists():
            return "dev"

        # Default to installed (production) mode
        return "installed"

    def get_cache_key(self, key: str) -> str:
        """Generate a namespaced cache key for update service.

        Args:
            key: The base key name

        Returns:
            str: Namespaced cache key with 'app_update:' prefix
        """
        return f"app_update:{key}"

    def _parse_version(self, version_str: str) -> tuple:
        """Parse a semantic version string into comparable components.

        Args:
            version_str: Version string like "0.2.0" or "v1.2.3"

        Returns:
            tuple: (major, minor, patch) as integers for comparison
        """
        # Remove 'v' prefix if present
        clean_version = version_str.lstrip('v')
        parts = clean_version.split('.')
        try:
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except ValueError:
            # If parsing fails, return zeros to indicate invalid version
            self.logger.warning(f"Failed to parse version string: {version_str}")
            return (0, 0, 0)

    def _compare_versions(self, current_version: str, latest_version: str) -> bool:
        """Compare two semantic versions.

        Args:
            current_version: Current app version
            latest_version: Latest available version from GitHub

        Returns:
            bool: True if latest_version is greater than current_version
        """
        current = self._parse_version(current_version)
        latest = self._parse_version(latest_version)
        return latest > current

    def check_for_updates(self, current_version: str) -> Dict[str, Any]:
        """Check GitHub for newer releases and cache the result.

        This method fetches the latest release information from the GitHub API
        for the radiant-node/radiant-node repository. It compares the latest
        release tag name with the provided current version to determine if an
        update is available. Results are cached in MemoryStore for 1 hour (3600s).

        Args:
            current_version: The current application version string

        Returns:
            dict: Contains 'update_available' (bool) and 'latest_version' (str)
                  If deployment mode is not 'installed', returns update_available=False

        Note:
            - Skips check if deployment mode is 'dev' or 'docker'
            - Gracefully handles API errors by returning no update available
            - Uses cache with 3600-second TTL to avoid excessive API calls
        """
        # Skip update checks for non-installed deployments
        deployment_mode = self.get_deployment_mode()
        if deployment_mode != "installed":
            self.logger.debug(f"Skipping update check in {deployment_mode} mode")
            return {"update_available": False, "latest_version": current_version}

        cache_key = self.get_cache_key("latest_version")

        # Check cache first
        cached_result = self.store.get(cache_key)
        if cached_result:
            try:
                import json as json_module
                result = json_module.loads(cached_result)
                self.logger.debug(f"Using cached update check result: {result}")
                return result
            except Exception as e:
                self.logger.warning(f"Failed to parse cached update result: {e}")

        # Fetch latest release from GitHub API
        github_api_url = "https://api.github.com/repos/radiant-node/radiant-node/releases/latest"

        try:
            with httpx.Client() as client:
                response = client.get(github_api_url, timeout=10.0)
                response.raise_for_status()
                release_data = response.json()

            # Extract version from tag_name (e.g., "v0.3.0" -> "0.3.0")
            latest_version_tag = release_data.get("tag_name", "")
            latest_version = latest_version_tag.lstrip('v')

            # Compare versions to determine if update is available
            update_available = self._compare_versions(current_version, latest_version)

            result = {
                "update_available": update_available,
                "latest_version": latest_version
            }

            # Cache the result for 1 hour (3600 seconds)
            import json as json_module
            self.store.set(cache_key, json_module.dumps(result), ex=3600)

            if update_available:
                self.logger.info(f"Update available: {current_version} -> {latest_version}")
            else:
                self.logger.debug(f"No update needed. Current: {current_version}, Latest: {latest_version}")

        except httpx.HTTPError as e:
            self.logger.error(f"GitHub API request failed: {e}")
            return {"update_available": False, "latest_version": current_version}
        except Exception as e:
            self.logger.error(f"Failed to check for updates: {e}")
            return {"update_available": False, "latest_version": current_version}

        return result

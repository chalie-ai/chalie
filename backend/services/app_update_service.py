"""
App Update Service — Handles version checking and in-place updates.

Detects deployment mode, checks GitHub releases for newer versions,
and performs tarball overlay updates for installed deployments.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AppUpdateService:
    """Service for managing application version detection and updates."""

    GITHUB_REPO = "dylangrech/chalie"
    CACHE_KEY = f"update:{GITHUB_REPO}:latest_release"
    CACHE_TTL_SECONDS = 3600  # 1 hour TTL

    def __init__(self):
        """Initialize the update service with the current app version.

        Imports APP_VERSION from backend.consumer module and stores it as an instance attribute.
        """
        from ..consumer import APP_VERSION
        self.app_version = APP_VERSION
        self._memory_store = None

    def _get_memory_store(self):
        """Lazy-load MemoryStore for caching."""
        if self._memory_store is None:
            try:
                from .memory_client import MemoryClientService
                self._memory_store = MemoryClientService.create_connection()
            except Exception as e:
                logger.warning(f"[UPDATE] Could not initialize MemoryStore: {e}")
                self._memory_store = None
        return self._memory_store

    def get_deployment_mode(self) -> str:
        """Detect the current deployment mode based on environment variables.

        Returns one of:
            - 'docker': If IS_DOCKER environment variable is set
            - 'installed': If APP_HOME environment variable is set
            - 'dev': Otherwise (default for development environments)

        Returns:
            str: The detected deployment mode ('docker', 'installed', or 'dev')
        """
        if os.environ.get("IS_DOCKER"):
            return "docker"
        elif os.environ.get("APP_HOME"):
            return "installed"
        else:
            return "dev"

    def check_for_updates(self) -> Dict[str, any]:
        """Check GitHub releases API for newer versions.

        Fetches the latest release from GitHub and compares it with the current version.
        Results are cached in MemoryStore with a 1-hour TTL to avoid excessive API calls.

        Returns:
            dict: A dictionary containing:
                - update_available (bool): True if a newer version exists
                - latest_version (str): The latest release version tag
                - current_version (str): The currently running version
        """
        deployment_mode = self.get_deployment_mode()

        # For docker and dev modes, skip the check gracefully
        if deployment_mode in ("docker", "dev"):
            logger.debug(f"[UPDATE] Skipping update check for {deployment_mode} mode")
            return {
                "update_available": False,
                "latest_version": self.app_version,
                "current_version": self.app_version
            }

        # Try to get cached result first
        cache = self._get_memory_store()
        if cache is not None:
            try:
                cached = cache.get(self.CACHE_KEY)
                if cached:
                    cached_data = json.loads(cached.decode('utf-8'))
                    logger.debug(f"[UPDATE] Using cached update info")
                    return {
                        "update_available": cached_data.get("update_available", False),
                        "latest_version": cached_data.get("latest_version", self.app_version),
                        "current_version": self.app_version
                    }
            except Exception as e:
                logger.warning(f"[UPDATE] Cache read error: {e}")

        # Fetch from GitHub API
        try:
            url = f"https://api.github.com/repos/{self.GITHUB_REPO}/releases/latest"
            with urllib.request.urlopen(url, timeout=10) as response:
                release_data = json.loads(response.read().decode('utf-8'))
                latest_tag = release_data.get("tag_name", "")

            # Extract version number (strip 'v' prefix if present)
            latest_version = re.sub(r'^v', '', latest_tag)

            # Compare versions using simple tuple comparison
            update_available = self._version_greater(latest_version, self.app_version)

            result = {
                "update_available": update_available,
                "latest_version": latest_version,
                "current_version": self.app_version
            }

            # Cache the result for 1 hour
            if cache is not None:
                try:
                    cache.setex(self.CACHE_KEY, self.CACHE_TTL_SECONDS, json.dumps(result))
                    logger.debug(f"[UPDATE] Cached update info with {self.CACHE_TTL_SECONDS}s TTL")
                except Exception as e:
                    logger.warning(f"[UPDATE] Cache write error: {e}")

            if update_available:
                logger.info(f"[UPDATE] Update available: {self.app_version} → {latest_version}")
            else:
                logger.debug(f"[UPDATE] No update available (current: {self.app_version}, latest: {latest_version})")

            return result

        except Exception as e:
            logger.error(f"[UPDATE] Failed to check for updates: {e}")
            return {
                "update_available": False,
                "latest_version": self.app_version,
                "current_version": self.app_version
            }

    def _version_greater(self, v1: str, v2: str) -> bool:
        """Compare two version strings.

        Args:
            v1: First version string (e.g., "0.3.0")
            v2: Second version string (e.g., "0.2.0")

        Returns:
            bool: True if v1 > v2, False otherwise
        """
        def parse_version(v):
            parts = re.split(r'[.-]', v)
            return [int(p) if p.isdigit() else 0 for p in parts]

        try:
            return tuple(parse_version(v1)) > tuple(parse_version(v2))
        except (ValueError, TypeError):
            logger.warning(f"[UPDATE] Could not parse versions: {v1}, {v2}")
            return False

    def perform_update(self, asset_url: Optional[str] = None) -> Dict[str, any]:
        """Download and apply the latest update via tarball overlay.

        For 'installed' mode: Downloads the release tarball from GitHub (or provided URL) 
        and extracts it to the APP_HOME directory, overwriting existing files in place.

        For 'docker' and 'dev' modes: Logs a message and returns without performing any action.

        Args:
            asset_url: Optional URL to the release tarball asset. If not provided, will fetch
                      from GitHub releases API using the latest version.

        Returns:
            dict: A dictionary containing:
                - success (bool): True if update completed successfully
                - message (str): Description of the result or error
                - requires_restart (bool): Whether application restart is required
        """
        deployment_mode = self.get_deployment_mode()

        # Check deployment mode and only proceed for 'installed' mode
        logger.info(f"[UPDATE] perform_update called with deployment mode: {deployment_mode}")

        if deployment_mode != "installed":
            if deployment_mode == "docker":
                logger.info("[UPDATE] Docker mode detected — update must be performed via container rebuild")
                return {
                    "success": False,
                    "message": "Updates in Docker mode require rebuilding the container"
                }
            else:  # dev mode
                logger.info("[UPDATE] Development mode detected — skipping automatic update")
                return {
                    "success": False,
                    "message": "Automatic updates are disabled in development mode"
                }

        # For installed mode, perform the tarball overlay update
        try:
            app_home = os.environ.get("APP_HOME", "/opt/chalie")
            logger.info(f"[UPDATE] Starting update process at {app_home}")

            # Get asset URL - use provided one or fetch from GitHub
            if asset_url is None:
                latest_version = self._get_latest_release_info()
                if not latest_version:
                    return {
                        "success": False,
                        "message": "Could not fetch release information from GitHub"
                    }
                # Construct download URL for the tarball
                asset_url = f"https://github.com/{self.GITHUB_REPO}/releases/download/v{latest_version}/chalie-{latest_version}.tar.gz"
                logger.info(f"[UPDATE] Using latest version {latest_version} to construct download URL")

            temp_dir = tempfile.mkdtemp(prefix="chalie_update_")
            tarball_path = os.path.join(temp_dir, "update.tar.gz")

            try:
                # Step 1: Download the tarball from the given URL
                logger.info(f"[UPDATE] Downloading tarball from {asset_url}")
                
                with urllib.request.urlopen(asset_url, timeout=60) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(tarball_path, 'wb') as f:
                        chunk_size = 8192
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Log progress for large files
                            if total_size > 0 and downloaded_size % (total_size // 10 + 1) == 0:
                                progress = int((downloaded_size / total_size) * 100)
                                logger.debug(f"[UPDATE] Download progress: {progress}%")

                download_size = os.path.getsize(tarball_path)
                logger.info(f"[UPDATE] Download complete ({download_size} bytes)")

                # Step 2: Extract the tarball to a temporary directory and overlay files
                logger.info(f"[UPDATE] Extracting tarball and overlaying files to {app_home}")
                
                self._extract_and_overlay(tarball_path, app_home)

                logger.info("[UPDATE] File extraction and overlay completed successfully")

                # Step 3: Application restart (to be implemented later)
                # TODO: Trigger application restart or signal parent process to reload
                # Example: subprocess.run(['systemctl', 'restart', 'chalie']) for systemd services
                logger.info("[UPDATE] Update applied. Application restart required to complete update.")

                return {
                    "success": True,
                    "message": f"Update completed successfully. Restart application to apply changes.",
                    "requires_restart": True
                }

            finally:
                # Cleanup temp directory
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"[UPDATE] Cleaned up temporary directory {temp_dir}")
                except Exception as e:
                    logger.warning(f"[UPDATE] Could not cleanup temp dir {temp_dir}: {e}")

        except urllib.error.URLError as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(f"[UPDATE] {error_msg}", exc_info=True)
            return {"success": False, "message": error_msg}

        except subprocess.CalledProcessError as e:
            error_msg = f"Extraction failed: {str(e)}"
            logger.error(f"[UPDATE] {error_msg}", exc_info=True)
            return {"success": False, "message": error_msg}

        except Exception as e:
            error_msg = f"Update failed: {str(e)}"
            logger.error(f"[UPDATE] {error_msg}", exc_info=True)
            return {"success": False, "message": error_msg}

    def _get_latest_release_info(self) -> Optional[str]:
        """Fetch the latest release version tag from GitHub.

        Returns:
            str or None: The version string (without 'v' prefix) if successful, None otherwise
        """
        try:
            url = f"https://api.github.com/repos/{self.GITHUB_REPO}/releases/latest"
            with urllib.request.urlopen(url, timeout=10) as response:
                release_data = json.loads(response.read().decode('utf-8'))
                tag_name = release_data.get("tag_name", "")
                return re.sub(r'^v', '', tag_name)
        except Exception as e:
            logger.error(f"[UPDATE] Failed to fetch latest release info: {e}")
            return None

    def _extract_and_overlay(self, tarball_path: str, target_dir: str):
        """Extract tarball and overlay files into the target directory.

        Args:
            tarball_path: Path to the downloaded .tar.gz file
            target_dir: Target directory where files should be extracted/overlaid
        """
        # Extract to a temp staging directory first
        import tempfile
        staging_dir = tempfile.mkdtemp(prefix="chalie_staging_")

        try:
            # Extract tarball
            subprocess.run(
                ['tar', 'xzf', tarball_path, '-C', staging_dir],
                check=True,
                capture_output=True
            )

            # Find the extracted directory (usually one top-level folder)
            extracted_contents = os.listdir(staging_dir)
            if len(extracted_contents) == 1:
                source_dir = os.path.join(staging_dir, extracted_contents[0])
            else:
                source_dir = staging_dir

            # Copy files to target directory (overlay)
            self._copy_tree(source_dir, target_dir)

        finally:
            import shutil
            try:
                shutil.rmtree(staging_dir)
            except Exception as e:
                logger.warning(f"[UPDATE] Could not cleanup staging dir: {e}")

    def _copy_tree(self, source: str, target: str):
        """Recursively copy files from source to target, overwriting existing files.

        Args:
            source: Source directory path
            target: Target directory path
        """
        import shutil
        for item in os.listdir(source):
            source_item = os.path.join(source, item)
            target_item = os.path.join(target, item)

            if os.path.isdir(source_item):
                # Create directory if it doesn't exist
                os.makedirs(target_item, exist_ok=True)
                # Recursively copy contents
                self._copy_tree(source_item, target_item)
            else:
                # Copy file (overwriting if exists)
                shutil.copy2(source_item, target_item)

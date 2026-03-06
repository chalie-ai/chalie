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

    def perform_update(self) -> Dict[str, Any]:
        """Perform an in-place update by downloading and extracting the latest release tarball.

        This method should only be called when deployment mode is 'installed'. It fetches
        the latest release from GitHub, downloads the radiant-node-linux-x64.tar.gz asset,
        extracts it to the installation directory (parent of current script), and triggers
        a restart. The update process runs as a background task to avoid blocking.

        Returns:
            dict: Contains 'success' (bool) and 'message' (str) describing the result

        Note:
            - Skips if deployment mode is not 'installed'
            - Runs extraction in background subprocess
            - Triggers application restart after successful update
        """
        # Check deployment mode first
        deployment_mode = self.get_deployment_mode()
        if deployment_mode != "installed":
            self.logger.warning(f"Update skipped: deployment mode is '{deployment_mode}', not 'installed'")
            return {"success": False, "message": f"Update only supported in 'installed' mode"}

        try:
            # Fetch latest release from GitHub API to get the download URL for the Linux tarball asset
            github_api_url = "https://api.github.com/repos/radiant-node/radiant-node/releases/latest"

            with httpx.Client() as client:
                response = client.get(github_api_url, timeout=10.0)
                response.raise_for_status()
                release_data = response.json()

            # Find the download URL for radiant-node-linux-x64.tar.gz asset
            assets = release_data.get("assets", [])
            download_url = None
            for asset in assets:
                if asset["name"] == "radiant-node-linux-x64.tar.gz":
                    download_url = asset["browser_download_url"]
                    break

            if not download_url:
                self.logger.error("Could not find radiant-node-linux-x64.tar.gz asset in release")
                return {"success": False, "message": "Linux x64 tarball not found in latest release"}

            # Download the tarball to /tmp/
            tarball_path = "/tmp/radiant-node-update.tar.gz"
            self.logger.info(f"Downloading update from {download_url} to {tarball_path}")

            with httpx.Client() as client:
                response = client.get(download_url, timeout=60.0)
                response.raise_for_status()
                with open(tarball_path, "wb") as f:
                    f.write(response.content)

            self.logger.info(f"Downloaded tarball to {tarball_path}")

            # Determine the installation directory (parent of current running script)
            install_dir = Path(__file__).resolve().parent.parent.parent

            # Extract the tarball using subprocess in background mode
            # Using --strip-components=1 to extract contents directly into install_dir
            self.logger.info(f"Extracting update to {install_dir}")

            process = subprocess.Popen(
                ["tar", "-xzf", tarball_path, "-C", str(install_dir), "--strip-components=1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for the extraction to complete (but don't block indefinitely)
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
                if process.returncode != 0:
                    self.logger.error(f"Extraction failed with code {process.returncode}: {stderr.decode()}")
                    return {"success": False, "message": f"Extraction failed: {stderr.decode()}"}
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.warning("Update extraction timed out, but may have completed partially")

            # Trigger application restart after successful update
            self._trigger_restart(install_dir)

            return {"success": True, "message": f"Update extracted to {install_dir}, restart triggered"}

        except httpx.HTTPError as e:
            self.logger.error(f"Failed to download update: {e}")
            return {"success": False, "message": f"Download failed: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Update process failed: {e}")
            return {"success": False, "message": f"Update failed: {str(e)}"}

    def _trigger_restart(self, install_dir: Path) -> None:
        """Trigger an application restart by creating a marker file.

        Args:
            install_dir: The installation directory path
        """
        # Create a marker file to signal that a restart is needed
        from datetime import datetime
        
        restart_marker = install_dir / ".restart_pending"
        try:
            with open(restart_marker, "w") as f:
                import json as json_module
                data = {
                    "timestamp": str(datetime.now().isoformat()),
                    "reason": "update_completed"
                }
                f.write(json_module.dumps(data))
            self.logger.info(f"Restart marker created at {restart_marker}")
        except Exception as e:
            self.logger.warning(f"Failed to create restart marker: {e}")

    def _run_background_update(self) -> None:
        """Run the update process in a background thread.

        This method spawns a daemon thread that executes perform_update()
        without blocking the main application flow.
        """
        import threading

        def update_task():
            result = self.perform_update()
            if result["success"]:
                self.logger.info(f"Background update completed: {result['message']}")
            else:
                self.logger.error(f"Background update failed: {result['message']}")

        thread = threading.Thread(target=update_task, daemon=True)
        thread.start()
        self.logger.info("Update started in background thread")

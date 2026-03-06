"""
App Update Service — Handles version checking and in-place updates.

Detects deployment mode, checks GitHub releases for newer versions,
and performs tarball overlay updates for installed deployments.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

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

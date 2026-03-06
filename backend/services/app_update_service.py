"""
App Update Service — Handles version checking and in-place updates.

Detects deployment mode, checks GitHub releases for newer versions,
and performs tarball overlay updates for installed deployments.
"""

import os


class AppUpdateService:
    """Service for managing application version detection and updates."""

    def __init__(self):
        """Initialize the update service with the current app version.

        Imports APP_VERSION from backend.consumer module and stores it as an instance attribute.
        """
        from ..consumer import APP_VERSION
        self.app_version = APP_VERSION

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

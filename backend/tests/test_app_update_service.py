# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Unit tests for AppUpdateService — in-place update system.

Tests cover:
  - Version parsing and comparison
  - Deployment mode detection (dev, docker, installed)
  - Update checking with cache fallback
  - Update application with mode rejection and concurrency guard
  - Database backup and cleanup
  - VERSION file reading
"""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from services.app_update_service import AppUpdateService


@pytest.mark.unit
class TestAppUpdateService:

    def test_parse_version_standard(self):
        svc = AppUpdateService()
        assert svc.parse_version("v0.2.0") == (0, 2, 0)

    def test_parse_version_no_prefix(self):
        svc = AppUpdateService()
        assert svc.parse_version("1.0.1") == (1, 0, 1)

    def test_parse_version_large_numbers(self):
        svc = AppUpdateService()
        assert svc.parse_version("v10.20.30") == (10, 20, 30)

    def test_parse_version_extra_segments(self):
        svc = AppUpdateService()
        # Extra segments should be included
        assert svc.parse_version("v1.2.3.4") == (1, 2, 3, 4)

    def test_parse_version_non_numeric(self):
        svc = AppUpdateService()
        # Non-numeric segments become 0
        assert svc.parse_version("v1.beta.3") == (1, 0, 3)

    def test_version_comparison(self):
        svc = AppUpdateService()
        assert svc.parse_version("v1.0.1") > svc.parse_version("v1.0.0")
        assert svc.parse_version("v2.0.0") > svc.parse_version("v1.99.99")
        assert svc.parse_version("v0.2.0") == svc.parse_version("0.2.0")
        assert svc.parse_version("v0.2.0") < svc.parse_version("v0.3.0")

    @patch('services.app_update_service.os.path.exists', return_value=False)
    def test_detect_deployment_mode_dev(self, _mock_exists, tmp_path):
        """When .git/ exists in app root, mode is dev."""
        (tmp_path / ".git").mkdir()
        with patch('services.app_update_service.APP_ROOT', tmp_path):
            assert AppUpdateService.detect_deployment_mode() == "dev"

    @patch('services.app_update_service.os.path.exists', return_value=True)
    def test_detect_deployment_mode_docker(self, _mock_exists):
        """When /.dockerenv exists, mode is docker."""
        assert AppUpdateService.detect_deployment_mode() == "docker"

    @patch('services.app_update_service.os.path.exists', return_value=False)
    def test_detect_deployment_mode_installed(self, _mock_exists, tmp_path):
        """When neither .git nor /.dockerenv exist, mode is installed."""
        # tmp_path has no .git/ directory
        with patch('services.app_update_service.APP_ROOT', tmp_path):
            assert AppUpdateService.detect_deployment_mode() == "installed"

    @patch('services.app_update_service.MemoryClientService')
    def test_check_for_update_cached(self, mock_mem):
        """Returns cached result when available in MemoryStore."""
        cached = json.dumps({
            "current_version": "0.2.0",
            "latest_version": "1.0.0",
            "latest_tag": "v1.0.0",
            "update_available": True,
            "release_notes": "cached notes",
            "release_url": "",
            "deployment_mode": "installed",
            "checked_at": "2026-03-13T00:00:00+00:00"
        })
        mock_store = MagicMock()
        mock_store.get.return_value = cached
        mock_mem.create_connection.return_value = mock_store

        svc = AppUpdateService()
        result = svc.check_for_update()
        assert result["update_available"] is True
        assert result["latest_tag"] == "v1.0.0"

    @patch('services.app_update_service.MemoryClientService')
    @patch('services.app_update_service.urlopen')
    def test_check_for_update_new_version(self, mock_urlopen, mock_mem):
        """Detects when a newer version is available (cache miss)."""
        mock_store = MagicMock()
        mock_store.get.return_value = None  # no cache
        mock_mem.create_connection.return_value = mock_store

        release_data = json.dumps({
            "tag_name": "v1.0.0",
            "body": "Release notes here",
            "html_url": "https://github.com/chalie-ai/chalie/releases/tag/v1.0.0"
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = release_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        svc = AppUpdateService()
        with patch.object(svc, 'get_current_version', return_value='0.2.0'):
            result = svc.check_for_update()
        assert result["update_available"] is True
        assert result["latest_tag"] == "v1.0.0"

    @patch('services.app_update_service.MemoryClientService')
    @patch('services.app_update_service.urlopen')
    def test_check_for_update_same_version(self, mock_urlopen, mock_mem):
        """No update when versions match (cache miss)."""
        mock_store = MagicMock()
        mock_store.get.return_value = None  # no cache
        mock_mem.create_connection.return_value = mock_store

        release_data = json.dumps({
            "tag_name": "v0.2.0",
            "body": "",
            "html_url": "https://github.com/chalie-ai/chalie/releases/tag/v0.2.0"
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = release_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        svc = AppUpdateService()
        with patch.object(svc, 'get_current_version', return_value='0.2.0'):
            result = svc.check_for_update()
        assert result["update_available"] is False

    @patch('services.app_update_service.MemoryClientService')
    @patch('services.app_update_service.urlopen')
    def test_check_for_update_network_failure_uses_cache(self, mock_urlopen, mock_mem):
        """Falls back to cached result on network failure (cache expired, re-fetch fails)."""
        from urllib.error import URLError

        cached = json.dumps({
            "current_version": "0.2.0",
            "latest_version": "0.2.0",
            "update_available": False,
            "checked_at": "2026-03-13T00:00:00+00:00"
        })
        mock_store = MagicMock()
        # First call (cache-first check): None → triggers API call
        # Second call (error fallback): returns stale cached result
        mock_store.get.side_effect = [None, cached]
        mock_mem.create_connection.return_value = mock_store

        mock_urlopen.side_effect = URLError("Network down")

        svc = AppUpdateService()
        with patch.object(svc, 'get_current_version', return_value='0.2.0'):
            result = svc.check_for_update()
        assert result["update_available"] is False
        assert result["current_version"] == "0.2.0"

    @patch('services.app_update_service.MemoryClientService')
    def test_apply_update_rejected_docker(self, mock_mem):
        """Docker mode rejects in-place updates."""
        mock_store = MagicMock()
        mock_mem.create_connection.return_value = mock_store

        svc = AppUpdateService()
        with patch.object(svc, 'detect_deployment_mode', return_value='docker'):
            result = svc.apply_update("v1.0.0")
        assert result["ok"] is False
        assert result["deployment_mode"] == "docker"

    @patch('services.app_update_service.MemoryClientService')
    def test_apply_update_rejected_dev(self, mock_mem):
        """Dev mode rejects in-place updates."""
        mock_store = MagicMock()
        mock_mem.create_connection.return_value = mock_store

        svc = AppUpdateService()
        with patch.object(svc, 'detect_deployment_mode', return_value='dev'):
            result = svc.apply_update("v1.0.0")
        assert result["ok"] is False
        assert result["deployment_mode"] == "dev"

    @patch('services.app_update_service.MemoryClientService')
    def test_concurrent_update_blocked(self, mock_mem):
        """Second update call blocked while first is in progress."""
        mock_store = MagicMock()
        mock_store.get.return_value = "1"  # in_progress flag set
        mock_mem.create_connection.return_value = mock_store

        svc = AppUpdateService()
        with patch.object(svc, 'detect_deployment_mode', return_value='installed'):
            result = svc.apply_update("v1.0.0")
        assert result["ok"] is False
        assert "in progress" in result["message"].lower()

    def test_backup_database(self, tmp_path):
        """Creates backup and cleans old ones."""
        # Setup fake data dir
        data_dir = tmp_path / "backend" / "data"
        data_dir.mkdir(parents=True)
        db_file = data_dir / "chalie.db"
        db_file.write_text("fake db content")

        # Create an old backup
        old_backup = data_dir / "chalie.db.pre-v0.1.0"
        old_backup.write_text("old backup")

        with patch('services.app_update_service.APP_ROOT', tmp_path), \
             patch.object(AppUpdateService, 'get_current_version', return_value='0.2.0'):
            backup_path = AppUpdateService.backup_database()

        assert backup_path.exists()
        assert backup_path.name == "chalie.db.pre-0.2.0"
        assert backup_path.read_text() == "fake db content"
        # Old backup should be cleaned
        assert not old_backup.exists()

    def test_get_current_version(self, tmp_path):
        """Reads version from VERSION file."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("1.0.1\n")

        with patch('services.app_update_service.APP_ROOT', tmp_path):
            assert AppUpdateService.get_current_version() == "1.0.1"

    def test_get_current_version_missing_file(self, tmp_path):
        """Falls back to 0.0.0 when VERSION file missing."""
        with patch('services.app_update_service.APP_ROOT', tmp_path):
            assert AppUpdateService.get_current_version() == "0.0.0"

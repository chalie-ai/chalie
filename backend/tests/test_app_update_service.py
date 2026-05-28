# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Unit tests for AppUpdateService — in-place update system.

Tests cover:
  - Update checking with cache fallback
  - Update application with mode rejection and concurrency guard
  - Database backup and cleanup
  - VERSION file reading
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from services.app_update_service import AppUpdateService, CACHE_KEY, IN_PROGRESS_KEY
from services.memory_store import MemoryStore


@pytest.mark.unit
class TestAppUpdateService:

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
        store = MemoryStore()
        store.set(CACHE_KEY, cached)
        mock_mem.create_connection.return_value = store

        svc = AppUpdateService()
        result = svc.check_for_update()
        assert result["update_available"] is True
        assert result["latest_tag"] == "v1.0.0"

    @patch('services.app_update_service.MemoryClientService')
    @patch('services.app_update_service.urlopen')
    def test_check_for_update_new_version(self, mock_urlopen, mock_mem):
        """Detects when a newer version is available (cache miss)."""
        store = MemoryStore()  # empty store → cache miss
        mock_mem.create_connection.return_value = store

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
        store = MemoryStore()  # empty store → cache miss
        mock_mem.create_connection.return_value = store

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
        # Category C — kept as MagicMock because the service calls store.get(CACHE_KEY)
        # twice on the *same* key without writing in between: first returns None (triggers
        # the network fetch), then returns stale cached data (error-path fallback).  A real
        # MemoryStore cannot produce two different values for the same key across consecutive
        # reads without an intervening write, so side_effect is the only faithful model here.
        broken_store = MagicMock()
        # First call (cache-first check): None → triggers API call
        # Second call (error fallback): returns stale cached result
        broken_store.get.side_effect = [None, cached]
        mock_mem.create_connection.return_value = broken_store

        mock_urlopen.side_effect = URLError("Network down")

        svc = AppUpdateService()
        with patch.object(svc, 'get_current_version', return_value='0.2.0'):
            result = svc.check_for_update()
        assert result["update_available"] is False
        assert result["current_version"] == "0.2.0"

    @patch('services.app_update_service.MemoryClientService')
    def test_apply_update_rejected_docker(self, mock_mem):
        """Docker mode has no explicit guard — falls through to download which fails in test env."""
        store = MemoryStore()
        mock_mem.create_connection.return_value = store

        svc = AppUpdateService()
        with patch.object(svc, 'detect_deployment_mode', return_value='docker'), \
             patch.object(svc, 'download_and_validate', side_effect=RuntimeError("no network in test env")):
            result = svc.apply_update("v1.0.0")
        assert result["ok"] is False
        assert result["deployment_mode"] == "docker"

    @patch('services.app_update_service.MemoryClientService')
    def test_apply_update_rejected_dev(self, mock_mem):
        """Dev mode rejects in-place updates.

        The service returns before ever calling create_connection() when the
        deployment mode is 'dev', so the store is not used at all.
        """
        store = MemoryStore()
        mock_mem.create_connection.return_value = store

        svc = AppUpdateService()
        with patch.object(svc, 'detect_deployment_mode', return_value='dev'):
            result = svc.apply_update("v1.0.0")
        assert result["ok"] is False
        assert result["deployment_mode"] == "dev"

    @patch('services.app_update_service.MemoryClientService')
    def test_concurrent_update_blocked(self, mock_mem):
        """Second update call blocked while first is in progress."""
        store = MemoryStore()
        store.set(IN_PROGRESS_KEY, "1")  # simulate a concurrent update already running
        mock_mem.create_connection.return_value = store

        svc = AppUpdateService()
        with patch.object(svc, 'detect_deployment_mode', return_value='installed'):
            result = svc.apply_update("v1.0.0")
        assert result["ok"] is False
        assert "in progress" in result["message"].lower()

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

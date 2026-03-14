# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
App Update Worker — checks for new Chalie releases every 6 hours.

When an update is found, publishes an ``app_update`` event to the
``output:events`` pub/sub channel so the frontend can display a banner.
"""

import json
import time
import logging

logger = logging.getLogger(__name__)


def app_update_worker(shared_state=None):
    """Background worker — checks for new Chalie releases every 6 hours.

    On finding an update, publishes an ``app_update`` event to the
    ``output:events`` pub/sub channel so the frontend can show a banner.
    """
    time.sleep(300)  # 5min initial delay — let startup settle

    while True:
        try:
            from services.app_update_service import AppUpdateService
            info = AppUpdateService().check_for_update()
            if info.get('update_available'):
                _push_update_event(info)
                logger.info(
                    "[APP_UPDATE] New version available: %s",
                    info.get('latest_tag'),
                )
        except Exception as e:
            logger.debug("[APP_UPDATE] Check failed: %s", e)
        time.sleep(6 * 3600)  # 6 hours


def _push_update_event(info):
    """Publish update notification to output:events for WebSocket delivery."""
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        event = {
            "type": "app_update",
            "current_version": info.get("current_version"),
            "latest_version": info.get("latest_version"),
            "latest_tag": info.get("latest_tag"),
            "release_notes": info.get("release_notes", ""),
            "release_url": info.get("release_url", ""),
            "deployment_mode": info.get("deployment_mode", "installed"),
        }
        store.publish("output:events", json.dumps(event))
    except Exception as e:
        logger.warning("[APP_UPDATE] Failed to push event: %s", e)

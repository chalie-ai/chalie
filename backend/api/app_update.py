"""
App Update API Blueprint — Exposes update service functionality via REST endpoints.

Endpoints:
- GET /api/v1/update/check: Check for available updates
- POST /api/v1/update/install: Trigger an update (background task)
"""

import logging
import threading
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

app_update_bp = Blueprint('app_update', __name__, url_prefix='/api/v1/update')


def _run_update_in_background():
    """Run the update process in a background thread.

    Executes AppUpdateService.perform_update() asynchronously to allow
    the API request to return immediately with 202 Accepted status.
    
    Logs success or failure of the update operation for debugging purposes.
    """
    try:
        from ..services.app_update_service import AppUpdateService
        
        service = AppUpdateService()
        
        result = service.perform_update()
        
        if result.get('success'):
            logger.info(f"[UPDATE] Update completed successfully: {result}")
        else:
            logger.error(f"[UPDATE] Update failed: {result}")
            
    except Exception as e:
        logger.exception(f"[UPDATE] Background update thread error: {e}")


@app_update_bp.route('/check', methods=['GET'])
@require_session
def check_for_updates():
    """Check for available updates.

    Calls AppUpdateService.check_for_updates() and returns the current version,
    latest version (if installed mode), and whether an update is available.
    
    Returns:
        JSON response with:
            - update_available (bool): True if a newer version exists
            - latest_version (str): The latest release version tag
            - current_version (str): The currently running version
            - deployment_mode (str): Current deployment mode ('docker', 'installed', or 'dev')
    
    Status Codes:
        200: Success
        500: Internal server error during update check
    """
    try:
        from ..services.app_update_service import AppUpdateService
        
        service = AppUpdateService()
        
        result = service.check_for_updates()
        
        return jsonify({
            "update_available": result.get("update_available", False),
            "latest_version": result.get("latest_version", service.app_version),
            "current_version": service.app_version,
            "deployment_mode": service.get_deployment_mode()
        }), 200
        
    except Exception as e:
        logger.error(f"[UPDATE] Check for updates error: {e}")
        return jsonify({
            "error": str(e),
            "update_available": False,
            "current_version": 'unknown'
        }), 500


@app_update_bp.route('/install', methods=['POST'])
@require_session
def install_update():
    """Trigger an update installation.

    Calls AppUpdateService.perform_update() as a background task so the API request
    returns immediately with 202 Accepted status. The actual update runs asynchronously.
    
    Returns:
        JSON response with:
            - status (str): 'accepted' if update initiated, 'no_update' if none available
            - message (str): Descriptive message about the operation
    
    Status Codes:
        200: No update available or success
        202: Update accepted and running in background
        500: Internal server error during install request
    """
    try:
        from ..services.app_update_service import AppUpdateService
        
        service = AppUpdateService()
        
        # Check if update is available first
        check_result = service.check_for_updates()
        
        if not check_result.get("update_available", False):
            return jsonify({
                "status": "no_update",
                "message": "No update available"
            }), 200
        
        # Start background thread for the update
        update_thread = threading.Thread(
            target=_run_update_in_background,
            daemon=True
        )
        update_thread.start()
        
        logger.info(f"[UPDATE] Update initiated in background thread")
        
        return jsonify({
            "status": "accepted",
            "message": "Update initiated"
        }), 202
        
    except Exception as e:
        logger.error(f"[UPDATE] Install update error: {e}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

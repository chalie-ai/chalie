"""
App Update API Blueprint — Exposes update service functionality via REST endpoints.

Endpoints:
- GET /api/update/check: Check for available updates
- POST /api/update/install: Trigger an update (background task)
"""

import logging
import threading
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

app_update_bp = Blueprint('app_update', __name__, url_prefix='/api/update')


def _run_update_in_background():
    """Run the update process in a background thread."""
    try:
        from services.app_update_service import AppUpdateService
        
        service = AppUpdateService()
        
        # Get current version from environment or default
        current_version = request.environ.get('CURRENT_VERSION', '0.0.0')
        
        result = service.perform_update(current_version)
        
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

    Returns the current version, latest version (if installed mode), and whether an update is available.
    
    Response format:
        {
            "update_available": bool,
            "latest_version": str,
            "current_version": str,
            "deployment_mode": str
        }
    """
    try:
        from services.app_update_service import AppUpdateService
        
        service = AppUpdateService()
        
        # Get current version from environment or default
        current_version = request.environ.get('CURRENT_VERSION', '0.0.0')
        
        result = service.check_for_updates(current_version)
        
        return jsonify({
            "update_available": result.get("update_available", False),
            "latest_version": result.get("latest_version", current_version),
            "current_version": current_version,
            "deployment_mode": service.get_deployment_mode()
        }), 200
        
    except Exception as e:
        logger.error(f"[UPDATE] Check for updates error: {e}")
        return jsonify({
            "error": str(e),
            "update_available": False,
            "current_version": request.environ.get('CURRENT_VERSION', 'unknown')
        }), 500


@app_update_bp.route('/install', methods=['POST'])
@require_session
def install_update():
    """Trigger an update installation.

    Runs the update process in a background thread and returns immediately with 202 Accepted.
    
    Response:
        {
            "status": "accepted",
            "message": "Update initiated"
        }
    """
    try:
        from services.app_update_service import AppUpdateService
        
        service = AppUpdateService()
        
        # Check if update is available first
        current_version = request.environ.get('CURRENT_VERSION', '0.0.0')
        check_result = service.check_for_updates(current_version)
        
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

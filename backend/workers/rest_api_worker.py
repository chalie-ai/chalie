"""
REST API Worker - Flask server entry point.

Spawns the Flask app from the api/ package and runs it on the configured host/port.
"""

import sys
import logging
from services.config_service import ConfigService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rest_api_worker(shared_state=None):
    """
    Main entry point for REST API worker.

    Can be run standalone: python -m workers.rest_api_worker
    Or integrated into consumer.py via multiprocessing.

    Args:
        shared_state: Shared state dict from multiprocessing.Manager (optional)
    """
    try:
        logger.info("[REST API] Starting REST API worker...")

        # Load configuration
        config = ConfigService.connections()
        api_config = config.get("rest_api", {})

        host = api_config.get('host', '0.0.0.0')
        port = api_config.get('port', 8080)

        logger.info(f"[REST API] Starting Flask server on {host}:{port}")

        # Ensure migrations are run before starting the server
        from services.database_service import get_shared_db_service
        try:
            db = get_shared_db_service()
            db.run_pending_migrations()
        except Exception as e:
            logger.warning(f"[REST API] Migration warning: {e}")

        # Create Flask app from api package (avoids pickling issues)
        from api import create_app
        app = create_app()

        # Run Flask app
        app.run(host=host, port=port, debug=False, threaded=True)

    except KeyboardInterrupt:
        logger.info("[REST API] Shutting down...")
    except Exception as e:
        logger.error(f"[REST API] Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    rest_api_worker()

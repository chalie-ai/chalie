"""
REST API Worker - Flask server entry point.

Spawns the Flask app from the api/ package and runs it on the configured host/port.
"""

import logging
import sys
from typing import cast

from runtime_config import RuntimeConfig
from utils.logger import Logger

Logger.start()
logger = logging.getLogger(__name__)


class RestApiWorker:
    """Flask server entry point.

    Can be run standalone: python -m workers.rest_api_worker
    Or integrated into run.py as a daemon thread.
    """

    @classmethod
    def run(cls) -> None:
        logger.info("[REST API] Starting REST API worker...")

        host = cast("str", RuntimeConfig.get("host", "0.0.0.0"))
        port = cast("int", RuntimeConfig.get("port", 31025))

        logger.info(f"[REST API] Starting Flask server on {host}:{port}")

        # Create Flask app from api package (avoids pickling issues)
        from api import create_app
        app = create_app()

        # Run Flask app
        app.run(host=host, port=port, debug=False, threaded=True)

    @staticmethod
    def main() -> None:
        try:
            RestApiWorker.run()
        except KeyboardInterrupt:
            logger.info("[REST API] Shutting down...")
        except Exception as e:
            logger.exception(f"[REST API] Fatal error: {e}")
            sys.exit(1)

"""
Dashboard Flask Server — gateway + app launcher + Chalie proxy.

Separate process from Chalie's backend. Serves the chat UI, acts as the
gateway for interface daemons, and proxies Chalie API requests.

Usage:
    python frontend/server/app.py --port=3000 --chalie=http://localhost:8081

The dashboard:
    1. Serves the chat UI static files (frontend/interface/)
    2. Proxies /api/* requests to Chalie (transparent to the browser)
    3. Exposes /register for daemon self-registration
    4. Exposes /gw/<id>/* gateway routes for daemon communication
    5. Exposes /api/apps/* for interface management from the UI
"""

import argparse
import logging
import os
import sys

from flask import Flask, send_from_directory, request, Response

import requests as http_requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard")


def create_app(chalie_url: str, port: int, data_dir: str) -> Flask:
    """Create and configure the dashboard Flask app."""

    app = Flask(__name__, static_folder=None)
    app.config["CHALIE_URL"] = chalie_url
    app.config["DASHBOARD_PORT"] = port

    # ── Database ───────────────────────────────────────────────────────────

    from . import db
    db_path = os.path.join(data_dir, "dashboard.db")
    db.init_db(db_path)

    # ── Chalie client ──────────────────────────────────────────────────────

    from .chalie_client import ChalieClient

    boot_token = db.get_config("boot_token")
    client = ChalieClient(chalie_url, token=boot_token)
    app.config["CHALIE_CLIENT"] = client

    # ── Register blueprints ────────────────────────────────────────────────

    from .gateway import gateway_bp, init_gateway
    from .interfaces import interfaces_bp, init_interfaces

    init_gateway(client, port, data_dir)
    init_interfaces(client, port, data_dir)

    app.register_blueprint(gateway_bp)
    app.register_blueprint(interfaces_bp)

    # ── Static file serving (chat UI) ──────────────────────────────────────

    _frontend_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "interface")
    )
    _shared_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "shared")
    )

    @app.route("/", methods=["GET"])
    def index():
        return send_from_directory(_frontend_dir, "index.html")

    @app.route("/<path:filename>", methods=["GET"])
    def static_files(filename):
        # Try interface dir first, then shared
        filepath = os.path.join(_frontend_dir, filename)
        if os.path.isfile(filepath):
            return send_from_directory(_frontend_dir, filename)
        # Try shared dir
        shared_path = os.path.join(_shared_dir, filename)
        if os.path.isfile(shared_path):
            return send_from_directory(_shared_dir, filename)
        # Fall through to Chalie proxy for /api/* etc.
        return _proxy_to_chalie()

    # ── Chalie API proxy ───────────────────────────────────────────────────
    # Transparently forward /api/* requests to Chalie so the browser JS
    # works identically whether talking to Chalie directly or via dashboard.

    @app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def proxy_api(path):
        # Don't proxy dashboard-owned routes
        if path.startswith("apps"):
            return  # Let the interfaces blueprint handle it
        return _proxy_to_chalie()

    @app.route("/ws", methods=["GET"])
    def proxy_ws():
        """WebSocket proxy — redirect to Chalie's WS endpoint."""
        # Flask can't proxy WebSockets natively. The chat UI JS will
        # need to connect to Chalie's WS directly. We return the URL.
        return Response(status=426)  # Upgrade required

    def _proxy_to_chalie():
        """Forward the current request to Chalie's backend."""
        target = f"{chalie_url}{request.full_path}"
        try:
            resp = http_requests.request(
                method=request.method,
                url=target,
                headers={k: v for k, v in request.headers if k.lower() != "host"},
                data=request.get_data(),
                cookies=request.cookies,
                allow_redirects=False,
                timeout=30,
            )
            # Build response, preserving headers and cookies
            excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
            headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in excluded_headers
            }
            return Response(resp.content, resp.status_code, headers)
        except Exception as e:
            logger.error("[Proxy] Failed to reach Chalie at %s: %s", target, e)
            return Response(
                f'{{"error": "Chalie backend unreachable: {e}"}}',
                502,
                content_type="application/json",
            )

    # ── Shared static assets ───────────────────────────────────────────────

    @app.route("/shared/<path:filename>", methods=["GET"])
    def shared_static(filename):
        return send_from_directory(_shared_dir, filename)

    return app


def main():
    parser = argparse.ArgumentParser(description="Chalie Dashboard Server")
    parser.add_argument("--port", type=int, default=3000, help="Dashboard port")
    parser.add_argument("--chalie", default="http://localhost:8081", help="Chalie backend URL")
    parser.add_argument("--data-dir", default=None, help="Data directory for dashboard.db")
    args = parser.parse_args()

    # Default data dir: alongside the server code
    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    app = create_app(args.chalie, args.port, data_dir)

    logger.info("Dashboard server starting on port %d → Chalie at %s", args.port, args.chalie)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    # Allow running as: python -m frontend.server.app or python frontend/server/app.py
    if __package__ is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        __package__ = "server"
    main()

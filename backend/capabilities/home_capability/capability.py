from __future__ import annotations

import logging

import yaml

from capabilities.base import AbstractCapability
from services.file_mapper_service import FileMapperService
from capabilities.home_capability import ha_rest_handler as rest
from capabilities.home_capability.ha_ws_handler import HaWebSocketHandler

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("home_capability", "manifest.yaml")

_K_URL = "home:url"
_K_TOKEN = "home:token"
_K_VERIFY_SSL = "home:verify_ssl"


class HomeCapability(AbstractCapability):

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict | None = None
        self._ws_handler = HaWebSocketHandler()
        self._url: str = ""
        self._token: str = ""
        self._verify_ssl: bool = True

    # ── Identity ─────────────────────────────────────────────────────

    def get_id(self) -> str:
        return "home"

    def get_manifest(self) -> dict:
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, encoding="utf-8") as fh:
                self._manifest_cache = yaml.safe_load(fh)
        return self._manifest_cache

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict) -> None:
        url = (credentials.get("url") or "").strip().rstrip("/")
        token = (credentials.get("token") or "").strip()
        if not url:
            raise ValueError("Home Assistant URL is required")
        if not token:
            raise ValueError("Long-Lived Access Token is required")

        verify_ssl = credentials.get("verify_ssl", True)
        if isinstance(verify_ssl, str):
            verify_ssl = verify_ssl.lower() not in ("0", "false", "no")

        try:
            rest.probe(url, token, verify_ssl)
        except Exception as exc:
            raise ValueError(f"[home] Connection probe failed: {exc}") from exc

        self.store_credential(_K_URL, url)
        self.store_credential(_K_TOKEN, token)
        self.store_credential(_K_VERIFY_SSL, "1" if verify_ssl else "0")

        self._url = url
        self._token = token
        self._verify_ssl = verify_ssl
        self.connect()

    def connect(self) -> bool:
        url = self.load_credential(_K_URL)
        token = self.load_credential(_K_TOKEN)
        ssl_raw = self.load_credential(_K_VERIFY_SSL)
        if not url or not token:
            self._connected = False
            return False

        self._url = url
        self._token = token
        self._verify_ssl = ssl_raw != "0"

        try:
            rest.probe(self._url, self._token, self._verify_ssl)
            self._connected = True
        except Exception as exc:
            logger.warning("[home] connect probe failed: %s", exc)
            self._connected = False
        return self._connected

    def disconnect(self) -> None:
        """Stop WebSocket handler, clear connection state, and remove credentials."""
        self._connected = False
        self._ws_handler.stop()
        self.delete_credentials()
        logger.info("[home] Disconnected and credentials removed.")

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list:
        if not self.is_connected():
            return []
        try:
            return rest.list_devices(
                self._url, self._token, self._verify_ssl, limit=200
            ).get("devices", [])
        except Exception as exc:
            logger.warning("[home] ingest failed: %s", exc)
            return []

    def understand(self, items: list) -> list:
        return items

    def _do_monitor(self) -> None:
        if self._ws_handler.is_alive:
            return
        rest.probe(self._url, self._token, self._verify_ssl)

    def act(self, action: str, params: dict) -> dict:
        tool_map = {t["name"]: t["handler"] for t in self.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            return {"error": f"Unknown action: {action}"}
        return handler(topic="", params=params)

    # ── Tools ────────────────────────────────────────────────────────

    def _require_connection(self) -> dict | None:
        """Return an error dict when not connected, else None."""
        if not self.is_connected():
            return {"error": "Home capability not connected. Configure it in the Brain dashboard."}
        return None

    def _tool_list_devices(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        return rest.list_devices(
            self._url, self._token, self._verify_ssl,
            domain=params.get("domain"),
            area=params.get("area"),
            limit=int(params.get("limit", 50)),
        )

    def _tool_get_state(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        eid = params.get("entity_id")
        if not eid:
            return {"error": "entity_id is required"}
        return rest.get_state(self._url, self._token, self._verify_ssl, eid)

    def _tool_control(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        eid = params.get("entity_id")
        svc = params.get("service")
        if not eid or not svc:
            return {"error": "entity_id and service are required"}
        return rest.control(
            self._url, self._token, self._verify_ssl,
            eid, svc, params.get("service_data"),
        )

    def _tool_list_automations(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        return rest.list_automations(self._url, self._token, self._verify_ssl)

    def _tool_trigger_automation(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        aid = params.get("automation_id")
        if not aid:
            return {"error": "automation_id is required"}
        return rest.trigger_automation(self._url, self._token, self._verify_ssl, aid)

    def _tool_subscribe_events(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        eid = params.get("entity_id")
        if not eid:
            return {"error": "entity_id is required"}
        ws_url = self._url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/api/websocket"
        self._ws_handler.start(ws_url, self._token)
        self._ws_handler.subscribe(eid)
        return {"status": "subscribed", "entity_id": eid}

    def get_tools(self) -> list:
        return [
            {
                "name": "list_devices",
                "description": "List smart home entities, optionally filtered by domain or area.",
                "parameters": {},
                "handler": self._tool_list_devices,
                "timeout": 30,
            },
            {
                "name": "get_state",
                "description": "Get the current state and attributes of a single entity.",
                "parameters": {},
                "handler": self._tool_get_state,
                "timeout": 15,
            },
            {
                "name": "control",
                "description": "Call a service on an entity (e.g. turn_on, turn_off).",
                "parameters": {},
                "handler": self._tool_control,
                "timeout": 15,
            },
            {
                "name": "list_automations",
                "description": "List all automation entities with their enabled state.",
                "parameters": {},
                "handler": self._tool_list_automations,
                "timeout": 30,
            },
            {
                "name": "trigger_automation",
                "description": "Manually trigger an automation by automation_id.",
                "parameters": {},
                "handler": self._tool_trigger_automation,
                "timeout": 15,
            },
            {
                "name": "subscribe_events",
                "description": "Subscribe to real-time state_changed events for an entity.",
                "parameters": {},
                "handler": self._tool_subscribe_events,
                "timeout": 15,
            },
        ]

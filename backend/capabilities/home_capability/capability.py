"""HomeCapability -- Home Assistant integration via REST + WebSocket.

REST for commands/queries (list_devices, get_state, control, list_automations,
trigger_automation). Persistent WebSocket for subscribe_events and efficient
_do_monitor() liveness checks.

Credential storage: home:url, home:token, home:verify_ssl (encrypted via
VaultService in tool_configs).
"""

from __future__ import annotations

import logging
import pathlib

import yaml

from capabilities.base import AbstractCapability
from capabilities.home_capability import ha_rest_handler as rest
from capabilities.home_capability.ha_ws_handler import HaWebSocketHandler

logger = logging.getLogger(__name__)

_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

_K_URL = "home:url"
_K_TOKEN = "home:token"
_K_VERIFY_SSL = "home:verify_ssl"


class HomeCapability(AbstractCapability):
    """Home Assistant capability.

    Attributes:
        _url (str): Home Assistant base URL, e.g. ``http://homeassistant.local:8123``.
        _token (str): Long-Lived Access Token.
        _verify_ssl (bool): Whether to verify SSL certificates on HTTPS connections.
    """

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
        """Validate credentials, probe HA REST API, and persist.

        Args:
            credentials: Must contain ``url`` (str) and ``token`` (str).
                Optional ``verify_ssl`` (bool, default True).

        Raises:
            ValueError: If url or token is missing, or the HA instance rejects them.
        """
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
        """Load stored credentials and probe the HA REST API.

        Returns:
            bool: ``True`` if the probe succeeds, ``False`` otherwise.
        """
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
        """Fetch all entity states from HA REST API.

        Returns:
            list[dict]: Entity summary dicts, or ``[]`` when not connected.
        """
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
        """Probe HA REST API unless a WebSocket connection is already alive."""
        if self._ws_handler.is_alive:
            return
        rest.probe(self._url, self._token, self._verify_ssl)

    def act(self, action: str, params: dict) -> dict:
        """Dispatch a write action to the matching tool handler.

        Args:
            action: Tool name, e.g. ``"control"``, ``"trigger_automation"``.
            params: Action-specific parameters.

        Returns:
            dict: Handler result, or ``{"error": ...}`` for unknown actions.
        """
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

    def _resolve_automation_id(self, value: str) -> dict:
        """Resolve a friendly-name substring to a HA automation entity_id.

        Returns a dict with either ``entity_id`` (resolved) or ``error``.
        When ``value`` already looks like an entity_id (starts with
        ``automation.``), it is returned as-is without a network call.
        """
        if value.startswith("automation."):
            return {"entity_id": value}

        result = rest.list_automations(self._url, self._token, self._verify_ssl)
        automations = result.get("automations", [])
        needle = value.lower()
        matches = [
            a for a in automations
            if needle in a.get("friendly_name", "").lower()
            or needle == a["entity_id"].lower()
        ]

        if len(matches) == 0:
            return {"error": f"No automation found matching '{value}'"}
        if len(matches) > 1:
            names = [m.get("friendly_name", m["entity_id"]) for m in matches]
            return {
                "error": (
                    f"Multiple automations match '{value}': {names}. "
                    "Use the entity_id instead."
                )
            }
        return {"entity_id": matches[0]["entity_id"], "resolved_from": value}

    def _tool_trigger_automation(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        aid = (params.get("automation_id") or "").strip()
        if not aid:
            return {"error": "automation_id is required"}

        resolved = self._resolve_automation_id(aid)
        if "error" in resolved:
            return resolved

        result = rest.trigger_automation(
            self._url, self._token, self._verify_ssl, resolved["entity_id"]
        )
        if "resolved_from" in resolved:
            result["resolved_from"] = resolved["resolved_from"]
        return result

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
        """Return tool definitions for all 6 home actions.

        Returns:
            list[dict]: Tool dicts with ``name``, ``description``, ``parameters``,
            ``handler``, and ``timeout`` keys.
        """
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

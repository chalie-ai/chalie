"""UbiquitiCapability -- UniFi network controller integration via REST.

Provides control and monitoring of UniFi network devices, connected clients,
WiFi networks, port forwarding rules, and traffic management rules.

Supports two authentication methods:
- API Key (UniFi OS): Uses the X-API-Key header.
- Username/Password: Session-cookie authentication.

Credential storage: ubiquiti:url, ubiquiti:auth_method, ubiquiti:api_key,
ubiquiti:username, ubiquiti:password, ubiquiti:site, ubiquiti:verify_ssl
(encrypted via VaultService in tool_configs).
"""

from __future__ import annotations

import logging
import pathlib

import yaml

from capabilities.base import AbstractCapability
from capabilities.ubiquiti_capability import unifi_rest_handler as rest

logger = logging.getLogger(__name__)

_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

_K_URL = "ubiquiti:url"
_K_AUTH_METHOD = "ubiquiti:auth_method"
_K_API_KEY = "ubiquiti:api_key"
_K_USERNAME = "ubiquiti:username"
_K_PASSWORD = "ubiquiti:password"
_K_SITE = "ubiquiti:site"
_K_VERIFY_SSL = "ubiquiti:verify_ssl"


class UbiquitiCapability(AbstractCapability):
    """Ubiquiti UniFi capability.

    Attributes:
        _url: UniFi controller base URL, e.g. ``https://192.168.1.1``.
        _auth_method: Either ``"api_key"`` or ``"credentials"``.
        _api_key: API key for UniFi OS authentication.
        _username: Username for credential-based authentication.
        _password: Password for credential-based authentication.
        _site: UniFi site name, defaults to ``"default"``.
        _verify_ssl: Whether to verify SSL certificates.
    """

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict | None = None
        self._url: str = ""
        self._auth_method: str = "api_key"
        self._api_key: str | None = None
        self._username: str | None = None
        self._password: str | None = None
        self._site: str = "default"
        self._verify_ssl: bool = False

    # ── Identity ─────────────────────────────────────────────────────

    def get_id(self) -> str:
        return "ubiquiti"

    def get_manifest(self) -> dict:
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, encoding="utf-8") as fh:
                self._manifest_cache = yaml.safe_load(fh)
        return self._manifest_cache

    # ── Auth helpers ─────────────────────────────────────────────────

    @property
    def _auth(self) -> dict:
        """Connection kwargs shared by every REST call."""
        return {
            "api_key": self._api_key,
            "username": self._username,
            "password": self._password,
            "verify_ssl": self._verify_ssl,
        }

    def _require_mac_cmd(self, params: dict, action_name: str) -> dict | None:
        """Validate that ``mac`` and ``command`` are present.

        Returns an error dict on failure, ``None`` on success.
        """
        if not params.get("mac"):
            return {"status": "error", "error": f"mac is required for {action_name}"}
        if not params.get("command"):
            return {"status": "error", "error": f"command is required for {action_name}"}
        return None

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict) -> None:
        """Validate credentials, probe the controller, and persist.

        Raises:
            ValueError: If required fields are missing or the controller
                rejects the provided credentials.
        """
        url = (credentials.get("url") or "").strip().rstrip("/")
        auth_method = (credentials.get("auth_method") or "api_key").strip()
        api_key = (credentials.get("api_key") or "").strip() or None
        username = (credentials.get("username") or "").strip() or None
        password = (credentials.get("password") or "").strip() or None
        site = (credentials.get("site") or "default").strip() or "default"

        verify_ssl = credentials.get("verify_ssl", False)
        if isinstance(verify_ssl, str):
            verify_ssl = verify_ssl.lower() not in ("0", "false", "no")

        if not url:
            raise ValueError("UniFi Controller URL is required")
        if auth_method == "api_key" and not api_key:
            raise ValueError("API Key is required when authentication method is 'api_key'")
        if auth_method == "credentials" and (not username or not password):
            raise ValueError(
                "Username and Password are required when authentication method is 'credentials'"
            )
        if auth_method not in ("api_key", "credentials"):
            raise ValueError(f"Unknown authentication method: {auth_method!r}")

        logger.info(
            "[ubiquiti] configure: url=%s auth=%s site=%s verify_ssl=%r (raw=%r)",
            url, auth_method, site, verify_ssl, credentials.get("verify_ssl"),
        )
        try:
            rest.probe(url, site, api_key=api_key, username=username,
                       password=password, verify_ssl=verify_ssl)
        except Exception as exc:
            logger.warning("[ubiquiti] probe error detail: %s %s", type(exc).__name__, exc)
            if hasattr(exc, "response") and exc.response is not None:
                logger.warning("[ubiquiti] response body: %s", exc.response.text[:500])
            raise ValueError(f"[ubiquiti] Connection probe failed: {exc}") from exc

        self.store_credential(_K_URL, url)
        self.store_credential(_K_AUTH_METHOD, auth_method)
        self.store_credential(_K_API_KEY, api_key or "")
        self.store_credential(_K_USERNAME, username or "")
        self.store_credential(_K_PASSWORD, password or "")
        self.store_credential(_K_SITE, site)
        self.store_credential(_K_VERIFY_SSL, "1" if verify_ssl else "0")

        self._url = url
        self._auth_method = auth_method
        self._api_key = api_key
        self._username = username
        self._password = password
        self._site = site
        self._verify_ssl = verify_ssl
        self._connected = True

    def connect(self) -> bool:
        """Load stored credentials and probe the UniFi controller."""
        url = self.load_credential(_K_URL)
        if not url:
            self._connected = False
            return False

        self._url = url
        self._auth_method = self.load_credential(_K_AUTH_METHOD) or "api_key"
        self._api_key = self.load_credential(_K_API_KEY) or None
        self._username = self.load_credential(_K_USERNAME) or None
        self._password = self.load_credential(_K_PASSWORD) or None
        self._site = self.load_credential(_K_SITE) or "default"
        self._verify_ssl = self.load_credential(_K_VERIFY_SSL) == "1"

        try:
            rest.probe(self._url, self._site, **self._auth)
            self._connected = True
        except Exception as exc:
            logger.warning("[ubiquiti] connect probe failed: %s", exc)
            self._connected = False
        return self._connected

    def disconnect(self) -> None:
        self._connected = False
        self.delete_credentials()
        logger.info("[ubiquiti] Disconnected and credentials removed.")

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list:
        if not self.is_connected():
            return []
        try:
            return rest.list_devices(self._url, self._site, **self._auth).get("devices", [])
        except Exception as exc:
            logger.warning("[ubiquiti] ingest failed: %s", exc)
            return []

    def understand(self, items: list) -> list:
        return items

    def _do_monitor(self) -> None:
        if not self._url:
            return
        rest.probe(self._url, self._site, **self._auth)

    def act(self, action: str, params: dict) -> dict:
        tool_map = {t["name"]: t["handler"] for t in self.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            return {"error": f"Unknown action: {action}"}
        return handler(topic="", params=params)

    # ── Tool handlers ────────────────────────────────────────────────

    def _require_connection(self) -> dict | None:
        if not self.is_connected():
            return {"status": "error", "error": "Ubiquiti controller not connected. Configure it in Brain → Capabilities."}
        return None

    def _th_list_devices(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        return rest.list_devices(self._url, self._site, **self._auth)

    def _th_list_clients(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        return rest.list_clients(self._url, self._site, **self._auth,
                                 active_only=params.get("active_only", True))

    def _th_get_info(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        return rest.get_info(self._url, self._site, params.get("target", "health"), **self._auth)

    def _th_control_client(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection() or self._require_mac_cmd(params, "control_client")
        if err:
            return err
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return rest.control_client(self._url, self._site, params["mac"], params["command"],
                                   **self._auth, **extra)

    def _th_control_device(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection() or self._require_mac_cmd(params, "control_device")
        if err:
            return err
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return rest.control_device(self._url, self._site, params["mac"], params["command"],
                                   **self._auth, **extra)

    def _th_manage_wlan(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return rest.list_wlans(self._url, self._site, **self._auth)
        if sub != "update":
            return {"status": "error", "error": f"manage_wlan supports 'list' and 'update', not '{sub}'"}
        return rest.update_wlan(self._url, self._site, params.get("wlan_id", ""),
                                params.get("updates", {}), **self._auth)

    def _th_manage_port_forward(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return rest.list_port_forwards(self._url, self._site, **self._auth)
        if sub == "create":
            return rest.create_port_forward(self._url, self._site, params.get("rule", {}), **self._auth)
        return rest.update_port_forward(self._url, self._site, params.get("rule_id", ""),
                                        params.get("updates", {}), **self._auth)

    def _th_manage_traffic_rule(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return rest.list_traffic_rules(self._url, self._site, **self._auth)
        if sub != "update":
            return {"status": "error", "error": f"manage_traffic_rule supports 'list' and 'update', not '{sub}'"}
        return rest.update_traffic_rule(self._url, self._site, params.get("rule_id", ""),
                                        params.get("updates", {}), **self._auth)

    def _th_authorize_guest(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        mac = params.get("mac", "")
        if not mac:
            return {"status": "error", "error": "mac is required for authorize_guest"}
        extra = {}
        if params.get("up_kbps") is not None:
            extra["up"] = params["up_kbps"]
        if params.get("down_kbps") is not None:
            extra["down"] = params["down_kbps"]
        if params.get("quota_mb") is not None:
            extra["bytes"] = params["quota_mb"]
        return rest.control_client(self._url, self._site, mac, "authorize-guest",
                                   **self._auth, minutes=params.get("minutes", 60), **extra)

    # ── Tool definitions ─────────────────────────────────────────────

    def get_tools(self) -> list:
        return [
            {"name": "list_devices", "handler": self._th_list_devices, "timeout": 15,
             "description": "List all UniFi network devices (APs, switches, gateways) with status.",
             "parameters": {}},
            {"name": "list_clients", "handler": self._th_list_clients, "timeout": 15,
             "description": "List devices connected to the network.",
             "parameters": {"active_only": {"type": "boolean", "description": "When true, return only currently connected clients.", "default": True}}},
            {"name": "get_info", "handler": self._th_get_info, "timeout": 15,
             "description": "Get detailed device info by MAC address or site health overview.",
             "parameters": {"target": {"type": "string", "description": 'Use "health" for a site health summary, or a MAC address (e.g. aa:bb:cc:dd:ee:ff) for a specific device.'}}},
            {"name": "control_client", "handler": self._th_control_client, "timeout": 15,
             "description": "Block, unblock, or disconnect a network client.",
             "parameters": {
                 "mac": {"type": "string", "description": "MAC address of the client to control."},
                 "command": {"type": "string", "enum": ["block-sta", "unblock-sta", "kick-sta"],
                             "description": "block-sta: block the client, unblock-sta: unblock a blocked client, kick-sta: force-disconnect the client."}}},
            {"name": "control_device", "handler": self._th_control_device, "timeout": 30,
             "description": "Restart, locate, or power-cycle a network device.",
             "parameters": {
                 "mac": {"type": "string", "description": "MAC address of the device to control."},
                 "command": {"type": "string", "enum": ["restart", "set-locate", "unset-locate", "power-cycle"],
                             "description": "restart: reboot the device, set-locate: enable locate LED, unset-locate: disable locate LED, power-cycle: power-cycle a PoE port (requires port_idx)."},
                 "port_idx": {"type": "integer", "description": "PoE port index; required for power-cycle command."}}},
            {"name": "manage_wlan", "handler": self._th_manage_wlan, "timeout": 15,
             "description": "List, enable/disable, or update WiFi networks.",
             "parameters": {
                 "sub_action": {"type": "string", "enum": ["list", "update"], "description": "list: retrieve all WLANs; update: modify a WLAN."},
                 "wlan_id": {"type": "string", "description": "WLAN ID; required for update."},
                 "updates": {"type": "object", "description": "Fields to update: enabled (bool), x_passphrase (str), name (str), hide_ssid (bool)."}}},
            {"name": "manage_port_forward", "handler": self._th_manage_port_forward, "timeout": 15,
             "description": "List, create, enable/disable port forwarding rules.",
             "parameters": {
                 "sub_action": {"type": "string", "enum": ["list", "update", "create"], "description": "list: retrieve all rules; create: add a new rule; update: modify an existing rule."},
                 "rule_id": {"type": "string", "description": "Rule ID; required for update."},
                 "rule": {"type": "object", "description": "Port forward rule definition; required for create."},
                 "updates": {"type": "object", "description": "Fields to update on an existing rule."}}},
            {"name": "manage_traffic_rule", "handler": self._th_manage_traffic_rule, "timeout": 15,
             "description": "List or enable/disable traffic management rules.",
             "parameters": {
                 "sub_action": {"type": "string", "enum": ["list", "update"], "description": "list: retrieve all rules; update: modify a rule."},
                 "rule_id": {"type": "string", "description": "Rule ID; required for update."},
                 "updates": {"type": "object", "description": "Fields to update on the rule (e.g. enabled)."}}},
            {"name": "authorize_guest", "handler": self._th_authorize_guest, "timeout": 15,
             "description": "Authorize a guest device on the WiFi network.",
             "parameters": {
                 "mac": {"type": "string", "description": "MAC address of the guest device to authorize."},
                 "minutes": {"type": "integer", "description": "Duration of the guest authorization in minutes.", "default": 60},
                 "up_kbps": {"type": "integer", "description": "Upload bandwidth cap in Kbps (optional)."},
                 "down_kbps": {"type": "integer", "description": "Download bandwidth cap in Kbps (optional)."},
                 "quota_mb": {"type": "integer", "description": "Total data quota in MB (optional)."}}},
        ]

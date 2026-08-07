

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import yaml

from capabilities.base import AbstractCapability
from capabilities.ubiquiti_capability.unifi_rest_handler import UnifiRestHandler
from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from typing import TypedDict

    class _AuthKwargs(TypedDict):
        api_key: str | None
        username: str | None
        password: str | None
        verify_ssl: bool

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("ubiquiti_capability", "manifest.yaml")

_K_URL = "ubiquiti:url"
_K_AUTH_METHOD = "ubiquiti:auth_method"
_K_API_KEY = "ubiquiti:api_key"
_K_USERNAME = "ubiquiti:username"
_K_PASSWORD = "ubiquiti:password"
_K_SITE = "ubiquiti:site"
_K_VERIFY_SSL = "ubiquiti:verify_ssl"


class UbiquitiCapability(AbstractCapability):

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict[str, object] | None = None
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

    def get_manifest(self) -> dict[str, object]:
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, encoding="utf-8") as fh:
                self._manifest_cache = cast(dict[str, object], yaml.safe_load(fh))
        return self._manifest_cache

    # ── Auth helpers ─────────────────────────────────────────────────

    @property
    def _auth(self) -> "_AuthKwargs":
        return {
            "api_key": self._api_key,
            "username": self._username,
            "password": self._password,
            "verify_ssl": self._verify_ssl,
        }

    def _require_mac_cmd(self, params: dict[str, object], action_name: str) -> dict[str, object] | None:
        if not params.get("mac"):
            return {"status": "error", "error": f"mac is required for {action_name}"}
        if not params.get("command"):
            return {"status": "error", "error": f"command is required for {action_name}"}
        return None

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict[str, object]) -> None:
        url = (cast(str, credentials.get("url")) or "").strip().rstrip("/")
        auth_method = (cast(str, credentials.get("auth_method")) or "api_key").strip()
        api_key = (cast(str, credentials.get("api_key")) or "").strip() or None
        username = (cast(str, credentials.get("username")) or "").strip() or None
        password = (cast(str, credentials.get("password")) or "").strip() or None
        site = (cast(str, credentials.get("site")) or "default").strip() or "default"

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
            UnifiRestHandler.probe(url, site, api_key=api_key, username=username,
                       password=password, verify_ssl=cast(bool, verify_ssl))
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
        self._verify_ssl = cast(bool, verify_ssl)
        self._connected = True

    def connect(self) -> bool:
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
            UnifiRestHandler.probe(self._url, self._site, **self._auth)
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

    def ingest(self) -> list[object]:
        if not self.is_connected():
            return []
        try:
            return cast(list[object], UnifiRestHandler.list_devices(self._url, self._site, **self._auth).get("devices", []))
        except Exception as exc:
            logger.warning("[ubiquiti] ingest failed: %s", exc)
            return []

    def understand(self, items: list[object]) -> list[object]:
        return items

    def _do_monitor(self) -> None:
        if not self._url:
            return
        UnifiRestHandler.probe(self._url, self._site, **self._auth)

    # ── Tool handlers ────────────────────────────────────────────────

    def _require_connection(self) -> dict[str, object] | None:
        if not self.is_connected():
            return {"status": "error", "error": "Ubiquiti controller not connected. Configure it in Brain → Capabilities."}
        return None

    def _th_list_devices(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        return UnifiRestHandler.list_devices(self._url, self._site, **self._auth)

    def _th_list_clients(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        return UnifiRestHandler.list_clients(self._url, self._site, **self._auth,
                                 active_only=cast(bool, params.get("active_only", True)))

    def _th_get_info(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        return UnifiRestHandler.get_info(self._url, self._site, cast(str, params.get("target", "health")), **self._auth)

    def _th_control_client(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection() or self._require_mac_cmd(params, "control_client")
        if err:
            return err
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return UnifiRestHandler.control_client(self._url, self._site, cast(str, params["mac"]), cast(str, params["command"]),
                                   **self._auth, **extra)

    def _th_control_device(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection() or self._require_mac_cmd(params, "control_device")
        if err:
            return err
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return UnifiRestHandler.control_device(self._url, self._site, cast(str, params["mac"]), cast(str, params["command"]),
                                   **self._auth, **extra)

    def _th_manage_wlan(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return UnifiRestHandler.list_wlans(self._url, self._site, **self._auth)
        if sub != "update":
            return {"status": "error", "error": f"manage_wlan supports 'list' and 'update', not '{sub}'"}
        return UnifiRestHandler.update_wlan(self._url, self._site, cast(str, params.get("wlan_id", "")),
                                cast(dict[str, object], params.get("updates", {})), **self._auth)

    def _th_manage_port_forward(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return UnifiRestHandler.list_port_forwards(self._url, self._site, **self._auth)
        if sub == "create":
            return UnifiRestHandler.create_port_forward(self._url, self._site, cast(dict[str, object], params.get("rule", {})), **self._auth)
        return UnifiRestHandler.update_port_forward(self._url, self._site, cast(str, params.get("rule_id", "")),
                                        cast(dict[str, object], params.get("updates", {})), **self._auth)

    def _th_manage_traffic_rule(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return UnifiRestHandler.list_traffic_rules(self._url, self._site, **self._auth)
        if sub != "update":
            return {"status": "error", "error": f"manage_traffic_rule supports 'list' and 'update', not '{sub}'"}
        return UnifiRestHandler.update_traffic_rule(self._url, self._site, cast(str, params.get("rule_id", "")),
                                        cast(dict[str, object], params.get("updates", {})), **self._auth)

    def _th_authorize_guest(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        mac = params.get("mac", "")
        if not mac:
            return {"status": "error", "error": "mac is required for authorize_guest"}
        extra: dict[str, object] = {}
        if params.get("up_kbps") is not None:
            extra["up"] = params["up_kbps"]
        if params.get("down_kbps") is not None:
            extra["down"] = params["down_kbps"]
        if params.get("quota_mb") is not None:
            extra["bytes"] = params["quota_mb"]
        return UnifiRestHandler.control_client(self._url, self._site, cast(str, mac), "authorize-guest",
                                   **self._auth, minutes=params.get("minutes", 60), **extra)

    # ── Tool definitions ─────────────────────────────────────────────

    def get_tools(self) -> list[dict[str, object]]:
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

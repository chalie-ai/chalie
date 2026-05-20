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
        _url (str): UniFi controller base URL, e.g. ``https://192.168.1.1``.
        _auth_method (str): Either ``"api_key"`` or ``"credentials"``.
        _api_key (str | None): API key for UniFi OS authentication.
        _username (str | None): Username for credential-based authentication.
        _password (str | None): Password for credential-based authentication.
        _site (str): UniFi site name, defaults to ``"default"``.
        _verify_ssl (bool): Whether to verify SSL certificates on HTTPS connections.
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

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict) -> None:
        """Validate credentials, probe the UniFi controller, and persist.

        Args:
            credentials: Must contain ``url`` (str) and ``auth_method`` (str).
                If ``auth_method`` is ``"api_key"``, ``api_key`` is required.
                If ``auth_method`` is ``"credentials"``, ``username`` and
                ``password`` are required. Optional: ``site`` (str, default
                ``"default"``), ``verify_ssl`` (bool, default ``False``).

        Raises:
            ValueError: If required fields are missing, or the controller
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
        if auth_method == "api_key":
            if not api_key:
                raise ValueError("API Key is required when authentication method is 'api_key'")
        elif auth_method == "credentials":
            if not username or not password:
                raise ValueError(
                    "Username and Password are required when authentication method is 'credentials'"
                )
        else:
            raise ValueError(f"Unknown authentication method: {auth_method!r}")

        try:
            rest.probe(
                url, site,
                api_key=api_key,
                username=username,
                password=password,
                verify_ssl=verify_ssl,
            )
        except Exception as exc:
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
        self.connect()

    def connect(self) -> bool:
        """Load stored credentials and probe the UniFi controller.

        Returns:
            bool: ``True`` if the probe succeeds, ``False`` otherwise.
        """
        url = self.load_credential(_K_URL)
        if not url:
            self._connected = False
            return False

        auth_method = self.load_credential(_K_AUTH_METHOD) or "api_key"
        api_key = self.load_credential(_K_API_KEY) or None
        username = self.load_credential(_K_USERNAME) or None
        password = self.load_credential(_K_PASSWORD) or None
        site = self.load_credential(_K_SITE) or "default"
        ssl_raw = self.load_credential(_K_VERIFY_SSL)
        verify_ssl = ssl_raw != "0"

        self._url = url
        self._auth_method = auth_method
        self._api_key = api_key if api_key else None
        self._username = username if username else None
        self._password = password if password else None
        self._site = site
        self._verify_ssl = verify_ssl

        try:
            rest.probe(
                self._url, self._site,
                api_key=self._api_key,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
            )
            self._connected = True
        except Exception as exc:
            logger.warning("[ubiquiti] connect probe failed: %s", exc)
            self._connected = False
        return self._connected

    def disconnect(self) -> None:
        """Clear connection state and remove all stored credentials."""
        self._connected = False
        self.delete_credentials()
        logger.info("[ubiquiti] Disconnected and credentials removed.")

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list:
        """Fetch all network devices from the UniFi controller.

        Returns:
            list[dict]: Device summary dicts, or ``[]`` when not connected.
        """
        if not self.is_connected():
            return []
        try:
            return rest.list_devices(
                self._url, self._site,
                api_key=self._api_key,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
            ).get("devices", [])
        except Exception as exc:
            logger.warning("[ubiquiti] ingest failed: %s", exc)
            return []

    def understand(self, items: list) -> list:
        return items

    def _do_monitor(self) -> None:
        """Probe the UniFi controller to check liveness."""
        rest.probe(
            self._url, self._site,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
        )

    def act(self, action: str, params: dict) -> dict:
        """Dispatch a write action to the matching tool handler.

        Args:
            action: Tool name, e.g. ``"control_client"``, ``"manage_wlan"``.
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
        """Return an error dict when not connected, else ``None``."""
        if not self.is_connected():
            return {
                "status": "error",
                "error": (
                    "Ubiquiti controller not connected. "
                    "Configure it in Brain → Capabilities."
                ),
            }
        return None

    def _th_list_devices(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        return rest.list_devices(
            self._url, self._site,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
        )

    def _th_list_clients(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        active_only = params.get("active_only", True)
        return rest.list_clients(
            self._url, self._site,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
            active_only=active_only,
        )

    def _th_get_info(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        target = params.get("target", "health")
        return rest.get_info(
            self._url, self._site, target,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
        )

    def _th_control_client(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        mac = params.get("mac", "")
        command = params.get("command", "")
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return rest.control_client(
            self._url, self._site, mac, command,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
            **extra,
        )

    def _th_control_device(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        mac = params.get("mac", "")
        command = params.get("command", "")
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return rest.control_device(
            self._url, self._site, mac, command,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
            **extra,
        )

    def _th_manage_wlan(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        sub_action = params.get("sub_action", "list")
        if sub_action == "list":
            return rest.list_wlans(
                self._url, self._site,
                api_key=self._api_key,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
            )
        wlan_id = params.get("wlan_id", "")
        updates = params.get("updates", {})
        return rest.update_wlan(
            self._url, self._site, wlan_id, updates,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
        )

    def _th_manage_port_forward(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        sub_action = params.get("sub_action", "list")
        if sub_action == "list":
            return rest.list_port_forwards(
                self._url, self._site,
                api_key=self._api_key,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
            )
        if sub_action == "create":
            rule = params.get("rule", {})
            return rest.create_port_forward(
                self._url, self._site, rule,
                api_key=self._api_key,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
            )
        rule_id = params.get("rule_id", "")
        updates = params.get("updates", {})
        return rest.update_port_forward(
            self._url, self._site, rule_id, updates,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
        )

    def _th_manage_traffic_rule(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        sub_action = params.get("sub_action", "list")
        if sub_action == "list":
            return rest.list_traffic_rules(
                self._url, self._site,
                api_key=self._api_key,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
            )
        rule_id = params.get("rule_id", "")
        updates = params.get("updates", {})
        return rest.update_traffic_rule(
            self._url, self._site, rule_id, updates,
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
        )

    def _th_authorize_guest(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        mac = params.get("mac", "")
        minutes = params.get("minutes", 60)
        up_kbps = params.get("up_kbps")
        down_kbps = params.get("down_kbps")
        quota_mb = params.get("quota_mb")
        extra = {}
        if up_kbps is not None:
            extra["up"] = up_kbps
        if down_kbps is not None:
            extra["down"] = down_kbps
        if quota_mb is not None:
            extra["bytes"] = quota_mb
        return rest.control_client(
            self._url, self._site, mac, "authorize-guest",
            api_key=self._api_key,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
            minutes=minutes,
            **extra,
        )

    def get_tools(self) -> list:
        """Return tool definitions for all 9 Ubiquiti actions.

        Returns:
            list[dict]: Tool dicts with ``name``, ``description``,
            ``parameters``, ``handler``, and ``timeout`` keys.
        """
        return [
            {
                "name": "list_devices",
                "description": (
                    "List all UniFi network devices (APs, switches, gateways) with status."
                ),
                "parameters": {},
                "handler": self._th_list_devices,
                "timeout": 15,
            },
            {
                "name": "list_clients",
                "description": "List devices connected to the network.",
                "parameters": {
                    "active_only": {
                        "type": "boolean",
                        "description": "When true, return only currently connected clients.",
                        "default": True,
                    },
                },
                "handler": self._th_list_clients,
                "timeout": 15,
            },
            {
                "name": "get_info",
                "description": (
                    "Get detailed device info by MAC address or site health overview."
                ),
                "parameters": {
                    "target": {
                        "type": "string",
                        "description": (
                            'Use "health" for a site health summary, '
                            "or a MAC address (e.g. aa:bb:cc:dd:ee:ff) "
                            "for a specific device."
                        ),
                    },
                },
                "handler": self._th_get_info,
                "timeout": 15,
            },
            {
                "name": "control_client",
                "description": "Block, unblock, or disconnect a network client.",
                "parameters": {
                    "mac": {
                        "type": "string",
                        "description": "MAC address of the client to control.",
                    },
                    "command": {
                        "type": "string",
                        "enum": ["block-sta", "unblock-sta", "kick-sta"],
                        "description": (
                            "block-sta: block the client, "
                            "unblock-sta: unblock a blocked client, "
                            "kick-sta: force-disconnect the client."
                        ),
                    },
                },
                "handler": self._th_control_client,
                "timeout": 15,
            },
            {
                "name": "control_device",
                "description": "Restart, locate, or power-cycle a network device.",
                "parameters": {
                    "mac": {
                        "type": "string",
                        "description": "MAC address of the device to control.",
                    },
                    "command": {
                        "type": "string",
                        "enum": ["restart", "set-locate", "unset-locate", "power-cycle"],
                        "description": (
                            "restart: reboot the device, "
                            "set-locate: enable locate LED, "
                            "unset-locate: disable locate LED, "
                            "power-cycle: power-cycle a PoE port (requires port_idx)."
                        ),
                    },
                    "port_idx": {
                        "type": "integer",
                        "description": "PoE port index; required for power-cycle command.",
                    },
                },
                "handler": self._th_control_device,
                "timeout": 30,
            },
            {
                "name": "manage_wlan",
                "description": "List, enable/disable, or update WiFi networks.",
                "parameters": {
                    "sub_action": {
                        "type": "string",
                        "enum": ["list", "update"],
                        "description": "list: retrieve all WLANs; update: modify a WLAN.",
                    },
                    "wlan_id": {
                        "type": "string",
                        "description": "WLAN ID; required for update.",
                    },
                    "updates": {
                        "type": "object",
                        "description": (
                            "Fields to update: enabled (bool), "
                            "x_passphrase (str), name (str), hide_ssid (bool)."
                        ),
                    },
                },
                "handler": self._th_manage_wlan,
                "timeout": 15,
            },
            {
                "name": "manage_port_forward",
                "description": "List, create, enable/disable port forwarding rules.",
                "parameters": {
                    "sub_action": {
                        "type": "string",
                        "enum": ["list", "update", "create"],
                        "description": (
                            "list: retrieve all rules; "
                            "create: add a new rule; "
                            "update: modify an existing rule."
                        ),
                    },
                    "rule_id": {
                        "type": "string",
                        "description": "Rule ID; required for update.",
                    },
                    "rule": {
                        "type": "object",
                        "description": "Port forward rule definition; required for create.",
                    },
                    "updates": {
                        "type": "object",
                        "description": "Fields to update on an existing rule.",
                    },
                },
                "handler": self._th_manage_port_forward,
                "timeout": 15,
            },
            {
                "name": "manage_traffic_rule",
                "description": "List or enable/disable traffic management rules.",
                "parameters": {
                    "sub_action": {
                        "type": "string",
                        "enum": ["list", "update"],
                        "description": "list: retrieve all rules; update: modify a rule.",
                    },
                    "rule_id": {
                        "type": "string",
                        "description": "Rule ID; required for update.",
                    },
                    "updates": {
                        "type": "object",
                        "description": "Fields to update on the rule (e.g. enabled).",
                    },
                },
                "handler": self._th_manage_traffic_rule,
                "timeout": 15,
            },
            {
                "name": "authorize_guest",
                "description": "Authorize a guest device on the WiFi network.",
                "parameters": {
                    "mac": {
                        "type": "string",
                        "description": "MAC address of the guest device to authorize.",
                    },
                    "minutes": {
                        "type": "integer",
                        "description": "Duration of the guest authorization in minutes.",
                        "default": 60,
                    },
                    "up_kbps": {
                        "type": "integer",
                        "description": "Upload bandwidth cap in Kbps (optional).",
                    },
                    "down_kbps": {
                        "type": "integer",
                        "description": "Download bandwidth cap in Kbps (optional).",
                    },
                    "quota_mb": {
                        "type": "integer",
                        "description": "Total data quota in MB (optional).",
                    },
                },
                "handler": self._th_authorize_guest,
                "timeout": 15,
            },
        ]

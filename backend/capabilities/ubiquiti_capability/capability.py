

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

from capabilities.base import AbstractCapability
from capabilities.ubiquiti_capability import unifi_rest_handler as rest
from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("ubiquiti_capability", "manifest.yaml")

_K_URL = "ubiquiti:url"
_K_USERNAME = "ubiquiti:username"
_K_PASSWORD = "ubiquiti:password"
_K_SITE = "ubiquiti:site"
_K_VERIFY_SSL = "ubiquiti:verify_ssl"
_K_SESSION = "ubiquiti:session"


class UbiquitiCapability(AbstractCapability):

    def __init__(self) -> None:
        super().__init__(_MANIFEST_PATH)
        self._url: str = ""
        self._username: str = ""
        self._password: str = ""
        self._site: str = "default"
        self._verify_ssl: bool = False
        self._session: rest.Session | None = None

    # ── Identity ─────────────────────────────────────────────────────

    def get_id(self) -> str:
        return "ubiquiti"

    # ── Console session ──────────────────────────────────────────────

    def _console(self, origin: dict[str, object] | None, should_stop: Callable[[], bool] | None) -> rest.Console:
        return rest.Console(
            url=self._url,
            site=self._site,
            username=self._username,
            password=self._password,
            verify_ssl=self._verify_ssl,
            stored=self._stored_session,
            keep=self._keep_session,
            origin=origin,
            should_stop=should_stop,
        )

    def _stored_session(self) -> rest.Session | None:
        return self._session

    def _keep_session(self, session: rest.Session) -> None:
        """Store a fresh console session over the one it replaces."""
        self.store_credential(_K_SESSION, json.dumps(session._asdict()))
        self._session = session

    def _require_mac_cmd(self, params: dict[str, object], action_name: str) -> dict[str, object] | None:
        if not params.get("mac"):
            return {"status": "error", "error": f"mac is required for {action_name}"}
        if not params.get("command"):
            return {"status": "error", "error": f"command is required for {action_name}"}
        return None

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict[str, object]) -> None:
        url = (cast(str, credentials.get("url")) or "").strip().rstrip("/")
        username = (cast(str, credentials.get("username")) or "").strip()
        password = cast(str, credentials.get("password")) or ""
        site = (cast(str, credentials.get("site")) or "default").strip() or "default"

        verify_ssl = credentials.get("verify_ssl", False)
        if isinstance(verify_ssl, str):
            verify_ssl = verify_ssl.lower() not in ("0", "false", "no")

        if not url:
            raise ValueError("UniFi Controller URL is required")
        if not username or not password:
            raise ValueError("Username and Password are required")

        logger.info(
            "[ubiquiti] configure: url=%s site=%s verify_ssl=%r (raw=%r)",
            url, site, verify_ssl, credentials.get("verify_ssl"),
        )

        # A kept session belongs to the console and username it was made with.
        if self._session is not None and not self._session.made_with(url, username):
            self.delete_credentials()
            self._session = None

        self.store_credential(_K_URL, url)
        self.store_credential(_K_USERNAME, username)
        self.store_credential(_K_PASSWORD, password)
        self.store_credential(_K_SITE, site)
        self.store_credential(_K_VERIFY_SSL, "1" if verify_ssl else "0")

        self._url = url
        self._username = username
        self._password = password
        self._site = site
        self._verify_ssl = cast(bool, verify_ssl)
        self._connected = True

    def connect(self) -> bool:
        self._url = self.load_credential(_K_URL) or ""
        self._username = self.load_credential(_K_USERNAME) or ""
        self._password = self.load_credential(_K_PASSWORD) or ""
        self._site = self.load_credential(_K_SITE) or "default"
        self._verify_ssl = self.load_credential(_K_VERIFY_SSL) == "1"
        session = self.load_credential(_K_SESSION)
        self._session = None if session is None else rest.Session(**json.loads(session))
        self._connected = bool(self._url and self._username and self._password)
        return self._connected

    def disconnect(self) -> None:
        """Forget the setup but keep the console session, so setting up the same
        console and username again needs no new sign-in."""
        self._connected = False
        self.delete_credentials(keep=(_K_SESSION,))
        logger.info("[ubiquiti] Disconnected; setup removed, console session kept.")

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list[object]:
        """Nothing is read in the background: a sign-in may wait on the user's
        2FA code, so the console is reached only when a tool runs."""
        return []

    def understand(self, items: list[object]) -> list[object]:
        return items

    def _do_monitor(self) -> None:
        """Nothing is checked: a sign-in may wait on the user's 2FA code, so the
        console is reached only when a tool runs."""

    # ── Tool handlers ────────────────────────────────────────────────
    # ``origin`` and ``should_stop`` come from the run calling the tool: a code
    # ask names its turn, and a stop ends the wait for a code.

    def _require_connection(self) -> dict[str, object] | None:
        if not self.is_connected():
            return {"status": "error", "error": "Ubiquiti controller not connected. Configure it in Brain → Capabilities."}
        return None

    def _th_list_devices(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        return rest.list_devices(self._console(origin, should_stop))

    def _th_list_clients(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        return rest.list_clients(self._console(origin, should_stop),
                                 active_only=cast(bool, params.get("active_only", True)))

    def _th_get_info(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        return rest.get_info(self._console(origin, should_stop), cast(str, params.get("target", "health")))

    def _th_control_client(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection() or self._require_mac_cmd(params, "control_client")
        if err:
            return err
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return rest.control_client(self._console(origin, should_stop), cast(str, params["mac"]),
                                   cast(str, params["command"]), **extra)

    def _th_control_device(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection() or self._require_mac_cmd(params, "control_device")
        if err:
            return err
        extra = {k: v for k, v in params.items() if k not in ("action", "mac", "command")}
        return rest.control_device(self._console(origin, should_stop), cast(str, params["mac"]),
                                   cast(str, params["command"]), **extra)

    def _th_manage_wlan(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return rest.list_wlans(self._console(origin, should_stop))
        if sub != "update":
            return {"status": "error", "error": f"manage_wlan supports 'list' and 'update', not '{sub}'"}
        return rest.update_wlan(self._console(origin, should_stop), cast(str, params.get("wlan_id", "")),
                                cast(dict[str, object], params.get("updates", {})))

    def _th_manage_port_forward(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return rest.list_port_forwards(self._console(origin, should_stop))
        if sub == "create":
            return rest.create_port_forward(self._console(origin, should_stop),
                                            cast(dict[str, object], params.get("rule", {})))
        return rest.update_port_forward(self._console(origin, should_stop), cast(str, params.get("rule_id", "")),
                                        cast(dict[str, object], params.get("updates", {})))

    def _th_manage_traffic_rule(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        err = self._require_connection()
        if err:
            return err
        sub = params.get("sub_action", "list")
        if sub == "list":
            return rest.list_traffic_rules(self._console(origin, should_stop))
        if sub != "update":
            return {"status": "error", "error": f"manage_traffic_rule supports 'list' and 'update', not '{sub}'"}
        return rest.update_traffic_rule(self._console(origin, should_stop), cast(str, params.get("rule_id", "")),
                                        cast(dict[str, object], params.get("updates", {})))

    def _th_authorize_guest(
        self, params: dict[str, object], telemetry: object = None,
        origin: dict[str, object] | None = None, should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
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
        return rest.control_client(self._console(origin, should_stop), cast(str, mac), "authorize-guest",
                                   minutes=params.get("minutes", 60), **extra)

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

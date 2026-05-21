"""
Interface Registry Service — manages external interface lifecycle.

Handles pairing (bluetooth-style), capability discovery, health monitoring,
tool execution dispatch, and teardown. Interface capabilities are registered
as tools in the ToolRegistryService so they appear in the ACT prompt.
"""

import hashlib
import json
import logging
import secrets
import uuid
from datetime import timedelta
from typing import Optional

import requests

from services.database_service import get_shared_db_service
from services.time_utils import utc_now, parse_utc
from utils.data_utils import parse_json_column
from services.write_queue_service import get_write_queue

logger = logging.getLogger(__name__)

_HEALTH_CHECK_TIMEOUT = 5  # seconds
_PAIRING_KEY_TTL = 600  # 10 minutes

# Module-level failure counter — must survive across InterfaceRegistryService
# instantiations (the service is not a singleton).
_failure_counts: dict[str, int] = {}


def _hash(value: str) -> str:
    """Return the SHA-256 hex digest of *value*.

    Args:
        value: Plain-text string to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class InterfaceRegistryService:
    """Manages interface lifecycle: pairing, capability sync, health, teardown."""

    def __init__(self, db=None):
        """Initialise the service, optionally injecting a DatabaseService.

        Args:
            db: DatabaseService instance.  When ``None`` the shared singleton is
                used.  Pass an in-memory instance in tests.
        """
        self._db = db or get_shared_db_service()
        self._write_queue = get_write_queue()

    # ------------------------------------------------------------------
    # Pairing key generation
    # ------------------------------------------------------------------

    def generate_pairing_key(self) -> dict:
        """Generate a one-time pairing key with 10 min expiry.

        The caller (brain dashboard) displays the key together with Chalie's
        host:port for the user to enter into the interface.

        Returns:
            Dict with ``pairing_key``, ``host``, ``port``, and
            ``expires_in_seconds``.
        """
        import runtime_config

        raw_key = secrets.token_urlsafe(24)  # shorter than wrapper tokens — typed by hand
        key_hash = _hash(raw_key)
        now = utc_now()
        expires = now + timedelta(seconds=_PAIRING_KEY_TTL)

        def _purge_and_insert(_kh=key_hash, _now=now, _exp=expires, _db=self._db):
            """Purge expired pairing keys and insert the new one atomically.

            Args:
                _kh: SHA-256 hex digest of the new raw key.
                _now: Current UTC datetime.
                _exp: Expiry UTC datetime for the new key.
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                # Purge expired keys to prevent unbounded table growth
                cursor.execute(
                    "DELETE FROM interface_pairing_keys WHERE expires_at < ?",
                    (_now.isoformat(),),
                )
                cursor.execute(
                    "INSERT INTO interface_pairing_keys (key_hash, created_at, expires_at) VALUES (?, ?, ?)",
                    (_kh, _now.isoformat(), _exp.isoformat()),
                )
                cursor.close()

        self._write_queue.submit_sync(_purge_and_insert)

        return {
            "pairing_key": raw_key,
            "host": runtime_config.get("host", "0.0.0.0"),
            "port": runtime_config.get("port", 8081),
            "expires_in_seconds": _PAIRING_KEY_TTL,
        }

    # ------------------------------------------------------------------
    # Pairing handshake
    # ------------------------------------------------------------------

    def pair(self, pairing_key: str, name: str, host: str, port: int) -> dict:
        """Complete the pairing handshake with an external interface.

        Steps:
            1. Validate pairing key (exists, not expired, not used).
            2. Mark key as used.
            3. Health-check the interface.
            4. Fetch capabilities.
            5. Create interface record.
            6. Issue signal_token via WrapperAuthService (reuse existing auth).
            7. Register capabilities as tools.
            8. Set status to ``online``.

        Args:
            pairing_key: The one-time key previously generated via
                :meth:`generate_pairing_key`.
            name: Human-readable label for the interface.
            host: Interface hostname or IP address.
            port: Interface port number.

        Returns:
            Dict with ``interface_id``, ``signal_token``, ``status``, and
            ``capabilities`` (list of capability names).

        Raises:
            ValueError: If the pairing key is invalid, expired, already used,
                or the interface is unreachable.
        """
        now = utc_now()
        key_hash = _hash(pairing_key)

        # Validate pairing key (read-only — stays on caller thread)
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT expires_at, used_at FROM interface_pairing_keys WHERE key_hash = ?",
                (key_hash,),
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            raise ValueError("Invalid pairing key")

        expires_at = parse_utc(row[0])
        if row[1]:  # used_at is set
            raise ValueError("Pairing key already used")
        if now > expires_at:
            raise ValueError("Pairing key expired")

        def _mark_used(_kh=key_hash, _now=now, _db=self._db):
            """Mark the pairing key as used on the write-queue thread.

            Args:
                _kh: SHA-256 hex digest of the key to mark.
                _now: Timestamp to store as used_at.
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE interface_pairing_keys SET used_at = ? WHERE key_hash = ?",
                    (_now.isoformat(), _kh),
                )
                cursor.close()

        self._write_queue.submit_sync(_mark_used)

        # Health-check interface
        base_url = f"http://{host}:{port}"
        try:
            resp = requests.get(f"{base_url}/health", timeout=_HEALTH_CHECK_TIMEOUT)
            resp.raise_for_status()
            health = resp.json()
        except Exception as e:
            raise ValueError(f"Interface at {host}:{port} is not reachable: {e}")

        # Fetch capabilities
        try:
            resp = requests.get(f"{base_url}/capabilities", timeout=_HEALTH_CHECK_TIMEOUT)
            resp.raise_for_status()
            capabilities = resp.json()
            if not isinstance(capabilities, list):
                raise ValueError("capabilities response must be a JSON array")
        except Exception as e:
            raise ValueError(f"Could not fetch capabilities from {host}:{port}: {e}")

        # Create interface record
        interface_id = str(uuid.uuid4())
        caps_json = json.dumps(capabilities, sort_keys=True)
        caps_hash = _hash(caps_json)
        metadata = {
            "version": health.get("version", "unknown"),
            "description": health.get("name", name),
        }

        def _insert_interface(
            _id=interface_id, _name=name, _host=host, _port=port,
            _ch=caps_hash, _now=now, _meta=metadata, _db=self._db,
        ):
            """Insert the new interface record on the write-queue thread.

            Args:
                _id: Pre-generated interface UUID.
                _name: Human-readable label for the interface.
                _host: Interface hostname.
                _port: Interface port number.
                _ch: SHA-256 of the capabilities JSON for change detection.
                _now: Pairing timestamp.
                _meta: Metadata dict (version, description).
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO interfaces
                        (id, name, host, port, status, capabilities_hash,
                         last_seen_at, paired_at, metadata)
                       VALUES (?, ?, ?, ?, 'online', ?, ?, ?, ?)""",
                    (
                        _id, _name, _host, _port, _ch,
                        _now.isoformat(), _now.isoformat(), json.dumps(_meta),
                    ),
                )
                cursor.close()

        self._write_queue.submit_sync(_insert_interface)

        # Issue signal token via WrapperAuthService (reuse existing infrastructure).
        # If this fails, remove the interface record — a paired interface without
        # a signal token is useless.
        try:
            from services.wrapper_auth_service import WrapperAuthService

            wrapper_svc = WrapperAuthService(self._db)
            signal_token, _wrapper_id = wrapper_svc.create_token(
                name=f"Interface: {name}",
                capabilities={"signals": ["*"]},
                permissions={"query": [], "update": ["context", "feedback"], "broadcast": False},
                wrapper_id_override=f"iface_{interface_id[:8]}",
            )
            # Store signal_token_hash in interface record
            _tok_hash = _hash(signal_token)

            def _set_token_hash(_th=_tok_hash, _id=interface_id, _db=self._db):
                """Store the hashed signal token on the interface record.

                Args:
                    _th: SHA-256 hex digest of the signal token.
                    _id: Interface UUID.
                    _db: DatabaseService instance.
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE interfaces SET signal_token_hash = ? WHERE id = ?",
                        (_th, _id),
                    )
                    cursor.close()

            self._write_queue.submit_sync(_set_token_hash)
        except Exception as e:
            # Rollback: remove orphan interface record
            def _rollback_delete(_id=interface_id, _db=self._db):
                """Delete the orphaned interface record on rollback.

                Args:
                    _id: Interface UUID to remove.
                    _db: DatabaseService instance.
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM interfaces WHERE id = ?", (_id,))
                    cursor.close()

            self._write_queue.submit_sync(_rollback_delete)
            raise ValueError(f"Failed to create signal token: {e}") from e

        # Register capabilities as tools
        cap_names = self._register_tools(interface_id, capabilities)

        _failure_counts[interface_id] = 0
        logger.info(
            "[INTERFACE] Paired with '%s' at %s:%d — %d capabilities registered",
            name, host, port, len(cap_names),
        )

        return {
            "interface_id": interface_id,
            "signal_token": signal_token,
            "status": "online",
            "capabilities": cap_names,
        }

    # ------------------------------------------------------------------
    # Capability management
    # ------------------------------------------------------------------

    def _register_tools(self, interface_id: str, capabilities: list[dict]) -> list[str]:
        """Register interface capabilities as tools.

        Stores in ``interface_tools`` table AND registers in the
        ToolRegistryService so they appear in the ACT prompt.

        Args:
            interface_id: The interface UUID.
            capabilities: List of capability manifest dicts from the interface.

        Returns:
            List of registered capability names.
        """
        now = utc_now().isoformat()
        # Compute cap_names from the capabilities manifest (pure Python, no DB).
        cap_names = [cap.get("name", "") for cap in capabilities if cap.get("name", "")]

        def _upsert_tools(_id=interface_id, _caps=capabilities, _now=now, _db=self._db):
            """Bulk insert-or-replace capability tool records on the write-queue thread.

            Args:
                _id: Interface UUID that owns these tools.
                _caps: List of capability manifest dicts from the interface.
                _now: ISO-format timestamp for registered_at.
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                for cap in _caps:
                    cap_name = cap.get("name", "")
                    if not cap_name:
                        continue
                    cursor.execute(
                        """INSERT OR REPLACE INTO interface_tools
                           (interface_id, tool_name, manifest_json, registered_at)
                           VALUES (?, ?, ?, ?)""",
                        (_id, cap_name, json.dumps(cap), _now),
                    )
                cursor.close()

        self._write_queue.submit_sync(_upsert_tools)

        # Register in ToolRegistryService
        try:
            from services.tool_registry_service import ToolRegistryService

            registry = ToolRegistryService()
            for cap in capabilities:
                cap_name = cap.get("name", "")
                if not cap_name:
                    continue
                registry.register_interface_tool(interface_id, cap)
        except Exception as e:
            logger.warning("[INTERFACE] Failed to register tools in registry: %s", e)

        return cap_names

    def _unregister_tools(self, interface_id: str):
        """Remove all tools for an interface from the ToolRegistryService.

        Args:
            interface_id: The interface UUID whose tools should be removed.
        """
        try:
            from services.tool_registry_service import ToolRegistryService

            registry = ToolRegistryService()
            registry.remove_interface_tools(interface_id)
        except Exception as e:
            logger.warning("[INTERFACE] Failed to unregister tools from registry: %s", e)

    def fetch_capabilities(self, interface_id: str) -> list[dict]:
        """Re-fetch capabilities from an interface.

        Used on reconnect or manual refresh.  If the capabilities hash changed
        the tool registry is updated automatically.

        Args:
            interface_id: The interface UUID.

        Returns:
            List of capability manifest dicts, or empty list on failure.
        """
        iface = self.get_interface(interface_id)
        if not iface:
            return []

        base_url = f"http://{iface['host']}:{iface['port']}"
        try:
            resp = requests.get(f"{base_url}/capabilities", timeout=_HEALTH_CHECK_TIMEOUT)
            resp.raise_for_status()
            capabilities = resp.json()
        except Exception as e:
            logger.warning("[INTERFACE] Could not fetch capabilities from %s: %s", iface["name"], e)
            return []

        # Always re-register tools (idempotent) so they reach the
        # in-memory registry and tool profiler — covers process restarts.
        caps_json = json.dumps(capabilities, sort_keys=True)
        new_hash = _hash(caps_json)

        if new_hash != iface.get("capabilities_hash"):
            self._unregister_tools(interface_id)

            def _update_caps_hash(_nh=new_hash, _id=interface_id, _db=self._db):
                """Persist the updated capabilities hash.

                Args:
                    _nh: New SHA-256 hex digest of the capabilities JSON.
                    _id: Interface UUID.
                    _db: DatabaseService instance.
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE interfaces SET capabilities_hash = ? WHERE id = ?",
                        (_nh, _id),
                    )
                    cursor.close()

            self._write_queue.submit(_update_caps_hash)

        self._register_tools(interface_id, capabilities)
        logger.info(
            "[INTERFACE] Registered %d capabilities for '%s'",
            len(capabilities), iface["name"],
        )

        return capabilities

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def health_check(self, interface_id: str) -> bool:
        """Ping interface health endpoint.

        Updates status on state transitions: marks ``offline`` after 3
        consecutive failures, marks ``online`` on recovery (and re-fetches
        capabilities).

        Args:
            interface_id: The interface UUID.

        Returns:
            ``True`` if the interface is healthy, ``False`` otherwise.
        """
        iface = self.get_interface(interface_id)
        if not iface or iface["status"] == "revoked":
            return False

        base_url = f"http://{iface['host']}:{iface['port']}"
        try:
            resp = requests.get(f"{base_url}/health", timeout=_HEALTH_CHECK_TIMEOUT)
            resp.raise_for_status()
            health_data = resp.json()
            if health_data.get("status") != "ok":
                raise ValueError("Health check returned non-ok status")
        except Exception as e:
            # Increment failure count
            logger.debug("[INTERFACE] Health check failed for %s: %s", interface_id[:8], e)
            count = _failure_counts.get(interface_id, 0) + 1
            _failure_counts[interface_id] = count

            if count >= 3 and iface["status"] != "offline":
                self._set_status(interface_id, "offline")
                self._unregister_tools(interface_id)
                logger.warning("[INTERFACE] '%s' is offline (3 consecutive failures)", iface["name"])
            return False

        # Success — reset failure count
        _failure_counts[interface_id] = 0
        now = utc_now().isoformat()

        def _update_last_seen(_now=now, _id=interface_id, _db=self._db):
            """Persist the successful health-check timestamp.

            Args:
                _now: ISO-format UTC timestamp to record as last_seen_at.
                _id: Interface UUID.
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE interfaces SET last_seen_at = ? WHERE id = ?",
                    (_now, _id),
                )
                cursor.close()

        self._write_queue.submit(_update_last_seen)

        # Recovery: was offline, now online
        if iface["status"] == "offline":
            self._set_status(interface_id, "online")
            # Re-fetch capabilities (may have changed during downtime)
            self.fetch_capabilities(interface_id)
            logger.info("[INTERFACE] '%s' recovered — back online", iface["name"])

        return True

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, interface_id: str):
        """Unpair an interface.

        Removes all registered tools, revokes the signal token, and deletes
        the interface record from the database.

        Args:
            interface_id: The interface UUID.
        """
        iface = self.get_interface(interface_id)
        if not iface:
            return

        self._unregister_tools(interface_id)

        # Revoke signal token if exists
        try:
            from services.wrapper_auth_service import WrapperAuthService

            wrapper_svc = WrapperAuthService(self._db)
            wrapper_svc.revoke(f"iface_{interface_id[:8]}")
        except Exception as e:
            logger.debug("[INTERFACE] Signal token revocation failed for %s: %s", interface_id[:8], e)

        def _delete_all(_id=interface_id, _db=self._db):
            """Delete interface tools and the interface record together.

            Args:
                _id: Interface UUID to remove.
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM interface_tools WHERE interface_id = ?", (_id,))
                cursor.execute("DELETE FROM interfaces WHERE id = ?", (_id,))
                cursor.close()

        self._write_queue.submit_sync(_delete_all)

        _failure_counts.pop(interface_id, None)
        logger.info("[INTERFACE] Removed '%s'", iface["name"])

    # ------------------------------------------------------------------
    # Listing / retrieval
    # ------------------------------------------------------------------

    def list_interfaces(self) -> list[dict]:
        """Return all interfaces with status and capability count.

        Returns:
            List of dicts with keys: ``id``, ``name``, ``host``, ``port``,
            ``status``, ``last_seen_at``, ``paired_at``, ``metadata``,
            ``capabilities_hash``, ``tool_count``.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT i.id, i.name, i.host, i.port, i.status, i.last_seen_at,
                          i.paired_at, i.metadata, i.capabilities_hash,
                          COUNT(it.tool_name) as tool_count
                   FROM interfaces i
                   LEFT JOIN interface_tools it ON i.id = it.interface_id
                   GROUP BY i.id
                   ORDER BY i.paired_at ASC"""
            )
            rows = cursor.fetchall()
            cursor.close()

        return [
            {
                "id": r[0],
                "name": r[1],
                "host": r[2],
                "port": r[3],
                "status": r[4],
                "last_seen_at": r[5],
                "paired_at": r[6],
                "metadata": parse_json_column(r[7]),
                "capabilities_hash": r[8],
                "tool_count": r[9],
            }
            for r in rows
        ]

    def get_interface(self, interface_id: str) -> Optional[dict]:
        """Retrieve a single interface by its UUID.

        Args:
            interface_id: The interface UUID.

        Returns:
            Dict with interface details, or ``None`` if not found.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, name, host, port, status, signal_token_hash,
                          capabilities_hash, last_seen_at, paired_at, metadata
                   FROM interfaces WHERE id = ?""",
                (interface_id,),
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "host": row[2],
            "port": row[3],
            "status": row[4],
            "signal_token_hash": row[5],
            "capabilities_hash": row[6],
            "last_seen_at": row[7],
            "paired_at": row[8],
            "metadata": parse_json_column(row[9]),
        }

    def get_interface_tools(self, interface_id: str) -> list[dict]:
        """Get all tool manifests for an interface.

        Args:
            interface_id: The interface UUID.

        Returns:
            List of dicts with ``name`` and ``manifest`` keys.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tool_name, manifest_json FROM interface_tools WHERE interface_id = ?",
                (interface_id,),
            )
            rows = cursor.fetchall()
            cursor.close()

        return [
            {"name": r[0], "manifest": json.loads(r[1])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_status(self, interface_id: str, status: str):
        """Update interface status.

        Args:
            interface_id: The interface UUID.
            status: New status string (``online``, ``offline``, ``revoked``).
        """
        def _update_status(_st=status, _id=interface_id, _db=self._db):
            """Persist the new interface status.

            Args:
                _st: New status string (``online``, ``offline``, ``revoked``).
                _id: Interface UUID.
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE interfaces SET status = ? WHERE id = ?",
                    (_st, _id),
                )
                cursor.close()

        self._write_queue.submit_sync(_update_status)

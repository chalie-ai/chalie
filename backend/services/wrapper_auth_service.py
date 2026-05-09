"""
Wrapper Authentication Service — bearer token auth for external wrappers.

External wrappers (IDE, trading terminal, automation scripts, etc.) connect to
Chalie programmatically using bearer tokens rather than the cookie-based session
auth used by the chat UI.

Tokens are generated as URL-safe random strings (48 bytes → 64 chars).  Only the
SHA-256 hash is persisted; the raw token is shown exactly once at creation time
and never stored.

Each wrapper has:
  - A stable ``wrapper_id`` for identity across token rotations.
  - A ``capabilities`` JSON payload declaring which signals/intents it can emit.
  - A ``permissions`` JSON payload controlling which API operations it may call.
"""

import hashlib
import json
import logging
import secrets
import uuid
from typing import Optional

from services.database_service import get_shared_db_service
from services.log_utils import safe
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


def _hash_token(raw_token: str) -> str:
    """Return the SHA-256 hex digest of a raw bearer token.

    Args:
        raw_token: The plain-text bearer token.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class WrapperAuthService:
    """Manages wrapper bearer-token lifecycle: creation, validation, and revocation."""

    def __init__(self, db=None):
        """Initialise the service, optionally injecting a DatabaseService.

        Args:
            db: DatabaseService instance.  When ``None`` the shared singleton is
                used.  Pass an in-memory instance in tests.
        """
        self._db = db or get_shared_db_service()

    # ------------------------------------------------------------------
    # Token creation
    # ------------------------------------------------------------------

    def create_token(
        self,
        name: str,
        capabilities: Optional[dict] = None,
        permissions: Optional[dict] = None,
        metadata: Optional[dict] = None,
        wrapper_id_override: Optional[str] = None,
    ) -> tuple[str, str]:
        """Create a new wrapper token and persist its hash.

        Args:
            name: Human-readable label for the wrapper (e.g. "IDE Wrapper").
            capabilities: Dict declaring wrapper capabilities, e.g.
                ``{"signals": ["context_change"], "intents": ["read_memory"]}``.
                Defaults to an empty dict.
            permissions: Dict controlling allowed operations, e.g.
                ``{"query": ["memory", "threads"], "update": [], "broadcast": False}``.
                Defaults to an empty dict.
            metadata: Optional dict with version, description, etc.
                Defaults to an empty dict.
            wrapper_id_override: Optional fixed ``wrapper_id`` to use instead of
                the auto-generated one.  Reserved for built-in wrappers such as
                ``"__chat_ui__"`` that must have a stable, well-known identifier.

        Returns:
            A ``(raw_token, wrapper_id)`` tuple.  ``raw_token`` is the plain-text
            bearer token that must be given to the wrapper — it is **not** stored
            and cannot be recovered later.  ``wrapper_id`` is the stable identifier
            for this wrapper.
        """
        raw_token = secrets.token_urlsafe(48)
        token_hash = _hash_token(raw_token)
        wrapper_id = wrapper_id_override if wrapper_id_override else f"wrp_{uuid.uuid4().hex}"
        record_id = str(uuid.uuid4())
        now = utc_now().isoformat()

        capabilities = capabilities or {}
        permissions = permissions or {}
        metadata = metadata or {}

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO wrapper_tokens
                    (id, name, token_hash, wrapper_id, capabilities,
                     permissions, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    name,
                    token_hash,
                    wrapper_id,
                    json.dumps(capabilities),
                    json.dumps(permissions),
                    json.dumps(metadata),
                    now,
                ),
            )
            cursor.close()

        logger.info("[WrapperAuth] Created wrapper token: id=%s name=%r", wrapper_id, name)
        return raw_token, wrapper_id

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def validate_bearer(self, request) -> Optional[str]:
        """Validate the ``Authorization: Bearer <token>`` header on a Flask request.

        On success, slides ``last_seen_at`` to the current UTC time.

        Args:
            request: The Flask ``request`` proxy object.

        Returns:
            The ``wrapper_id`` string if the token is valid and not revoked,
            otherwise ``None``.
        """
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        raw_token = auth_header[len("Bearer "):]
        if not raw_token:
            return None

        token_hash = _hash_token(raw_token)
        now = utc_now().isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT wrapper_id FROM wrapper_tokens
                WHERE token_hash = ?
                  AND revoked_at IS NULL
                """,
                (token_hash,),
            )
            row = cursor.fetchone()

            if row is None:
                cursor.close()
                return None

            wrapper_id = row[0]

            # Slide last_seen_at
            cursor.execute(
                "UPDATE wrapper_tokens SET last_seen_at = ? WHERE wrapper_id = ?",
                (now, wrapper_id),
            )
            cursor.close()

        return wrapper_id

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    def check_permission(self, wrapper_id: str, operation: str, resource: str) -> bool:
        """Check whether a wrapper may perform ``operation`` on ``resource``.

        Permission structure::

            {
                "query":     ["memory", "threads"],   # readable resources
                "update":    ["memory"],               # writable resources
                "broadcast": true                      # may push events
            }

        Args:
            wrapper_id: The stable wrapper identifier.
            operation: One of ``"query"``, ``"update"``, or ``"broadcast"``.
            resource: The resource name (e.g. ``"memory"``, ``"threads"``).
                Ignored when ``operation`` is ``"broadcast"``.

        Returns:
            ``True`` if the permission is granted, ``False`` otherwise (including
            when the wrapper does not exist or has been revoked).
        """
        wrapper = self.get_wrapper(wrapper_id)
        if wrapper is None:
            return False

        perms = wrapper.get("permissions", {})

        if operation == "broadcast":
            return bool(perms.get("broadcast", False))

        allowed_resources = perms.get(operation, [])
        return resource in allowed_resources

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------

    def revoke(self, wrapper_id: str) -> bool:
        """Revoke a wrapper token by setting its ``revoked_at`` timestamp.

        Args:
            wrapper_id: The stable wrapper identifier to revoke.

        Returns:
            ``True`` if the wrapper was found and revoked, ``False`` if it did
            not exist or was already revoked.
        """
        now = utc_now().isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE wrapper_tokens
                SET revoked_at = ?
                WHERE wrapper_id = ?
                  AND revoked_at IS NULL
                """,
                (now, wrapper_id),
            )
            affected = cursor.rowcount
            cursor.close()

        if affected:
            logger.info("[WrapperAuth] Revoked wrapper: %s", safe(wrapper_id))
        return affected > 0

    # ------------------------------------------------------------------
    # Listing / retrieval
    # ------------------------------------------------------------------

    def list_wrappers(self) -> list[dict]:
        """Return all non-revoked wrappers.

        Returns:
            List of dicts with keys: ``id``, ``wrapper_id``, ``name``,
            ``capabilities``, ``permissions``, ``metadata``, ``last_seen_at``,
            ``created_at``.  JSON fields are decoded into Python dicts.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, wrapper_id, name, capabilities, permissions,
                       metadata, last_seen_at, created_at
                FROM wrapper_tokens
                WHERE revoked_at IS NULL
                ORDER BY created_at ASC
                """
            )
            rows = cursor.fetchall()
            cursor.close()

        return [self._row_to_dict(row) for row in rows]

    def get_wrapper(self, wrapper_id: str) -> Optional[dict]:
        """Retrieve a single non-revoked wrapper by ``wrapper_id``.

        Args:
            wrapper_id: The stable wrapper identifier.

        Returns:
            Dict with the same keys as :meth:`list_wrappers`, or ``None`` if
            the wrapper does not exist or has been revoked.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, wrapper_id, name, capabilities, permissions,
                       metadata, last_seen_at, created_at
                FROM wrapper_tokens
                WHERE wrapper_id = ?
                  AND revoked_at IS NULL
                """,
                (wrapper_id,),
            )
            row = cursor.fetchone()
            cursor.close()

        if row is None:
            return None
        return self._row_to_dict(row)

    def update_capabilities(self, wrapper_id: str, capabilities: dict) -> bool:
        """Replace the capabilities payload for a wrapper.

        Args:
            wrapper_id: The stable wrapper identifier.
            capabilities: New capabilities dict.

        Returns:
            ``True`` if the wrapper was found and updated, ``False`` otherwise.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE wrapper_tokens
                SET capabilities = ?
                WHERE wrapper_id = ?
                  AND revoked_at IS NULL
                """,
                (json.dumps(capabilities), wrapper_id),
            )
            affected = cursor.rowcount
            cursor.close()

        return affected > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row) -> dict:
        """Convert a sqlite3.Row (or sequence) to a clean wrapper dict.

        Args:
            row: A row with columns in the order returned by the SELECT
                statements in this service.

        Returns:
            Dict with JSON fields decoded and timestamps left as ISO strings.
        """
        return {
            "id": row[0],
            "wrapper_id": row[1],
            "name": row[2],
            "capabilities": json.loads(row[3]) if row[3] else {},
            "permissions": json.loads(row[4]) if row[4] else {},
            "metadata": json.loads(row[5]) if row[5] else {},
            "last_seen_at": row[6],
            "created_at": row[7],
        }

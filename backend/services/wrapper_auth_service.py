import hashlib
import json
import logging

from utils.data_utils import parse_json_column
import secrets
import uuid
from typing import Optional

from services.database_service import get_shared_db_service
from services.log_utils import safe
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class WrapperAuthService:
    def __init__(self, db=None):
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

        The raw bearer token is **not** stored server-side and cannot be recovered later.
        """
        raw_token = secrets.token_urlsafe(48)
        token_hash = _hash_token(raw_token)
        wrapper_id = wrapper_id_override or f"wrp_{uuid.uuid4().hex}"
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
        return {
            "id": row[0],
            "wrapper_id": row[1],
            "name": row[2],
            "capabilities": parse_json_column(row[3]),
            "permissions": parse_json_column(row[4]),
            "metadata": parse_json_column(row[5]),
            "last_seen_at": row[6],
            "created_at": row[7],
        }

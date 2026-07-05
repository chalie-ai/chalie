import hashlib
import json
import logging
import secrets
import sqlite3
import uuid
from typing import Optional, cast

import flask

from services.database import Database
from services.log_utils import safe
from services.time_utils import utc_now
from utils.data_utils import parse_json_column

logger = logging.getLogger(__name__)


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class WrapperAuthService:
    """Wrapper (external bearer) token CRUD — reaches the DB through the
    static :class:`~services.database.Database` gateway, no instance state."""

    # ------------------------------------------------------------------
    # Token creation
    # ------------------------------------------------------------------

    def create_token(
        self,
        name: str,
        metadata: Optional[dict[str, object]] = None,
        wrapper_id_override: Optional[str] = None,
    ) -> tuple[str, str]:
        """Create a new wrapper token and persist its hash.

        The raw bearer token is **not** stored server-side and cannot be recovered later.
        """
        raw_token = secrets.token_urlsafe(48)
        token_hash = _hash_token(raw_token)
        wrapper_id = wrapper_id_override if wrapper_id_override else f"wrp_{uuid.uuid4().hex}"
        record_id = str(uuid.uuid4())
        now = utc_now().isoformat()

        metadata = metadata or {}

        with Database.transaction() as conn:
            conn.execute(
                """
                INSERT INTO wrapper_tokens
                    (id, name, token_hash, wrapper_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    name,
                    token_hash,
                    wrapper_id,
                    json.dumps(metadata),
                    now,
                ),
            )

        logger.info("[WrapperAuth] Created wrapper token: id=%s name=%r", wrapper_id, name)
        return raw_token, wrapper_id

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def validate_bearer(self, request: "flask.Request") -> Optional[str]:
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

        with Database.transaction() as conn:
            row = conn.execute(
                """
                SELECT wrapper_id FROM wrapper_tokens
                WHERE token_hash = ?
                  AND revoked_at IS NULL
                """,
                (token_hash,),
            ).fetchone()

            if row is None:
                return None

            wrapper_id = cast(str, row[0])

            # Slide last_seen_at
            conn.execute(
                "UPDATE wrapper_tokens SET last_seen_at = ? WHERE wrapper_id = ?",
                (now, wrapper_id),
            )

        return wrapper_id

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------

    def revoke(self, wrapper_id: str) -> bool:
        now = utc_now().isoformat()

        with Database.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE wrapper_tokens
                SET revoked_at = ?
                WHERE wrapper_id = ?
                  AND revoked_at IS NULL
                """,
                (now, wrapper_id),
            )
            affected = cursor.rowcount

        if affected:
            logger.info("[WrapperAuth] Revoked wrapper: %s", safe(wrapper_id))
        return affected > 0

    # ------------------------------------------------------------------
    # Listing / retrieval
    # ------------------------------------------------------------------

    def list_wrappers(self) -> list[dict[str, object]]:
        conn = Database.conn()
        rows = conn.execute(
            """
            SELECT id, wrapper_id, name, metadata, last_seen_at, created_at
            FROM wrapper_tokens
            WHERE revoked_at IS NULL
            ORDER BY created_at ASC
            """
        ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_wrapper(self, wrapper_id: str) -> Optional[dict[str, object]]:
        conn = Database.conn()
        row = conn.execute(
            """
            SELECT id, wrapper_id, name, metadata, last_seen_at, created_at
            FROM wrapper_tokens
            WHERE wrapper_id = ?
              AND revoked_at IS NULL
            """,
            (wrapper_id,),
        ).fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row[0],
            "wrapper_id": row[1],
            "name": row[2],
            "metadata": parse_json_column(row[3]),
            "last_seen_at": row[4],
            "created_at": row[5],
        }

"""
Browser Credential Vault — Encrypted storage for site credentials.

Credentials are encrypted with the vault DEK (AES-256-GCM) via
:func:`~services.vault_service.get_vault_service`.  Domain-scoped: cookies for
github.com are never injected into other domains.

Credential types:
  - password: username + password, injected via fill steps
  - cookie:   session cookies captured after login, replayed in browser context
  - header:   custom HTTP headers injected into requests to the domain
"""

import json
import logging

from services.time_utils import utc_now

logger = logging.getLogger(__name__)


class CredentialVault:
    """Encrypted credential storage backed by SQLite."""

    def __init__(self, db_service):
        self.db = db_service

    def store(self, account_id: int, domain: str, label: str,
              credential_type: str, data: dict) -> int:
        """AES-256-GCM–encrypt *data* via VaultService and persist the credential.

        The credential dict is serialised to JSON, encrypted with the vault DEK,
        and base64-encoded so it can be stored safely in the TEXT
        ``encrypted_data`` column.

        Args:
            account_id:       ID of the owning account row.
            domain:           Site domain (e.g. ``"github.com"``).
            label:            Human-readable identifier for this credential.
            credential_type:  One of ``"password"``, ``"cookie"``, or ``"header"``.
            data:             Arbitrary dict of credential fields to protect.

        Returns:
            The SQLite row-id of the inserted or updated ``browser_credentials`` row.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the vault has not
                been unlocked before this call.
        """
        import base64
        from services.vault_service import get_vault_service
        now = utc_now().isoformat()
        encrypted = base64.b64encode(
            get_vault_service().encrypt_str(json.dumps(data))
        ).decode()

        with self.db.connection() as conn:
            cursor = conn.cursor()
            # Upsert — update if exists, insert if not
            cursor.execute("""
                INSERT INTO browser_credentials
                    (account_id, domain, label, credential_type, encrypted_data,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, domain, label)
                DO UPDATE SET
                    credential_type = excluded.credential_type,
                    encrypted_data = excluded.encrypted_data,
                    updated_at = excluded.updated_at
            """, (account_id, domain, label, credential_type, encrypted, now, now))
            row_id = cursor.lastrowid
            cursor.close()
        return row_id

    def load(self, account_id: int, domain: str, label: str | None = None) -> list[dict]:
        """Decrypt and return stored credentials for *domain*.

        Each row's ``encrypted_data`` is decoded from base64 and decrypted via
        :class:`~services.vault_service.VaultService`.  Rows whose value is stored
        as raw bytes (produced by the one-time legacy migration) are passed to the
        vault directly without a base64 step.  Rows that fail decryption (locked
        vault, corrupted data) are silently skipped and logged at WARNING level.

        Args:
            account_id: ID of the owning account row.
            domain:     Site domain to filter by (e.g. ``"github.com"``).
            label:      Optional label to narrow the query to a single credential.
                        Pass ``None`` to return all credentials for *domain*.

        Returns:
            List of dicts, each containing ``id``, ``domain``, ``label``,
            ``credential_type``, and ``data`` (the decrypted credential dict).
        """
        import base64
        from services.vault_service import get_vault_service
        with self.db.connection() as conn:
            cursor = conn.cursor()
            if label:
                cursor.execute("""
                    SELECT id, domain, label, credential_type, encrypted_data
                    FROM browser_credentials
                    WHERE account_id = ? AND domain = ? AND label = ?
                """, (account_id, domain, label))
            else:
                cursor.execute("""
                    SELECT id, domain, label, credential_type, encrypted_data
                    FROM browser_credentials
                    WHERE account_id = ? AND domain = ?
                """, (account_id, domain))

            results = []
            for row in cursor.fetchall():
                try:
                    vault = get_vault_service()
                    encrypted_val = row[4]
                    # Raw bytes / memoryview → migrated BLOB; pass directly.
                    if isinstance(encrypted_val, (bytes, memoryview)):
                        raw = bytes(encrypted_val)
                    else:
                        # String → base64-encoded new format; decode before passing.
                        raw = base64.b64decode(encrypted_val.encode())
                    decrypted = json.loads(vault.decrypt_str(raw))
                except Exception as e:
                    logger.warning("[CRED VAULT] Decryption failed for %s/%s: %s",
                                   row[1], row[2], e)
                    continue
                results.append({
                    "id": row[0],
                    "domain": row[1],
                    "label": row[2],
                    "credential_type": row[3],
                    "data": decrypted,
                })
            cursor.close()

            # Touch last_used_at
            if results:
                now = utc_now().isoformat()
                ids = [r["id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE browser_credentials SET last_used_at = ? WHERE id IN ({placeholders})",
                    [now] + ids,
                )

        return results

    def list_credentials(self, account_id: int) -> list[dict]:
        """List credentials without decrypting.  Safe for API responses."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, domain, label, credential_type, created_at, last_used_at
                FROM browser_credentials
                WHERE account_id = ?
                ORDER BY domain, label
            """, (account_id,))
            results = [
                {
                    "id": row[0], "domain": row[1], "label": row[2],
                    "credential_type": row[3], "created_at": row[4],
                    "last_used_at": row[5],
                }
                for row in cursor.fetchall()
            ]
            cursor.close()
        return results

    def delete(self, account_id: int, credential_id: int) -> bool:
        """Delete a credential.  Returns True if deleted."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM browser_credentials
                WHERE id = ? AND account_id = ?
            """, (credential_id, account_id))
            deleted = cursor.rowcount > 0
            cursor.close()
        return deleted

    def inject_cookies(self, context, account_id: int, domain: str):
        """Inject stored cookies into a Playwright browser context.

        Loads all 'cookie' type credentials for the domain and adds them
        to the context.  No-op if no cookies found.
        """
        creds = self.load(account_id, domain)
        cookies = [c for c in creds if c["credential_type"] == "cookie"]
        if not cookies:
            return

        all_cookies = []
        for cred in cookies:
            cookie_list = cred["data"].get("cookies", [])
            all_cookies.extend(cookie_list)

        if all_cookies:
            try:
                context.add_cookies(all_cookies)
                logger.info("[CRED VAULT] Injected %d cookies for %s",
                            len(all_cookies), domain)
            except Exception as e:
                logger.warning("[CRED VAULT] Cookie injection failed for %s: %s",
                               domain, e)

    def capture_cookies(self, context, account_id: int, domain: str, label: str):
        """Capture cookies from browser context and store encrypted.

        Called after a successful login to persist the session.
        """
        try:
            all_cookies = context.cookies()
            # Filter to cookies relevant to this domain
            domain_cookies = [
                c for c in all_cookies
                if domain in (c.get("domain", "").lstrip("."))
            ]
            if domain_cookies:
                self.store(account_id, domain, label, "cookie",
                           {"cookies": domain_cookies})
                logger.info("[CRED VAULT] Captured %d cookies for %s/%s",
                            len(domain_cookies), domain, label)
        except Exception as e:
            logger.warning("[CRED VAULT] Cookie capture failed: %s", e)


# -- Snapshot storage (for monitor action) ------------------------------------

class SnapshotStore:
    """Simple key-value store for page monitoring snapshots."""

    def __init__(self, db_service):
        self.db = db_service

    def get(self, account_id: int, snapshot_key: str) -> dict | None:
        """Load the most recent snapshot for a key."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT url, content_hash, content_text, captured_at
                FROM browser_snapshots
                WHERE account_id = ? AND snapshot_key = ?
            """, (account_id, snapshot_key))
            row = cursor.fetchone()
            cursor.close()
        if not row:
            return None
        return {
            "url": row[0],
            "content_hash": row[1],
            "content_text": row[2],
            "captured_at": row[3],
        }

    def save(self, account_id: int, snapshot_key: str, url: str,
             content_hash: str, content_text: str):
        """Save or update a snapshot (upsert)."""
        now = utc_now().isoformat()
        with self.db.connection() as conn:
            conn.execute("""
                INSERT INTO browser_snapshots
                    (account_id, snapshot_key, url, content_hash, content_text, captured_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, snapshot_key)
                DO UPDATE SET
                    url = excluded.url,
                    content_hash = excluded.content_hash,
                    content_text = excluded.content_text,
                    captured_at = excluded.captured_at
            """, (account_id, snapshot_key, url, content_hash, content_text, now))

    def delete(self, account_id: int, snapshot_key: str) -> bool:
        """Delete a snapshot."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM browser_snapshots
                WHERE account_id = ? AND snapshot_key = ?
            """, (account_id, snapshot_key))
            deleted = cursor.rowcount > 0
            cursor.close()
        return deleted

    def list_snapshots(self, account_id: int) -> list[dict]:
        """List all snapshots for an account."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT snapshot_key, url, captured_at
                FROM browser_snapshots
                WHERE account_id = ?
                ORDER BY captured_at DESC
            """, (account_id,))
            results = [
                {"snapshot_key": row[0], "url": row[1], "captured_at": row[2]}
                for row in cursor.fetchall()
            ]
            cursor.close()
        return results

    def prune_stale(self, max_age_days: int = 30):
        """Delete snapshots older than max_age_days."""
        with self.db.connection() as conn:
            conn.execute("""
                DELETE FROM browser_snapshots
                WHERE captured_at < datetime('now', ?)
            """, (f"-{max_age_days} days",))

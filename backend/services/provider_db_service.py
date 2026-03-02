"""Provider database service — manages provider configuration in DB (SQLite)."""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ProviderDbService:
    """Manages provider configuration in database."""

    def __init__(self, database_service):
        self.db = database_service
        self._enc_key = None

    def _get_enc_key(self):
        """Lazily load encryption key from .key file."""
        if self._enc_key is None:
            from services.encryption_key_service import get_encryption_key
            self._enc_key = get_encryption_key()
        return self._enc_key

    def _encrypt(self, value: str) -> str:
        """Encrypt a value using the encryption key (Python-level, HMAC-based obfuscation).

        Uses Fernet-style encryption via the standard library.
        Falls back to base64 encoding if cryptography is not available.
        """
        if value is None:
            return None
        try:
            import base64
            import hashlib
            from cryptography.fernet import Fernet
            # Derive a Fernet-compatible key from the encryption key
            key_bytes = hashlib.sha256(self._get_enc_key().encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            f = Fernet(fernet_key)
            return f.encrypt(value.encode()).decode()
        except ImportError:
            # Fallback: base64 encode (not truly secure, but preserves data)
            # TODO: Install cryptography package for proper encryption
            import base64
            return base64.urlsafe_b64encode(value.encode()).decode()

    def _decrypt(self, value: str) -> str:
        """Decrypt a value encrypted by _encrypt."""
        if value is None:
            return None
        try:
            import base64
            import hashlib
            from cryptography.fernet import Fernet
            key_bytes = hashlib.sha256(self._get_enc_key().encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            f = Fernet(fernet_key)
            return f.decrypt(value.encode()).decode()
        except ImportError:
            import base64
            return base64.urlsafe_b64decode(value.encode()).decode()

    def _row_to_provider(self, row) -> Dict[str, Any]:
        """Convert a database row to a provider dict, decrypting api_key."""
        api_key_raw = row['api_key'] if isinstance(row, dict) else row[5]
        api_key = self._decrypt(api_key_raw) if api_key_raw else None

        if isinstance(row, dict):
            return {
                "id": row['id'],
                "name": row['name'],
                "platform": row['platform'],
                "model": row['model'],
                "host": row['host'],
                "api_key": api_key,
                "dimensions": row['dimensions'],
                "timeout": row['timeout'],
                "is_active": bool(row['is_active']),
            }
        return {
            "id": row[0],
            "name": row[1],
            "platform": row[2],
            "model": row[3],
            "host": row[4],
            "api_key": api_key,
            "dimensions": row[6],
            "timeout": row[7],
            "is_active": bool(row[8]),
        }

    def get_all_providers(self) -> List[Dict[str, Any]]:
        """Get all active providers."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, api_key, "
                "dimensions, timeout, is_active "
                "FROM providers WHERE is_active = 1 ORDER BY name"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_provider(row) for row in rows]

    def list_providers_summary(self) -> List[Dict[str, Any]]:
        """Get all active providers without decrypting api_key (for REST listings)."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, "
                "(api_key IS NOT NULL) AS has_api_key, "
                "dimensions, timeout, is_active "
                "FROM providers WHERE is_active = 1 ORDER BY name"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "platform": row[2],
                    "model": row[3],
                    "host": row[4],
                    "api_key": "***" if row[5] else None,
                    "dimensions": row[6],
                    "timeout": row[7],
                    "is_active": bool(row[8]),
                }
                for row in rows
            ]

    def get_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get provider by name."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, api_key, "
                "dimensions, timeout, is_active "
                "FROM providers WHERE name = ? AND is_active = 1",
                (name,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return self._row_to_provider(row)

    def get_provider_by_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get provider by ID."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, api_key, "
                "dimensions, timeout, is_active "
                "FROM providers WHERE id = ?",
                (provider_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return self._row_to_provider(row)

    def create_provider(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new provider."""
        api_key_val = data.get("api_key")
        encrypted_key = self._encrypt(api_key_val) if api_key_val else None

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO providers (name, platform, model, host, api_key, dimensions, timeout, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["name"],
                    data["platform"],
                    data["model"],
                    data.get("host"),
                    encrypted_key,
                    data.get("dimensions"),
                    data.get("timeout", 120),
                    1 if data.get("is_active", True) else 0,
                )
            )
            new_id = cursor.lastrowid
            cursor.close()

        # Fetch the newly created row and return it
        return self.get_provider_by_id(new_id)

    def update_provider(self, provider_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a provider."""
        updates = []
        params = []

        for key in ["name", "platform", "model", "host", "dimensions", "timeout"]:
            if key in data:
                updates.append(f"{key} = ?")
                params.append(data[key])

        if "is_active" in data:
            updates.append("is_active = ?")
            params.append(1 if data["is_active"] else 0)

        # Handle api_key separately for encryption
        if "api_key" in data:
            if data["api_key"] is None:
                updates.append("api_key = NULL")
            else:
                updates.append("api_key = ?")
                params.append(self._encrypt(data["api_key"]))

        if not updates:
            return self.get_provider_by_id(provider_id)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(provider_id)

        query = f"UPDATE providers SET {', '.join(updates)} WHERE id = ?"

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            cursor.close()

        return self.get_provider_by_id(provider_id)

    def delete_provider(self, provider_id: int) -> bool:
        """Delete a provider (sets is_active to FALSE)."""
        # Check if provider is referenced by any job assignment
        assignment = self.get_job_assignment_by_provider_id(provider_id)
        if assignment:
            raise ValueError(f"Cannot delete provider {provider_id}; it is referenced by job '{assignment['job_name']}'")

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE providers SET is_active = 0 WHERE id = ?",
                (provider_id,)
            )
            cursor.close()
        return True

    def get_all_job_assignments(self) -> List[Dict[str, Any]]:
        """Get all job->provider assignments."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_name, provider_id FROM job_provider_assignments"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [
                {
                    "job_name": row[0],
                    "provider_id": row[1],
                }
                for row in rows
            ]

    def get_job_assignment(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get provider assignment for a job."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_name, provider_id FROM job_provider_assignments WHERE job_name = ?",
                (job_name,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return {
                "job_name": row[0],
                "provider_id": row[1],
            }

    def get_job_assignment_by_provider_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get job assignment by provider ID (for deletion check)."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_name, provider_id FROM job_provider_assignments WHERE provider_id = ? LIMIT 1",
                (provider_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return {
                "job_name": row[0],
                "provider_id": row[1],
            }

    def set_job_assignment(self, job_name: str, provider_id: int) -> Dict[str, Any]:
        """Create or update a job->provider assignment."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            # Check if assignment exists
            cursor.execute(
                "SELECT id FROM job_provider_assignments WHERE job_name = ?",
                (job_name,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update
                cursor.execute(
                    "UPDATE job_provider_assignments SET provider_id = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE job_name = ?",
                    (provider_id, job_name)
                )
            else:
                # Insert
                cursor.execute(
                    "INSERT INTO job_provider_assignments (job_name, provider_id) VALUES (?, ?)",
                    (job_name, provider_id)
                )

            cursor.close()
            return {
                "job_name": job_name,
                "provider_id": provider_id,
            }

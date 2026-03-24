"""Provider database service — manages provider configuration in DB (SQLite)."""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _infer_vision_support(platform: str, model: str) -> bool:
    """Infer vision support from platform and model name."""
    model_lower = model.lower()
    if platform in ('anthropic', 'claude_code'):
        return any(x in model_lower for x in ('sonnet', 'opus', 'haiku'))
    elif platform == 'openai':
        return any(x in model_lower for x in ('gpt-4', 'gpt-5'))
    elif platform == 'gemini':
        return True  # All Gemini models support vision
    elif platform == 'ollama':
        return any(x in model_lower for x in ('llava', 'vision', 'bakllava'))
    return False


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
        """Encrypt a value using Fernet (symmetric, HMAC-authenticated).

        Requires the ``cryptography`` package (mandatory dependency).
        """
        if value is None:
            return None
        import base64
        import hashlib
        from cryptography.fernet import Fernet
        # Derive a Fernet-compatible key from the encryption key
        key_bytes = hashlib.sha256(self._get_enc_key().encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        f = Fernet(fernet_key)
        return f.encrypt(value.encode()).decode()

    def _decrypt(self, value: str) -> str:
        """Decrypt a value encrypted by _encrypt.

        Handles legacy values gracefully: if Fernet decryption fails
        (e.g. value was stored as plain base64 or plaintext before Fernet
        was enforced), falls back to base64 decode, then returns raw value
        so existing installations are not bricked on upgrade.
        """
        if value is None:
            return None
        import base64
        import hashlib
        from cryptography.fernet import Fernet
        key_bytes = hashlib.sha256(self._get_enc_key().encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        f = Fernet(fernet_key)
        try:
            return f.decrypt(value.encode()).decode()
        except Exception:
            # Legacy fallback: try base64 decode, then return raw
            try:
                return base64.b64decode(value).decode()
            except Exception:
                return value

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
                "supports_vision": bool(row.get('supports_vision', 0)),
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
            "supports_vision": bool(row[9]) if len(row) > 9 else False,
        }

    def get_all_providers(self) -> List[Dict[str, Any]]:
        """Get all active providers."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, api_key, "
                "dimensions, timeout, is_active, supports_vision "
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
                "dimensions, timeout, is_active, supports_vision "
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
                    "supports_vision": bool(row[9]),
                }
                for row in rows
            ]

    def get_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get provider by name."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, api_key, "
                "dimensions, timeout, is_active, supports_vision "
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
                "dimensions, timeout, is_active, supports_vision "
                "FROM providers WHERE id = ? AND is_active = 1",
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

        # Auto-infer vision support if not explicitly provided
        if 'supports_vision' in data:
            vision = 1 if data['supports_vision'] else 0
        else:
            vision = 1 if _infer_vision_support(data.get('platform', ''), data.get('model', '')) else 0

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO providers (name, platform, model, host, api_key, dimensions, timeout, is_active, supports_vision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["name"],
                    data["platform"],
                    data["model"],
                    data.get("host"),
                    encrypted_key,
                    data.get("dimensions"),
                    data.get("timeout", 120),
                    1 if data.get("is_active", True) else 0,
                    vision,
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

        if "supports_vision" in data:
            updates.append("supports_vision = ?")
            params.append(1 if data["supports_vision"] else 0)

        if "is_active" in data:
            updates.append("is_active = ?")
            params.append(1 if data["is_active"] else 0)

        # Auto-infer vision support if platform or model changed and supports_vision not explicit
        if 'supports_vision' not in data and ('platform' in data or 'model' in data):
            current = self.get_provider_by_id(provider_id)
            if current:
                platform = data.get('platform', current.get('platform', ''))
                model = data.get('model', current.get('model', ''))
                updates.append("supports_vision = ?")
                params.append(1 if _infer_vision_support(platform, model) else 0)

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

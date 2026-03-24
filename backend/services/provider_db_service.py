"""Provider database service — manages provider configuration in DB (SQLite)."""

import json
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


def _parse_models(models_json: Optional[str], default_model: str) -> List[str]:
    """Parse models JSON array, falling back to [default_model] if NULL."""
    if models_json:
        try:
            parsed = json.loads(models_json)
            if isinstance(parsed, list) and parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return [default_model]


def compute_job_group(caps: Dict[str, str]) -> str:
    """Derive capability group from a job's caps dict.

    Priority order (first match wins):
      1. vision != none  → 'vision'
      2. reasoning=high OR creativity=high → 'reasoning'
      3. structured=high OR classification=high → 'analytical'
      4. everything else → 'utility'
    """
    if caps.get('vision', 'none') != 'none':
        return 'vision'
    if caps.get('reasoning') == 'high' or caps.get('creativity') == 'high':
        return 'reasoning'
    if caps.get('structured') == 'high' or caps.get('classification') == 'high':
        return 'analytical'
    return 'utility'


def load_jobs_for_group(group_name: str) -> List[str]:
    """Return job IDs belonging to a capability group."""
    import os
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'cognitive_jobs.json')
    with open(config_path, 'r') as f:
        all_jobs = json.load(f).get('jobs', [])
    return [j['id'] for j in all_jobs if compute_job_group(j.get('caps', {})) == group_name]


class ProviderDbService:
    """Manages provider configuration in database."""

    # Column list used by all SELECT queries — order matters for positional access
    _PROVIDER_COLS = (
        "id, name, platform, model, models, host, api_key, "
        "dimensions, timeout, is_active, supports_vision"
    )

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
        """Convert a database row to a provider dict, decrypting api_key.

        Column order: id, name, platform, model, models, host, api_key,
                      dimensions, timeout, is_active, supports_vision
        """
        if isinstance(row, dict):
            api_key = self._decrypt(row['api_key']) if row.get('api_key') else None
            default_model = row['model']
            return {
                "id": row['id'],
                "name": row['name'],
                "platform": row['platform'],
                "model": default_model,
                "models": _parse_models(row.get('models'), default_model),
                "host": row['host'],
                "api_key": api_key,
                "dimensions": row['dimensions'],
                "timeout": row['timeout'],
                "is_active": bool(row['is_active']),
                "supports_vision": bool(row.get('supports_vision', 0)),
            }
        # Positional access (tuple row)
        api_key = self._decrypt(row[6]) if row[6] else None
        default_model = row[3]
        return {
            "id": row[0],
            "name": row[1],
            "platform": row[2],
            "model": default_model,
            "models": _parse_models(row[4], default_model),
            "host": row[5],
            "api_key": api_key,
            "dimensions": row[7],
            "timeout": row[8],
            "is_active": bool(row[9]),
            "supports_vision": bool(row[10]) if len(row) > 10 else False,
        }

    def get_all_providers(self) -> List[Dict[str, Any]]:
        """Get all active providers."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._PROVIDER_COLS} "
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
                "SELECT id, name, platform, model, models, host, "
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
                    "models": _parse_models(row[4], row[3]),
                    "host": row[5],
                    "api_key": "***" if row[6] else None,
                    "dimensions": row[7],
                    "timeout": row[8],
                    "is_active": bool(row[9]),
                    "supports_vision": bool(row[10]),
                }
                for row in rows
            ]

    def get_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get provider by name."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._PROVIDER_COLS} "
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
                f"SELECT {self._PROVIDER_COLS} "
                "FROM providers WHERE id = ? AND is_active = 1",
                (provider_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return self._row_to_provider(row)

    def create_provider(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new provider.

        Accepts either:
          - ``model`` (str): single model, stored as default and models=[model]
          - ``models`` (list[str]): multiple models, first becomes the default
          - Both: ``models`` takes precedence for the list, ``model`` for the default
        """
        models_list = data.get("models")
        default_model = data.get("model")

        if models_list and isinstance(models_list, list) and models_list:
            if not default_model:
                default_model = models_list[0]
        elif default_model:
            models_list = [default_model]
        else:
            raise ValueError("Either 'model' or 'models' is required")

        api_key_val = data.get("api_key")
        encrypted_key = self._encrypt(api_key_val) if api_key_val else None

        # Auto-infer vision support if not explicitly provided
        if 'supports_vision' in data:
            vision = 1 if data['supports_vision'] else 0
        else:
            vision = 1 if _infer_vision_support(data.get('platform', ''), default_model) else 0

        models_json = json.dumps(models_list)

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO providers (name, platform, model, models, host, api_key, "
                "dimensions, timeout, is_active, supports_vision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["name"],
                    data["platform"],
                    default_model,
                    models_json,
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

        # Handle models list
        if "models" in data:
            models_list = data["models"]
            if isinstance(models_list, list) and models_list:
                updates.append("models = ?")
                params.append(json.dumps(models_list))
                # If model not explicitly set, update default to first in list
                if "model" not in data:
                    updates.append("model = ?")
                    params.append(models_list[0])

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

    # ── Job Assignments ──────────────────────────────────────────

    def get_all_job_assignments(self) -> List[Dict[str, Any]]:
        """Get all job->provider assignments."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_name, provider_id, model FROM job_provider_assignments"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [
                {
                    "job_name": row[0],
                    "provider_id": row[1],
                    "model": row[2],  # None if using provider default
                }
                for row in rows
            ]

    def get_job_assignment(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get provider assignment for a job."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_name, provider_id, model FROM job_provider_assignments WHERE job_name = ?",
                (job_name,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return {
                "job_name": row[0],
                "provider_id": row[1],
                "model": row[2],
            }

    def get_job_assignment_by_provider_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get job assignment by provider ID (for deletion check)."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_name, provider_id, model FROM job_provider_assignments WHERE provider_id = ? LIMIT 1",
                (provider_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return {
                "job_name": row[0],
                "provider_id": row[1],
                "model": row[2],
            }

    def set_job_assignment(self, job_name: str, provider_id: int, model: Optional[str] = None) -> Dict[str, Any]:
        """Create or update a job->provider assignment with optional model override."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM job_provider_assignments WHERE job_name = ?",
                (job_name,)
            )
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    "UPDATE job_provider_assignments SET provider_id = ?, model = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE job_name = ?",
                    (provider_id, model, job_name)
                )
            else:
                cursor.execute(
                    "INSERT INTO job_provider_assignments (job_name, provider_id, model) VALUES (?, ?, ?)",
                    (job_name, provider_id, model)
                )

            cursor.close()
            return {
                "job_name": job_name,
                "provider_id": provider_id,
                "model": model,
            }

    def set_group_assignments(self, group_name: str, provider_id: int, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Batch-assign a provider+model to all jobs in a capability group.

        Group membership is derived from cognitive_jobs.json caps.
        """
        job_ids = load_jobs_for_group(group_name)
        if not job_ids:
            raise ValueError(f"Unknown or empty group: '{group_name}'")

        results = []
        for job_id in job_ids:
            result = self.set_job_assignment(job_id, provider_id, model)
            results.append(result)
        return results

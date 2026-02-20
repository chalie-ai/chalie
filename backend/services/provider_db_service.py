"""Provider database service — manages provider configuration in DB."""

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

    def get_all_providers(self) -> List[Dict[str, Any]]:
        """Get all active providers."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT id, name, platform, model, host, "
                     "CASE WHEN api_key IS NULL THEN NULL "
                     "     ELSE pgp_sym_decrypt(api_key, :enc_key) END AS api_key, "
                     "dimensions, timeout, is_active "
                     "FROM providers WHERE is_active = TRUE ORDER BY name"),
                {"enc_key": self._get_enc_key()}
            )
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "platform": row[2],
                    "model": row[3],
                    "host": row[4],
                    "api_key": row[5],
                    "dimensions": row[6],
                    "timeout": row[7],
                    "is_active": row[8],
                }
                for row in rows
            ]

    def list_providers_summary(self) -> List[Dict[str, Any]]:
        """Get all active providers without decrypting api_key (for REST listings)."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT id, name, platform, model, host, "
                     "(api_key IS NOT NULL) AS has_api_key, "
                     "dimensions, timeout, is_active "
                     "FROM providers WHERE is_active = TRUE ORDER BY name")
            )
            rows = result.fetchall()
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
                    "is_active": row[8],
                }
                for row in rows
            ]

    def get_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get provider by name."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT id, name, platform, model, host, "
                     "CASE WHEN api_key IS NULL THEN NULL "
                     "     ELSE pgp_sym_decrypt(api_key, :enc_key) END AS api_key, "
                     "dimensions, timeout, is_active "
                     "FROM providers WHERE name = :name AND is_active = TRUE"),
                {"name": name, "enc_key": self._get_enc_key()}
            )
            row = result.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "name": row[1],
                "platform": row[2],
                "model": row[3],
                "host": row[4],
                "api_key": row[5],
                "dimensions": row[6],
                "timeout": row[7],
                "is_active": row[8],
            }

    def get_provider_by_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get provider by ID."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT id, name, platform, model, host, "
                     "CASE WHEN api_key IS NULL THEN NULL "
                     "     ELSE pgp_sym_decrypt(api_key, :enc_key) END AS api_key, "
                     "dimensions, timeout, is_active "
                     "FROM providers WHERE id = :id"),
                {"id": provider_id, "enc_key": self._get_enc_key()}
            )
            row = result.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "name": row[1],
                "platform": row[2],
                "model": row[3],
                "host": row[4],
                "api_key": row[5],
                "dimensions": row[6],
                "timeout": row[7],
                "is_active": row[8],
            }

    def create_provider(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new provider."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("INSERT INTO providers (name, platform, model, host, api_key, dimensions, timeout, is_active) "
                     "VALUES (:name, :platform, :model, :host, "
                     "        CASE WHEN :api_key_val IS NULL THEN NULL "
                     "             ELSE pgp_sym_encrypt(:api_key_val, :enc_key) END, "
                     "        :dimensions, :timeout, :is_active) "
                     "RETURNING id, name, platform, model, host, "
                     "          CASE WHEN api_key IS NULL THEN NULL "
                     "               ELSE pgp_sym_decrypt(api_key, :enc_key) END AS api_key, "
                     "          dimensions, timeout, is_active"),
                {
                    "name": data["name"],
                    "platform": data["platform"],
                    "model": data["model"],
                    "host": data.get("host"),
                    "api_key_val": data.get("api_key"),
                    "dimensions": data.get("dimensions"),
                    "timeout": data.get("timeout", 120),
                    "is_active": data.get("is_active", True),
                    "enc_key": self._get_enc_key(),
                }
            )
            session.commit()
            row = result.fetchone()
            return {
                "id": row[0],
                "name": row[1],
                "platform": row[2],
                "model": row[3],
                "host": row[4],
                "api_key": row[5],
                "dimensions": row[6],
                "timeout": row[7],
                "is_active": row[8],
            }

    def update_provider(self, provider_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a provider."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            updates = []
            params = {"id": provider_id}

            for key in ["name", "platform", "model", "host", "dimensions", "timeout", "is_active"]:
                if key in data:
                    updates.append(f"{key} = :{key}")
                    params[key] = data[key]

            # Handle api_key separately for encryption
            if "api_key" in data:
                if data["api_key"] is None:
                    updates.append("api_key = NULL")
                else:
                    updates.append("api_key = pgp_sym_encrypt(:api_key_val, :enc_key)")
                    params["api_key_val"] = data["api_key"]

            if not updates:
                return self.get_provider_by_id(provider_id)

            # Always include enc_key for RETURNING clause
            params["enc_key"] = self._get_enc_key()

            query = f"UPDATE providers SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP " \
                    f"WHERE id = :id RETURNING id, name, platform, model, host, " \
                    f"CASE WHEN api_key IS NULL THEN NULL " \
                    f"     ELSE pgp_sym_decrypt(api_key, :enc_key) END AS api_key, " \
                    f"dimensions, timeout, is_active"

            result = session.execute(text(query), params)
            session.commit()
            row = result.fetchone()
            return {
                "id": row[0],
                "name": row[1],
                "platform": row[2],
                "model": row[3],
                "host": row[4],
                "api_key": row[5],
                "dimensions": row[6],
                "timeout": row[7],
                "is_active": row[8],
            }

    def delete_provider(self, provider_id: int) -> bool:
        """Delete a provider (sets is_active to FALSE)."""
        # Check if provider is referenced by any job assignment
        assignment = self.get_job_assignment_by_provider_id(provider_id)
        if assignment:
            raise ValueError(f"Cannot delete provider {provider_id}; it is referenced by job '{assignment['job_name']}'")

        with self.db.get_session() as session:
            from sqlalchemy import text
            session.execute(
                text("UPDATE providers SET is_active = FALSE WHERE id = :id"),
                {"id": provider_id}
            )
            session.commit()
        return True

    def get_all_job_assignments(self) -> List[Dict[str, Any]]:
        """Get all job→provider assignments."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT job_name, provider_id FROM job_provider_assignments")
            )
            rows = result.fetchall()
            return [
                {
                    "job_name": row[0],
                    "provider_id": row[1],
                }
                for row in rows
            ]

    def get_job_assignment(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get provider assignment for a job."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT job_name, provider_id FROM job_provider_assignments WHERE job_name = :job_name"),
                {"job_name": job_name}
            )
            row = result.fetchone()
            if not row:
                return None
            return {
                "job_name": row[0],
                "provider_id": row[1],
            }

    def get_job_assignment_by_provider_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get job assignment by provider ID (for deletion check)."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT job_name, provider_id FROM job_provider_assignments WHERE provider_id = :provider_id LIMIT 1"),
                {"provider_id": provider_id}
            )
            row = result.fetchone()
            if not row:
                return None
            return {
                "job_name": row[0],
                "provider_id": row[1],
            }

    def set_job_assignment(self, job_name: str, provider_id: int) -> Dict[str, Any]:
        """Create or update a job→provider assignment."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            # Check if assignment exists
            result = session.execute(
                text("SELECT id FROM job_provider_assignments WHERE job_name = :job_name"),
                {"job_name": job_name}
            )
            existing = result.fetchone()

            if existing:
                # Update
                session.execute(
                    text("UPDATE job_provider_assignments SET provider_id = :provider_id, updated_at = CURRENT_TIMESTAMP "
                         "WHERE job_name = :job_name"),
                    {"job_name": job_name, "provider_id": provider_id}
                )
            else:
                # Insert
                session.execute(
                    text("INSERT INTO job_provider_assignments (job_name, provider_id) VALUES (:job_name, :provider_id)"),
                    {"job_name": job_name, "provider_id": provider_id}
                )

            session.commit()
            return {
                "job_name": job_name,
                "provider_id": provider_id,
            }


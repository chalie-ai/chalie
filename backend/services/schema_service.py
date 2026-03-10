"""
Schema Service — SQLite database initialization.

Replaces PostgreSQL-specific schema management. For SQLite:
- database_exists() checks if the DB file exists
- create_database() creates the file and loads schema.sql
- initialize_schema() runs the consolidated schema.sql
"""

import logging
import os
from pathlib import Path

from services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class SchemaService:
    """Manages SQLite database schema initialization and versioning."""

    def __init__(self, database_service: DatabaseService, embedding_dimensions: int = 768):
        self.db_service = database_service
        self.embedding_dimensions = embedding_dimensions
        self._schema_path = Path(__file__).resolve().parent.parent / "schema.sql"

    def database_exists(self, db_name: str = None) -> bool:
        """Check if the SQLite database file exists."""
        return os.path.exists(self.db_service.db_path)

    def create_database(self, db_name: str = None):
        """Create the database by running the consolidated schema."""
        self.initialize_schema()
        logger.info(f"Database created at {self.db_service.db_path}")

    def initialize_schema(self):
        """
        Initialize database schema from schema.sql (idempotent).
        All CREATE TABLE statements use IF NOT EXISTS.
        """
        if not self._schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self._schema_path}")

        schema_sql = self._schema_path.read_text()

        with self.db_service.connection() as conn:
            conn.executescript(schema_sql)

            # Create sqlite-vec virtual tables for vector search
            self._create_vec_tables(conn)

            logger.info("Schema initialized successfully")

    def ensure_vec_tables(self):
        """Ensure all sqlite-vec virtual tables exist. Idempotent — safe to call on every startup."""
        with self.db_service.connection() as conn:
            self._create_vec_tables(conn)

    def _create_vec_tables(self, conn):
        """Create sqlite-vec companion virtual tables for vector columns."""
        vec_tables = [
            ("episodes_vec", self.embedding_dimensions),
            ("semantic_concepts_vec", self.embedding_dimensions),
            ("topics_vec", self.embedding_dimensions),
            ("user_traits_vec", self.embedding_dimensions),
            ("tool_capability_profiles_vec", self.embedding_dimensions),
            ("cognitive_reflexes_vec", self.embedding_dimensions),
            ("documents_vec", self.embedding_dimensions),
            ("document_chunks_vec", self.embedding_dimensions),
            ("scheduled_items_vec", self.embedding_dimensions),
            ("persistent_tasks_vec", self.embedding_dimensions),
        ]

        for table_name, dims in vec_tables:
            try:
                conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}
                    USING vec0(embedding float[{dims}])
                """)
            except Exception as e:
                logger.warning(f"Could not create vec table {table_name}: {e}")

    def schema_version(self) -> int:
        """Get current schema version (0 if not initialized)."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Check if schema_version table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='schema_version'
                """)
                if cursor.fetchone() is None:
                    return 0

                cursor.execute("SELECT MAX(version) FROM schema_version")
                row = cursor.fetchone()
                version = row[0] if row else None
                return version if version is not None else 0

        except Exception as e:
            logger.error(f"Failed to get schema version: {e}")
            return 0

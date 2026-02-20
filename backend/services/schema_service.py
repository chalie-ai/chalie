"""
Schema Service - Database schema initialization and migrations.
Responsibility: Schema management only (SRP).
"""

import logging
from services.database_service import DatabaseService


class SchemaService:
    """Manages database schema initialization and versioning."""

    def __init__(self, database_service: DatabaseService, embedding_dimensions: int = 768):
        """
        Initialize schema service.

        Args:
            database_service: DatabaseService instance for connection management
            embedding_dimensions: Dimension of embedding vectors (default 768)
        """
        self.db_service = database_service
        self.embedding_dimensions = embedding_dimensions

    def database_exists(self, db_name: str) -> bool:
        """
        Check if a database exists.

        Args:
            db_name: Name of database to check

        Returns:
            True if database exists, False otherwise
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                exists = cursor.fetchone() is not None
                cursor.close()
                return exists
        except Exception as e:
            logging.error(f"Failed to check database existence: {e}")
            return False

    def create_database(self, db_name: str):
        """
        Create a new database.

        Args:
            db_name: Name of database to create

        Raises:
            Exception if creation fails
        """
        conn = None
        try:
            conn = self.db_service.get_connection()
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {db_name}")
            cursor.close()
            logging.info(f"Database '{db_name}' created successfully")
        except Exception as e:
            logging.error(f"Failed to create database: {e}")
            raise
        finally:
            if conn:
                self.db_service.release_connection(conn)

    def initialize_schema(self):
        """
        Initialize episodic memory schema (idempotent).

        Creates tables, indexes, and extensions if they don't exist.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Enable vector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create episodes table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS episodes (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                        -- Core episode fields
                        intent TEXT NOT NULL,
                        context JSONB NOT NULL,
                        action TEXT NOT NULL,
                        emotion JSONB NOT NULL,
                        outcome TEXT NOT NULL,
                        gist TEXT NOT NULL,

                        -- Scoring fields (1-10 scale)
                        salience INTEGER NOT NULL CHECK (salience BETWEEN 1 AND 10),
                        freshness INTEGER NOT NULL CHECK (freshness BETWEEN 1 AND 10),

                        -- Vector embedding (dimension from config)
                        embedding vector({self.embedding_dimensions}),

                        -- Metadata
                        topic TEXT NOT NULL,
                        exchange_id TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        last_accessed_at TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        deleted_at TIMESTAMP,

                        -- Computed activation score
                        activation_score FLOAT DEFAULT 1.0
                    )
                """)

                # Create HNSW index for vector similarity
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                        WHERE deleted_at IS NULL
                """)

                # Create indexes for structured filtering
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_topic
                        ON episodes(topic)
                        WHERE deleted_at IS NULL
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_activation
                        ON episodes(activation_score DESC)
                        WHERE deleted_at IS NULL
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_composite
                        ON episodes(topic, activation_score DESC, created_at DESC)
                        WHERE deleted_at IS NULL
                """)

                # Create GIN indexes for full-text search
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_intent_fts
                        ON episodes
                        USING GIN(to_tsvector('english', intent))
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_gist_fts
                        ON episodes
                        USING GIN(to_tsvector('english', gist))
                """)

                # Create cortex_iterations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cortex_iterations (
                        -- Primary key
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                        -- Foreign keys & context
                        topic TEXT NOT NULL,
                        exchange_id TEXT,
                        session_id TEXT,

                        -- Loop metadata
                        loop_id UUID NOT NULL,
                        iteration_number INTEGER NOT NULL,

                        -- Timing
                        started_at TIMESTAMP DEFAULT NOW(),
                        completed_at TIMESTAMP,
                        execution_time_ms FLOAT,

                        -- Confidence & paths
                        chosen_mode TEXT,
                        chosen_confidence FLOAT,
                        alternative_paths JSONB,

                        -- Cost breakdown
                        iteration_cost FLOAT,
                        diminishing_cost FLOAT,
                        uncertainty_cost FLOAT,
                        action_base_cost FLOAT,
                        total_cost FLOAT,
                        cumulative_cost FLOAT,

                        -- Efficiency
                        efficiency_score FLOAT,
                        expected_confidence_gain FLOAT,

                        -- Net value components
                        task_value FLOAT,
                        future_leverage FLOAT,
                        effort_estimate TEXT,
                        effort_multiplier FLOAT,
                        iteration_penalty FLOAT,
                        exploration_bonus FLOAT,
                        net_value FLOAT,

                        -- Decision data
                        decision_override BOOLEAN,
                        overridden_mode TEXT,
                        termination_reason TEXT,

                        -- Actions executed
                        actions_executed JSONB,
                        action_count INTEGER,
                        action_success_count INTEGER,

                        -- Full response
                        frontal_cortex_response JSONB,

                        -- Metadata
                        config_snapshot JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Create indexes for cortex_iterations
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cortex_iterations_loop
                    ON cortex_iterations(loop_id, iteration_number)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cortex_iterations_topic
                    ON cortex_iterations(topic, created_at DESC)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cortex_iterations_exchange
                    ON cortex_iterations(exchange_id)
                """)

                # Create schema version table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Set initial version
                cursor.execute("""
                    INSERT INTO schema_version (version)
                    VALUES (1)
                    ON CONFLICT (version) DO NOTHING
                """)

                cursor.close()
                logging.info("Episodic memory schema initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize schema: {e}")
            raise

    def schema_version(self) -> int:
        """
        Get current schema version.

        Returns:
            Current schema version (0 if not initialized)
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Check if schema_version table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'schema_version'
                    )
                """)
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    cursor.close()
                    return 0

                # Get latest version
                cursor.execute("SELECT MAX(version) FROM schema_version")
                version = cursor.fetchone()[0]
                cursor.close()

                return version if version is not None else 0

        except Exception as e:
            logging.error(f"Failed to get schema version: {e}")
            return 0

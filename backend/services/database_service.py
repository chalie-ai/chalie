"""
Database Service - PostgreSQL connection pooling and connection management.
Responsibility: Connection management only (SRP).
"""

import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker


def get_merged_db_config():
    """
    Merge database config from connections.json and episodic-memory.json.

    Connection details (host, port, database, username, password) come from connections.json.
    Pool settings (pool_size, max_overflow, pool_timeout) come from episodic-memory.json.
    episodic-memory.json can override the database name if needed.

    Returns:
        dict: Merged database configuration
    """
    from services.config_service import ConfigService

    episodic_config = ConfigService.get_agent_config("episodic-memory")
    connections_config = ConfigService.connections()

    # Get pool settings from episodic-memory.json
    episodic_db_config = episodic_config.get('database', {})

    # Get connection details from connections.json
    postgresql_config = connections_config.get('postgresql', {})

    if not postgresql_config:
        raise ValueError("PostgreSQL connection details not found in connections.json")

    # Build merged config
    db_config = {
        'host': postgresql_config['host'],
        'port': postgresql_config.get('port', 5432),
        'database': episodic_db_config.get('database') or postgresql_config['database'],
        'username': postgresql_config.get('username'),
        'password': postgresql_config.get('password'),
        'pool_size': episodic_db_config.get('pool_size', 10),
        'max_overflow': episodic_db_config.get('max_overflow', 20),
        'pool_timeout': episodic_db_config.get('pool_timeout', 30)
    }

    return db_config


# ── Shared singleton pool (per-worker) ──────────────────────────
_shared_db_service = None


def get_shared_db_service() -> 'DatabaseService':
    """
    Get or create a shared DatabaseService singleton.

    Reuses a single connection pool across all callers within a worker process,
    eliminating the 8-12 pool open/close cycles per request.
    """
    global _shared_db_service
    if _shared_db_service is None or _shared_db_service.engine is None:
        db_config = get_merged_db_config()
        _shared_db_service = DatabaseService(db_config)
        logging.info("[DB] Created shared DatabaseService singleton")
    return _shared_db_service


class DatabaseService:
    """Manages PostgreSQL connection pooling via SQLAlchemy engine."""

    def __init__(self, config: dict):
        """
        Initialize connection pool.

        Args:
            config: Database configuration dict with keys:
                    host, port, database, user, password,
                    pool_size, max_overflow, pool_timeout
        """
        self.config = config
        self.engine = None
        self.session_factory = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Create SQLAlchemy engine with connection pool."""
        try:
            username = self.config.get('username') or self.config.get('user')
            if not username:
                raise ValueError("Database username not provided (use 'username' or 'user' field)")

            url = URL.create(
                drivername="postgresql+psycopg2",
                username=username,
                password=self.config.get('password'),
                host=self.config['host'],
                port=self.config.get('port', 5432),
                database=self.config['database']
            )

            pool_size = self.config.get('pool_size', 10)
            max_overflow = self.config.get('max_overflow', 20)
            pool_timeout = self.config.get('pool_timeout', 30)

            self.engine = create_engine(
                url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            self.session_factory = sessionmaker(bind=self.engine)

            logging.info(
                f"Database pool initialized: pool_size={pool_size}, "
                f"max_overflow={max_overflow} to {self.config['host']}/{self.config['database']}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize database pool: {e}")
            raise

    def get_connection(self):
        """
        Get a connection from the pool.

        Returns:
            psycopg2 connection object (pool-managed)

        Raises:
            Exception if pool is exhausted or connection fails
        """
        if not self.engine:
            raise RuntimeError("Connection pool not initialized")

        try:
            conn = self.engine.raw_connection()
            conn.autocommit = False
            return conn
        except Exception as e:
            logging.error(f"Failed to get connection from pool: {e}")
            raise

    def release_connection(self, conn):
        """
        Return a connection to the pool.

        Args:
            conn: psycopg2 connection object
        """
        if conn:
            try:
                conn.close()
            except Exception as e:
                logging.error(f"Failed to release connection: {e}")

    def execute(self, sql, params=None):
        """Execute a write statement (INSERT/UPDATE/DELETE) with auto-commit."""
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
            finally:
                cursor.close()

    def fetch_all(self, sql, params=None):
        """Execute a SELECT and return all rows as a list of dicts."""
        import psycopg2.extras
        with self.connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            try:
                cursor.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
            finally:
                cursor.close()

    @contextmanager
    def connection(self):
        """
        Context manager for database connections.

        Auto-commits on success, auto-rolls-back on exception,
        auto-releases in finally.

        Usage:
            with db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SQL", (params,))
                cursor.close()
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    @contextmanager
    def get_session(self):
        """
        Context manager for SQLAlchemy sessions.

        Auto-commits on success, auto-rolls-back on exception,
        auto-closes in finally.

        Usage:
            with db_service.get_session() as session:
                result = session.execute(text("SELECT ..."))
        """
        if not self.session_factory:
            raise RuntimeError("Session factory not initialized")

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close_pool(self):
        """Close all connections in the pool."""
        if self.engine:
            try:
                self.engine.dispose()
                logging.info("Database pool closed")
            except Exception as e:
                logging.error(f"Failed to close pool: {e}")

    def run_pending_migrations(self):
        """
        Run any pending database migrations from migrations/ directory.
        Creates migrations tracking table if needed.
        """
        import os
        from pathlib import Path

        migrations_dir = Path(__file__).resolve().parent.parent / "migrations"

        if not migrations_dir.exists():
            logging.info("No migrations directory found, skipping migrations")
            return

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Create migrations tracking table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()

            # Get list of applied migrations
            cursor.execute("SELECT filename FROM schema_migrations")
            applied = {row[0] for row in cursor.fetchall()}

            # Find all .sql migration files
            migration_files = sorted(migrations_dir.glob("*.sql"))

            pending_count = 0
            for migration_file in migration_files:
                filename = migration_file.name

                if filename in applied:
                    continue

                # Apply migration
                logging.info(f"Applying migration: {filename}")
                with open(migration_file, 'r') as f:
                    sql = f.read()

                cursor.execute(sql)

                # Record migration as applied
                cursor.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (filename,)
                )
                conn.commit()
                pending_count += 1
                logging.info(f"Migration applied: {filename}")

            if pending_count == 0:
                logging.info("No pending migrations")
            else:
                logging.info(f"Applied {pending_count} migrations")

            cursor.close()

        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Migration failed: {e}")
            raise
        finally:
            if conn:
                self.release_connection(conn)

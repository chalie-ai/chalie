"""Encryption key service — manages DB_ENCRYPTION_KEY generation and persistence.

The encryption key is stored IN the database itself (encryption_keys table).
This guarantees key and encrypted data can never be separated — wherever the
DB goes, the key goes. On upgrade from older versions, the key is migrated
from the legacy .key file into the database automatically.
"""

import logging
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

# Singleton cache to avoid repeated DB reads
_key_cache = None


def _get_legacy_key_file_path() -> Path:
    """Return path to the legacy ``.key`` file (pre-v0.3.0 installs)."""
    import os
    db_path = os.environ.get("CHALIE_DB_PATH")
    if db_path:
        return Path(db_path).resolve().parent / ".key"
    return Path(__file__).resolve().parent.parent / "data" / ".key"


def _ensure_table(conn):
    """Create the encryption_keys table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encryption_keys (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            key_value TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def get_encryption_key(database_service=None) -> str:
    """Get the encryption key, loading from DB or migrating from .key file.

    Priority:
    1. Return cached key if available
    2. Load from database (encryption_keys table)
    3. Migrate from legacy .key file into database
    4. Generate new key and store in database

    Args:
        database_service: Optional DatabaseService instance. If not provided,
            uses the shared singleton. Pass explicitly during early boot
            before the singleton is fully initialized.

    Returns:
        str: The encryption key
    """
    global _key_cache
    if _key_cache:
        return _key_cache

    # Get DB service
    if database_service is None:
        from services.database_service import get_shared_db_service
        database_service = get_shared_db_service()

    with database_service.connection() as conn:
        _ensure_table(conn)

        # Try to load from database
        cursor = conn.cursor()
        cursor.execute("SELECT key_value FROM encryption_keys WHERE id = 1")
        row = cursor.fetchone()
        cursor.close()

        if row:
            _key_cache = row[0]
            logger.info("[Encryption] Loaded encryption key from database")
            return _key_cache

        # Try to migrate from legacy .key file
        legacy_path = _get_legacy_key_file_path()
        key = None

        if legacy_path.exists():
            try:
                with open(legacy_path, 'r') as f:
                    key = f.read().strip()
                if key:
                    logger.info(f"[Encryption] Migrating key from {legacy_path} into database")
            except Exception as e:
                logger.error(f"[Encryption] Failed to read legacy {legacy_path}: {e}")
                key = None

        # Generate new key if no legacy file
        if not key:
            key = secrets.token_urlsafe(32)
            logger.info("[Encryption] Generated new encryption key")

        # Store in database
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO encryption_keys (id, key_value) VALUES (1, ?)",
            (key,)
        )
        cursor.close()
        logger.info("[Encryption] Encryption key stored in database (co-located with encrypted data)")

    _key_cache = key
    return key

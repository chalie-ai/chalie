"""Encryption key service — manages DB_ENCRYPTION_KEY generation and persistence."""

import logging
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

# Singleton cache to avoid repeated file reads
_key_cache = None


def get_key_file_path() -> Path:
    """Return the path to the persistent ``.key`` file inside the data directory.

    Uses the same data directory as the database: ``CHALIE_DB_PATH`` env var
    determines the directory when set, otherwise falls back to
    ``<backend_root>/data/.key``.

    Returns:
        :class:`~pathlib.Path` pointing to the ``.key`` file.
    """
    import os
    db_path = os.environ.get("CHALIE_DB_PATH")
    if db_path:
        return Path(db_path).resolve().parent / ".key"
    return Path(__file__).resolve().parent.parent / "data" / ".key"


def get_encryption_key() -> str:
    """
    Get the encryption key from .key file.

    Priority:
    1. Return cached key if available
    2. Load from .key file if it exists
    3. Generate new key, save to .key with restrictive permissions (0600)

    Returns:
        str: The encryption key

    Raises:
        SystemExit: If key cannot be set up
    """
    global _key_cache
    if _key_cache:
        return _key_cache

    key_file = get_key_file_path()

    # Try to load from .key file
    if key_file.exists():
        try:
            with open(key_file, 'r') as f:
                key = f.read().strip()
            if key:
                _key_cache = key
                logger.info(f"[Encryption] Loaded encryption key from {key_file}")
                return key
        except Exception as e:
            logger.error(f"[Encryption] Failed to read {key_file}: {e}")
            raise SystemExit(1)

    # Generate new key
    try:
        key = secrets.token_urlsafe(32)

        # Ensure data directory exists
        key_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to .key file with restrictive permissions
        key_file.write_text(key)
        key_file.chmod(0o600)

        _key_cache = key
        logger.info(f"[Encryption] Generated new encryption key and saved to {key_file}")
        logger.info(f"[Encryption] File permissions: 0600 (read/write owner only)")
        logger.warning(f"[Encryption] IMPORTANT: Back up {key_file} to a secure location")
        logger.warning(f"[Encryption] Loss of this key means encrypted data cannot be recovered")

        return key
    except Exception as e:
        logger.error(f"[Encryption] Failed to generate or save encryption key: {e}")
        raise SystemExit(1)

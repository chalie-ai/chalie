"""
Tool Config Service — PostgreSQL-backed per-tool configuration storage.

Provides get/set/delete for tool config keys (credentials, endpoints, etc.).
Config values are injected into tool containers at invocation time.
"""

import hmac
import logging
import secrets
import time

logger = logging.getLogger(__name__)


class ToolConfigService:
    RESERVED_KEYS = {
        "_enabled", "_webhook_key",
        "_oauth_access_token", "_oauth_refresh_token",
        "_oauth_token_expires_at", "_oauth_connected_at", "_oauth_scopes",
    }

    def __init__(self, database_service):
        self.db = database_service

    def get_tool_config(self, tool_name: str) -> dict:
        """
        Fetch all config key-value pairs for a tool.

        Returns:
            dict of {key: value}, empty dict on error or no config.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT config_key, config_value FROM tool_configs WHERE tool_name = %s",
                    (tool_name,)
                )
                rows = cursor.fetchall()
                cursor.close()
                return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.debug(f"[TOOL CONFIG] get_tool_config('{tool_name}'): {e}")
            return {}

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Return True if the tool is enabled (default), False if _enabled=false in DB."""
        cfg = self.get_tool_config(tool_name)
        return cfg.get("_enabled", "true").lower() != "false"

    def _set_enabled_flag(self, tool_name: str, enabled: bool) -> bool:
        """Write _enabled flag directly, bypassing the reserved-key guard."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                value = "true" if enabled else "false"
                cursor.execute(
                    """
                    INSERT INTO tool_configs (tool_name, config_key, config_value)
                    VALUES (%s, '_enabled', %s)
                    ON CONFLICT (tool_name, config_key)
                    DO UPDATE SET config_value = EXCLUDED.config_value,
                                  updated_at = NOW()
                    """,
                    (tool_name, value)
                )
                cursor.close()
            return True
        except Exception as e:
            logger.error(f"[TOOL CONFIG] _set_enabled_flag('{tool_name}', {enabled}): {e}", exc_info=True)
            return False

    def set_tool_config(self, tool_name: str, config: dict) -> bool:
        """
        Upsert config key-value pairs for a tool.

        Args:
            tool_name: Tool identifier
            config: Dict of {key: value} to store

        Returns:
            True on success, False on error.

        Raises:
            ValueError: If any key in config is a reserved internal key.
        """
        reserved = set(config.keys()) & self.RESERVED_KEYS
        if reserved:
            raise ValueError(f"Reserved config keys cannot be set directly: {sorted(reserved)}")
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                for key, value in config.items():
                    cursor.execute(
                        """
                        INSERT INTO tool_configs (tool_name, config_key, config_value)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (tool_name, config_key)
                        DO UPDATE SET config_value = EXCLUDED.config_value,
                                      updated_at = NOW()
                        """,
                        (tool_name, key, str(value))
                    )
                cursor.close()
            return True
        except Exception as e:
            logger.error(f"[TOOL CONFIG] set_tool_config('{tool_name}'): {e}", exc_info=True)
            return False

    def generate_webhook_key(self, tool_name: str) -> str:
        """
        Generate and store a webhook API key for a tool.

        Returns the generated key (shown once — caller must present it to the user).
        Subsequent calls regenerate the key, invalidating the old one.
        """
        key = secrets.token_urlsafe(32)
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO tool_configs (tool_name, config_key, config_value)
                    VALUES (%s, '_webhook_key', %s)
                    ON CONFLICT (tool_name, config_key)
                    DO UPDATE SET config_value = EXCLUDED.config_value,
                                  updated_at = NOW()
                    """,
                    (tool_name, key)
                )
                cursor.close()
            logger.info(f"[TOOL CONFIG] Generated webhook key for '{tool_name}'")
            return key
        except Exception as e:
            logger.error(f"[TOOL CONFIG] generate_webhook_key('{tool_name}'): {e}", exc_info=True)
            raise

    def get_webhook_key(self, tool_name: str) -> str | None:
        """Return the stored webhook key for a tool, or None if not set."""
        cfg = self.get_tool_config(tool_name)
        return cfg.get("_webhook_key")

    def validate_webhook_key(self, tool_name: str, provided_key: str) -> bool:
        """
        Validate a provided webhook API key using constant-time comparison.

        Returns True if valid, False otherwise.
        """
        stored = self.get_webhook_key(tool_name)
        if not stored or not provided_key:
            return False
        return hmac.compare_digest(stored.encode(), provided_key.encode())

    def validate_webhook_hmac(self, tool_name: str, timestamp: str, raw_body: bytes, signature: str) -> bool:
        """
        Validate an HMAC-SHA256 webhook signature with replay protection.

        Expected signature: HMAC-SHA256(webhook_key, "{timestamp}.{body_hex}")
        Timestamp must be within 300 seconds of now.

        Returns True if valid, False otherwise.
        """
        stored_key = self.get_webhook_key(tool_name)
        if not stored_key or not signature or not timestamp:
            return False

        # Replay protection: reject requests older than 5 minutes
        try:
            ts = int(timestamp)
            if abs(time.time() - ts) > 300:
                logger.warning(f"[TOOL CONFIG] HMAC timestamp too old for '{tool_name}': {ts}")
                return False
        except (ValueError, TypeError):
            return False

        # Compute expected signature
        msg = f"{timestamp}.{raw_body.hex()}".encode()
        expected = hmac.new(stored_key.encode(), msg, "sha256").hexdigest()
        return hmac.compare_digest(expected.encode(), signature.encode())

    def delete_tool_config_key(self, tool_name: str, key: str) -> bool:
        """
        Delete a single config key for a tool.

        Returns:
            True if a row was deleted, False otherwise.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM tool_configs WHERE tool_name = %s AND config_key = %s",
                    (tool_name, key)
                )
                rowcount = cursor.rowcount
                cursor.close()
                return rowcount > 0
        except Exception as e:
            logger.warning(f"[TOOL CONFIG] delete_tool_config_key('{tool_name}', '{key}'): {e}")
            return False

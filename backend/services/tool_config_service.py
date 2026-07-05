"""
Tool Config Service — SQLite-backed per-tool configuration storage.

Config values are injected into tool containers at invocation time.
"""

from __future__ import annotations

import logging

from services.database import Database

logger = logging.getLogger(__name__)


class ToolConfigService:
    """SQLite-backed per-tool configuration store.

    Reserved keys are write-protected through the public ``set_tool_config``
    interface; dedicated helper methods (``_set_enabled_flag``,
    ``_set_source_metadata``, etc.) must be used to update them.
    """

    RESERVED_KEYS: set[str] = {
        "_enabled",
        "_oauth_access_token", "_oauth_refresh_token",
        "_oauth_token_expires_at", "_oauth_connected_at", "_oauth_scopes",
        "_source_type", "_source_url", "_installed_tag",
    }

    def get_tool_config(self, tool_name: str) -> dict[str, str]:
        try:
            conn = Database.conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT config_key, config_value FROM tool_configs WHERE tool_name = ?",
                (tool_name,)
            )
            rows = cursor.fetchall()
            cursor.close()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.debug(f"[TOOL CONFIG] get_tool_config('{tool_name}'): {e}")
            return {}

    def _set_enabled_flag(self, tool_name: str, enabled: bool) -> bool:
        """Write _enabled flag directly, bypassing the reserved-key guard."""
        try:
            with Database.transaction() as conn:
                cursor = conn.cursor()
                value = "true" if enabled else "false"
                cursor.execute(
                    """
                    INSERT INTO tool_configs (tool_name, config_key, config_value)
                    VALUES (?, '_enabled', ?)
                    ON CONFLICT (tool_name, config_key)
                    DO UPDATE SET config_value = EXCLUDED.config_value,
                                  updated_at = datetime('now')
                    """,
                    (tool_name, value)
                )
                cursor.close()
            return True
        except Exception as e:
            logger.exception(f"[TOOL CONFIG] _set_enabled_flag('{tool_name}', {enabled}): {e}")
            return False

    def _set_source_metadata(self, tool_name: str, source_type: str, source_url: str, installed_tag: str) -> bool:
        """Write source tracking keys (_source_type, _source_url, _installed_tag), bypassing the reserved-key guard."""
        try:
            with Database.transaction() as conn:
                cursor = conn.cursor()
                for key, value in [
                    ("_source_type", source_type),
                    ("_source_url", source_url),
                    ("_installed_tag", installed_tag),
                ]:
                    cursor.execute(
                        """
                        INSERT INTO tool_configs (tool_name, config_key, config_value)
                        VALUES (?, ?, ?)
                        ON CONFLICT (tool_name, config_key)
                        DO UPDATE SET config_value = EXCLUDED.config_value,
                                      updated_at = datetime('now')
                        """,
                        (tool_name, key, value)
                    )
                cursor.close()
            return True
        except Exception as e:
            logger.exception(f"[TOOL CONFIG] _set_source_metadata('{tool_name}'): {e}")
            return False

    def set_tool_config(self, tool_name: str, config: dict[str, object]) -> bool:
        """
        Raises:
            ValueError: If any key in config is a reserved internal key.
        """
        reserved = set(config.keys()) & self.RESERVED_KEYS
        if reserved:
            raise ValueError(f"Reserved config keys cannot be set directly: {sorted(reserved)}")
        try:
            with Database.transaction() as conn:
                cursor = conn.cursor()
                for key, value in config.items():
                    cursor.execute(
                        """
                        INSERT INTO tool_configs (tool_name, config_key, config_value)
                        VALUES (?, ?, ?)
                        ON CONFLICT (tool_name, config_key)
                        DO UPDATE SET config_value = EXCLUDED.config_value,
                                      updated_at = datetime('now')
                        """,
                        (tool_name, key, str(value))
                    )
                cursor.close()
            return True
        except Exception as e:
            logger.exception(f"[TOOL CONFIG] set_tool_config('{tool_name}'): {e}")
            return False

    def delete_tool_config_key(self, tool_name: str, key: str) -> bool:
        try:
            with Database.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM tool_configs WHERE tool_name = ? AND config_key = ?",
                    (tool_name, key)
                )
                rowcount = cursor.rowcount
                cursor.close()
                return rowcount > 0
        except Exception as e:
            logger.warning(f"[TOOL CONFIG] delete_tool_config_key('{tool_name}', '{key}'): {e}")
            return False

    def delete_tool_config(self, tool_name: str) -> bool:
        """Delete ALL config rows for a tool, including reserved system keys (used during uninstall)."""
        try:
            with Database.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM tool_configs WHERE tool_name = ?",
                    (tool_name,)
                )
                rowcount = cursor.rowcount
                cursor.close()
                return rowcount > 0
        except Exception as e:
            logger.warning(f"[TOOL CONFIG] delete_tool_config('{tool_name}'): {e}")
            return False

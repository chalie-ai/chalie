"""
Tool Config Service — PostgreSQL-backed per-tool configuration storage.

Provides get/set/delete for tool config keys (credentials, endpoints, etc.).
Config values are injected into tool containers at invocation time.
"""

import logging

logger = logging.getLogger(__name__)


class ToolConfigService:
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

    def set_tool_config(self, tool_name: str, config: dict) -> bool:
        """
        Upsert config key-value pairs for a tool.

        Args:
            tool_name: Tool identifier
            config: Dict of {key: value} to store

        Returns:
            True on success, False on error.
        """
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

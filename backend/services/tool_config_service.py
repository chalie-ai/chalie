"""
Tool Config Service — SQLite-backed per-tool configuration storage.

Provides get/set/delete for tool config keys (credentials, endpoints, etc.).
Config values are injected into tool containers at invocation time.
"""

import logging

logger = logging.getLogger(__name__)


class ToolConfigService:
    """SQLite-backed per-tool configuration store.

    Provides get/set/delete operations for arbitrary tool config keys
    (API credentials, endpoint URLs, feature flags, etc.) as well as
    reserved system keys such as ``_enabled`` and OAuth token fields.

    Reserved keys are write-protected through the public ``set_tool_config``
    interface; dedicated helper methods (``_set_enabled_flag``,
    ``_set_source_metadata``, etc.) must be used to update them.
    """

    RESERVED_KEYS = {
        "_enabled",
        "_oauth_access_token", "_oauth_refresh_token",
        "_oauth_token_expires_at", "_oauth_connected_at", "_oauth_scopes",
        "_source_type", "_source_url", "_installed_tag",
    }

    def __init__(self, database_service):
        """Initialise the service with a shared database connection.

        Args:
            database_service: A ``DatabaseService`` instance whose
                ``connection()`` context manager provides a SQLite
                connection to the ``tool_configs`` table.
        """
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
                    "SELECT config_key, config_value FROM tool_configs WHERE tool_name = ?",
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
            logger.error(f"[TOOL CONFIG] _set_enabled_flag('{tool_name}', {enabled}): {e}", exc_info=True)
            return False

    def _set_source_metadata(self, tool_name: str, source_type: str, source_url: str, installed_tag: str) -> bool:
        """Write source tracking keys (_source_type, _source_url, _installed_tag), bypassing the reserved-key guard."""
        try:
            with self.db.connection() as conn:
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
            logger.error(f"[TOOL CONFIG] _set_source_metadata('{tool_name}'): {e}", exc_info=True)
            return False

    def get_source_metadata(self, tool_name: str) -> dict:
        """Return source tracking fields for a tool: _source_type, _source_url, _installed_tag."""
        cfg = self.get_tool_config(tool_name)
        return {k: cfg[k] for k in ("_source_type", "_source_url", "_installed_tag") if k in cfg}

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
        """Delete ALL config rows for a tool (used during uninstall).

        Removes every row in ``tool_configs`` whose ``tool_name`` matches the
        provided value, including reserved system keys such as ``_enabled``
        and OAuth fields.

        Args:
            tool_name: The tool identifier whose config rows should be purged.

        Returns:
            True if at least one row was deleted, False if no rows existed or
            an exception occurred.
        """
        try:
            with self.db.connection() as conn:
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

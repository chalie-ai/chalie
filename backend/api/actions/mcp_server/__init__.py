"""MCP server action package — verb operations for the inbound MCP server.

Directory name is underscored (Python packages can't have dashes); the URL
segment stays ``mcp-server`` because ``api/routes.py`` mounts these under that
slug, matching the singleton settings endpoint in ``api/endpoints/mcp_settings.py``.
"""

# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MCP server — exposes Chalie as an MCP endpoint for external agents.

Conversations are stored in per-agent channels and optionally disclosed to
the user.
"""

from mcp_server.server import create_mcp_server, run_mcp_server

__all__ = ["create_mcp_server", "run_mcp_server"]

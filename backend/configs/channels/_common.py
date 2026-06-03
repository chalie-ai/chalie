from __future__ import annotations

# ── Default tool visibility (mirrors MessageProcessor class defaults) ──────────

DEFAULT_ALWAYS_AVAILABLE: list[str] = [
    "find_skills",
    "find_tools",
    "memory",
]

# Pattern/graph-writing tools — only PatternConfig and GeoConfig may call them.
# Every other discovery-capable loop (anything carrying find_tools) blocks them
# so find_tools cannot surface them outside the pattern channels.
PATTERN_WRITE_TOOLS: frozenset[str] = frozenset({"save_pattern", "save_graph"})

# Delegate (subagent-as-tool) names — blocked on every discovery-capable loop
# except the user-facing ones (UserConfig, EAMPConfig) so background loops can
# never spawn delegate work.
DELEGATE_TOOLS: frozenset[str] = frozenset({"web_search", "web_browse"})

# Raw web tools exclusive to the delegate agents (WebSearchConfig drives
# ``search``, WebBrowseConfig drives ``browser``).  Every discovery-capable loop
# blocks them so they reach the web only through the delegate tools, never
# directly via find_tools.
DELEGATE_INTERNAL_TOOLS: frozenset[str] = frozenset({"browser", "search"})

DEFAULT_DISCOVERABLE: list[str] = [
    "bash",
    "browser",
    "calendar",
    "chalie_docs",
    "code_eval",
    "contacts",
    "document",
    "email",
    "file_permissions",
    "file_write",
    "home",
    "list",
    "mcp_manager",
    "news",
    "place",
    "programming_docs_search",
    "read",
    "review_tool_calls",
    "review_transcript",
    "schedule",
    "search",
    "search_files",
    "skill_builder",
    "timer",
    "ubiquiti",
    "weather",
    "web_browse",
    "web_download",
    "web_search",
]

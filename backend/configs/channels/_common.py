from __future__ import annotations

# ── Per-provider system-prompt placeholders ───────────────────────────────────

_CONTENT_FIELD_PLACEHOLDER = "{{provider_content_field_name}}"


def substitute_provider_content_field(body: str, job: str = "unified") -> str:
    """Replace ``{{provider_content_field_name}}`` with the active provider's
    ``CONTENT_FIELD_LABEL`` (e.g. ``content[].text`` for Anthropic,
    ``message.content`` for Ollama) so the model is told the exact JSON field
    where its user-visible prose lands.

    ``job`` is a logging label only — the provider is resolved from the active
    selection, not from this key.  Best-effort: if the placeholder is absent or
    the provider lookup fails, the body passes through unchanged.
    """
    if _CONTENT_FIELD_PLACEHOLDER not in body:
        return body
    try:
        from services.providers import Providers  # noqa: PLC0415
        provider = Providers.instance()._resolve(job)
        label = getattr(provider, "CONTENT_FIELD_LABEL", None)
    except Exception:
        label = None
    if not label:
        return body
    return body.replace(_CONTENT_FIELD_PLACEHOLDER, label)


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

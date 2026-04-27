"""
Tool Library Service — Single source of truth for first-party tool metadata.

First-party tool execution is handled exclusively by AbilityRegistry.
This module owns:
  - TOOL_METADATA: prompt descriptions and input_schema for all first-party tools
  - Dynamic registration API (register_tool / unregister_tool) for capability plugins

Dynamic registration
--------------------
Capability plugins (and tests) may register and unregister tools at runtime via
:func:`register_tool` and :func:`unregister_tool`.  All mutations and reads of
:data:`TOOL_HANDLERS` / :data:`TOOL_METADATA` are protected by the
module-level :data:`_registry_lock` so concurrent calls across threads are safe.

Use :func:`get_all_tool_names` rather than the :data:`ALL_TOOL_NAMES` constant
whenever the set of registered tools may have changed since import time.
"""

import threading

# Optional: detect headless browser availability (requires playwright + chromium)
try:
    import playwright  # noqa: F401 — availability check
    _BROWSER_AVAILABLE = True
except ImportError:
    _BROWSER_AVAILABLE = False


# -- Handler registry ----------------------------------------------------------
# First-party tools are dispatched via AbilityRegistry.
# TOOL_HANDLERS holds only capability-plugin handlers registered at runtime.

TOOL_HANDLERS: dict = {}

#: Snapshot of built-in tool names taken at import time.  This constant is
#: preserved for backward compatibility but reflects only the tools registered
#: before any dynamic ``register_tool()`` calls.  Prefer
#: :func:`get_all_tool_names` for code that runs after capability loading.
ALL_TOOL_NAMES: frozenset = frozenset(TOOL_HANDLERS.keys())

#: Module-level lock that serialises all reads and writes to
#: :data:`TOOL_HANDLERS` and :data:`TOOL_METADATA`.
_registry_lock: threading.Lock = threading.Lock()


def get_handler(name: str):
    """Return the execute() callable for a named tool, or ``None``.

    Args:
        name: The registered tool name (e.g. ``"weather"``).

    Returns:
        The ``execute`` callable for the tool, or ``None`` if the tool is not
        registered.
    """
    with _registry_lock:
        return TOOL_HANDLERS.get(name)


# -- Metadata (replaces manifest.json) ----------------------------------------

TOOL_METADATA: dict = {
    "weather": {
        "name": "weather",
        "description": "Get current weather conditions for any location — temperature, wind, rain, humidity, UV.",
        "documentation": (
            "Fetches current weather conditions for a given location or the user's "
            "automatically detected location, returning temperature, feels-like, "
            "humidity, wind speed and direction, UV index, precipitation, visibility, "
            "and boolean flags (is_raining, is_hot, is_cold, is_windy, is_clear). "
            "Use when the user asks about the weather, temperature, or outdoor "
            "conditions, or when a question implies environmental context — for "
            "example, 'I'm going for a run', 'should I bring a jacket?', or "
            "'planning a picnic today'. If no location is specified, telemetry "
            "coordinates are used automatically. CRITICAL: Always set the location "
            "parameter when the user names any place, including short follow-up "
            "phrasing like 'what about London?', 'how about Paris?', or 'and in "
            "Tokyo?' — extract the place name and pass it. Results are cached per "
            "location for 10 minutes."
        ),
        "category": "context",
        "icon": "fa-cloud",
        "trigger": {"type": "on_demand"},
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "City or place name (e.g. 'Malta', 'London'). Omit to "
                        "use the user's current location automatically."
                    ),
                },
            },
            "required": [],
        },
        "returns": {
            "location": {"type": "string"},
            "condition": {"type": "string"},
            "temperature_c": {"type": "number"},
            "temperature_f": {"type": "number"},
            "feels_like_c": {"type": "number"},
            "humidity_pct": {"type": "integer"},
            "wind_kmh": {"type": "number"},
            "wind_direction": {"type": "string"},
            "visibility_km": {"type": "number"},
            "uv_index": {"type": "integer"},
            "precip_mm": {"type": "number"},
            "observation_time": {"type": "string"},
            "is_raining": {"type": "boolean"},
            "is_daylight": {"type": "boolean"},
            "is_hot": {"type": "boolean", "description": "feels_like_c >= 30"},
            "is_cold": {"type": "boolean", "description": "feels_like_c <= 10"},
            "is_windy": {"type": "boolean", "description": "wind_kmh >= 30"},
            "is_clear": {"type": "boolean"},
        },
        "constraints": {"timeout_seconds": 40},
        "output": {
            "synthesize": False,
            "ephemeral": True,
            "card": {"enabled": True, "title": "Weather", "accent_color": "#4a90d4",
                     "background_color": "rgba(74, 144, 212, 0.10)"},
        },
        "ambient": {"enabled": False},
        "tips": [
            "Cached per location for 10 minutes",
            "Location is detected automatically from client telemetry",
            "Always set location when the user names any place",
            "Never pass raw coordinates as the location param",
        ],
    },

    "search": {
        "name": "search",
        "description": (
            "Search Wikipedia, GitHub, Reddit, arXiv, news, books, and more using "
            "plain natural language. Automatically routes to the best source(s)."
        ),
        "documentation": (
            "Multi-provider search with semantic routing across 12 sources. Write "
            "queries as plain natural language — the router selects the best "
            "provider(s) automatically. Pair with read skill to fetch full content "
            "from promising URLs."
        ),
        "category": "research",
        "icon": "fa-magnifying-glass",
        "trigger": {"type": "on_demand"},
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Plain natural language search query.",
                },
                "provider": {
                    "type": "string",
                    "enum": ["wikipedia", "wikidata", "arxiv", "github", "hackernews", "reddit", "google_news", "stackoverflow", "open_library", "musicbrainz", "itunes", "nominatim", "ddg"],
                    "description": "Limit your search to one provider. Only use this if the user asks you to use 1 specific provider.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results per provider (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        "returns": {
            "results": {"type": "array", "description": "List of {title, snippet, url, provider, date}"},
            "count": {"type": "integer"},
            "providers_used": {"type": "array", "description": "List of provider names that returned results"},
            "_meta": {"type": "object", "description": "Routing scores, latency, method"},
        },
        "constraints": {"timeout_seconds": 20},
        "config_schema": {},
        "output": {
            "synthesize": True,
            "card": {"enabled": True, "mode": "immediate", "title": "Search",
                     "accent_color": "#1a8fff", "background_color": "rgba(26,143,255,0.06)"},
        },
        "tips": [
            "Omit provider to let semantic routing pick the best source(s) automatically",
            "Pair with read skill to fetch full content from the most relevant result",
        ],
    },

    "code_eval": {
        "name": "code_eval",
        "description": "Run Python code to compute, verify formulas, or test logic. Use print() for output.",
        "documentation": (
            "Runs Python code in a restricted sandbox. Use this as a scratchpad "
            "when holding a computation in working memory would be unreliable.\n\n"
            "Ideal for: verifying complex formulas, testing algorithm logic, unit "
            "conversions, validating data transformations.\n\n"
            "Available modules: math, statistics, json, decimal, fractions, "
            "itertools, functools, collections.\n\n"
            "Write Python. Use print() to emit results. No file I/O, no subprocess, "
            "no imports — the safe modules above are pre-loaded."
        ),
        "category": "productivity",
        "icon": "fa-solid fa-code",
        "trigger": {"type": "on_demand"},
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to emit results.",
                },
            },
            "required": ["code"],
        },
        "returns": {
            "text": {"type": "string", "description": "Captured print() output"},
            "error": {"type": "string", "description": "Compile or runtime error, if any"},
        },
        "constraints": {"timeout_seconds": 15},
        "output": {"synthesize": False, "card": {"enabled": False}},
        "tips": [
            "Always use print() — bare expressions produce no output",
            "math, statistics, itertools, functools, collections, decimal, fractions are pre-loaded",
        ],
    },

    "programming_docs_search": {
        "name": "programming_docs_search",
        "description": (
            "Search official documentation for PHP, Python, JavaScript, Go, Rust, "
            "Java, Ruby, C#, Dart, C/C++, Bash, SQL and frameworks Laravel, Django, "
            "Flask, NumPy, Pandas, Node.js, React, Vue, Spring, Rails, Flutter."
        ),
        "documentation": (
            "Use this tool when you need to look up a function, class, method, "
            "module, or concept in a programming language or framework's official "
            "documentation. Pass the language/framework name and the query. Returns "
            "extracted documentation text from the official source."
        ),
        "icon": "fa-solid fa-book-open",
        "trigger": {"type": "on_demand"},
        "input_schema": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["php", "python", "javascript", "typescript", "go", "rust", "java", "ruby", "csharp", "dart", "c", "cpp", "bash", "sql", "laravel", "django", "flask", "numpy", "pandas", "node", "react", "vue", "spring", "rails", "flutter"],
                    "description": "Programming language or framework to search documentation for.",
                },
                "query": {
                    "type": "string",
                    "description": "What to look up — function name, class, method, module, or concept in plain language.",
                },
            },
            "required": ["language", "query"],
        },
        "returns": {
            "text": {"type": "string"},
            "url": {"type": "string"},
            "source": {"type": "string"},
        },
        "constraints": {"timeout_seconds": 15},
        "output": {"synthesize": False, "ephemeral": True, "card": {"enabled": False}},
        "tips": [
            "Use when you need exact function signatures or parameter descriptions",
            "Supports frameworks directly: django, flask, numpy, pandas, node, react, vue, laravel, spring, rails, flutter",
            "Prefer this over the search tool when you know the language/framework",
        ],
    },

    "news": {
        "name": "news",
        "description": "Search news articles across global sources.",
        "documentation": (
            "Search news by query. Supply a category only when the query is broad "
            "(e.g. \"What's happening in tech today?\"). "
            "Omit category for specific queries (e.g. \"Sam Altman fired\")."
        ),
        "trigger": {"type": "on_demand"},
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for.",
                },
                "category": {
                    "type": "string",
                    "enum": ["tech", "business", "sports", "science", "entertainment", "us", "uk"],
                    "description": "Narrow to a news category. Use only for broad topic browsing.",
                },
            },
            "required": ["query"],
        },
        "returns": {
            "text": {"type": "string", "description": "Formatted news results"},
            "title": {"type": "string", "description": "Result section title"},
            "error": {"type": "string", "description": "Error message if any"},
        },
        "output": {
            "synthesize": True,
            "card": {"enabled": False},
        },
        "tips": [
            "Use for any news-related query — current events, headlines, what's happening",
            "Add a category for broad topic browsing (e.g. 'tech', 'sports')",
        ],
        "ambient": {"enabled": False},
    },
}


if _BROWSER_AVAILABLE:
    TOOL_METADATA["browser"] = {
        "name": "browser",
        "timeout": 90,
        "description": (
            "Render JavaScript-heavy web pages, take screenshots, fill forms, "
            "and monitor pages for changes. Use when the read skill returns "
            "empty or broken content, when interaction with a web page is needed, "
            "or when you need to visually verify page state."
        ),
        "documentation": (
            "Headless browser for the modern web. Four actions:\n\n"
            "**render** — Load URL with full JavaScript execution, extract clean text. "
            "Use when the read skill fails on JS-heavy pages (SPAs, dynamic content, "
            "Cloudflare-protected sites). Returns extracted text + page links.\n\n"
            "**screenshot** — Capture visual state of a page as PNG. Optionally run OCR "
            "to extract text from the image. Use for dashboards, visual verification, "
            "or when text extraction fails.\n\n"
            "**interact** — Fill forms, click buttons, select dropdowns, navigate "
            "multi-step flows. Provide an ordered list of steps. Use for login flows, "
            "search filtering, form submission, multi-page navigation. Steps execute "
            "sequentially; stops on first failure with partial results.\n\n"
            "**monitor** — Track page changes over time. Renders a page, extracts "
            "content, compares against the previous snapshot. Use in goal pursuits "
            "to watch prices, availability, status changes. Requires a unique "
            "snapshot_key to identify what's being monitored.\n\n"
            "The read skill is faster for static pages — try it first. "
            "Use browser when JavaScript rendering or page interaction is required.\n\n"
            "Supports custom wait strategies: 'networkidle' (default), "
            "'domcontentloaded', 'selector:<css>' (wait for element), "
            "'timeout:<ms>' (fixed wait)."
        ),
        "category": "research",
        "icon": "fa-globe",
        "trigger": {"type": "on_demand"},
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["render", "screenshot", "interact", "monitor"],
                    "description": "render: load page with JS and extract text. screenshot: capture page as PNG. interact: fill forms, click buttons, navigate flows. monitor: track page changes over time.",
                },
                "url": {
                    "type": "string",
                    "description": "URL to load.",
                },
                "wait_for": {
                    "type": "string",
                    "description": "Wait strategy: 'networkidle' (default), 'domcontentloaded', 'selector:<css>' (wait for element), 'timeout:<ms>' (fixed wait).",
                    "default": "networkidle",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to scope extraction or screenshot to a specific element.",
                },
                "extract": {
                    "type": "string",
                    "enum": ["text", "html"],
                    "description": "Extraction format for render action.",
                    "default": "text",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to extract.",
                    "default": 8000,
                },
                "steps": {
                    "type": "array",
                    "description": "Interaction steps for interact action. Each step: {action: click|fill|select|check|wait|scroll|press|hover|type, selector: 'css', value: 'text', timeout: ms}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "selector": {"type": "string"},
                            "value": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["action"],
                    },
                },
                "snapshot_key": {
                    "type": "string",
                    "description": "Unique key for monitor action snapshots (e.g. 'price-watch-amazon-xyz').",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture full scrollable page in screenshot.",
                    "default": False,
                },
                "screenshot_after": {
                    "type": "boolean",
                    "description": "Take screenshot after interaction steps complete.",
                    "default": False,
                },
                "ocr": {
                    "type": "boolean",
                    "description": "Run OCR on screenshot to extract text from image.",
                    "default": False,
                },
                "save_session": {
                    "type": "boolean",
                    "description": "Save session cookies after interaction for future visits.",
                    "default": False,
                },
                "credential_label": {
                    "type": "string",
                    "description": "Load stored credentials for this label.",
                },
            },
            "required": ["action", "url"],
        },
        "returns": {
            "text": {"type": "string", "description": "Extracted page text"},
            "screenshot_b64": {"type": "string", "description": "Base64 PNG screenshot"},
            "ocr_text": {"type": "string", "description": "OCR-extracted text from screenshot"},
            "title": {"type": "string", "description": "Page title"},
            "url": {"type": "string", "description": "Final URL after redirects"},
            "links": {"type": "array", "description": "Navigable page links [{text, url}]"},
            "changed": {"type": "boolean", "description": "Whether monitored page changed (monitor)"},
            "diff": {"type": "string", "description": "Unified diff of changes (monitor)"},
            "change_ratio": {"type": "number", "description": "Fraction of content changed (monitor)"},
            "first_check": {"type": "boolean", "description": "True if no previous snapshot (monitor)"},
            "steps_completed": {"type": "integer", "description": "Steps completed (interact)"},
            "steps_total": {"type": "integer", "description": "Total steps (interact)"},
            "step_error": {"type": "string", "description": "First failed step error (interact)"},
            "error": {"type": "string", "description": "Error message if any"},
        },
        "constraints": {"timeout_seconds": 90},
        "config_schema": {},
        "output": {
            "synthesize": True,
            "card": {
                "enabled": True,
                "mode": "immediate",
                "title": "Browser",
                "accent_color": "#00C8FF",
                "background_color": "rgba(0, 200, 255, 0.06)",
            },
        },
        "ambient": {"enabled": False},
        "tips": [
            "Use 'render' when the read skill returned empty or garbled content from a JS-heavy site",
            "Use 'screenshot' + ocr=true for visual content that resists text extraction",
            "Use 'interact' for multi-step workflows: login, form fill, apply filters, navigate pages",
            "Use 'monitor' with snapshot_key in goal pursuits to track page changes over time",
            "Always try the read skill first — it is much faster for static pages",
            "Use wait_for='selector:.price' to wait for a specific element before extracting",
            "For shopping: use interact with fill + select steps to apply filters, then extract results",
            "For flight search: interact to set dates/routes, wait for results, extract and compare",
            "Use save_session=true after login to persist cookies for future visits",
        ],
    }


# ---------------------------------------------------------------------------
# Dynamic registration API
# ---------------------------------------------------------------------------

def register_tool(name: str, handler, metadata: dict) -> None:
    """Register a tool handler and its metadata at runtime.

    Both :data:`TOOL_HANDLERS` and :data:`TOOL_METADATA` are updated
    atomically under :data:`_registry_lock`, making this function safe to call
    from any thread (e.g. capability workers, test fixtures).

    If *name* is already registered the existing entries are **replaced**
    without raising an error, so callers can safely re-register after a
    reconnect.

    Args:
        name:     Unique tool identifier (must match ``metadata["name"]``).
        handler:  Callable with signature
                  ``execute(topic, params, config=None, telemetry=None) -> dict``.
        metadata: Tool metadata dict following the schema used in
                  :data:`TOOL_METADATA` (must include at least ``"name"`` and
                  ``"description"``).

    Returns:
        None
    """
    with _registry_lock:
        TOOL_HANDLERS[name] = handler
        TOOL_METADATA[name] = metadata


def unregister_tool(name: str) -> None:
    """Remove a tool handler and its metadata from the registry.

    Both :data:`TOOL_HANDLERS` and :data:`TOOL_METADATA` are updated
    atomically under :data:`_registry_lock`.  If *name* is not currently
    registered the call is a no-op (no exception is raised).

    Args:
        name: The tool name to remove.

    Returns:
        None
    """
    with _registry_lock:
        TOOL_HANDLERS.pop(name, None)
        TOOL_METADATA.pop(name, None)


def get_all_tool_names() -> set:
    """Return the current set of registered tool names.

    Unlike the module-level :data:`ALL_TOOL_NAMES` constant (which is a
    snapshot taken at import time), this function acquires :data:`_registry_lock`
    and computes the set from the live :data:`TOOL_HANDLERS` dict, so it
    reflects any tools added or removed via :func:`register_tool` /
    :func:`unregister_tool` after module load.

    Returns:
        A new ``set`` of registered tool name strings.
    """
    with _registry_lock:
        return set(TOOL_HANDLERS.keys())

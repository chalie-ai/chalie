"""
Tool Library Service — Single source of truth for first-party tool metadata.

Like innate_skills/registry.py for skills, this module declares all first-party
tools in Python code. No manifest.json, no runner.py, no per-tool directories.

Tools are simple callable Python modules in backend/tools/. Each exposes an
execute(topic, params, config, telemetry) -> dict function.
"""

from tools.weather import execute as _weather_execute
from tools.web_search import execute as _web_search_execute
from tools.code_eval import execute as _code_eval_execute
from tools.programming_docs_search import execute as _docs_execute
from tools.search.search import execute as _search_execute
from tools.news.news import execute as _news_execute


# -- Handler registry ----------------------------------------------------------

TOOL_HANDLERS = {
    "weather": _weather_execute,
    "web_search": _web_search_execute,
    "search": _search_execute,
    "code_eval": _code_eval_execute,
    "programming_docs_search": _docs_execute,
    "news": _news_execute,
}

ALL_TOOL_NAMES: frozenset = frozenset(TOOL_HANDLERS.keys())


def get_handler(name: str):
    """Return the execute() callable for a tool, or None."""
    return TOOL_HANDLERS.get(name)


# -- Metadata (replaces manifest.json) ----------------------------------------

TOOL_METADATA: dict = {
    "weather": {
        "name": "weather",
        "description": (
            "Get current weather conditions for a location. Use when the user "
            "asks about weather, temperature, outdoor conditions, or when "
            "environmental context would add warmth."
        ),
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
        "parameters": {
            "location": {
                "type": "string",
                "required": False,
                "description": (
                    "City or place name (e.g. 'Malta', 'London'). Omit entirely to "
                    "use the user's current location automatically. IMPORTANT: If "
                    "the user names any place — even in a short follow-up like "
                    "'what about London?' or 'how about Paris?' — extract that place "
                    "name and set it here."
                ),
            },
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
            "Search across multiple sources including Wikipedia, academic databases, "
            "code repositories, Reddit, news, books, music, and more. Automatically "
            "routes your query to the most relevant source(s). Falls back to general "
            "web search if needed. Use provider parameter to force a specific source."
        ),
        "documentation": (
            "Multi-provider search tool with semantic routing. Automatically selects "
            "the best source(s) for each query from 12 providers: Wikipedia, Wikidata, "
            "ArXiv, GitHub, Hacker News, Reddit, Google News, Stack Overflow, Open "
            "Library, MusicBrainz, iTunes, and Nominatim. Falls back to DuckDuckGo "
            "web search when no provider matches or when all selected providers return "
            "empty. Use the provider parameter to force a specific source (e.g. "
            "'wikipedia', 'reddit', 'arxiv', 'ddg'). Results include the source "
            "provider for each item. Pair with read skill to fetch full content from "
            "promising URLs."
        ),
        "category": "research",
        "icon": "fa-magnifying-glass",
        "trigger": {"type": "on_demand"},
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "What to search for.",
            },
            "provider": {
                "type": "string",
                "required": False,
                "description": (
                    "Force a specific provider. Use 'ddg' for general web search. "
                    "Omit to let the system auto-route to the best source(s)."
                ),
            },
            "limit": {
                "type": "integer",
                "required": False,
                "default": 5,
                "description": "Max results per provider (default 5, max 10).",
            },
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
            "Omit provider to let semantic routing pick the best source(s)",
            "Use provider='ddg' when you specifically need general web search",
            "Use provider='reddit' for opinions, recommendations, community advice",
            "Use provider='arxiv' for academic papers in physics, CS, math",
            "Pair with read skill to fetch full content from the most relevant result",
        ],
    },

    "web_search": {
        "name": "web_search",
        "description": "Deprecated — use 'search' tool instead. DuckDuckGo-only web search.",
        "documentation": (
            "DEPRECATED: Use the 'search' tool instead, which includes DuckDuckGo as "
            "a fallback alongside 12 other providers with semantic routing. This tool "
            "is kept for backward compatibility only. "
            "Searches the web via DuckDuckGo and returns titles, snippets, URLs, "
            "and optional images. Use for general web queries when no domain-specific "
            "tool applies. Supports time_range filtering (day/week/month/year). "
            "Pair with read skill to fetch full page content from promising results. "
            "DuckDuckGo may rate-limit on burst usage — space queries."
        ),
        "category": "research",
        "icon": "fa-magnifying-glass",
        "trigger": {"type": "on_demand"},
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "Search query",
            },
            "limit": {
                "type": "integer",
                "required": False,
                "default": 5,
                "description": "Number of results to return (1-8).",
            },
            "time_range": {
                "type": "string",
                "required": False,
                "description": "Filter results by time: 'day', 'week', 'month', or 'year'.",
            },
        },
        "returns": {
            "results": {"type": "array", "description": "List of {title, snippet, url, domain}"},
            "count": {"type": "integer"},
            "_meta": {"type": "object", "description": "Observability fields"},
        },
        "constraints": {"timeout_seconds": 15},
        "config_schema": {},
        "output": {
            "synthesize": True,
            "card": {"enabled": True, "mode": "immediate", "title": "Web Search",
                     "accent_color": "#1a8fff", "background_color": "rgba(26,143,255,0.06)"},
        },
        "tips": [
            "Set time_range='day' or 'week' for recent results on fast-moving topics",
            "Pair with read skill to fetch full content from the most relevant result",
        ],
    },

    "code_eval": {
        "name": "code_eval",
        "description": (
            "Execute a Python snippet to verify formulas, test algorithms, or "
            "compute results precisely. Use instead of mental arithmetic for any "
            "multi-step or nested calculation."
        ),
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
        "parameters": {
            "code": {
                "type": "string",
                "required": True,
                "description": "Python code to execute. Use print() to emit results.",
            },
            "label": {
                "type": "string",
                "required": False,
                "description": "Short description of what this snippet is computing.",
            },
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
            "Look up official programming language and framework documentation. "
            "Searches the canonical documentation site and returns the relevant "
            "page content. Supports 12 languages (PHP, Python, JavaScript/TypeScript, "
            "Go, Rust, Java, Ruby, C#, Dart, C/C++, Bash, SQL) and 11 major "
            "frameworks (Laravel, Django, Flask, NumPy, Pandas, Node.js, React, "
            "Vue, Spring, Rails, Flutter)."
        ),
        "documentation": (
            "Use this tool when you need to look up a function, class, method, "
            "module, or concept in a programming language or framework's official "
            "documentation. Pass the language/framework name (or alias like 'py', "
            "'js', 'django', 'react', 'rails') and the query. Returns extracted "
            "documentation text from the official source."
        ),
        "icon": "fa-solid fa-book-open",
        "trigger": {"type": "on_demand"},
        "parameters": {
            "language": {
                "type": "string",
                "required": True,
                "description": (
                    "Programming language or framework name (e.g. 'python', 'django', "
                    "'javascript', 'react', 'php', 'go', 'rust', 'java', 'spring', "
                    "'ruby', 'rails', 'csharp', 'dart', 'flutter', 'c', 'cpp', 'bash', 'sql')"
                ),
            },
            "query": {
                "type": "string",
                "required": True,
                "description": (
                    "Function, class, method, module, or concept to look up "
                    "(e.g. 'array_map', 'asyncio.gather', 'Vec::push')"
                ),
            },
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
            "Prefer this over web_search when you know the language/framework",
        ],
    },

    "news": {
        "name": "news",
        "description": (
            "Search and browse news from 56 global RSS sources across 8 categories. "
            "Get real-time headlines, search for specific topics, discover trending "
            "stories, and explore available sources."
        ),
        "documentation": (
            "News search and browsing tool with 4 actions:\n\n"
            "**search** — Find news articles matching a query. Optionally filter by source.\n"
            "  Required: query (string)\n"
            "  Optional: source (string), limit (integer, default 10)\n\n"
            "**digest** — Get a curated news digest with international headlines and local news.\n"
            "  Optional: category (string), source (string)\n\n"
            "**trending** — Discover trending stories clustered by topic coverage across multiple sources.\n"
            "  Optional: category (string, default 'international'), limit (integer, default 5), "
            "min_sources (integer, default 2)\n\n"
            "**sources** — List available news sources. Filter by category or search by name.\n"
            "  Optional: category (string), query (string via 'source' param)"
        ),
        "category": "research",
        "icon": "fa-solid fa-newspaper",
        "trigger": {"type": "on_demand"},
        "parameters": {
            "action": {
                "type": "string",
                "required": True,
                "description": "Action to perform: search, digest, trending, or sources",
            },
            "query": {
                "type": "string",
                "required": False,
                "description": "Search query (required for search action)",
            },
            "category": {
                "type": "string",
                "required": False,
                "description": (
                    "News category: international, us, uk, tech, business, "
                    "science, sports, entertainment"
                ),
            },
            "source": {
                "type": "string",
                "required": False,
                "description": "Source ID or name to filter by",
            },
            "limit": {
                "type": "integer",
                "required": False,
                "description": "Maximum number of results (default 10, max 20)",
                "default": 10,
            },
            "min_sources": {
                "type": "integer",
                "required": False,
                "description": "Minimum sources for a trending cluster (default 2)",
                "default": 2,
            },
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
            "Use 'search' with a query to find specific news topics",
            "Use 'trending' to see what stories are being covered by multiple sources",
            "Use 'digest' for a quick overview of current headlines",
        ],
        "ambient": {"enabled": False},
        "config_schema": {
            "preferred_source": {
                "type": "string",
                "description": "Default news source ID",
                "default": "bbc_world",
            },
            "topics": {
                "type": "string",
                "description": "Comma-separated interest topics for digest",
            },
            "location_override": {
                "type": "string",
                "description": "Override location for local news",
            },
        },
    },
}


def get_metadata(name: str) -> dict | None:
    """Return metadata dict for a tool, or None."""
    return TOOL_METADATA.get(name)

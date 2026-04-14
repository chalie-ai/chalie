"""
Tool Library Service — Single source of truth for first-party tool metadata.

Like innate_skills/registry.py for skills, this module declares all first-party
tools in Python code. No manifest.json, no runner.py, no per-tool directories.

Tools are simple callable Python modules in backend/tools/. Each exposes an
execute(topic, params, config, telemetry) -> dict function.

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

from tools.weather import execute as _weather_execute
from tools.code_eval import execute as _code_eval_execute
from tools.programming_docs_search import execute as _docs_execute
from tools.search.search import execute as _search_execute
from tools.news.news import execute as _news_execute

# Optional: headless browser (requires playwright + chromium)
try:
    from tools.browser.browser import execute as _browser_execute
    _BROWSER_AVAILABLE = True
except ImportError:
    _BROWSER_AVAILABLE = False


# -- Handler registry ----------------------------------------------------------

TOOL_HANDLERS = {
    "weather": _weather_execute,
    "search": _search_execute,
    "code_eval": _code_eval_execute,
    "programming_docs_search": _docs_execute,
    "news": _news_execute,
}

if _BROWSER_AVAILABLE:
    TOOL_HANDLERS["browser"] = _browser_execute

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


# -- Built-in tool profiles ---------------------------------------------------
# Hardcoded capability profiles for first-party tools.
# seed_builtin_profiles() in tool_profile_service.py reads this dict and upserts
# into tool_capability_profiles + vec table at startup, bypassing the LLM profiler.
# The manifest_hash is computed from TOOL_METADATA[name] so the existing staleness
# gate in bootstrap_all() naturally skips these tools after seeding.

BUILTIN_TOOL_PROFILES: dict = {
    "search": {
        "keywords": "search,web,google,lookup,find,internet,online,research",
        "short_summary": "Search the web — plain natural language queries, curated results from multiple online sources.",
        "full_profile": (
            "Search using natural language across multiple providers.\n\n"
            "Wikipedia: history of the roman empire, what is quantum entanglement, biography of Ada Lovelace, how do vaccines work, the french revolution causes and timeline, what is a black hole, photosynthesis process explained, who invented the printing press, overview of the united nations, difference between mitosis and meiosis, how does the internet work, history of the olympic games, what is cryptocurrency, the water cycle explained, who was Nikola Tesla, how does DNA replication work, what caused World War 1, the theory of relativity explained, history of ancient Egypt, how do airplanes fly, what is the greenhouse effect, biography of Marie Curie, what is democracy, how does the stock market work, the history of jazz music, what are human rights, how does a computer processor work, what is natural selection, the silk road trade route history, how do earthquakes happen\n"
            "Wikidata: population of Malta, capital of Japan, date of birth of Einstein, official languages of Switzerland, GDP of Brazil, area of Australia in square kilometers, founding date of NASA, height of Mount Everest, currency of South Korea, Nobel Prize winners in physics, ISO country code for Germany, mayor of London, number of UNESCO world heritage sites, timezone of Tokyo, national anthem of France\n"
            "ArXiv: transformer architecture attention mechanism, reinforcement learning from human feedback, quantum error correction, diffusion models for image generation, large language model alignment techniques, neural radiance fields 3D reconstruction, federated learning privacy guarantees, graph neural networks for molecular property prediction, sparse mixture of experts scaling laws, contrastive learning self-supervised representations, protein structure prediction deep learning, adversarial robustness in neural networks\n"
            "GitHub: react state management libraries, python async http client, rust web framework, go database migration tools, typescript orm comparison, open source video editor, self-hosted analytics platform, kubernetes deployment tools, machine learning model serving frameworks, static site generators, command line productivity tools, open source password manager, real-time collaboration library, browser extension development kit\n"
            "Hacker News: startup funding trends, new programming languages, self-hosting alternatives, tech layoffs discussion, indie hacker revenue milestones, YC batch reviews, developer tool launches, remote work productivity debates, open source sustainability models, AI product launches and demos, side project show and tell, silicon valley culture critiques\n"
            "Reddit: best mechanical keyboards under 100, experiences with intermittent fasting, home lab setup advice, recommended sci-fi books 2025, moving to a new country tips, best budgeting apps, beginner weightlifting programs, sourdough bread troubleshooting, vintage audio equipment recommendations, travel itinerary for Japan two weeks, best online courses for data science, apartment decorating on a budget\n"
            "Google News: latest EU tech regulation, SpaceX launch update, central bank interest rate decision, election results, climate summit outcomes, tech company earnings report, natural disaster updates, sports championship results, pharmaceutical drug approval, diplomatic negotiations update, stock market crash analysis, new trade agreement signed\n"
            "Stack Overflow: python multiprocessing vs threading, docker compose networking, git rebase vs merge, nginx reverse proxy configuration, react useEffect cleanup, postgresql index optimization, CSS grid vs flexbox layout, kubernetes pod restart policy, redis cache invalidation strategies, typescript generic type constraints, JWT token refresh pattern, elasticsearch query performance tuning\n"
            "Open Library: books by Isaac Asimov, history of computing literature, science fiction classics, philosophy introductory texts, cookbooks by Ottolenghi, children's books about space, graphic novels about war, best translations of Homer, biographies of women in science, poetry collections from the Beat generation, mystery novels set in Venice, economics textbooks for beginners\n"
            "MusicBrainz: albums by Radiohead, jazz recordings from 1960s, film soundtrack composers, discography of Miles Davis, classical piano recordings of Chopin, electronic music labels from Berlin, live albums recorded at Fillmore, collaborations between David Bowie and Brian Eno, Portuguese fado recordings, West African highlife compilations\n"
            "iTunes: top podcast episodes about AI, audiobook versions of Dune, new music releases, true crime podcast series, meditation and mindfulness apps, language learning podcasts in Spanish, indie rock new releases this month, history podcast series about Rome, comedy specials on Apple TV, business and entrepreneurship audiobooks\n"
            "Nominatim: where is Valletta Malta, coordinates of Times Square, location of Mount Fuji, address of the Eiffel Tower, nearest city to Lake Bled, latitude longitude of Sydney Opera House, where is Machu Picchu, postal code for central Amsterdam, geographic center of the United States, location of the Colosseum in Rome"
        ),
        "effort": "light",
        "domain": "Research",
        "descriptor": "search",
        "usage_scenarios": ["see full_profile"],
        "triage_triggers": ["see full_profile"],
    },

    "weather": {
        "keywords": "weather,forecast,temperature,rain,wind,humidity,climate",
        "short_summary": "Get current weather conditions for any location — temperature, wind, rain, humidity, UV.",
        "full_profile": (
            "Current weather conditions for any location worldwide.\n\n"
            "Returns temperature in Celsius and Fahrenheit, feels-like temperature, humidity percentage, wind speed and direction, UV index, precipitation, visibility, and condition flags: is_raining, is_hot, is_cold, is_windy, is_clear, is_daylight.\n\n"
            "what is the weather like, should I bring an umbrella, is it raining outside, how hot is it today, what is the temperature in London, weather in New York, is it cold in Moscow, will I need a jacket, how windy is it, what is the UV index today, weather conditions in Tokyo, is it sunny outside, how humid is it, outdoor conditions for a picnic, should I wear sunscreen today, what is the weather like in Paris, is it safe to go running outside, beach weather today, weather for cycling, current conditions in Sydney, temperature in Dubai, is it snowing in Oslo, weather report for Malta, how is the weather in Berlin, rain forecast for today, wind conditions for sailing"
        ),
        "effort": "trivial",
        "domain": "Context",
        "descriptor": "weather",
        "usage_scenarios": ["see full_profile"],
        "triage_triggers": ["see full_profile"],
    },

    "news": {
        "keywords": "news,headlines,rss,current events,articles,press",
        "short_summary": "Search news articles across global sources.",
        "full_profile": (
            "News search from global sources including Google News and 56 curated RSS feeds in 7 categories: tech, business, sports, science, entertainment, US, UK.\n\n"
            "Two parameters: query (required, what to search for) and category (optional enum, narrows to curated RSS feeds for broad topic browsing).\n\n"
            "When to use: any request about current events, news, headlines, what's happening. "
            "Examples: 'latest news about AI', 'what's happening in tech today', 'Sam Altman news', 'UK election results'."
        ),
        "effort": "light",
        "domain": "Research",
        "descriptor": "news",
        "usage_scenarios": ["see full_profile"],
        "triage_triggers": ["see full_profile"],
    },

    "code_eval": {
        "keywords": "code,python,eval,execute,run,compute,calculate,script",
        "short_summary": "Run Python code to compute, verify formulas, or test logic. Use print() for output.",
        "full_profile": (
            "Python sandbox for precise computation. Available modules: math, statistics, json, decimal, fractions, itertools, functools, collections.\n\n"
            "calculate compound interest over 10 years, convert celsius to fahrenheit, verify a quadratic formula, compute standard deviation of a dataset, test a sorting algorithm, generate fibonacci sequence, validate a regex pattern, compute factorial of large numbers, convert between number bases, calculate BMI from height and weight, solve simultaneous equations, compute permutations and combinations, verify matrix multiplication, calculate loan amortization schedule, convert unix timestamp to date, compute running average of a series, validate checksum algorithm, calculate distance between coordinates, compute percentage change between values, test string parsing logic, verify binary search implementation, calculate moving average, compute hash of a string, test date arithmetic, validate JSON structure, compute statistical median and mode, test list comprehension output, calculate area of geometric shapes, verify prime number checker, compute currency conversion rates"
        ),
        "effort": "trivial",
        "domain": "Productivity",
        "descriptor": "code_eval",
        "usage_scenarios": ["see full_profile"],
        "triage_triggers": ["see full_profile"],
    },

    "programming_docs_search": {
        "keywords": "docs,documentation,api,reference,php,python,javascript,go,rust,java,ruby,typescript,react,vue,angular,django,flask,laravel",
        "short_summary": "Search official documentation for PHP, Python, JavaScript, Go, Rust, Java, Ruby, C#, Dart, C/C++, Bash, SQL and frameworks Laravel, Django, Flask, NumPy, Pandas, Node.js, React, Vue, Spring, Rails, Flutter.",
        "full_profile": (
            "Official documentation lookup for programming languages and frameworks. Returns extracted content from the canonical documentation source.\n\n"
            "Supported languages: PHP, Python, JavaScript, TypeScript, Go, Rust, Java, Ruby, C#, Dart, C, C++, Bash, SQL.\n"
            "Supported frameworks: Laravel, Django, Flask, NumPy, Pandas, Node.js, React, Vue, Spring, Rails, Flutter.\n\n"
            "php array_map function, python asyncio gather, javascript promise.all, typescript generics, go goroutines and channels, rust ownership and borrowing, java streams api, ruby blocks and procs, csharp linq queries, dart futures and async, c pointer arithmetic, cpp smart pointers, bash array manipulation, sql window functions, laravel eloquent relationships, django queryset filtering, flask blueprint registration, numpy array broadcasting, pandas dataframe merge, nodejs event loop, react useEffect hook, vue computed properties, spring dependency injection, rails active record associations, flutter stateful widget lifecycle, python list comprehension, php pdo prepared statements, javascript fetch api, go error handling patterns, rust pattern matching, java optional class, ruby enumerable methods, django middleware, laravel queue workers, react context api, vue router navigation guards, flask request handling, numpy matrix operations, pandas groupby aggregation, spring boot configuration, rails migrations, flutter provider state management, python decorators, php composer autoloading, typescript utility types, go interfaces, rust traits and implementations, java collections framework, sql joins and subqueries, nodejs stream processing, react hooks custom implementation"
        ),
        "effort": "light",
        "domain": "Productivity",
        "descriptor": "programming_docs_search",
        "usage_scenarios": ["see full_profile"],
        "triage_triggers": ["see full_profile"],
    },
}

if _BROWSER_AVAILABLE:
    BUILTIN_TOOL_PROFILES["browser"] = {
        "keywords": "browser,webpage,screenshot,scrape,render,javascript,html,click,form",
        "short_summary": "Open web pages with full JavaScript rendering, take screenshots, fill forms, and monitor pages for changes.",
        "full_profile": (
            "Headless browser for JavaScript-heavy web pages. Use when the read skill returns empty or broken content from single-page apps, dynamic sites, or Cloudflare-protected pages. The read skill is faster for static pages — try it first.\n\n"
            "Four actions: render (load page with JS, extract text), screenshot (capture page as PNG with optional OCR), interact (fill forms, click buttons, navigate multi-step flows), monitor (track page changes over time with snapshots).\n\n"
            "render a single page application, extract content from a cloudflare protected site, load a react or angular web app, get text from a dynamically loaded page, render a dashboard with live data, extract content behind javascript rendering, load a page that requires cookies, render an e-commerce product page, extract pricing from a dynamic listing, load a web app login page, render a documentation site built with javascript, extract content from a news paywall preview, load a social media profile page, render a job listing site, extract flight prices from a booking site, take a screenshot of a web dashboard, capture visual state of a monitoring page, screenshot a chart or graph for analysis, capture a web page design for reference, screenshot an error page for debugging, take a full page screenshot of a long article, capture a responsive design at specific viewport, fill out a contact form, submit a search form and extract results, log in to a web application, navigate a multi-step checkout process, apply filters on a product listing, fill out a registration form, select options from dropdown menus, click through a setup wizard, monitor a product page for price drops, track stock availability changes, watch a job listing page for new postings, monitor a status page for outages, track shipping status updates, watch for appointment availability, monitor auction prices, track concert ticket availability"
        ),
        "effort": "moderate",
        "domain": "Research",
        "descriptor": "browser",
        "usage_scenarios": ["see full_profile"],
        "triage_triggers": ["see full_profile"],
    }


def get_metadata(name: str) -> dict | None:
    """Return the metadata dict for a named tool, or ``None``.

    Args:
        name: The registered tool name (e.g. ``"weather"``).

    Returns:
        The metadata ``dict`` for the tool, or ``None`` if the tool is not
        registered or has no metadata entry.
    """
    with _registry_lock:
        return TOOL_METADATA.get(name)


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

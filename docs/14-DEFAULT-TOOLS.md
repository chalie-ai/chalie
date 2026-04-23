# First-Party Tools

These tools ship with Chalie and are invoked in-process via the ACT loop. Most of them are surfaced to the LLM either via the mode gate (when the current user turn activates a cognitive intent they serve) or by the `find_tools` innate skill's semantic search — the full list is never pre-injected into context. A `browser` tool is also available when Playwright dependencies are installed.

The mode-gate mapping at time of writing: `search` and `news` serve `research` and `brainstorm`; `browser` serves `research` and `analyze`; `code_eval` serves `coding` and `math`; `programming_docs_search` serves `coding` and `research`. `weather` is intentionally unmapped and reaches the LLM only through `find_tools`. See `docs/09-TOOLS.md` for how the three loading tiers stack and when each fires.

## Weather

Fetches current conditions and tomorrow's forecast using Open-Meteo and wttr.in. No API key required. Results include temperature, precipitation, and wind at the user's location when telemetry is available.

## Search

Searches across multiple sources — Wikipedia, GitHub, Reddit, arXiv, Google News, Stack Overflow, Open Library, and more — using plain natural language. Semantic routing selects the best provider(s) automatically. Returns results with titles, URLs, snippets, and provenance. No API key required.

## News

Searches news articles across global sources including Google News and curated RSS feeds in seven categories: tech, business, sports, science, entertainment, US, UK. Use for current events, headlines, and what's happening now. No API key required.

## Code Eval

Executes Python snippets in a restricted sandbox. Used to verify formulas, test algorithms, or produce exact numerical results rather than approximations. Execution is isolated — no filesystem access, no network.

## Programming Docs Search

Searches and reads official documentation for 12 languages and 11 major frameworks. Languages: PHP, Python, JavaScript/TypeScript, Go, Rust, Java, Ruby, C#, Dart, C/C++, Bash, SQL. Frameworks: Django, Flask, NumPy, Pandas, Laravel, Node.js, React, Vue, Spring, Rails, Flutter. No API key required.

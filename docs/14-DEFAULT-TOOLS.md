# First-Party Abilities

These abilities ship with Chalie as `Ability` subclasses under `backend/abilities/` and are invoked in-process via the ACT loop. None are pre-injected into context — they reach the LLM exclusively through the `find_tools` innate ability's semantic search. A `browser` ability is also present when Playwright dependencies are installed.

See `docs/09-TOOLS.md` for how innate vs. discoverable tiers stack and when each fires.

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

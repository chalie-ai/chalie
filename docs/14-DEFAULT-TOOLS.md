# First-Party Tools

These tools ship with Chalie and are invoked in-process via the ACT loop. The LLM discovers them through `find_tools` (semantic search) — they are not pre-loaded into context.

## Weather

Fetches current conditions and tomorrow's forecast using Open-Meteo and wttr.in. No API key required. Results include temperature, precipitation, and wind at the user's location when telemetry is available.

## Web Search

Searches the web via DuckDuckGo. Privacy-focused and requires no API key. Returns a ranked list of results with titles, URLs, and snippets that the LLM can reason over or follow up on.

## Code Eval

Executes Python snippets in a restricted sandbox. Used to verify formulas, test algorithms, or produce exact numerical results rather than approximations. Execution is isolated — no filesystem access, no network.

## Programming Docs Search

Searches and reads official documentation for 12 languages and 11 major frameworks. Languages: PHP, Python, JavaScript/TypeScript, Go, Rust, Java, Ruby, C#, Dart, C/C++, Bash, SQL. Frameworks: Django, Flask, NumPy, Pandas, Laravel, Node.js, React, Vue, Spring, Rails, Flutter. No API key required.

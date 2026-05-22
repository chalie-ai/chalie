# First-Party Abilities

These abilities ship with Chalie as `Ability` subclasses under `backend/abilities/` and are invoked in-process via the ACT loop. Tool tiers are centralised on `MessageProcessor`: `ALWAYS_AVAILABLE` defaults to `["find_skills", "find_tools", "memory"]` and `DISCOVERABLE` defaults to all 20 first-party abilities. Most processors inherit these defaults; subclasses override only where needed (e.g. `_BLOCKED` to exclude specific tools). The `find_tools` tool schema includes a compact index of all discoverable tools (built from each ability's `SEARCH_TOOLTIP`) so the LLM knows what is available before calling it.

See `docs/09-TOOLS.md` for how the always-available and discoverable tiers stack and when each fires.

## Find Skills

Returns curated step-by-step playbooks for complex tasks (research, planning, analysis, writing). Queries `abilities/assets/skills.sqlite` (vec + FTS5 RRF fusion) built from YAML files in `backend/abilities/skills/`. Results are annotated with personalisation rules from `skill_associations` when `SkillAssociationService` has mapped the user's behavioural patterns to a skill. ALWAYS_AVAILABLE on all user-facing processors — not discoverable, because returning procedural playbooks is infrastructure like `find_tools` and `memory`. Falls back to FTS-only when embedding generation fails. Build: `python -m utils.build_skills_db`; drift check: `python -m utils.build_skills_db --check`.

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

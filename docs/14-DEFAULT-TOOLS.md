# First-Party Abilities

These abilities ship with Chalie as `Ability` subclasses under `backend/abilities/` and are invoked in-process via the ACT loop. Tool tiers are centralised on `MessageProcessor`: `ALWAYS_AVAILABLE` defaults to `["find_skills", "find_tools", "memory"]` and `DISCOVERABLE` defaults to all 21 first-party abilities. Most processors inherit these defaults; subclasses override only where needed (e.g. `_BLOCKED` to exclude specific tools). `find_tools` and `find_skills` both inherit from `SearchableAbility` (`abilities/_search.py`) which provides shared vec+FTS5 RRF fusion search.

See `docs/09-TOOLS.md` for how the always-available and discoverable tiers stack and when each fires.

## Find Skills

Returns curated and user-created step-by-step tool-calling playbooks for complex tasks. Queries `abilities/assets/skills.sqlite` (vec + FTS5 RRF fusion). Each playbook is a numbered sequence of exact tool-calling instructions (e.g. "Use the `search` tool to find competitors…"). Results are annotated with personalisation rules from `skill_associations` when `SkillAssociationService` has mapped the user's behavioural patterns to a skill. Filters by `enabled=1` so disabled skills are excluded. ALWAYS_AVAILABLE on all user-facing processors — not discoverable, because returning procedural playbooks is infrastructure like `find_tools` and `memory`. Falls back to FTS-only when embedding generation fails. Build: `python -m utils.build_skills_db`; drift check: `python -m utils.build_skills_db --check`.

## Skill Builder

Create, edit, delete, and list user-defined skill playbooks. User skills are stored as YAML files in `data/skills/user/` with the same frontmatter format as curated skills in `abilities/skills/`. On create/edit, skills are indexed into `skills.sqlite` for `find_skills` routing. Only `source='user'` skills can be edited or deleted. Actions: `create` (title + use_for + content required), `edit` (title required, partial updates), `delete` (title required), `list` (no params). All four actions default to `allow` in chat policy. DISCOVERABLE in UMP + DMN. The Brain Skills tab (`/api/skills`) provides a parallel REST CRUD surface for managing skills from the dashboard.

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

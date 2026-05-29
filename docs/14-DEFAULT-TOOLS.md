# First-Party Abilities

These abilities ship with Chalie as `Ability` subclasses under `backend/abilities/` and are invoked in-process via the ACT loop. Tool tiers are centralised on `MessageProcessor`: `ALWAYS_AVAILABLE` defaults to `["find_skills", "find_tools", "memory"]` and `DISCOVERABLE` defaults to all 22 first-party abilities. Most processors inherit these defaults; subclasses override only where needed (e.g. `_BLOCKED` to exclude specific tools). `find_tools` and `find_skills` both inherit from `SearchableAbility` (`abilities/_search.py`) which provides shared vec+FTS5 RRF fusion search.

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

## Search Files

Cross-platform alternative to `bash find`/`bash grep` — ensures consistent behaviour across macOS, Linux, and Windows including mounted drives and connected storage. No path restrictions: the LLM may search any directory on the system.

Two actions: `glob` (filename pattern matching via `fnmatch`) and `grep` (content search via regex). `query` is required; `directory` is optional (defaults to `$HOME`). Optional `max_files` (default 10) caps the number of returned files; optional `context_lines` (default 3, grep only) controls how many lines above and below each match are shown.

`glob` returns a JSON list of absolute file paths (most-recently-modified first). `grep` returns per-file results with line-numbered context snippets around each match, plus a hint to call `read` for full file contents. Grep skips files >5 MiB; symlinks are not followed (loop-safe). All contexts default to `allow` for both actions. DISCOVERABLE on every user-facing processor.

## Git

Read and write GitHub and GitLab repositories. Requires the Git capability to be enabled and configured in Brain → Capabilities.

**Authentication:** works without a token (public repos, read-only). Add a Personal Access Token to unlock private repo access and all write actions.

**Self-hosted:** set the `host` field (e.g. `https://gitlab.example.com`) to target a self-hosted GitHub Enterprise or GitLab instance. The host must be reachable over HTTPS; private/loopback ranges are blocked via the shared SSRF guard.

**Architecture:** hybrid — git CLI (`clone`, `diff`, `commit`, `push`) plus REST API (`get_repo`, `list_branches`, `list_commits`, `get_commit`, `list_prs`, `get_pr`, `list_issues`, `get_issue`, `list_releases`, `search_repos`, `read_file`, `create_pr`, `merge_pr`). 17 actions total.

**Workspace lifecycle:** `clone` creates a shallow temp directory (`--depth 1` by default; `full=true` for a complete clone). Use existing file tools (`file_write`, `search_files`, `bash`) to edit files in the workspace, then call `commit` (stage explicit paths only) and `push`. Force-push and pushes to default/protected branches are unconditionally blocked. Stale workspaces older than 6 hours are cleaned at the start of each `clone`.

**Policy:** all READ actions are `allow` in every channel. WRITE actions (`commit`, `push`, `create_pr`, `merge_pr`) are `ask` in chat and subagent, `deny` in subconscious and external_agent. TIMEOUT = 120 s. DISCOVERABLE only.

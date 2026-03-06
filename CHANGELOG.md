# Changelog

All notable changes to this project will be documented in this file.

[Unreleased]

### Added

- temporal pattern mining — learn behavioral rhythms from ambient signals
- rename google_news → news_tool and promote to trusted tool
- migrate Wikipedia tool to trusted execution
- add action intent detection and improve startup reliability
- add zscore and zremrangebyscore to MemoryStore
- add 14-DEFAULT-TOOLS.md — first-party default tools catalog
- weather tool as trusted subprocess + default tool auto-install
- single-process refactor + read innate skill + text extractor
- embodiments marketplace — tag-based install, catalog, update checker
- backend readiness probe — hold spark overlay until system is truly ready (#5)
- document processing with synthesis confirmation & supersession (#6)
- deferred card rendering + autobiography as Chalie's self-narrative
- temporal pattern mining — behavioral patterns from interaction history as user traits
- procedural memory in context assembly — learned action reliability in RESPOND mode
- decision explanation — routing audit and autonomous actions in introspect skill
- conversational inspection — user_traits layer in recall skill
- ship-readiness hardening across 4 recent features + persistent task execution
- unified activity feed — what Chalie did while you were away
- conversational belief correction — user's word always wins
- complete deletion scope and add data export endpoint
- wire semantic concept injection into context assembly and response generation
- cognitive reflexes — learned fast path via semantic abstraction
- psycopg2 cursor pattern in system.py + deprecate gemini-2.0-flash
- plan decomposition for persistent tasks + observability fixes
- message routing simplification & robustness (4 phases)
- rate limit graceful fallback + duplicate schedule prevention
- voice container fail-safe — nginx resolver + opt-out override
- STT Docker service
- expand test suite to 843 passing tests, remove valueless stubs, add CI gate
- persistent task strip + continuity surfaced — make Chalie's work and understanding visible
- Added vision doc
- cognitive legibility for Brain dashboard — make Chalie's thinking visible
- oauth support for tools
- close tool learning feedback loops — performance data flows into tool selection
- critic loop, persistent tasks, event bridge, OAuth, background LLM queue
- triage routing — add triage_triggers for vocabulary bridging, remove _pick_default_tool
- PWA share target + ambient behavioral sensor
- tool webhook endpoint + interactive bidirectional dialog
- spark — first-contact welcome, nurture cadence, skill suggestions
- ambient inference + place learning — deterministic context pipeline
- text race
- prompt
- forget moment — 10s undo window with card pending state
- add moment-enrichment LLM config and Brain UI registration
- Moments — pinned message bookmarks with semantic recall
- add automated build log generation via GitHub Actions
- implement hot tool registration without restart
- add GitHub Actions workflow to sync docs to chalie-web

### Changed

- purge Redis, RQ, PostgreSQL and private network references
- unified ACT orchestrator — eliminate triplicated loop, centralize constants, fix card duplication (#10)
- mode router coverage — ACT scoring, IGNORE, ACKNOWLEDGE, and hysteresis
- Fixes
- Fix for voice
- expand unit test suite to 945 passing — workers, services, API, mock depth
- remove tool-specific handler tests — tools are agnostic capsules
- Increase feedback clarity when acting and deciding when to put human-in-the-loop
- skill calling
- thread that may manifest in impromptu human interactions.
- Markdown support in responses
- Adaptive Response & Tool Disabling across multi-instance installs
- switch from Claude to Gemini for build log generation
- use Claude Code CLI instead of direct API calls
- automate build log generation
- Fix for multi-instance deployments
- partial markdown support in fe
- fix for tool-agnostic structure & improve token usage
- multi-instance on same host support
- tool parsing fix
- provider issues
- DB fixes & provider switch cache issues
- docs refinement
- Delete plan.md
- Doc fixes (#1)
- Scheduler and list cards
- fixes
- Fixes to tool pipeline
- Innate skils
- Clean repo

### Fixed

- create-faq-file
- fix-link-injection-script
- automate-docs-cross-linking
- revert-docs-and-rerun-script
- fix-seo-script-bugs
- create-seo-optimization-script
- expand-docs-table
- remove-duplicate-install
- add-feature-and-community-sections
- refactor-readme-header
- replace signal.alarm with thread-based timeouts, fix WS auth close, routing JSON deserialize, schema always-apply
- increase brain navbar background opacity for scroll separation
- expand onboarding schedule with missing identity traits
- use consistent CHALIE_WEB_TOKEN in release workflow
- stack provider cards vertically on mobile (#12)
- WebSocket close signals, sqlite-vec aarch64 build, and auth error propagation
- strip tool markers from act_history to prevent JSON-formatted responses
- attenuate reward penalty for externally rate-limited tools (Issue 004)
- 11 bugs — OOM crash loops, RQ poison-pill, cognitive drift, push cleanup, frontend hardening
- wire persistent task progress events to frontend task strip (#2)
- wire persistent task progress events to frontend task strip
- voice defaults — Whisper base (fits 4GB Docker), Jasper voice
- voice service build — use GitHub wheel, correct model ID, bump timeout
- tolerate markdown-fenced JSON in LLM profile build responses
- build_skill_profile
- background_llm_worker signature — accept shared_state arg
- tool profile LLM builder — connect provider, remove max_predict_tokens, force rebuild
- URL detection in self-eval, heuristic fallback, and forced profile rebuild
- preserve newlines and whitespace in user messages
- JSON recovery layer, timezone-aware quiet hours, context budget, demographic traits
- moment card — markdown rendering, forget button, footer layout
- moment card repetition — collapsible message, fix title fallback
- correct Content-Length to use byte length and strip Gemini JSON fences
- use ANTHROPIC_API_KEY instead of OAuth token for API auth
- embodiments tab now reflects installed tools correctly

### Documentation

- update architecture, workers, and cognitive docs for plans 06-10
- architecture — ambient awareness, spark, tool webhook, interactive dialog

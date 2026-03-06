# Changelog

All notable changes to this project will be documented in this file.

[Unreleased]

### Added

- Temporal pattern mining — learn behavioral rhythms from ambient signals
- Rename google_news → news_tool and promote to trusted tool
- Migrate Wikipedia tool to trusted execution
- Add action intent detection and improve startup reliability
- Add zscore and zremrangebyscore to MemoryStore
- Add 14-DEFAULT-TOOLS.md — first-party default tools catalog
- Weather tool as trusted subprocess + default tool auto-install
- Single-process refactor + read innate skill + text extractor
- Embodiments marketplace — tag-based install, catalog, update checker
- Backend readiness probe — hold spark overlay until system is truly ready (#5)
- Document processing with synthesis confirmation & supersession (#6)
- Deferred card rendering + autobiography as Chalie's self-narrative
- Temporal pattern mining — behavioral patterns from interaction history as user traits
- Procedural memory in context assembly — learned action reliability in RESPOND mode
- Decision explanation — routing audit and autonomous actions in introspect skill
- Conversational inspection — user_traits layer in recall skill
- Ship-readiness hardening across 4 recent features + persistent task execution
- Unified activity feed — what Chalie did while you were away
- Conversational belief correction — user's word always wins
- Complete deletion scope and add data export endpoint
- Wire semantic concept injection into context assembly and response generation
- Cognitive reflexes — learned fast path via semantic abstraction
- Psycopg2 cursor pattern in system.py + deprecate gemini-2.0-flash
- Plan decomposition for persistent tasks + observability fixes
- Message routing simplification & robustness (4 phases)
- Rate limit graceful fallback + duplicate schedule prevention
- Voice container fail-safe — nginx resolver + opt-out override
- Ship built-in local voice — zero-config TTS/STT Docker service
- Expand test suite to 843 passing tests, remove valueless stubs, add CI gate
- Persistent task strip + continuity surfaced — make Chalie's work and understanding visible
- Added vision doc
- Cognitive legibility for Brain dashboard — make Chalie's thinking visible
- Oauth support for tools
- Close tool learning feedback loops — performance data flows into tool selection
- Critic loop, persistent tasks, event bridge, OAuth, background LLM queue
- Triage routing — add triage_triggers for vocabulary bridging, remove _pick_default_tool
- PWA share target + ambient behavioral sensor
- Tool webhook endpoint + interactive bidirectional dialog
- Spark — first-contact welcome, nurture cadence, skill suggestions
- Ambient inference + place learning — deterministic context pipeline
- Tool cards — generic interactive conventions + card/text race
- Scheduler types — replace reminder/task with notification/prompt
- Forget moment — 10s undo window with card pending state
- Add moment-enrichment LLM config and Brain UI registration
- Moments — pinned message bookmarks with semantic recall
- Add automated build log generation via GitHub Actions
- Implement hot tool registration without restart
- Add GitHub Actions workflow to sync docs to chalie-web

### Changed

- Purge Redis, RQ, PostgreSQL and private network references
- Unified ACT orchestrator — eliminate triplicated loop, centralize constants, fix card duplication (#10)
- Mode router coverage — ACT scoring, IGNORE, ACKNOWLEDGE, and hysteresis
- Fixes
- Fix for voice
- Expand unit test suite to 945 passing — workers, services, API, mock depth
- Remove tool-specific handler tests — tools are agnostic capsules
- Increase feedback clarity when acting and deciding when to put human-in-the-loop
- Bug fix for multi-step tool / skill calling
- Curiosity & Self-defined goals / thread that may manifest in impromptu human interactions.
- Markdown support in responses
- Adaptive Response & Tool Disabling across multi-instance installs
- Switch from Claude to Gemini for build log generation
- Use Claude Code CLI instead of direct API calls
- Automate build log generation
- Fix for multi-instance deployments
- Partial markdown support in fe
- Fix for tool-agnostic structure & improve token usage
- Multi-instance on same host support
- Tool parsing fix
- Provider issues
- DB fixes & provider switch cache issues
- Docs refinement
- Delete plan.md
- Doc fixes (#1)
- Scheduler and list cards
- Fixes to tool pipeline
- Innate skils
- Clean repo

### Fixed

- Generate-changelog-from-git
- Create-faq-file
- Fix-link-injection-script
- Automate-docs-cross-linking
- Revert-docs-and-rerun-script
- Fix-seo-script-bugs
- Create-seo-optimization-script
- Expand-docs-table
- Remove-duplicate-install
- Add-feature-and-community-sections
- Refactor-readme-header
- Replace signal.alarm with thread-based timeouts, fix WS auth close, routing JSON deserialize, schema always-apply
- Increase brain navbar background opacity for scroll separation
- Expand onboarding schedule with missing identity traits
- Use consistent CHALIE_WEB_TOKEN in release workflow
- Stack provider cards vertically on mobile (#12)
- WebSocket close signals, sqlite-vec aarch64 build, and auth error propagation
- Strip tool markers from act_history to prevent JSON-formatted responses
- Attenuate reward penalty for externally rate-limited tools (Issue 004)
- 11 bugs — OOM crash loops, RQ poison-pill, cognitive drift, push cleanup, frontend hardening
- Wire persistent task progress events to frontend task strip (#2)
- Wire persistent task progress events to frontend task strip
- Voice defaults — Whisper base (fits 4GB Docker), Jasper voice
- Voice service build — use GitHub wheel, correct model ID, bump timeout
- Tolerate markdown-fenced JSON in LLM profile build responses
- Force-rebuild bypasses inner staleness guard in build_profile/build_skill_profile
- Background_llm_worker signature — accept shared_state arg
- Tool profile LLM builder — connect provider, remove max_predict_tokens, force rebuild
- URL detection in self-eval, heuristic fallback, and forced profile rebuild
- Preserve newlines and whitespace in user messages
- JSON recovery layer, timezone-aware quiet hours, context budget, demographic traits
- Moment card — markdown rendering, forget button, footer layout
- Moment card repetition — collapsible message, fix title fallback
- Correct Content-Length to use byte length and strip Gemini JSON fences
- Use ANTHROPIC_API_KEY instead of OAuth token for API auth
- Embodiments tab now reflects installed tools correctly

### Documentation

- Update architecture, workers, and cognitive docs for plans 06-10
- Architecture — ambient awareness, spark, tool webhook, interactive dialog

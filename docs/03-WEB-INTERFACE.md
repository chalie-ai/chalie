# Chalie Web Interface

Chalie's frontend is four independent single-page applications — chat, brain, onboarding, and login — all built on the **Radiant design system**: a cinematic dark UI where light is used with restraint, so when something glows, it matters.

## Auth + Provider Gate

Every page loads a shared auth gate before its own bootstrap. The gate calls `/auth/status` once and redirects as needed:

| Page | No account | No session | No providers | All good |
|---|---|---|---|---|
| chat (`/`) | → onboarding | → login | → brain | enter chat |
| brain (`/brain/`) | → onboarding | → login | providers tab only | full dashboard |
| onboarding (`/on-boarding/`) | stay | — | — | — |
| onboarding (account exists) | — | → login | → login | → login |
| login (`/login/`) | stay | stay | — | → chat |

When brain enters **providers-only mode**, every tab except Providers is hidden and a persistent toast prompts setup. All app code waits for the gate to complete before booting.

## Radiant Design System

The canvas is near-black — darker than a typical dark mode. Color enters the UI in two forms only: atmospheric orbs drifting in the background canvas, and precision accents on interactive elements. Surfaces carry no tint; they are almost-transparent glass over the dark floor.

Three accent colors do all expressive work: **violet** (primary, buttons and active states), **magenta** (presence and indicators), and **cyan** (processing states). Each element carries one glow, one thin edge highlight. No stacked shadows, no multi-color gradients on small elements. When nothing is active, the newest Chalie message draws the eye with a single thin violet top edge.

**Atmospheric depth** comes from a background canvas rendering four low-alpha orbs — two warm (violet, magenta), two cool (cyan, indigo) — drifting on slow 25–35 second cycles. They provide color temperature without competing with the UI.

Transitions are uniform and quick. Buttons use a solid accent fill and glow only on hover. The restraint is deliberate: luxury is making the user notice the one thing that changed.

## Layout

The chat interface uses three fixed regions:

- **Title bar** (top, fixed) — app name, status indicators, optional media controls during voice playback, navigation
- **Chat area** (scrollable middle) — messages in chronological order; Chalie messages sit left with a thin violet top-edge gradient, user messages right; depth fades at scroll boundaries
- **Prompt box** (bottom, fixed) — full-width text input; mic button on the left, send on the right

## ACT Cycle UI

When the user sends a message, Chalie immediately renders a **chrome-less ACT cycle** — no bubble, no border, no background. Just a small blinking violet logo that signals "Chalie is working". Narrations and tool calls accumulate beside and below the logo until the ACT loop completes; on completion, the entire ACT UI vanishes and is replaced by a normal Chalie speech-form bubble carrying the final response.

The shape:

```
While the ACT loop runs:                    Final:
●  Got it — checking the weather…           ┌──────────────┐
   Search          1.5s                     │ <response>   │
   Read            2.2s                     │              │
   weather         error                    └──────────────┘
   find_tools      2.4s
   Read            ⟳
```

- **Logo** (`act-logo`) — 12 px violet disc, 1.4 s sine pulse. It IS the placeholder; there is no "Working on it…" text.
- **Narrative** (`act-narrative`) — italic text at 75 % opacity sitting on the same row as the logo. **Each new narration replaces the previous one** — narratives do not stack.
- **Tool list** (`act-tools`) — cumulative across the entire ACT loop (NOT per-iteration). Indented 24 px under the logo so it visually belongs to the row.
- **Final response**: `replaceActWithResponse` removes the `act-cycle` node and appends a fresh `.speech-form--chalie` bubble in its place.

Tool rows are minimal monospace lines — name LEFT, state RIGHT — with no chrome:

| State | Class | Color | Status slot |
|---|---|---|---|
| Running | `act-tool--running` | white @ 75 % | small white spinner |
| Done | `act-tool--done` | `var(--success)` (#34d399) @ 90 % | duration in seconds (e.g. `1.5s`) |
| Error | `act-tool--error` | `var(--error)` (#fb7185) @ 50 % | literal `error` (the actual error string is not surfaced) |

Sub-100 ms tools enforce a 150 ms minimum visible duration so the spinner-to-state transition is always perceptible. Rows are display-only — no click-to-expand, no tooltips.

**CSS classes:** `act-cycle` (chrome-less host), `act-row` (logo + narrative line), `act-logo` (blinking disc), `act-narrative` (italic text), `act-tools` (cumulative list), `act-tool` + `act-tool--running` / `--done` / `--error`, `act-tool__name`, `act-tool__status`, `act-spinner`. Animations: `@keyframes act-logo-pulse` (1.4 s opacity + box-shadow), `@keyframes act-spinner-spin` (0.9 s linear).

**Frontend wiring:** `renderer.js` exposes `createActCycle()`, `setActNarrative(actEl, text, step)`, `appendToolPill(actEl, callId, name)`, `resolveToolPill(callId, ms, ok)`, `replaceActWithResponse(actEl, blocks, meta)`, `replaceActWithError(actEl, message)`. `chat.js` calls `createActCycle()` once on send and stores the element for the whole turn — `onNarration` calls `setActNarrative` (which mutates the single text slot in-place), `onToolStart` calls `appendToolPill(actEl, …)` (single flat list, no nesting under narrations), `onDone` calls `replaceActWithResponse` to swap the ACT UI for the final bubble. The action flow in `app.js` (deterministic skill invocation) shares the same primitives.

## Presence Dot

A small dot beneath the chat input communicates what Chalie is doing:

| State | Color | Animation |
|---|---|---|
| Resting | Magenta | Slow breathing pulse |
| Processing | Cyan | Faster pulse |
| Thinking | Violet | Variable-intensity glow |
| Retrieving memory | Cyan | Expanding ripple |
| Planning | Violet → cyan | Cycling shimmer |
| Responding | Amber | Waveform bars |

## Cards System

Structured responses render as typed cards rather than prose. All cards share the same dark surface and violet accent language.

- **Scheduled item** — title, time, recurrence, status (pending/completed), edit/delete actions
- **List** — name, type, item count, recent items preview, add/manage actions
- **Goal** — title, progress bar, target date, status (active/completed/abandoned), update actions
- **Knowledge** — concept name and strength, related concepts, last accessed

## Rich-Media Cards

Rich-media cards are tool-driven structured renders that appear inline with text bubbles in the chat surface. They follow the same Radiant conventions (near-black surface, 1 px violet/cyan accent edge, hover glow) and are the only component class that may use entrance animation.

**Segment iteration.** `appendChalieForm(content, meta, opts)` and `prependChalieForm(content, meta, opts)` in `frontend/interface/renderer.js` iterate `meta.segments` when present. Each segment is either `{type:"text"}` — rendered as a normal text bubble via `_appendTextBubble` — or `{type:"rich", tag, payload, synthesis}` — rendered via `_appendRichCard`. When `segments` is absent the renderer falls back to a single text bubble using `content`, keeping the change backward-compatible.

**Module registry.** `frontend/interface/rich_media/registry.js` maps tag prefixes to card modules:

```
frontend/interface/rich_media/
  registry.js        — tag-prefix → module map
  weather.js / .css  — ambient-sky weather card
  article.js / .css  — news + search (shared layout)
  scheduler.js / .css — schedule card (date block + same-day list)
  list.js / .css     — checklist with click-to-toggle persistence
  timer.js / .css    — live countdown card (pause/stop, Web-Audio alarm)
  base_card.css      — shared Radiant card chrome (border, padding, entrance keyframes)
  icons/weather/     — semantic SVGs (clear-day, clear-night, partly-cloudy-*, rain, snow, fog, thunderstorm, cloudy)
```

`registry.js` extracts the prefix from the tag (e.g. `weather` from `weather_1`) and delegates to the matching module's `render(payload, synthesis, root)`. Unknown prefixes fall back silently to a text bubble using the synthesis string.

**Animation scope.** Entrance: card fades in and lifts 8 px on mount (200 ms ease-out). The weather card additionally runs ambient loops on the hero canvas — sun bob (12 s), cloud drift (26–38 s) — because the canvas *is* the answer ("vibe, not data"). All other cards stay still. No live ticking, no parallax. Animation lives entirely inside the card module; the framework imposes nothing.

**Weather payload (ambient-sky variant).** Beyond the original fields, `WeatherAbility` (Open-Meteo path) now emits:

- `sunrise`, `sunset` — local ISO timestamps from the Open-Meteo `daily` block, used by the FE to pick a phase (`dawn` / `day` / `sunset` / `night`).
- `hourly` — array of 8 `{hour, temp_c, code}` entries starting from the slot containing the current observation. Drives the bottom rail. wttr.in fallback returns `[]`; the rail simply hides.

The card renders a gradient sky tinted by phase, drifting cloud blobs, a sun (day/dawn/sunset) or moon (night), a low skyline silhouette, and an overlay with the big top-left temperature, location + day/time top-right, and a narrative caption (the LLM synthesis or a phase-aware fallback). The hour rail highlights the current hour (violet) and the daily peak (amber).

**Timer card.** `TimerAbility` returns `{title, duration_seconds}` to the LLM — the wall-clock anchor is deliberately hidden so the model can't fill, edit, or reason about it. `RichMediaParser` calls `TimerAbility.enrich_rich_payload(payload, row)` (the generic per-ability hook on `Ability`), which grafts `started_at` from `row.created_at` (normalised to ISO 8601 UTC) before the segment reaches the FE. `timer.js` then computes `remaining = duration_seconds − (now − started_at)` on every 1-second tick, so a page reload resumes the countdown from real elapsed (no DB, no worker — purely ephemeral, in contrast to `schedule` which persists in `scheduled_items`). Pause is in-memory only; reloading while paused resumes the timer. When `remaining ≤ 0` the ring switches to amber and a Web-Audio oscillator (3 × 880 Hz beeps every 1.4 s) plays until the user clicks Stop.

**Interactive cards — silent action channel.** Cards that mutate server state (currently the list card's check toggles) dispatch a `chalie:silent-action` `CustomEvent` on `document` with `detail = { payload, onMessage, onError, onDone }`. `app.js` listens and forwards `payload` to `ws.sendAction(...)` without rendering an ACT cycle or chat bubble — the card already owns the visual feedback (optimistic UI, revert on error). The matching backend path is unchanged: `WS message {type:'action'} → _handle_action → ActDispatcherService.dispatch_action('action_button', …)`. Channel `'action_button'` is **not** `'user'`, so no rich-media ordinal is injected and the response is a plain dict, never a card.

The action handler writes to the canonical `lists`/`list_items` tables, but the LLM-channel rich-media result is a snapshot frozen in `tool_calls.result`. So on conversation refresh, `ListAbility.enrich_rich_payload` re-fetches the live list via `ListService.get_list(name)` and overrides the snapshot — both render paths converge on the same data, no duplicate read paths. The same `Ability.enrich_rich_payload` hook services the timer's `started_at` injection; cards that need no live state simply inherit the identity default.

**Lazy-load conventions.** Media inside cards that should load asynchronously uses the standard data-attributes: `[data-lazy-embed]` for embeds, `[data-lazy-thumb]` for thumbnails, `[data-lazy-src]` for media URLs. The existing lazy-load observer in the chat surface handles these without card-specific wiring.

**Article thumbnails (search + news cards).** The search and news abilities surface up to three candidate images alongside their results — each result's per-provider image (GitHub avatar, Reddit `preview.images[0].source.url`, Open Library cover, RSS `media:thumbnail`/`enclosure`, etc.) is shortlisted by `services.image_candidate_service.build_image_candidates`, fetched in parallel (3s timeout, 5 MB cap), and OCR'd via the local RapidOCR service. Each candidate is `{url, source_title, ocr_text}` — `source_title` is the result's title so the LLM can match thumbnail → article without relying on OCR text (most news photos return empty OCR). When fewer than three native thumbnails are present (Google News RSS strips them, for example), `services.og_image_service.resolve_og_images` fills the gap by fetching the top result URLs in parallel and scraping `og:image` / `twitter:image` / relative paths (resolved against the post-redirect page URL); Google's SOCS cookie is sent so EU/UK requests bypass the consent gate. The instruction trailer asks the LLM to add `data-image='<that url>'` to the rich-media span if a candidate's source matches its synthesis. `RichMediaParser._apply_image_choice` then validates the chosen URL against the candidate list (rejects anything not shortlisted — closes the prompt-injection path) and promotes it to `payload.image_url` while always stripping `image_candidates` from the FE payload. `<span ... data-image="...">` is the only LLM-controlled thumbnail signal; the regex permits the attribute optionally so older spans without it continue to work unchanged.

See `docs/superpowers/specs/2026-05-02-rich-media-cards-design.md` for the full card protocol, weather payload shape, and acceptance criteria.

## Voice I/O

Voice is optional. If the voice service is unavailable, the mic button and all speaker icons are hidden automatically.

**Microphone (speech-to-text)** — the mic button sits left of the prompt box. Click to record, click again to stop. The transcript is pasted into the prompt box; the user reviews it and sends. The mic track is released immediately after each recording.

**Speaker (text-to-speech)** — a speaker icon appears below each Chalie message. Clicking it opens a centered overlay player with play/pause, seek ±10 s, a progress bar, and a close button. Only one message plays at a time; opening a new one cancels the previous. Playback streams: `/voice/synthesize` returns `{ok, total}` immediately and the backend publishes each sentence-sized WAV chunk on the `output:events` pub/sub channel (as `tts_chunk`, terminated by `tts_done`). The WebSocket forwards those frames to the client; VoicePlayer decodes each chunk into an `AudioBuffer` and chains playback via `AudioBufferSourceNode.onended`, so audio starts as soon as the first chunk is ready rather than waiting for the full blob. `AudioContext.resume()` runs synchronously inside the click handler, satisfying iOS Safari's autoplay policy.

## File Attachments

Images (max 3 per message) attach via three equivalent paths: the `+` menu file picker, **drag-and-drop** onto the input dock or anywhere on the viewport (a full-page overlay lights up on `dragenter`), or **paste** from the clipboard into the prompt box. Non-image files dropped anywhere fall through to the document upload endpoint.

Each attached image renders as a thumbnail in a strip above the prompt box with a cyan spinner and `analyzing` class while the backend runs OCR and scene analysis. The send button unblocks as soon as the upload returns an `image_id` — the user is never made to wait for analysis. A WebSocket `image_ready` event clears the spinner; a 90-second safety timeout replaces it with a warning badge if analysis never completes (the image stays attached, context may be incomplete). Analysis failure surfaces as a red badge.

The global drop overlay uses a refcount on `dragenter` / `dragleave` with `dragend` and `window.blur` fallbacks, so it always clears — even if the browser swallows the `drop` event.

## Applications

### Chat (`/`)

The primary interface. Title bar + scrollable chat + prompt box. Presence dot indicates cognitive state. Canvas atmosphere renders behind everything. Home view shows recent conversations. Supports all card types and optional voice I/O.

### Brain (`/brain/`)

The cognitive dashboard. Tabs expose episodic memory with decay visualization, semantic concepts, routing audit trails, tool history, settings, and tool management.

The **Personality** subtab (under Cognition) exposes five sliders — warmth, mood, expressiveness, curiosity, humor — each ranging from −2 to +2. The selected combination maps to a voice paragraph prepended to the system prompt for user-facing conversations. Background processors (memory encoding, goal pursuit, scheduled tasks) are unaffected.

The **Errors** subtab (under Cognition) shows the most recent ERROR and CRITICAL log entries from `/tmp/chalie.log`, newest first, capped at 200 entries. Served by `GET /system/observability/errors` (`@require_session`). Reads only the last ~256 KB of the log file. Returns an empty list on a missing file rather than a 500.

### Onboarding (`/on-boarding/`)

Account creation only — username and password. If an account already exists, the page bounces to login. After creating an account it redirects to Brain so the user can add a provider.

### Login (`/login/`)

A dedicated form so browser autofill and OS credential managers (macOS Keychain, etc.) can offer stored credentials. Accepts a `?next=` parameter to resume the original destination after sign-in. Bounces to chat when a session is already active.

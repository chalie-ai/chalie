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

## Tool Pills + Fastpath Shape

When the user sends a message, Chalie immediately renders a **fastpath bubble** — a single Chalie speech-form that persists for the entire ACT loop. Everything that happens during the turn (narrations, tool calls) lives **inside** this bubble. The final response replaces it wholesale; nothing leaks into a sibling bubble.

The shape grows iteration by iteration:

```
Iteration 1:                  Iteration 2:                   Final:
┌──────────────┐              ┌──────────────┐               ┌──────────────┐
│ Working on…  │              │ Working on…  │               │ <response>   │
│   ▸ tool 1   │              │   ▸ tool 1   │               │              │
│   ▸ tool 2   │              │   [narr_1]   │               │              │
└──────────────┘              │     ▸ tool 3 │               └──────────────┘
                              │     ▸ tool 4 │
                              └──────────────┘
```

- **Iter 1, pre-narration**: tool pills attach directly to the fastpath.
- **Iter 2+, narration arrived**: each narration is a sub-bubble nested inside the fastpath; the iteration's tool pills nest under that narration.
- **Final response**: `resolvePendingForm` wipes the fastpath's inner content and renders the response blocks in its place — narrations and pills disappear with it.

Pills themselves are minimal monospace rows — name on the left, state on the right — with no chrome or fill. The defining visual is a thin iridescent gradient running along the bottom edge of each row (cyan → lavender, magenta on error), the same accent language as the rest of the Radiant surface.

- **Active** (`tool-pill--active`) — full-opacity gradient with a small lavender spinner.
- **Done** (`tool-pill--done`) — gradient fades to ~28% opacity; status slot shows the elapsed duration in seconds (e.g. `0.7s`).
- **Error** (`tool-pill--error`) — gradient swaps to a pure magenta band; status slot reads `error`.

The thinking-dots placeholder lives inside the fastpath until the first pill or narration arrives — at which point it is removed (so the 2 s `upgradePendingText` timer becomes a no-op via its `if (!dots) return` guard). If the timer fires first, `upgradePendingText` swaps the dots for an "On it…" label using `replaceChild` so any sub-bubbles or pills already nested inside the fastpath survive the upgrade.

Sub-100 ms tools enforce a 150 ms minimum visible duration so the spinner-to-state transition is always perceptible. Pills are display-only — no click-to-expand, no tooltips.

**CSS classes:** `tool-pill-row` (vertical flex column hosting the pill stack), `tool-pill`, `tool-pill__name`, `tool-pill__status`, `tool-pill__spinner`, `tool-pill--active`, `tool-pill--done`, `tool-pill--error`. The gradient is rendered via `tool-pill::after` so the row itself stays unbordered. Nested narration sub-bubbles use the `narration-bubble` class with a thin violet left edge marking each iteration.

**Frontend wiring:** `renderer.js` exposes `appendNarrationBubble(parentEl, text, step)`, `appendToolPill(parentEl, callId, name)`, and `resolveToolPill(callId, ms, ok)`. `chat.js` keeps `pendingForm` (the fastpath) connected for the entire turn — `onNarration` nests bubbles inside it via `appendNarrationBubble(pendingForm, …)`, `onToolStart` routes to `_lastNarrationBubble` when set or to the fastpath otherwise, and `onDone` calls `resolvePendingForm` to wipe + replace the inner content with the final blocks.

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

### Onboarding (`/on-boarding/`)

Account creation only — username and password. If an account already exists, the page bounces to login. After creating an account it redirects to Brain so the user can add a provider.

### Login (`/login/`)

A dedicated form so browser autofill and OS credential managers (macOS Keychain, etc.) can offer stored credentials. Accepts a `?next=` parameter to resume the original destination after sign-in. Bounces to chat when a session is already active.

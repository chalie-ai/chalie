# Web Interface

The frontend is four independent single-page apps under `frontend/` — plain HTML/CSS/JS served directly by Flask, no build step, no bundler. They share the **Radiant** design language: a near-black canvas, drifting atmospheric orbs, and three accent colors (violet for primary actions, magenta for presence, cyan for processing).

| App | URL | Purpose |
|---|---|---|
| **Chat** | `/` | The conversation: messages, live tool activity, rich cards, voice, attachments |
| **Brain** | `/brain/` | The dashboard: providers, memory, cognition, policies, skills, MCP, scheduler, documents |
| **Onboarding** | `/on-boarding/` | One-time account creation |
| **Login** | `/login/` | Session login (supports `?next=` redirect) |

## Auth & Provider Gate

Every page calls `/auth/status` before booting and redirects as needed: no account → onboarding, no session → login, no providers → Brain (locked to the Providers tab until the first provider is saved).

## The Chat Surface

**ACT cycle.** While a turn runs, the UI shows a chrome-less working state — a pulsing violet logo, a one-line narration that updates in place, and a cumulative list of tool calls (name, optional `act_summary`, then duration or error). When the turn finishes, the whole ACT UI is replaced by the final reply bubble. The data comes from WebSocket events: `act_tool_start`, `act_tool_end`, `act_narration`, then `message` + `done`.

**Rich-media cards.** Tools can return a structured payload that renders as an inline card instead of prose. The registry (`frontend/interface/rich_media/registry.js`) maps tag prefixes to card modules:

| Card | Source tools |
|---|---|
| `weather` | weather — ambient sky scene with hourly rail |
| `article` | search + news — shared article layout with thumbnails |
| `schedule` | schedule — date block + same-day list |
| `list` | list — checklist with click-to-toggle persistence |
| `timer` | timer — live countdown with alarm (purely client-side) |
| `calendar` | calendar — event card |
| `contacts` | contacts — contact card |

Interactive cards (e.g. checking a list item) post silent actions back to the server without rendering a chat bubble — the card owns its own feedback.

**Attachments.** Up to 3 images per message via file picker, drag-and-drop, or paste. Thumbnails show an analyzing state while the backend runs vision/OCR; sending is never blocked on analysis. Non-image files route to document upload.

**Voice.** Optional and off by default (see [01-QUICK-START.md](01-QUICK-START.md)). When enabled: the mic button transcribes speech into the prompt box (Moonshine STT), and a speaker icon under each reply opens an overlay player (Kokoro TTS, single WAV per message). If voice dependencies aren't installed, all voice controls hide automatically.

## The Brain Dashboard

| Tab | What it does |
|---|---|
| **Providers** | Add/test/select LLM providers; also contains the Vision and Delegate provider selectors under the LLM Providers section |
| **Cognition** | Subtabs: Memory, Tools, World state, Personality, Errors, Usage, Compacted Summary |
| **Scheduler** | Reminders and scheduled tasks (All / Pending / Fired / Failed / Cancelled) |
| **Lists** | The user's persistent checklists |
| **Documents** | Ingested documents and watched folders (Active / Processing / Uploads / Deleted) |
| **Capabilities** | Connect external services (mail, Home Assistant, UniFi) |
| **Policies** | Per-action permission control — allow / ask / deny across Chat, Background, and External Agent contexts, plus a blocked-actions log |
| **Skills** | Curated and user-created playbooks, skill associations |
| **MCP** | External MCP tool-server connections |

The **Personality** sliders (warmth, mood, expressiveness, curiosity, humor; −2 to +2) map to a voice paragraph prepended to the system prompt for user-facing conversations only — background processing is unaffected.

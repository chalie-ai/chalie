# Web Interface

The frontend is a **pnpm workspace** under `frontend/` built with Vue 3, Vite 5, TypeScript (strict), Pinia, Vue Router, and SCSS. Flask serves the compiled builds verbatim — no runtime asset injection.

## Workspace layout

```
frontend/
  packages/
    shared/           (@chalie/shared) — ApiClient, WebSocketService, PlatformAdapter,
                       SCSS theme tokens, base UI components (BaseButton, BaseCard, …)
  apps/
    interface/        (@chalie/interface) — chat SPA + login/on-boarding multi-page entries
    brain/            (@chalie/brain) — admin SPA, asset base /brain/
```

Build: `pnpm -r build` from `frontend/` — emits `apps/interface/dist` and `apps/brain/dist`.

## URL map

| URL | Build | Purpose |
|---|---|---|
| `/` | `apps/interface/dist` | Chat SPA |
| `/brain/` | `apps/brain/dist` | Admin SPA (auth-gated at the serve layer) |
| `/login/` | `apps/interface/dist/login/index.html` | Session login (supports `?next=` redirect) |
| `/on-boarding/` | `apps/interface/dist/on-boarding/index.html` | One-time account creation |

Flask routes are registered in `backend/api/__init__.py` (`_register_static_routes`). Assets carry Vite content hashes; index files are served `no-cache` so browsers always fetch the latest entry point. There is no server-side version injection.

## Auth flow

Auth is entirely client-side — the server guards its API endpoints independently.

- **Shared `ApiClient`** — on any HTTP 401 from an authenticated call, the client redirects to `/login/?next=<current-path>` (idempotent per client instance) and throws `AuthError`. This is the single place mid-session expiry is handled for both apps, replacing the Brain app's former per-call `withAuth()` wrapper. A few callers opt out with `{ redirectOnAuthError: false }` because their 401 is *data*, not expiry: the `/auth/status` gate probe (the router inspects the result to route), the public probes (`/ready`, `POST /health`, `/voice/health`), and the login / register / voice-settings onboarding calls (401 = bad credentials or no session yet).
- **Chat router** — `frontend/apps/interface/src/router.ts` runs a `beforeEach` guard on navigation: no account → redirect `/on-boarding/`, no session → redirect `/login/?next=…`, no providers → redirect `/brain/`.
- **Brain router** — `frontend/apps/brain/src/router.ts` runs the same guard; no providers locks the router to the Providers panel (no hard redirect, so the user can save a provider without leaving the app).
- **Login / on-boarding entries** — `src/login/main.ts` and `src/onboarding/main.ts` each run a pre-mount `/auth/status` check and redirect away before the Vue app mounts if the condition is already satisfied.
- **Serve layer** — `/brain/` and `/brain/<path>` additionally call `validate_session` at request time and redirect to `/login/` if the session is absent.

Network failures in any gate → stay and mount; the API endpoints are still protected.

## The Chat Surface

**ACT cycle.** While a turn runs, the UI shows a chrome-less working state — a pulsing violet logo, a one-line narration that updates in place, and a cumulative list of tool calls (name, optional `act_summary`, then duration or error). When the turn finishes, the whole ACT UI is replaced by the final reply bubble. The data comes from WebSocket events: `act_tool_start`, `act_tool_end`, `act_narration`, then `message` + `done`.

**Rich-media cards.** Tools can return a structured payload that renders as an inline card instead of prose. The registry (`frontend/apps/interface/src/components/rich/richRegistry.ts`) maps tag prefixes to card components. Eight prefixes resolve to seven components — `news` and `search` share `ArticleCard.vue`:

| Card | Tag prefix | Component | Renders |
|---|---|---|---|
| `weather` | `weather` | `WeatherCard.vue` | Ambient sky scene with an hourly rail |
| `article` | `news`, `search` | `ArticleCard.vue` | Article layout with optional image thumbnail |
| `schedule` | `schedule` | `SchedulerCard.vue` | Date block + same-day list |
| `list` | `list` | `ListCard.vue` | Checklist with click-to-toggle persistence |
| `timer` | `timer` | `TimerCard.vue` | Live countdown with a client-side alarm |
| `calendar` | `calendar` | `CalendarCard.vue` | Event card |
| `contacts` | `contacts` | `ContactsCard.vue` | Contact card |

Cards are loaded via `defineAsyncComponent` so each card and its scoped styles code-split into their own chunk. Interactive cards (e.g. checking a list item) post silent actions back to the server without rendering a chat bubble — the card owns its own feedback.

**Attachments.** Up to 3 images per message via file picker, drag-and-drop, or paste. Thumbnails show an analyzing state while the backend runs vision/OCR; sending is never blocked on analysis. Non-image files route to document upload.

**Voice.** Optional and off by default (see [01-QUICK-START.md](01-QUICK-START.md)). When enabled: the mic button transcribes speech into the prompt box (Moonshine STT), and a speaker icon under each reply opens an overlay player (Kokoro TTS, single WAV per message). If voice dependencies aren't installed, all voice controls hide automatically.

## The Brain Dashboard

`frontend/apps/brain/src/router.ts` defines the tab/route set:

| Tab | Route | What it does |
|---|---|---|
| **Providers** | `/providers` | Add/test/select LLM providers |
| **Vision** | `/vision` | Vision-provider configuration |
| **Cognition** | `/cognition` | Subtabs: Memory, Tools, World state, Personality, Errors, Usage, Compacted Summary |
| **Scheduler** | `/scheduler` | Reminders and scheduled tasks (All / Pending / Fired / Failed / Cancelled) |
| **Lists** | `/lists` | The user's persistent checklists |
| **Documents** | `/documents` | Ingested documents and watched folders (Active / Processing / Uploads / Deleted) |
| **Capabilities** | `/capabilities` | Connect external services (mail, Home Assistant, UniFi) |
| **Policies** | `/policies` | Per-action permission control — allow / ask / deny across Chat, Background, and External Agent contexts, plus a blocked-actions log |
| **Skills** | `/skills` | Curated and user-created playbooks, skill associations |
| **MCP** | `/mcp` | External MCP tool-server connections |

The **Personality** sliders (warmth, mood, expressiveness, curiosity, humor; −2 to +2) map to a voice paragraph prepended to the system prompt for user-facing conversations only — background processing is unaffected.

## Radiant design system

Both apps import the shared SCSS theme from `packages/shared/src/styles/`. The design language: a near-black canvas, drifting atmospheric orbs, and three accent colors — violet for primary actions, magenta for presence, cyan for processing. Dark and light themes are supported via CSS variables declared in the shared tokens; no hardcoded color values are used in component styles.

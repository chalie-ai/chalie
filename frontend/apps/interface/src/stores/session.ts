/**
 * Session store — WS coordinator, turn state-machine, drift router.
 *
 * Single WS owner rule: ONLY this store may touch WebSocketService handlers
 * (send/sendAction/abort/onDrift/onAny/onConnect/onDisconnect/connect/ensureAlive).
 * Everything else goes through this store or the event bus.
 */
import { defineStore } from 'pinia';
import { getWebSocket, useConnectionStore, AuthError } from '@chalie/shared';
import type { WsInboundEvent, WsPushEvent, WsMessageEvent } from '@chalie/shared';
import { on } from '../composables/useEventBus';
import { extractText } from '../composables/useMarkup';
import { conversation } from '../api/conversation';
import { getHost } from '../api/index';
import { showToast } from '../utils/toast';
import { useConversationStore, chalieFormPlaintext } from './conversation';
import type { AttachmentPreview, ChalieForm } from './conversation';
import { useQueueStore } from './queue';
import { useTasksStore } from './tasks';
import { useNotificationsStore } from './notifications';
import type { TipState, UpdateState } from './notifications';
import { usePermissionsStore } from './permissions';
import { useContextUsageStore } from './contextUsage';
import { useAmbientSensor } from '../composables/useAmbientSensor';

/** Guard: init() must be idempotent (HMR / Vue StrictMode). */
let _initialized = false;

/** Unbind fns for event-bus listeners registered in init() (for future cleanup). */
const _busUnbinds: Array<() => void> = [];

export const useSessionStore = defineStore('session', {
  state: () => ({
    isSending: false,
    /** turn_id of the turn THIS surface is sending — claimed on its `working`
     *  signal, cleared on its `done`. Disambiguates own vs peer turns so a peer's
     *  broadcast `done` can't release this surface's send guard. */
    _liveTurnId: null as number | null,
    /** Id of the optimistic user form (for the requestStop restore). */
    _lastUserFormId: null as number | null,
    /** Captured text from the last user turn (for requestStop restore). */
    _lastUserText: '',

    /** Turn-level provider/quota error, surfaced as a closable toast above the
     *  input dock. Null when there is nothing to show. */
    errorMessage: null as string | null,

    historyLoading: false,
    /** True while a deep-link thread fetch is in-flight (drives the panel
     *  spinner on first open of a thread outside the loaded pages). */
    threadExpanding: false,

    /** turn_id of the thread shown in the slide-over panel, or null when closed.
     *  The opener button opens the panel; the main feed dims behind it. */
    panelThreadId: null as number | null,

    /** True while the thread-search overlay is open (Cmd/Ctrl-K or the top-bar
     *  search button). The overlay self-fetches; this is pure open/close state. */
    searchOpen: false,

    /** Registered auth-failure callback (set by App bootstrap). */
    _onAuthFailure: null as (() => void) | null,
  }),

  actions: {
    /** Wire the WebSocket singleton and connect. Idempotent (HMR / StrictMode). */
    init(): void {
      if (_initialized) return;
      _initialized = true;

      const ws = getWebSocket();
      const conn = useConnectionStore();
      const contextUsage = useContextUsageStore();

      ws.onConnect(() => {
        conn.setConnected(true);
      });

      ws.onDisconnect(() => {
        conn.setConnected(false);
        // A mid-turn drop strands the spinner: the turn's `done` now lands on the
        // dead socket and is never resent, so clear this surface's working turn or
        // the anchor hangs. The rows persisted server-side reload on reconnect.
        if (this._liveTurnId != null) useConversationStore().setWorking(this._liveTurnId, false);
        this._liveTurnId = null;
        this._lastUserFormId = null;
        this.isSending = false;
      });

      ws.onDrift((data: WsPushEvent) => {
        this.routeDrift(data);
      });

      // onAny feeds the context-usage indicator — must never throw. refresh()
      // is coalesced inside the store, so a per-frame call is safe.
      ws.onAny((data: WsInboundEvent) => {
        try {
          void data;
          void contextUsage.refresh();
        } catch {
          /* never break the WS pipe */
        }
      });

      // Tab-refocus liveness check.
      globalThis.addEventListener('focus', () => ws.ensureAlive());

      // chalie:action — deterministic skill invocations. Registered inside init()
      // so the WS single-owner rule holds (only one listener ever bound).
      _busUnbinds.push(
        on('chalie:action', (payload) => {
          // Bus emits the full detail; legacy read e.detail.payload — support both.
          const p =
            (payload as { payload?: Record<string, unknown> }).payload ??
            (payload as Record<string, unknown>);
          void this.sendAction(p);
        }),
      );

      // chalie:silent-action — rich-card interactions, no chat bubble. Caller
      // supplies optional onMessage/onError/onDone for optimistic card UI.
      _busUnbinds.push(
        on('chalie:silent-action', (detail) => {
          const d = detail as {
            payload?: Record<string, unknown>;
            onMessage?: (data: WsMessageEvent) => void;
            onError?: (data: { message: string; recoverable?: boolean }) => void;
            onDone?: (data: { duration_ms?: number }) => void;
          };
          if (!d.payload) return;
          getWebSocket().sendAction(d.payload, {
            onMessage: d.onMessage ?? (() => { /* no-op */ }),
            onError: d.onError ?? (() => { /* no-op */ }),
            onDone: d.onDone ?? (() => { /* no-op */ }),
          });
        }),
      );

      ws.connect();
    },

    /** Register an auth-failure callback (called by App bootstrap). */
    onAuthFailure(cb: () => void): void {
      this._onAuthFailure = cb;
    },

    /**
     * Send a user turn. Signal-only: the optimistic user bubble renders instantly;
     * everything else — the spinner, the rows, the reply — flows back through the
     * broadcast `working`/`updated`/`done` signals (→ routeDrift → refetch).
     *
     * Busy path: if this scope is working (the surface is single-flight, or the
     * target thread's turn is still streaming in the background), the text is
     * queued client-side instead of interrupting. It renders faded at the scope's
     * tail and dispatches — as ONE newline-joined message — when the scope settles
     * (_drainQueues). Queues are per-scope: the spine drains into a new spine turn,
     * a thread into a reply on that turn.
     */
    async sendMessage(
      text: string,
      source: 'text' | 'voice' = 'text',
      files: File[] = [],
      previews: AttachmentPreview[] = [],
      threadId: number | null = null,
    ): Promise<void> {
      if (!text && !files.length) return;

      const convo = useConversationStore();
      const ws = getWebSocket();
      const body = text || '[File attached]';

      if (this.isSending || (threadId != null && convo.isTurnWorking(threadId))) {
        useQueueStore().enqueue(threadId, body);
        return;
      }

      this.isSending = true;
      this._liveTurnId = threadId; // null for a new thread — claimed on `working`
      this._lastUserText = text;
      this._lastUserFormId = convo.appendUser(body, previews, {
        inWorkingMemory: true,
        turnId: threadId,
      });
      ws.send(body, source, (m) => this._onSendFailure(m), files, threadId);
    },

    /** A local send failure (offline / POST rejected) — no signal will ever
     *  arrive, so release this surface's guard and surface the message. */
    _onSendFailure(message: string): void {
      this.errorMessage = message;
      if (this._liveTurnId != null) useConversationStore().setWorking(this._liveTurnId, false);
      this._liveTurnId = null;
      this._lastUserFormId = null;
      this.isSending = false;
    },

    /** `done(turn_id)` — settle the turn. Clear its spinner; release this
     *  surface's send guard only when its OWN turn finished (peers' `done`
     *  broadcasts must not). Fire ambient + autoscroll, and an OS notification
     *  for the final reply when the tab is unfocused. */
    async _finishTurn(turnId: number | null): Promise<void> {
      const convo = useConversationStore();
      const id = turnId ?? this._liveTurnId;
      if (id != null) {
        // A forked thread that settled while the user is looking elsewhere keeps a
        // standing Activity card (blue `done`); one they're viewing, or a plain
        // spine turn watched inline, just settles.
        if (id !== this.panelThreadId && convo.isForkedThread(id)) convo.markThreadDone(id);
        else convo.setWorking(id, false);
      }
      if (id != null && id === this._liveTurnId) {
        this._liveTurnId = null;
        this._lastUserFormId = null;
        this.isSending = false;
      }
      this._drainQueues();
      useAmbientSensor().recordResponse();
      document.dispatchEvent(new CustomEvent('session:turn-done'));

      if (id != null && !document.hasFocus()) {
        await convo.refetchTurn(id);
        const chalie = convo.forms.filter(
          (f): f is ChalieForm => f.kind === 'chalie' && f.turnId === id,
        );
        const last = chalie[chalie.length - 1];
        if (last) this._notifyBackground(chalieFormPlaintext(last));
      }
    },

    /** Dispatch one settled scope's queued messages. The surface is single-flight,
     *  so drain at most one scope per call — the next scope follows on its own
     *  `done`. A thread scope waits until its turn stops working; the spine drains
     *  whenever the surface is free. Each scope ships as ONE newline-joined turn. */
    _drainQueues(): void {
      if (this.isSending) return;
      const queue = useQueueStore();
      const convo = useConversationStore();
      for (const key of queue.pendingScopes) {
        const threadId = key === 'main' ? null : Number(key.slice(1));
        if (threadId != null && convo.isTurnWorking(threadId)) continue;
        void this.sendMessage(queue.take(threadId), 'text', [], [], threadId);
        return;
      }
    },

    /** Turn-level error (provider failure, quota/429) — surface it as the one dock
     *  toast. Not an act row: the following `done` clears the spinner. Only
     *  `auth_failed` redirects to login. */
    _handleTurnError(data: WsPushEvent): void {
      this.errorMessage = (data as { message?: string }).message ?? 'Something went wrong.';
      if ((data as { auth_failed?: boolean }).auth_failed) this._onAuthFailure?.();
    },

    /**
     * Stop + undo the whole in-flight turn (user message + everything the chain
     * rendered after it). Emits 'session:turn-interrupted' so InputDock can
     * restore the textarea value.
     */
    async requestStop(): Promise<void> {
      const convo = useConversationStore();
      const ws = getWebSocket();

      // Restore the in-flight user bubble's text into the composer; _lastUserText
      // is only the fallback for when that optimistic form is already gone.
      let restoredText = this._lastUserText;
      if (this._lastUserFormId != null) {
        const u = convo.forms.find((f) => f.id === this._lastUserFormId);
        if (u?.kind === 'user') restoredText = u.text;
      }

      // Undo the whole turn: its forms + thread shell + working state. dropLiveTurn
      // clears a bound turn; the explicit filter catches a new thread that never
      // bound (no `working` signal yet), whose bubble is still turn_id-less.
      if (this._liveTurnId != null) convo.dropLiveTurn(this._liveTurnId);
      if (this._lastUserFormId != null) {
        convo.forms = convo.forms.filter((f) => f.id !== this._lastUserFormId);
      }
      this._liveTurnId = null;
      this._lastUserFormId = null;
      this._lastUserText = '';

      ws.abort();
      this.isSending = false;

      document.dispatchEvent(
        new CustomEvent('session:turn-interrupted', { detail: { text: restoredText } }),
      );

      await this._postInterrupt();
    },

    /** POST /chat/interrupt — best-effort, never throws. */
    async _postInterrupt(): Promise<void> {
      try {
        const host = getHost();
        const base = host ? host.replace(/\/$/, '') : '';
        await fetch(base + '/chat/interrupt', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
      } catch {
        // Best-effort — swallow.
      }
    },

    /**
     * Create an ACT cycle, block concurrent sends, and call ws.sendAction with
     * callbacks that resolve the ACT into a Chalie form or error.
     */
    async sendAction(payload: Record<string, unknown>): Promise<void> {
      if (this.isSending) return;

      const convo = useConversationStore();
      const ws = getWebSocket();

      this.isSending = true;
      const actId = convo.appendAct();

      ws.sendAction(payload, {
        onMessage: (data) => {
          const d = data as WsMessageEvent & {
            content?: string;
            mode?: string;
            confidence?: number;
          };
          convo.replaceActWithResponse(actId, {
            content: d.content ?? '',
            mode: d.mode ?? 'ACT',
            confidence: d.confidence ?? 0.95,
          });
        },
        onError: (data) => {
          convo.resolveAct(actId);
          this.errorMessage = data.message;
        },
        onDone: () => {
          this.isSending = false;
        },
      });
    },

    /**
     * Load (or paginate) the thread list. Initial load fetches the 20 most-recent
     * collapsed threads; scroll-up pagination prepends 20 more. Expanded threads
     * keep their forms in the conversation store and are excluded from the
     * collapsed rows.
     */
    async loadRecentConversation(): Promise<void> {
      const convo = useConversationStore();
      if (convo.threadsExhausted || this.historyLoading) return;
      this.historyLoading = true;

      const LIMIT = 20;
      const isInitialLoad = convo.threadsOffset === 0;

      try {
        const data = await conversation.threads(LIMIT, convo.threadsOffset > 0 ? convo.threadsOffset : undefined);
        const items = data.threads ?? [];

        if (items.length === 0 && convo.threadsOffset === 0) {
          convo.threadsExhausted = true;
          return;
        }

        if (isInitialLoad) convo.appendThreadList(items);
        else convo.prependThreadList(items);

        // Hydrate the page in one round-trip — gather the page's ids from the
        // minimal-metadata list, batch-fetch the full blocks, and loop
        // upsertTurn. The thread items already render as pills from the metadata;
        // the blocks fill in the forms so expand (and the panel) are instant.
        const pageIds = items
          .map((t) => t.turn_id)
          .filter((id): id is number => id != null);
        if (pageIds.length) {
          const { blocks } = await conversation.batch(pageIds);
          for (const block of blocks) convo.upsertTurn(block);
        }

        convo.threadsOffset += data.threads_returned;

        if (!data.has_more) {
          convo.threadsExhausted = true;
        }

        if (isInitialLoad && items.length > 0) {
          document.dispatchEvent(new CustomEvent('session:history-initial-loaded'));
        }
      } catch (err) {
        if (err instanceof AuthError) {
          this._onAuthFailure?.();
        } else {
          console.error('[Session] Failed to load thread list:', err);
        }
      } finally {
        this.historyLoading = false;
      }
    },

    /** Open a thread in the slide-over panel, loading its rows on first open.
     *  The panel id is set first so it slides in immediately and shows its own
     *  loader; an already-hydrated thread (batch-loaded) opens instantly, a
     *  deep-link one fetches on demand. */
    async openThreadPanel(turnId: number): Promise<void> {
      this.panelThreadId = turnId;
      const convo = useConversationStore();
      convo.seenThread(turnId);
      if (convo.isHydrated(turnId)) return;
      this.threadExpanding = true;
      try {
        await convo.refetchTurn(turnId);
      } finally {
        this.threadExpanding = false;
      }
    },

    /** Close the slide-over panel. Forms stay loaded so the thread can fall
     *  back to its inline/pill render in the main feed. */
    closeThreadPanel(): void {
      this.panelThreadId = null;
    },

    /** Open / close the thread-search overlay. */
    openSearch(): void {
      this.searchOpen = true;
    },
    closeSearch(): void {
      this.searchOpen = false;
    },

    /**
     * Route a drift push event. Turn signals and content-free events route first;
     * the only push that still carries content is the scheduler `notification`
     * (reminder/task-done) — it background-notifies, chimes unconditionally and
     * refreshes the task strip. Every other frame is a signal handled above.
     */
    routeDrift(data: WsPushEvent): void {
      if (this._routeTurnSignal(data)) return;
      if (this._routeSimpleEvent(data)) return;

      if (data.type !== 'notification') return;
      const content = (data as { content?: string }).content ?? '';
      if (content) this._notifyBackground(content);
      useNotificationsStore().chime();
      void useTasksStore().loadActiveTasks();
    },

    /**
     * Turn-lifecycle signals — stateless and turn-addressed. Each is broadcast to
     * EVERY surface; only the surface that owns the turn (`_liveTurnId`) releases
     * its send guard on `done`. Returns true when handled.
     *   created(id)        → carries a NEW thread's allocated id: claim it onto the
     *                        optimistic live turn (unbound until now), bind, spinner on
     *   working(id)        → an EXISTING turn (reply, id already known) (re)activates:
     *                        bind (idempotent) + spinner on
     *   tool_called(id…)   → push a live act-trail pill (name + summary) onto its
     *                        `transcript_row_id`'s trail, start its timer (NO fetch)
     *   tool_done(id)      → stop that pill's timer, matched by id (NO fetch)
     *   updated(id)        → fetch the block + atomic monotonic replace; that
     *                        refetch (in upsertTurn) drops the now-persisted rows'
     *                        live pills — the §6.5 data-driven handoff, no manual clear
     *   done(id)           → spinner off + settle
     *   error              → one dock toast (a `done` follows to clear the spinner)
     */
    _routeTurnSignal(data: WsPushEvent): boolean {
      const convo = useConversationStore();
      const turnId = (data as { turn_id?: number | null }).turn_id ?? null;
      switch (data.type as string) {
        case 'created':
        case 'working':
          if (turnId == null) return true;
          // A new thread's optimistic turn is still unbound — claim its allocated
          // id (created). A reply already carries _liveTurnId from sendMessage.
          if (this.isSending && this._liveTurnId == null) this._liveTurnId = turnId;
          convo.bindLiveTurn(turnId);
          convo.setWorking(turnId, true);
          // Pull the block so EVERY surface renders the user bubble from the API —
          // the deleted user-echo's job, now signal-driven. The sender's optimistic
          // forms reconcile via upsertTurn's monotonic, atomic re-splice.
          void convo.refetchTurn(turnId);
          return true;
        case 'tool_called':
          if (turnId != null) {
            const tc = data as { id?: number; name?: string; summary?: string; transcript_row_id?: number };
            convo.startLiveTool(turnId, tc.transcript_row_id ?? null, tc.id ?? null, tc.name ?? '', tc.summary);
          }
          return true;
        case 'tool_done':
          convo.finishLiveTool((data as { id?: number }).id ?? null);
          return true;
        case 'updated':
          // The refetch lands the just-persisted rows; upsertTurn drops their live
          // pills (§6.5 step-4 handoff) — no manual clear, no double-render.
          if (turnId != null) void convo.refetchTurn(turnId);
          return true;
        case 'done':
          void this._finishTurn(turnId);
          return true;
        case 'error':
          this._handleTurnError(data);
          return true;
        default:
          return false;
      }
    },

    /** Route content-free event types; returns true when handled. */
    _routeSimpleEvent(data: WsPushEvent): boolean {
      switch (data.type as string) {
        case 'app_update':
          useNotificationsStore().handleUpdate(data as unknown as UpdateState);
          return true;
        // task + subagent lifecycle both feed the task drawer.
        case 'task':
        case 'subagent_start':
        case 'subagent_end':
          useTasksStore().applyDriftEvent(data);
          return true;
        case 'capability_alert':
          // No-op: dormant channel (no UI consumer).
          return true;
        case 'permission_request':
          usePermissionsStore().enqueue(data);
          return true;
        case 'quick_tip':
          useNotificationsStore().handleTip(data as unknown as TipState);
          return true;
        case 'provider_retry':
          showToast((data as { message?: string }).message ?? 'The AI provider had a problem — retrying…');
          return true;
        default:
          return false;
      }
    },

    /** Fire a background notification when the tab is not focused. */
    _notifyBackground(content: string): void {
      if (document.hasFocus()) return;
      // Plain text for the OS preview — drops <actions> labels, substitutes <img alt>.
      const plain = extractText(content);
      if (plain) useNotificationsStore().pushBackground(plain);
    },
  },
});

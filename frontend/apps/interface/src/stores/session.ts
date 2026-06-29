/**
 * Session store — WS coordinator, turn state-machine, drift router.
 *
 * Single WS owner rule: ONLY this store may touch WebSocketService handlers
 * (send/sendAction/abort/onDrift/onAny/onConnect/onDisconnect/connect/ensureAlive).
 * Everything else goes through this store or the event bus.
 *
 * Lane model: every independent conversation surface (the main spine + each
 * open thread reply) is a "lane" keyed by laneKey(threadId). The four former
 * global single-flight fields (isSending, _liveTurnId, _lastUserFormId,
 * _lastUserText) are now per-lane so a working thread never blocks the spine
 * and vice-versa.
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
import { useQueueStore, laneKey } from './queue';
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

const FILE_PLACEHOLDER = '[File attached]';

interface LaneState {
  /** Bound turn_id; null for a new main-spine turn until its `created` claims one. */
  liveTurnId: number | null;
  /** Id of the optimistic user form (for the requestStop restore). */
  userFormId: number | null;
  /** Captured text from the last user turn (for requestStop restore). */
  userText: string;
}

export const useSessionStore = defineStore('session', {
  state: () => ({
    /** Per-lane single-flight state. Key = laneKey(threadId). A key's presence
     *  means that lane is actively sending. */
    lanes: {} as Record<string, LaneState>,
    /** True while a sendAction (ACT cycle) is in flight — not a lane turn. */
    _actionBusy: false,

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

  getters: {
    /**
     * True when the MAIN spine is busy (lane present) or an ACT cycle is running.
     * Consumed by PresenceBar (logo pulse) and ConversationFeed (live-turn spinner)
     * — both mean "the main spine is doing something". Unchanged public shape.
     */
    isSending: (state): boolean => 'main' in state.lanes || state._actionBusy,
  },

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
        // A mid-turn drop strands spinners: `done` lands on the dead socket and is
        // never resent. Clear ALL lanes' working state so nothing hangs.
        const convo = useConversationStore();
        for (const k in this.lanes) {
          const id = this.lanes[k].liveTurnId;
          if (id != null) convo.setWorking(id, false);
        }
        this.lanes = {};
        this._actionBusy = false;
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
     * True when a given surface is currently busy — gates sends and drains.
     * Main spine shares its surface with ACT cycles (no stable turn id);
     * threads have a stable id plus the conversation working-set.
     */
    isLaneBusy(threadId: number | null): boolean {
      return threadId == null
        ? this.isSending
        : laneKey(threadId) in this.lanes || useConversationStore().isTurnWorking(threadId);
    },

    /**
     * Send a user turn. Signal-only: the optimistic user bubble renders instantly;
     * everything else — the spinner, the rows, the reply — flows back through the
     * broadcast `working`/`updated`/`done` signals (→ routeDrift → refetch).
     *
     * Busy path: if this lane is working, the text is queued client-side instead
     * of interrupting. It renders faded at the scope's tail and dispatches — as ONE
     * newline-joined message — when the lane settles (_drainQueues). Each lane
     * drains independently; a busy thread never blocks the spine and vice-versa.
     */
    async sendMessage(
      text: string,
      source: 'text' | 'voice' = 'text',
      files: File[] = [],
      previews: AttachmentPreview[] = [],
      threadId: number | null = null,
    ): Promise<void> {
      if (!text && !files.length) return;

      const body = text || FILE_PLACEHOLDER;

      if (this.isLaneBusy(threadId)) {
        useQueueStore().enqueue(threadId, body);
        return;
      }

      const convo = useConversationStore();
      const key = laneKey(threadId);
      this.lanes[key] = {
        liveTurnId: threadId,   // null for a new spine turn — claimed on `created`
        userFormId: convo.appendUser(body, previews, {
          inWorkingMemory: true,
          turnId: threadId,
        }),
        userText: text,
      };
      getWebSocket().send(body, source, (m) => this._onSendFailure(key, m), files, threadId);
    },

    /** A local send failure (offline / POST rejected) — no signal will ever
     *  arrive, so release this lane's guard and surface the message. */
    _onSendFailure(key: string, message: string): void {
      this.errorMessage = message;
      const lane = this.lanes[key];
      if (lane?.liveTurnId != null) useConversationStore().setWorking(lane.liveTurnId, false);
      delete this.lanes[key];
    },

    /** `done(turn_id)` — settle the turn. Clear its spinner; release the owning
     *  lane only when ITS own turn finished (peers' `done` broadcasts must not).
     *  Fire ambient + autoscroll, and an OS notification for the final reply when
     *  the tab is unfocused. */
    async _finishTurn(turnId: number | null): Promise<void> {
      const convo = useConversationStore();
      const owningKey = this._laneOwning(turnId);
      // Resolve id: a null turnId means it was the main spine's unbound turn; look
      // it up from the lane record. For a known thread id, turnId is already it.
      const id = turnId ?? (owningKey != null ? this.lanes[owningKey]?.liveTurnId ?? null : null);

      if (id != null) {
        // A forked thread that settled while the user is looking elsewhere keeps a
        // standing Activity card (blue `done`); one they're viewing, or a plain
        // spine turn watched inline, just settles.
        if (id !== this.panelThreadId && convo.isForkedThread(id)) convo.markThreadDone(id);
        else convo.setWorking(id, false);
      }

      // Release only the lane that OWNS this turn — peer `done` signals leave
      // other lanes' guards intact.
      if (owningKey != null) delete this.lanes[owningKey];

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

    /**
     * Find the lane key that owns the given turnId — used to release ONLY the
     * correct lane on `done` without disturbing peer lanes.
     * Main spine's lane has liveTurnId=null initially, then claimed on `created`;
     * thread lanes carry their id from the moment of send.
     */
    _laneOwning(turnId: number | null): string | null {
      if ('main' in this.lanes && (turnId == null || this.lanes['main'].liveTurnId === turnId)) {
        return 'main';
      }
      for (const k in this.lanes) {
        if (k !== 'main' && this.lanes[k].liveTurnId === turnId) return k;
      }
      return null;
    },

    /** Drain ALL pending scopes independently — each lane checks its own busy
     *  state, so a working thread never blocks a free spine drain and vice-versa. */
    _drainQueues(): void {
      for (const key of useQueueStore().pendingScopes) this._drainLane(key);
    },

    _drainLane(key: string): void {
      const threadId = key === 'main' ? null : Number(key.slice(1));
      if (this.isLaneBusy(threadId)) return;
      const text = useQueueStore().take(threadId);
      if (text) void this.sendMessage(text, 'text', [], [], threadId);
    },

    /** Turn-level error (provider failure, quota/429) — surface it as the one dock
     *  toast. Not an act row: the following `done` clears the spinner. Only
     *  `auth_failed` redirects to login. */
    _handleTurnError(data: WsPushEvent): void {
      this.errorMessage = (data as { message?: string }).message ?? 'Something went wrong.';
      if ((data as { auth_failed?: boolean }).auth_failed) this._onAuthFailure?.();
    },

    /**
     * Stop + undo the in-flight turn identified by `turnId` (user message +
     * everything the chain rendered after it). The act-trail knows only the turn
     * id, so its lane — the main spine vs a specific thread — is resolved here:
     * the local lane is authoritative for a turn THIS client started (correct
     * even before a thread's first reply rows land); for a turn we are only
     * viewing, the turn's thread-nature still targets the right backend lane.
     * Emits 'session:turn-interrupted' so InputDock can restore the textarea.
     */
    async requestStop(turnId: number | null = null): Promise<void> {
      const convo = useConversationStore();
      const ws = getWebSocket();

      const key = this._laneOwning(turnId);
      const lane = key != null ? this.lanes[key] : undefined;
      // The turn to cancel by id: the lane's bound id for a turn THIS client
      // started, else the id we're viewing. null only for a brand-new spine turn
      // not yet bound — no backend turn_id to target; the local undo below stands.
      const stopId = lane?.liveTurnId ?? turnId;

      // Restore the in-flight user bubble's text into the composer; lane.userText
      // is only the fallback for when that optimistic form is already gone.
      let restoredText = lane?.userText ?? '';
      if (lane?.userFormId != null) {
        const u = convo.forms.find((f) => f.id === lane.userFormId);
        if (u?.kind === 'user') restoredText = u.text;
      }

      // Undo the whole turn: its forms + thread shell + working state. dropLiveTurn
      // clears a bound turn; the explicit filter catches a new thread that never
      // bound (no `working` signal yet), whose bubble is still turn_id-less.
      if (lane?.liveTurnId != null) convo.dropLiveTurn(lane.liveTurnId);
      if (lane?.userFormId != null) {
        convo.forms = convo.forms.filter((f) => f.id !== lane.userFormId);
      }
      if (lane?.liveTurnId != null) convo.setWorking(lane.liveTurnId, false);
      if (key != null) delete this.lanes[key];

      ws.abort();

      document.dispatchEvent(
        new CustomEvent('session:turn-interrupted', { detail: { text: restoredText } }),
      );

      await this._postInterrupt(stopId);
    },

    /** DELETE /api/thread/<turn_id> — best-effort interrupt, never throws. The
     *  turn to cancel is the path; the backend signals that turn's cancel_event.
     *  A still-unbound spine turn (null id) has no backend row yet — the local
     *  undo in requestStop stands and we skip the call. */
    async _postInterrupt(turnId: number | null = null): Promise<void> {
      if (turnId == null) return;
      try {
        const host = getHost();
        const base = host ? host.replace(/\/$/, '') : '';
        await fetch(base + '/api/thread/' + turnId, {
          method: 'DELETE',
          credentials: 'same-origin',
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

      this._actionBusy = true;
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
          this._actionBusy = false;
        },
        onDone: () => {
          this._actionBusy = false;
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
     * EVERY surface; only the surface that owns the turn releases its lane on
     * `done`. Returns true when handled.
     *   created(id)        → carries a NEW thread's allocated id: claim it onto the
     *                        optimistic main lane (unbound until now), bind, spinner on
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
      const eventType = data.type as string;
      switch (eventType) {
        case 'created':
        case 'working':
          if (turnId == null) return true;
          // `created` = a NEW main-spine turn: claim its allocated id for the
          // optimistic main lane (liveTurnId was null until now).
          if (eventType === 'created' && 'main' in this.lanes && this.lanes['main'].liveTurnId == null) {
            this.lanes['main'].liveTurnId = turnId;
          }
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

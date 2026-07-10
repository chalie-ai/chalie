/**
 * Session store — WS coordinator + turn lane state-machine.
 *
 * Single WS owner rule: ONLY this store may touch WebSocketService's
 * send/sendAction/abort/onDrift/onAny/ensureAlive handlers, and it is the
 * only place `connect()` is called for the interface app. `onConnect` /
 * `onDisconnect` are NOT exclusive to this store — `@chalie/shared`'s
 * `useWebSocket()` composable also registers them, but only to drive the
 * connection-status pill (`useConnectionStore`); it never touches `onDrift`
 * or the send/abort surface. Drift routing itself lives in
 * `utils/driftDispatcher.ts` (see `init()` below) — this store no longer
 * inspects WS payload shapes at all.
 *
 * Lane model: every independent conversation surface (the main spine + each
 * open thread reply) is a "lane" keyed by laneKey(threadId). The four former
 * global single-flight fields (isSending, _liveTurnId, _lastUserFormId,
 * _lastUserText) are now per-lane so a working thread never blocks the spine
 * and vice-versa.
 */
import { defineStore } from 'pinia';
import type { WsInboundEvent, WsPushEvent } from '@chalie/shared';
import { AuthError, ConfigType, getWebSocket, useConnectionStore } from '@chalie/shared';
import { on } from '../composables/useEventBus';
import { extractText } from '../composables/useMarkup';
import { getHost } from '../api/index';
import { conversation as convoApi } from '../api/conversation';
import { useConversationFeed } from '../composables/useConversationFeed';
import { useActionCard } from '../composables/useActionCard';
import { dispatchDrift } from '../utils/driftDispatcher';
import { setTurnDone } from '../utils/turnDom';
import { laneKey, useQueueStore } from './queue';
import { useNotificationsStore } from './notifications';
import { useContextUsageStore } from './contextUsage';
import { useAmbientSensor } from '../composables/useAmbientSensor';

/** Guard: init() must be idempotent (HMR / Vue StrictMode). */
let _initialized = false;

/** Unbind fns for event-bus listeners registered in init() (for future cleanup). */
const _busUnbinds: Array<() => void> = [];

const FILE_PLACEHOLDER = '[File attached]';

interface LaneState {
  /** Bound turn_id; null for a new main-spine turn until the POST 200 body claims one. */
  liveTurnId: number | null;
  /** Captured text from the last user turn (for requestStop restore). */
  userText: string;
  /** ConfigType this lane belongs to — used to pick the correct feed on settle. */
  type: string;
}

export const useSessionStore = defineStore('session', {
  state: () => ({
    /** Per-lane single-flight state. Key = laneKey(threadId). A key's presence
     *  means that lane is actively sending. */
    lanes: {} as Record<string, LaneState>,

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

    /** ConfigType of the thread currently open in the panel (default user). */
    panelType: ConfigType.USER as string,

    /** True while the thread-search overlay is open (Cmd/Ctrl-K or the top-bar
     *  search button). The overlay self-fetches; this is pure open/close state. */
    searchOpen: false,

    /** True while the scheduler dock is open. */
    schedulerDockOpen: false,

    /** Registered auth-failure callback (set by App bootstrap). */
    _onAuthFailure: null as (() => void) | null,

    /**
     * Registered history-load callback (set by ConversationFeed.vue —
     * D17: the pagination cursor is UI-local, this store only owns the
     * shared `historyLoading` flag + the AuthError/initial-load event
     * semantics around it). Returns whether this call was the very first
     * load and whether it actually loaded anything, so the initial-load
     * event fires exactly once, only when there was something to show.
     */
    _loadRecentCallback: null as (() => Promise<{ isInitialLoad: boolean; loadedAny: boolean }>) | null,
  }),

  getters: {
    /**
     * True when the MAIN spine is busy (lane present).
     * Consumed by PresenceBar (logo pulse) and ConversationFeed (live-turn spinner).
     */
    isSending: (state): boolean => 'main' in state.lanes,
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
        void this._reconcileLanes();
      });

      ws.onDisconnect(() => {
        conn.setConnected(false);
        // A mid-turn drop strands spinners: the terminal `turn_execution` frame
        // lands on the dead socket and is never resent. Clear ONLY the transient
        // visual working state so nothing hangs — but keep the lane records
        // themselves (draft text + busy identity). `requestStop` (undo/restore,
        // D1) and the send queue guard (D3) both key off lane presence, and the
        // backend may still be mid-turn behind the dead socket. `_reconcileLanes`
        // settles anything that actually finished while offline once the WS
        // reconnects.
        for (const k in this.lanes) {
          const lane = this.lanes[k];
          if (lane.liveTurnId != null) useConversationFeed(lane.type).setWorking(lane.liveTurnId, false);
        }
      });

      ws.onDrift((data: WsPushEvent) => {
        dispatchDrift(data);
      });

      // onAny feeds the context-usage indicator — must never throw.
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

      // chalie:action — deterministic skill invocations routed through useActionCard.
      _busUnbinds.push(
        on('chalie:action', (payload) => {
          const p =
            (payload as { payload?: Record<string, unknown> }).payload ??
            (payload as Record<string, unknown>);
          useActionCard().run(p, (msg) => { this.errorMessage = msg; });
        }),
      );

      // chalie:silent-action — rich-card interactions, no chat bubble.
      _busUnbinds.push(
        on('chalie:silent-action', (detail) => {
          const d = detail as {
            payload?: Record<string, unknown>;
            onMessage?: (data: WsInboundEvent) => void;
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
     * Reconcile every surviving lane against the backend after a WS reconnect.
     * `onDisconnect` clears the transient spinner but deliberately keeps the
     * lane record (see there) since the backend may still be mid-turn; this
     * re-fetches each lane's bound turn and either restores its spinner (still
     * genuinely in-flight — `working` is a server-derived field, see
     * ConversationTurnBlock) or settles it via `_finishTurn` (it actually
     * completed/crashed/cancelled while offline). Lanes with no bound turn_id
     * yet (a main-spine send whose POST hadn't resolved when the drop hit) are
     * left alone — the pending `send()` promise resolves them independently.
     */
    async _reconcileLanes(): Promise<void> {
      for (const key of Object.keys(this.lanes)) {
        const lane = this.lanes[key];
        const turnId = lane.liveTurnId;
        if (turnId == null) continue;
        const convo = useConversationFeed(lane.type);
        try {
          await convo.fetchTurn(turnId);
        } catch {
          continue; // best-effort; leave the lane as-is, retried on the next reconnect
        }
        if (convo.blocks[turnId]?.working) {
          convo.setWorking(turnId, true);
        } else {
          void this._finishTurn(turnId, lane.type);
        }
      }
    },

    /**
     * True when a given surface is currently busy — gates sends and drains.
     * Main spine shares its surface with ACT cycles (no stable turn id);
     * threads have a stable id plus the conversation working-set.
     */
    isLaneBusy(threadId: number | null, type: string = ConfigType.USER): boolean {
      return threadId == null
        ? this.isSending
        : laneKey(threadId) in this.lanes || useConversationFeed(type).isTurnWorking(threadId);
    },

    /**
     * Send a user turn. Signal-only: the `turn_execution` working refetch renders
     * the user bubble from the API — no optimistic echo. Everything — the
     * spinner, the rows, the reply — flows back through the `updated` broadcast
     * signal and the `turn_execution` lifecycle frame (→ dispatchDrift → refetch).
     */
    async sendMessage(
      text: string,
      files: File[] = [],
      threadId: number | null = null,
      type: string = ConfigType.USER,
    ): Promise<void> {
      if (!text && !files.length) return;

      const body = text || FILE_PLACEHOLDER;

      if (this.isLaneBusy(threadId, type)) {
        useQueueStore().enqueue(threadId, body, type, files);
        return;
      }

      const key = laneKey(threadId);
      this.lanes[key] = {
        liveTurnId: threadId,
        userText: text,
        type,
      };
      const result = await getWebSocket().send(body, (m) => this._onSendFailure(key, m), files, threadId, type);
      if (result != null && key in this.lanes) {
        this.lanes[key].liveTurnId = result.turn_id;
      }
    },

    /** A local send failure (offline / POST rejected) — no signal will ever
     *  arrive, so release this lane's guard and surface the message. */
    _onSendFailure(key: string, message: string): void {
      this.errorMessage = message;
      const lane = this.lanes[key];
      if (lane?.liveTurnId != null) useConversationFeed(lane.type).setWorking(lane.liveTurnId, false);
      delete this.lanes[key];
    },

    /** `done(turn_id, type)` — settle the turn on its type's feed. Leave a
     *  standing done marker unless the user is viewing it; release the owning lane
     *  only when ITS own turn finished. Drain queues, record ambient, and fire an
     *  OS notification for the final reply when the tab is unfocused. Identical for
     *  every type — only the dock the settled thread lives in differs. */
    async _finishTurn(turnId: number | null, type: string = ConfigType.USER): Promise<void> {
      const convo = useConversationFeed(type);
      const owningKey = this._laneOwning(turnId);
      const id = turnId ?? (owningKey == null ? null : this.lanes[owningKey]?.liveTurnId ?? null);

      if (id != null) {
        if (id === this.panelThreadId) convo.setWorking(id, false);
        else convo.markThreadDone(id);
      }

      if (owningKey != null) delete this.lanes[owningKey];

      this._drainQueues();
      useAmbientSensor().recordResponse();

      if (id != null && !document.hasFocus()) {
        await convo.fetchTurn(id);
        const t = convo.turnSpeechText(id);
        if (t) this._notifyBackground(t);
      }
    },

    /**
     * Find the lane key that owns the given turnId — used to release ONLY the
     * correct lane on `done` without disturbing peer lanes.
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

    /** Drain ALL pending scopes independently. */
    _drainQueues(): void {
      for (const key of useQueueStore().pendingScopes) this._drainLane(key);
    },

    _drainLane(key: string): void {
      const threadId = key === 'main' ? null : Number(key.slice(1));
      const queue = useQueueStore();
      const type = queue.typeFor(threadId);
      if (this.isLaneBusy(threadId, type)) return;
      const { text, files } = queue.take(threadId);
      if (text) void this.sendMessage(text, files, threadId, type);
    },

    /**
     * Stop + undo the in-flight turn identified by `turnId`. Emits
     * 'session:turn-interrupted' so InputDock can restore the textarea. `type`
     * (default user) names the owning thread's ProcessorConfig — DELETE resolves
     * the channel from it server-side, and turn_id alone is only unique per
     * channel, so a non-user thread's stop must carry its own type through.
     */
    async requestStop(turnId: number | null = null, type: string = ConfigType.USER): Promise<void> {
      const ws = getWebSocket();

      const key = this._laneOwning(turnId);
      const lane = key == null ? undefined : this.lanes[key];
      const stopId = lane?.liveTurnId ?? turnId;
      const laneType = lane?.type ?? type;

      // Restore text from lane record (no optimistic form to read from any more).
      const restoredText = lane?.userText ?? '';

      // The dock scope id InputDock keys its `turnId` prop by: null for the main
      // spine, the thread's root turn_id for a thread reply dock. That is the
      // LANE key, not necessarily the live turn id passed in (they only diverge
      // for a still-resolving main-spine send, where `liveTurnId` is null until
      // the POST returns) — derive it from the owning lane key so the restore
      // event lands on the dock that actually owns this turn.
      const scopeId = key == null ? turnId : key === 'main' ? null : Number(key.slice(1));

      // Undo the whole turn: drop its block + all signal state. For a fork
      // turn (real prior content already settled) this optimistically blanks
      // that content too — there is no per-row "just the pending bubble"
      // removal, a fork's settled exchange and its still-open reply share one
      // block — but the reconcile below restores it within one round-trip
      // once the DELETE ack lands, well before a WS frame would.
      if (lane?.liveTurnId != null) useConversationFeed(laneType).dropLiveTurn(lane.liveTurnId);
      if (key != null) delete this.lanes[key];

      ws.abort();

      document.dispatchEvent(
        new CustomEvent('session:turn-interrupted', { detail: { text: restoredText, turnId: scopeId } }),
      );

      await this._postInterrupt(stopId, laneType);

      // Reconcile against the backend now, rather than waiting on the WS
      // 'cancelled' frame (`_handleCancelled`) — that frame can be delayed or
      // dropped on a flaky connection, and it's also what actually restores a
      // fork turn's content after the blanket drop above. Idempotent against
      // a WS frame that also arrives later.
      if (stopId != null) await this._handleCancelled(stopId, laneType);
    },

    /** DELETE /api/thread/<turn_id>?type=<type> — best-effort interrupt, never throws. */
    async _postInterrupt(turnId: number | null = null, type: string = ConfigType.USER): Promise<void> {
      if (turnId == null) return;
      try {
        const host = getHost();
        const base = host ? host.replace(/\/$/, '') : '';
        await fetch(base + '/api/thread/' + turnId + '?type=' + encodeURIComponent(type), {
          method: 'DELETE',
          credentials: 'same-origin',
        });
      } catch {
        // Best-effort — swallow.
      }
    },

    /**
     * D17 — the pagination cursor (offset/hasMore) is UI-local, owned by
     * ConversationFeed.vue's own `_loadRecent` callback (registered via
     * `registerHistoryLoader`); this store keeps only what's genuinely
     * shared: the `historyLoading` flag (also gates scroll-pagination) and
     * the AuthError / initial-load event semantics around it. Smaller diff
     * than moving `historyLoading` itself out — every other consumer
     * (InputDock, PresenceBar) reads it off this store already.
     */
    async loadRecentConversation(): Promise<void> {
      if (!this._loadRecentCallback || this.historyLoading) return;
      this.historyLoading = true;

      try {
        const { isInitialLoad, loadedAny } = await this._loadRecentCallback();
        if (isInitialLoad && loadedAny) {
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

    /** Register the feed's history-load callback (called once, on mount). */
    registerHistoryLoader(cb: () => Promise<{ isInitialLoad: boolean; loadedAny: boolean }>): void {
      this._loadRecentCallback = cb;
    },

    /**
     * Open a thread in the slide-over panel. ThreadPanel.vue owns the actual
     * fetch + surface upsert (it watches panelThreadId/panelType) — this just
     * sets identity and clears the standing "done" marker: the buffer's own
     * `seenThread` (interim — tasks.ts's Activity dock still reads it) plus
     * the DOM's `data-done` attribute (D16).
     */
    openThreadPanel(turnId: number, type: string = ConfigType.USER): void {
      this.panelThreadId = turnId;
      this.panelType = type;
      useConversationFeed(type).seenThread(turnId);
      setTurnDone(turnId, type, false);
    },

    /** Close the slide-over panel. */
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

    /** Open / close the scheduler dock. */
    openSchedulerDock(): void {
      this.schedulerDockOpen = true;
    },
    closeSchedulerDock(): void {
      this.schedulerDockOpen = false;
    },

    /**
     * Cancelled turn — lane/buffer bookkeeping ONLY (interim seam). Called
     * from `requestStop`'s post-interrupt reconcile AND from
     * `driftDispatcher._dispatchCancelled` (idempotent either order, or
     * both). The DOM-rendering side of a cancel — the force-upserted refetch
     * and `removeTurn` for an emptied turn — lives in the dispatcher now (see
     * its comment for the full rationale); this just releases the owning
     * lane, mirrors the buffer write for tasks.ts's Activity dock (still
     * buffer-driven this slice), and drains queues.
     */
    async _handleCancelled(turnId: number, type: string): Promise<void> {
      const convo = useConversationFeed(type);
      const owningKey = this._laneOwning(turnId);
      if (owningKey != null) delete this.lanes[owningKey];
      try {
        const block = await convoApi.thread(turnId, type);
        if (block.messages.length) {
          convo.forceUpsertTurn(block);
          convo.setWorking(turnId, false);
        } else {
          convo.dropLiveTurn(turnId);
        }
      } catch {
        // Best-effort — leave whatever is buffered; a later fetchTurn (panel
        // open, page refresh) will reconcile against the DB regardless.
        convo.setWorking(turnId, false);
      }
      this._drainQueues();
    },

    /** Fire a background notification when the tab is not focused. */
    _notifyBackground(content: string): void {
      if (document.hasFocus()) return;
      const plain = extractText(content);
      if (plain) useNotificationsStore().pushBackground(plain);
    },
  },
});

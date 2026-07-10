/**
 * Session store — WS coordinator + turn send/stop orchestration.
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
 * D3 — no lane model: busy/working state for every independent conversation
 * surface (the main spine + each open thread reply) lives as a DOM
 * attribute (`data-working`, see `utils/turnDom.ts`) rather than a store-held
 * record. `isSurfaceBusy` derives it via a DOM query for a stable turn_id
 * (a thread), or a registered surface's own container (the main spine, which
 * has no stable id until a brand-new send's POST resolves one).
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
import { clearLiveTurn } from '../utils/liveActTrail';
import {
  isSurfaceWorking,
  isTurnWorking,
  liveWorkingKeys,
  setTurnDone,
  setTurnWorking,
  SPINE_SURFACE_ID,
} from '../utils/turnDom';
import { laneKey, useQueueStore } from './queue';
import { useNotificationsStore } from './notifications';
import { useContextUsageStore } from './contextUsage';
import { useAmbientSensor } from '../composables/useAmbientSensor';

/** Guard: init() must be idempotent (HMR / Vue StrictMode). */
let _initialized = false;

/** Unbind fns for event-bus listeners registered in init() (for future cleanup). */
const _busUnbinds: Array<() => void> = [];

const FILE_PLACEHOLDER = '[File attached]';

export const useSessionStore = defineStore('session', {
  state: () => ({
    /** Synchronous send-in-flight guard, keyed by laneKey(threadId). D3: this
     *  is NOT a lane record — it holds no turn identity or draft text, only a
     *  scope key. It bridges the gap between "we decided to send" and the
     *  DOM's own `data-working` state existing: held from the send decision
     *  until the turn's FIRST `turn_execution` frame is observed (see
     *  `_pendingByTurn` below), because the POST resolves as soon as the
     *  backend allocates the turn_id — execution proceeds in the background,
     *  so there is no ordering guarantee between the POST 200 and the WS
     *  'working' frame that stamps the DOM. */
    _pendingSends: new Set<string>(),

    /** `type:turnId` → laneKey for sends whose POST resolved but whose first
     *  `turn_execution` frame hasn't been observed yet — the dispatcher
     *  releases the matching `_pendingSends` hold via `_releasePendingSend`.
     *  Mirrors HEAD's lane binding ("bind the lane handle the moment the
     *  server allocates it, release on the finish signal") without a lane
     *  record. */
    _pendingByTurn: new Map<string, string>(),

    /** True when the main spine had a working turn at the moment the WS
     *  dropped — the spine's counterpart to `_offlineWorking` (it has no
     *  stable turn id for `isSurfaceBusy` to key off). Keeps the spine dock
     *  queueing rather than dropping drafts while the backend may still be
     *  mid-turn behind the dead socket; cleared once `_reconcileWorking`
     *  restores the real state. */
    _offlineSpineWorking: false,

    /** `type:turnId` keys that were in flight when the WS dropped. The
     *  disconnect handler clears every visual `data-working` marker — the
     *  same marker `_reconcileWorking` would otherwise scan — so what was
     *  working MUST be remembered here or nothing gets reconciled on
     *  reconnect (spinner restore / offline-settle / queue drain would all
     *  silently die). */
    _offlineWorking: new Set<string>(),

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
        void this._reconcileWorking();
      });

      ws.onDisconnect(() => {
        conn.setConnected(false);
        // A mid-turn drop strands spinners: the terminal `turn_execution` frame
        // lands on the dead socket and is never resent. Snapshot what was in
        // flight FIRST (liveWorkingKeys covers turns whose 'working' frame
        // arrived before any element rendered), THEN clear the transient
        // visual state so nothing hangs — `_reconcileWorking` walks the
        // snapshot on reconnect and either restores the spinner (backend
        // still genuinely mid-turn — `working` is a server-derived field,
        // see ConversationTurnBlock) or settles it via `_finishTurn`.
        for (const key of liveWorkingKeys()) this._offlineWorking.add(key);
        for (const el of Array.from(document.querySelectorAll<HTMLElement>('[data-working][data-turn-id]'))) {
          const turnId = Number(el.getAttribute('data-turn-id'));
          const type = el.getAttribute('data-type') ?? ConfigType.USER;
          if (!Number.isNaN(turnId)) this._offlineWorking.add(`${type}:${turnId}`);
        }
        // Spine snapshot too (it has no stable id for the key set) — read
        // BEFORE the clearing loop below strips the very markers it scans.
        if (isSurfaceWorking(SPINE_SURFACE_ID)) this._offlineSpineWorking = true;
        for (const key of this._offlineWorking) {
          const idx = key.indexOf(':');
          const type = key.slice(0, idx);
          const turnId = Number(key.slice(idx + 1));
          setTurnWorking(turnId, type, false);
          useConversationFeed(type).setWorking(turnId, false);
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
     * Reconcile every turn that was in flight when the WS dropped against
     * the backend after reconnect. Walks the `_offlineWorking` snapshot the
     * disconnect handler recorded — NOT the DOM: the disconnect handler
     * already cleared every `data-working` marker, so a DOM scan here would
     * find nothing. Re-fetches each candidate turn and either restores its
     * spinner (backend still genuinely mid-turn — `working` is a
     * server-derived field, see ConversationTurnBlock) or settles it via
     * `_finishTurn` (it actually completed/crashed/cancelled while offline).
     */
    async _reconcileWorking(): Promise<void> {
      for (const key of Array.from(this._offlineWorking)) {
        const idx = key.indexOf(':');
        const type = key.slice(0, idx);
        const turnId = Number(key.slice(idx + 1));

        const convo = useConversationFeed(type);
        try {
          await convo.fetchTurn(turnId);
        } catch {
          continue; // best-effort; key stays snapshotted, retried on the next reconnect
        }
        this._offlineWorking.delete(key);
        if (convo.blocks[turnId]?.working) {
          setTurnWorking(turnId, type, true);
          convo.setWorking(turnId, true);
        } else {
          void this._finishTurn(turnId, type);
        }
      }
      // Real state is restored above (still-working turns have their DOM
      // markers back), so the blanket offline flag can drop — best-effort
      // even when a fetch failed and its key stayed snapshotted.
      this._offlineSpineWorking = false;
    },

    /**
     * True when a given surface is currently busy — gates sends and drains.
     * Main spine shares its surface with ACT cycles (no stable turn id);
     * threads have a stable id and are checked directly against the DOM.
     */
    isSurfaceBusy(threadId: number | null, type: string = ConfigType.USER): boolean {
      if (this._pendingSends.has(laneKey(threadId))) return true;
      // Offline snapshots count as busy: the backend may still be mid-turn
      // behind the dead socket even though the visual markers were cleared —
      // a send now should queue (drained after `_reconcileWorking`), not
      // silently drop the draft on the disconnected transport.
      if (threadId == null) {
        return this._offlineSpineWorking || isSurfaceWorking(SPINE_SURFACE_ID);
      }
      return this._offlineWorking.has(`${type}:${threadId}`) || isTurnWorking(threadId, type);
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

      if (this.isSurfaceBusy(threadId, type)) {
        useQueueStore().enqueue(threadId, body, type, files);
        return;
      }

      const key = laneKey(threadId);
      this._pendingSends.add(key);
      let heldForFrame = false;
      try {
        const result = await getWebSocket().send(
          body, (m) => this._onSendFailure(m), files, threadId, type,
        );
        // POST resolved with the allocated turn_id but execution runs in the
        // background — keep the busy hold until the dispatcher observes the
        // turn's first `turn_execution` frame (unless one already beat the
        // POST response here). `null` result = local send failure; nothing
        // will ever arrive, release now.
        if (result && !isTurnWorking(result.turn_id, result.type)) {
          this._pendingByTurn.set(`${result.type}:${result.turn_id}`, key);
          heldForFrame = true;
        }
      } finally {
        if (!heldForFrame) this._pendingSends.delete(key);
      }
    },

    /** Release a send's POST-scoped busy hold once its turn's first
     *  `turn_execution` frame arrives (called by the dispatcher for EVERY
     *  execution state — a crash/settle that beat the 'working' frame must
     *  release too). No-op for turns with no hold registered. */
    _releasePendingSend(turnId: number, type: string): void {
      const key = this._pendingByTurn.get(`${type}:${turnId}`);
      if (key == null) return;
      this._pendingByTurn.delete(`${type}:${turnId}`);
      this._pendingSends.delete(key);
    },

    /** A local send failure (offline / POST rejected) — no signal will ever
     *  arrive for a turn that never got created; just surface the message. */
    _onSendFailure(message: string): void {
      this.errorMessage = message;
    },

    /** `done(turn_id, type)` — settle the turn on its type's feed. Leave a
     *  standing done marker unless the user is viewing it. Drain queues,
     *  record ambient, and fire an OS notification for the final reply when
     *  the tab is unfocused. Identical for every type — only the dock the
     *  settled thread lives in differs. */
    async _finishTurn(turnId: number, type: string = ConfigType.USER): Promise<void> {
      const convo = useConversationFeed(type);

      if (turnId === this.panelThreadId) convo.setWorking(turnId, false);
      else convo.markThreadDone(turnId);

      this._drainQueues();
      useAmbientSensor().recordResponse();

      if (!document.hasFocus()) {
        // Buffer-sourced deliberately (interim seam): the awaited fetchTurn
        // just populated this exact block, so the read can't race — whereas
        // the DOM copy is written by a SEPARATE, unawaited `updated`-signal
        // refetch (and may not exist at all for a turn no surface renders).
        await convo.fetchTurn(turnId);
        const t = convo.turnSpeechText(turnId);
        if (t) this._notifyBackground(t);
      }
    },

    /** Drain ALL pending scopes independently. */
    _drainQueues(): void {
      for (const key of useQueueStore().pendingScopes) this._drainLane(key);
    },

    _drainLane(key: string): void {
      const threadId = key === 'main' ? null : Number(key.slice(1));
      const queue = useQueueStore();
      const type = queue.typeFor(threadId);
      if (this.isSurfaceBusy(threadId, type)) return;
      const { text, files } = queue.take(threadId);
      if (text) void this.sendMessage(text, files, threadId, type);
    },

    /**
     * Stop + undo the in-flight turn identified by `turnId`. Emits
     * 'session:turn-interrupted' so InputDock can restore the textarea.
     * `type` (default user) names the owning thread's ProcessorConfig —
     * DELETE resolves the channel from it server-side, and turn_id alone is
     * only unique per channel, so a non-user thread's stop must carry its
     * own type through. `dockScope` (D6) is the dock's own identity — null
     * for the main spine, the thread's root turn_id for a thread reply dock
     * — read by the caller off the DOM's `data-dock-scope` marker (see
     * ThreadPanel.vue), since there is no lane record any more to derive it
     * from. `restoreText` is the exact text to hand back to that dock,
     * likewise read by the caller off the DOM (`data-user-text`, see
     * UserBubble.vue / turnDom's `lastUserText`) before this call.
     */
    async requestStop(
      turnId: number | null = null,
      type: string = ConfigType.USER,
      dockScope: number | null = null,
      restoreText: string = '',
    ): Promise<void> {
      const ws = getWebSocket();

      // D6: confirm turnId is genuinely still in flight (per the DOM's own
      // data-working marker) before firing the DELETE — a stale/late click
      // could otherwise target an already-settled turn.
      const stopId = turnId != null && isTurnWorking(turnId, type) ? turnId : null;

      const text = restoreText === FILE_PLACEHOLDER ? '' : restoreText;

      if (stopId != null) {
        // Optimistic: hide the spinner/stop control/live pill trail
        // immediately, rather than waiting on the DELETE round-trip or a WS
        // 'cancelled' frame that can be delayed or dropped on a flaky
        // connection. `_handleCancelled` below (idempotent alongside a later
        // WS frame, same as before) reconciles the actual content once
        // confirmed.
        setTurnWorking(stopId, type, false);
        clearLiveTurn(type, stopId);
        // Interim seam (tasks.ts / SchedulerDock) — dies with the buffer in
        // a later slice.
        useConversationFeed(type).dropLiveTurn(stopId);
      }

      ws.abort();

      document.dispatchEvent(
        new CustomEvent('session:turn-interrupted', { detail: { text, turnId: dockScope } }),
      );

      await this._postInterrupt(stopId, type);

      // Reconcile against the backend now, rather than waiting on the WS
      // 'cancelled' frame (`_handleCancelled`) — that frame can be delayed or
      // dropped on a flaky connection, and it's also what actually restores a
      // fork turn's content after the blanket drop above. Idempotent against
      // a WS frame that also arrives later.
      if (stopId != null) await this._handleCancelled(stopId, type);
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
     * Cancelled turn — buffer bookkeeping ONLY (interim seam). Called from
     * `requestStop`'s post-interrupt reconcile AND from
     * `driftDispatcher._dispatchCancelled` (idempotent either order, or
     * both). The DOM-rendering side of a cancel — the force-upserted refetch
     * and `removeTurn` for an emptied turn — lives in the dispatcher now (see
     * its comment for the full rationale); this mirrors the buffer write for
     * tasks.ts's Activity dock (still buffer-driven this slice) and drains
     * queues.
     */
    async _handleCancelled(turnId: number, type: string): Promise<void> {
      const convo = useConversationFeed(type);
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

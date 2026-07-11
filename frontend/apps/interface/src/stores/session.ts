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
import { useActionCard } from '../composables/useActionCard';
import { dispatchDrift, registerSessionHooks } from '../utils/driftDispatcher';
import { reconcileCancelledTurn } from '../utils/cancelReconcile';
import { clearLiveTurn } from '../utils/liveActTrail';
import { blockSpeechText } from '../utils/speech';
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

      registerSessionHooks({
        releasePendingSend: (turnId, type) => this._releasePendingSend(turnId, type),
        getPanelThreadId: () => this.panelThreadId,
        setErrorMessage: (message) => { this.errorMessage = message; },
        finishTurn: (turnId, type) => this._finishTurn(turnId, type),
        drainQueues: () => this._drainQueues(),
      });

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
     * find nothing. Re-fetches each candidate turn straight off the REST API
     * (the DOM contract's sole data source — no client-side cache to
     * consult) and either restores its spinner (backend still genuinely
     * mid-turn — `working` is a server-derived field, see
     * ConversationTurnBlock) or settles it: marks `data-done` (D16, unless
     * the panel has it open) and hands off to `_finishTurn` for the
     * offline-settle bookkeeping the live WS settle path would otherwise
     * have done.
     */
    async _reconcileWorking(): Promise<void> {
      for (const key of Array.from(this._offlineWorking)) {
        const idx = key.indexOf(':');
        const type = key.slice(0, idx);
        const turnId = Number(key.slice(idx + 1));

        let block;
        try {
          block = await convoApi.thread(turnId, type);
        } catch {
          continue; // best-effort; key stays snapshotted, retried on the next reconnect
        }
        this._offlineWorking.delete(key);
        if (block.working) {
          setTurnWorking(turnId, type, true);
        } else {
          if (turnId !== this.panelThreadId) setTurnDone(turnId, type, true);
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

    /** Settle bookkeeping for a completed/crashed/offline-settled turn.
     *  `data-done` itself is already stamped by the caller (D16, see
     *  `driftDispatcher`'s turn_execution branch and `_reconcileWorking`
     *  above) — this only drains queues, records ambient activity, and fires
     *  an OS notification for the final reply when the tab is unfocused.
     *  Identical for every type — only the dock the settled thread lives in
     *  differs. */
    async _finishTurn(turnId: number, type: string = ConfigType.USER): Promise<void> {
      this._drainQueues();
      useAmbientSensor().recordResponse();

      if (!document.hasFocus()) {
        // Fetched ONCE, here, for the notification — deliberately NOT read
        // off the DOM: the DOM copy is written by a SEPARATE, unawaited
        // `updated`-signal refetch and may not exist at all for a turn no
        // surface renders (a critic HIGH in S4).
        try {
          const block = await convoApi.thread(turnId, type);
          const t = blockSpeechText(block);
          if (t) this._notifyBackground(t);
        } catch {
          // Best-effort — no notification if the fetch fails.
        }
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

      // Optimistic: hide the spinner/live pill trail immediately rather than
      // waiting on the DELETE round-trip. The CONTENT refetch, however, must
      // NOT start yet — the backend only strips a cancelled turn's orphan
      // user row once cancel() has committed, so a fetch racing ahead of the
      // DELETE can force-upsert stale pre-cancel content that nothing
      // corrects if the WS 'cancelled' frame is dropped.
      if (stopId != null) {
        setTurnWorking(stopId, type, false);
        clearLiveTurn(type, stopId);
      }

      ws.abort();

      document.dispatchEvent(
        new CustomEvent('session:turn-interrupted', { detail: { text, turnId: dockScope } }),
      );

      await this._postInterrupt(stopId, type);

      if (stopId != null) {
        // Post-DELETE, the fetch reads authoritative post-cancel state — and
        // dedupes (same in-flight cache) with whatever reconcile a WS
        // 'cancelled' frame may have kicked off during the round-trip.
        await reconcileCancelledTurn(stopId, type);
        this._drainQueues();
      }
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
     * sets identity and clears the standing "done" marker (D16).
     */
    openThreadPanel(turnId: number, type: string = ConfigType.USER): void {
      this.panelThreadId = turnId;
      this.panelType = type;
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

    /** Fire a background notification when the tab is not focused. */
    _notifyBackground(content: string): void {
      if (document.hasFocus()) return;
      const plain = extractText(content);
      if (plain) useNotificationsStore().pushBackground(plain);
    },
  },
});

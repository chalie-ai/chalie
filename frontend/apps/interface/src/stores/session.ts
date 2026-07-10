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
import type { WsInboundEvent, WsPushEvent, WsTurnExecutionEvent } from '@chalie/shared';
import { AuthError, ConfigType, getWebSocket, useConnectionStore } from '@chalie/shared';
import { on } from '../composables/useEventBus';
import { extractText } from '../composables/useMarkup';
import { getHost } from '../api/index';
import { showToast } from '../utils/toast';
import { conversation as convoApi } from '../api/conversation';
import { useConversationFeed } from '../composables/useConversationFeed';
import { useActionCard } from '../composables/useActionCard';
import { laneKey, useQueueStore } from './queue';
import { useTasksStore } from './tasks';
import type { TipState, UpdateState } from './notifications';
import { useNotificationsStore } from './notifications';
import { usePermissionsStore } from './permissions';
import { useContextUsageStore } from './contextUsage';
import { useAmbientSensor } from '../composables/useAmbientSensor';

/** Guard: init() must be idempotent (HMR / Vue StrictMode). */
let _initialized = false;

/** Unbind fns for event-bus listeners registered in init() (for future cleanup). */
const _busUnbinds: Array<() => void> = [];

const FILE_PLACEHOLDER = '[File attached]';

/** Runtime membership check backing `_routeTurnExecution`'s discriminator — a
 *  `state` key alone does not prove a frame is a turn_execution row (e.g. the
 *  Home capability's `home_state_changed` push also carries an unrelated
 *  `state` string, an HA entity state that could coincidentally collide with
 *  one of these four literals). Combined with a required `turn_id` + `started_at`
 *  (execution-only fields no other push family emits) the match is unambiguous. */
const TURN_EXECUTION_STATES: ReadonlySet<string> = new Set([
  'working', 'completed', 'cancelled', 'crashed',
]);

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
        this.routeDrift(data);
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
     * signal and the `turn_execution` lifecycle frame (→ routeDrift → refetch).
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
     * Load (or paginate) the thread list via the composable. Initial load fetches
     * the 20 most-recent collapsed threads; scroll-up pagination prepends 20 more.
     */
    async loadRecentConversation(): Promise<void> {
      const convo = useConversationFeed();
      if (!convo.hasMore || this.historyLoading) return;
      this.historyLoading = true;

      try {
        const isInitialLoad = convo.sortedBlocks.value.length === 0;
        await (isInitialLoad ? convo.loadRecent() : convo.loadMore());

        if (isInitialLoad && convo.sortedBlocks.value.length > 0) {
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

    /** Open a thread in the slide-over panel, loading its rows on first open. */
    async openThreadPanel(turnId: number, type: string = ConfigType.USER): Promise<void> {
      this.panelThreadId = turnId;
      this.panelType = type;
      const convo = useConversationFeed(type);
      convo.seenThread(turnId);
      if (convo.isHydrated(turnId)) return;
      this.threadExpanding = true;
      try {
        await convo.fetchTurn(turnId);
      } finally {
        this.threadExpanding = false;
      }
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
     * Route a drift push event. The live tool-call frame is claimed first (it is
     * recognised by `tool_name`, checked before the `state`/`status` discriminants
     * the other turn frames key on). Every push is a content-free signal — the
     * scheduler runs prompts only now, so its fires arrive as ordinary turn
     * lifecycle frames on the `scheduled` type, not a distinct `notification` push.
     */
    routeDrift(data: WsPushEvent): void {
      if (this._routeToolCall(data)) return;
      if (this._routeTurnSignal(data)) return;
      if (this._routeTurnExecution(data)) return;
      this._routeSimpleEvent(data);
    },

    /**
     * Mid-turn progress signals — stateless and turn-addressed, discriminated on
     * `status` (the sole turn chokepoint; pushes key on `type`). `type` names the
     * ProcessorConfig surface to route by. Returns true when handled; a frame with
     * no `status` falls through (to `_routeTurnExecution`, then `_routeSimpleEvent`).
     */
    _routeTurnSignal(data: WsPushEvent): boolean {
      const status = (data as { status?: string }).status;
      if (status == null) return false;
      const convo = useConversationFeed((data as { type?: string }).type ?? ConfigType.USER);
      const turnId = (data as { turn_id?: number | null }).turn_id ?? null;
      switch (status) {
        case 'updated':
          if (turnId != null) void convo.fetchTurn(turnId);
          return true;
        case 'provider_retry':
          // Transient: a provider call failed mid-turn and is being resent. The
          // turn stays in flight (no error bubble) — just a toast.
          showToast((data as { message?: string }).message ?? 'The AI provider had a problem — retrying…');
          return true;
        default:
          return false;
      }
    },

    /**
     * Live tool-call frame — the `tool_calls` row's WS projection, pushed on every
     * state flip by ActTrail (see services/act_trail.py). It carries no kind tag, so
     * it is recognised by the presence of `tool_name` (claimed before the
     * `state`/`status` frames). `started` opens the live pill + elapsed timer;
     * `done`/`error` resolve it (ok = state === 'done'), the ONLY place the pill's
     * error state is set. A frame with no anchor turn (`turn_id` null) is dropped —
     * no pill to hang it on. Returns true when the frame is a tool call.
     */
    _routeToolCall(data: WsPushEvent): boolean {
      const toolName = (data as { tool_name?: string }).tool_name;
      if (toolName == null) return false;
      const turnId = (data as { turn_id?: number | null }).turn_id ?? null;
      if (turnId == null) return true;
      const convo = useConversationFeed((data as { type?: string }).type ?? ConfigType.USER);
      const frame = data as { id?: number; summary?: string; state?: string; transcript_row_id?: number | null };
      if (frame.state === 'started') {
        convo.startLiveTool(turnId, frame.id ?? null, toolName, frame.summary, frame.transcript_row_id ?? null);
      } else {
        convo.finishLiveTool(turnId, frame.id ?? null, frame.state === 'done');
      }
      return true;
    },

    /**
     * Cancelled turn — the DELETE-triggered terminal state
     * (backend `TurnExecutionService.cancel`, the single authority: the row
     * is already stamped CANCELLED, with `ended_at` set, by the time this
     * frame is broadcast) AND the post-`_postInterrupt` reconcile `requestStop`
     * runs regardless of whether that frame ever arrives. The backend's
     * `serialize_turn` strips a cancelled turn's trailing, reply-less user
     * row(s) before this refetch ever runs (real content a turn already wrote
     * before the cancel landed always stays), so re-fetching now IS the
     * authoritative post-cancel state. Written via `forceUpsertTurn` — the
     * genuinely UNGUARDED buffer write (composables/useConversationFeed.ts) —
     * rather than `upsertTurn`/`fetchTurn`: this is the one case where the
     * freshest read can legitimately be SMALLER than what's already buffered
     * (a fork turn's earlier-settled content is version-N buffered, the
     * now-stripped orphan reply never raises that version), and `upsertTurn`'s
     * monotonic version guard would silently reject such a write, leaving a
     * phantom orphan bubble on any tab that had buffered the higher version.
     * Empty messages → the turn never happened from the surface's
     * perspective, dropped exactly like an aborted send; any remaining
     * content (a fork turn whose reply-less LAST exchange alone was
     * cancelled) re-renders in place. Idempotent — safe to call twice for the
     * same turn (once from `requestStop`'s reconcile, once from this WS
     * frame, in either order).
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

    /**
     * Turn lifecycle — the DB-backed turn_executions row, pushed whole on every
     * state flip (see services/execution_tracker.py). This frame carries no
     * separate kind tag, so it is recognised structurally: `state` must be one
     * of the four lifecycle literals AND `turn_id` + `started_at` (execution-only
     * fields) must be present — a `state` key alone is not enough (see
     * TURN_EXECUTION_STATES; `home_state_changed` also carries a `state` string
     * with no `turn_id`/`started_at`, and claiming it here drained its queue into
     * a garbage _finishTurn call). `working` opens exactly like the old `working`
     * signal; `completed`/`crashed` settle the turn exactly like the old `done`
     * signal — a crash now reaches this path too, so a turn that dies before
     * producing a reply never leaves a spinner hanging. `cancelled` is its own
     * branch (`_handleCancelled`): a normal settle would leave a fully-rendered,
     * discarded response looking exactly like a completed one, when the whole
     * point of cancel is that it never counted. `crashed` additionally raises
     * the 'turn failed' dock toast — derived from the lifecycle state itself,
     * so no separate error frame is needed.
     */
    _routeTurnExecution(data: WsPushEvent): boolean {
      const exec = data as unknown as WsTurnExecutionEvent;
      if (
        typeof exec.state !== 'string' || !TURN_EXECUTION_STATES.has(exec.state)
        || exec.turn_id == null || exec.started_at == null
      ) {
        return false;
      }
      const type = exec.type ?? ConfigType.USER;
      if (exec.state === 'working') {
        useConversationFeed(type).setWorking(exec.turn_id, true);
        // Pull the block so every surface renders the user bubble from the API.
        void useConversationFeed(type).fetchTurn(exec.turn_id);
        return true;
      }
      if (exec.state === 'cancelled') {
        void this._handleCancelled(exec.turn_id, type);
        return true;
      }
      if (exec.state === 'crashed') this.errorMessage = 'Turn failed unexpectedly';
      void this._finishTurn(exec.turn_id, type);
      return true;
    },

    /** Route content-free push event types (keyed on `type`); returns true when
     *  handled. */
    _routeSimpleEvent(data: WsPushEvent): boolean {
      switch (data.type as string) {
        case 'app_update':
          useNotificationsStore().handleUpdate(data as unknown as UpdateState);
          return true;
        case 'subagent_start':
        case 'subagent_end':
          useTasksStore().applyDriftEvent(data);
          return true;
        case 'capability_alert':
          return true;
        case 'permission_request':
          usePermissionsStore().enqueue(data);
          return true;
        case 'quick_tip':
          useNotificationsStore().handleTip(data as unknown as TipState);
          return true;
        default:
          return false;
      }
    },

    /** Fire a background notification when the tab is not focused. */
    _notifyBackground(content: string): void {
      if (document.hasFocus()) return;
      const plain = extractText(content);
      if (plain) useNotificationsStore().pushBackground(plain);
    },
  },
});

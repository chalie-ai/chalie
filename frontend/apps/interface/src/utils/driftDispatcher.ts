/**
 * driftDispatcher — the DOM-effect WS router (Weave DOM contract).
 *
 * Governing design: NO shared client store for thread state. Turn state
 * lives as data- attributes on rendered DOM (see utils/turnDom.ts); WS
 * handlers look up elements by data-turn-id and mutate in place; REST
 * (api/conversation.ts) is the sole data source. This module is the ONLY
 * place that maps a raw WS drift push onto that DOM — session.ts no longer
 * routes drift itself (see its `init()`, which wires `ws.onDrift(dispatchDrift)`).
 *
 * This module does NOT import `stores/session.ts` — session.ts imports
 * `dispatchDrift` from here (it's the sole WS owner, see its docblock), so a
 * static import back would be circular. The handful of session-owned side
 * effects a turn_execution frame still needs (releasing a send's busy hold,
 * reading the open panel's turn id, surfacing a crash toast, settle
 * bookkeeping) are reached through `SessionHooks`, a small interface session
 * registers an implementation of via `registerSessionHooks()` from its own
 * `init()` — dependency injection instead of a static import, breaking the
 * cycle in both directions.
 */
import type { WsPushEvent, WsTurnExecutionEvent } from '@chalie/shared';
import { conversation as convoApi } from '../api/conversation';
import { showToast } from './toast';
import { clearLiveTurn, finishLiveTool, startLiveTool } from './liveActTrail';
import { reconcileCancelledTurn } from './cancelReconcile';
import { findTurnType, setTurnDone, setTurnWorking, upsertTurnToSurfaces } from './turnDom';
import { useNotificationsStore } from '../stores/notifications';
import type { UpdateState } from '../stores/notifications';
import { useTasksStore } from '../stores/tasks';
import { usePermissionsStore } from '../stores/permissions';
import { useContextUsageStore } from '../stores/contextUsage';

/** The session-owned side effects a turn_execution frame still needs — see
 *  module docblock for why this is dependency-injected rather than a static
 *  import of `stores/session.ts`. */
export interface SessionHooks {
  /** Release a send's POST-scoped busy hold — called for EVERY execution frame. */
  releasePendingSend(turnId: number, type: string): void;
  /** turn_id currently open in the slide-over panel, or null (D16 gate). */
  getPanelThreadId(): number | null;
  /** ConfigType of the panel identified by `getPanelThreadId` — paired with it
   *  so a settle/refetch can match the FULL (turn_id, type) identity rather
   *  than turn_id alone (turn_id is only unique PER TYPE). Required: leaving it
   *  off would silently collapse the D16 gate back to a turn_id-only match, so
   *  every hooks provider — production and test doubles alike — must supply it. */
  getPanelType(): string;
  /** Surface a turn-level provider/quota error as a closable toast. */
  setErrorMessage(message: string): void;
  /** Settle bookkeeping for a completed/crashed turn (queue drain, ambient
   *  sensor, background notification). */
  finishTurn(turnId: number, type: string): Promise<void>;
  /** Drain every pending queued send — called once a cancel has resolved. */
  drainQueues(): void;
}

let _hooks: SessionHooks | null = null;

/** Registered once, from `session.init()`. */
export function registerSessionHooks(hooks: SessionHooks): void {
  _hooks = hooks;
}

function hooks(): SessionHooks {
  if (!_hooks) {
    throw new Error('driftDispatcher: session hooks not registered — session.init() must run first');
  }
  return _hooks;
}

/**
 * Resolve a WS push frame's ConfigType identity when the frame itself omits
 * it — the `type` field on `WsTurnExecutionEvent`/`WsToolCallEvent` is
 * nullable, and a scheduler tick is exactly the kind of frame that can arrive
 * this way. turn_id is only unique PER TYPE, so a missing type can NEVER be
 * defaulted to `user` — that is precisely how a scheduled turn's tick would
 * refetch and paint over a same-numbered user turn. Resolution order: (1)
 * the already-rendered node's own stamped type (authoritative once a turn
 * has painted); (2) the open panel's type, when this frame's turn IS the
 * panel's turn. Returns null when neither resolves — the caller must drop
 * the frame rather than guess.
 */
function resolveFrameType(turnId: number, rawType: string | null | undefined): string | null {
  if (rawType != null) return rawType;
  const domType = findTurnType(turnId);
  if (domType != null) return domType;
  const h = hooks();
  if (h.getPanelThreadId() === turnId) return h.getPanelType();
  return null;
}

/** True when `(turnId, type)` identifies the turn currently open in the
 *  slide-over panel (D16 gate) — the FULL pair must match, since turn_id
 *  alone is only unique PER TYPE. */
function isOpenPanelTurn(turnId: number, type: string): boolean {
  const h = hooks();
  return h.getPanelThreadId() === turnId && h.getPanelType() === type;
}

/** Runtime membership check backing `isTurnExecutionEvent`'s discriminator — a
 *  `state` key alone does not prove a frame is a turn_execution row (e.g. the
 *  Home capability's `home_state_changed` push also carries an unrelated
 *  `state` string, an HA entity state that could coincidentally collide with
 *  one of these four literals). Combined with a required `turn_id` + `started_at`
 *  (execution-only fields no other push family emits) the match is unambiguous. */
const TURN_EXECUTION_STATES: ReadonlySet<string> = new Set([
  'working', 'completed', 'cancelled', 'crashed',
]);

/** A live tool-call frame is recognised by the presence of `tool_name` —
 *  checked before the `state`/`status` discriminants the other turn frames key on. */
export function isToolCallEvent(data: WsPushEvent): boolean {
  return (data as { tool_name?: string }).tool_name != null;
}

/** A mid-turn progress signal is recognised by the presence of `status`. */
export function isTurnSignal(data: WsPushEvent): boolean {
  return (data as { status?: string }).status != null;
}

/** A turn lifecycle frame — see TURN_EXECUTION_STATES above for why the
 *  `state` check alone is insufficient. */
export function isTurnExecutionEvent(data: WsPushEvent): boolean {
  const exec = data as unknown as WsTurnExecutionEvent;
  return (
    typeof exec.state === 'string' && TURN_EXECUTION_STATES.has(exec.state)
    && exec.turn_id != null && exec.started_at != null
  );
}

/**
 * Route a drift push event onto the DOM. The live tool-call frame is claimed
 * first, then the turn_signal `status` discriminant, then the structural
 * turn_execution check, then the content-free push family — identical
 * ordering to the router this replaces (formerly session.ts `routeDrift`).
 */
export function dispatchDrift(data: WsPushEvent): void {
  if (_dispatchToolCall(data)) return;
  if (_dispatchTurnSignal(data)) return;
  if (_dispatchTurnExecution(data)) return;
  _dispatchSimpleEvent(data);
}

/**
 * Live tool-call frame — the `tool_calls` row's WS projection, pushed on every
 * state flip by ActTrail (see services/act_trail.py). `started` opens the live
 * pill + elapsed timer; `done`/`error` resolve it (ok = state === 'done'), the
 * ONLY place the pill's error state is set. A frame with no anchor turn
 * (`turn_id` null) is dropped — no pill to hang it on. No fetch: purely visual.
 *
 * `type` is resolved rather than trusted — turn_id is only unique PER TYPE, so
 * a null/absent type is never coerced to `user`; it's looked up from the
 * rendered DOM node or the open panel instead (see `resolveFrameType`), and
 * the frame is dropped with a loud warning if neither source can name it.
 */
function _dispatchToolCall(data: WsPushEvent): boolean {
  if (!isToolCallEvent(data)) return false;
  const turnId = (data as { turn_id?: number | null }).turn_id ?? null;
  if (turnId == null) return true;
  const type = resolveFrameType(turnId, (data as { type?: string | null }).type);
  if (type == null) {
    console.warn('[driftDispatcher] tool_call frame for turn', turnId, 'has no resolvable type — dropped');
    return true;
  }
  const frame = data as {
    id?: number;
    tool_name?: string;
    summary?: string;
    state?: string;
    transcript_row_id?: number | null;
  };
  if (frame.state === 'started') {
    startLiveTool(type, turnId, frame.id ?? null, frame.tool_name ?? '', frame.summary, frame.transcript_row_id ?? null);
  } else {
    finishLiveTool(type, turnId, frame.id ?? null, frame.state === 'done');
  }
  return true;
}

/**
 * Mid-turn progress signals — stateless and turn-addressed, discriminated on
 * `status`. `updated` re-fetches the turn ONCE and fans it out to every
 * registered surface of its type; `provider_retry` is a transient toast (the
 * turn stays in flight, no error bubble); `context_usage` writes the token
 * reading straight to its store — no refetch, no DOM effect. Returns true when
 * handled.
 *
 * The types of `updated` and `context_usage` are resolved (never coerced to
 * `user`) since turn_id alone doesn't identify a turn across channels — see
 * `resolveFrameType`.
 */
function _dispatchTurnSignal(data: WsPushEvent): boolean {
  if (!isTurnSignal(data)) return false;
  const status = (data as { status?: string }).status;
  const turnId = (data as { turn_id?: number | null }).turn_id ?? null;
  switch (status) {
    case 'updated': {
      if (turnId == null) return true;
      const type = resolveFrameType(turnId, (data as { type?: string | null }).type);
      if (type == null) {
        console.warn('[driftDispatcher] turn_signal "updated" for turn', turnId, 'has no resolvable type — dropped');
        return true;
      }
      // An 'updated' signal is equally proof the send's turn is now tracked by
      // its own signals — so it must release the POST-scoped busy hold too, not
      // only the `turn_execution` path (see `_dispatchTurnExecution`, the other
      // caller). A turn can render AND settle from 'updated' frames alone when
      // its `turn_execution` frame is dropped or arrives before the turn is in
      // the DOM (unresolvable type). That path is otherwise the SOLE releaser
      // and `_reconcileWorking` never touches the hold, so without this the
      // hold — and thus the spine's `isSurfaceBusy` gate — strands forever,
      // silently queueing every later send. No-op when no hold is registered.
      hooks().releasePendingSend(turnId, type);
      void _refetchAndUpsert(turnId, type);
      return true;
    }
    case 'provider_retry':
      showToast((data as { message?: string }).message ?? 'The AI provider had a problem — retrying…');
      return true;
    case 'context_usage': {
      if (turnId == null) return true;
      const type = resolveFrameType(turnId, (data as { type?: string | null }).type);
      if (type == null) {
        console.warn('[driftDispatcher] turn_signal "context_usage" for turn', turnId, 'has no resolvable type — dropped');
        return true;
      }
      const usage = data as { tokens_input?: number; context_window?: number };
      if (usage.tokens_input == null || usage.context_window == null) return true;
      useContextUsageStore().record(type, turnId, usage.tokens_input, usage.context_window);
      return true;
    }
    default:
      return false;
  }
}

/**
 * Turn lifecycle — the DB-backed turn_executions row, pushed whole on every
 * state flip (see services/execution_tracker.py). `working` flips the
 * data-working attribute and re-fetches (renders the user bubble — no
 * optimistic echo). `completed` clears working, drops the live trail, marks
 * data-done UNLESS the settled turn is the one open in the panel (D16), then
 * hands settle bookkeeping to the `finishTurn` session hook. `crashed` does
 * the same but additionally raises the 'turn failed' toast. `cancelled` is
 * its own branch (see `_dispatchCancelled`) — a normal settle would leave a
 * fully-rendered, discarded response looking exactly like a completed one,
 * when the whole point of cancel is that it never counted.
 *
 * `type` is resolved (never coerced to `user`) since turn_id alone doesn't
 * identify a turn across channels — see `resolveFrameType`. The D16 panel
 * check below is likewise a (turn_id, type) pair via `isOpenPanelTurn`, not
 * turn_id alone, so a same-id turn in a different channel can't be mistaken
 * for the one open in the panel.
 */
function _dispatchTurnExecution(data: WsPushEvent): boolean {
  if (!isTurnExecutionEvent(data)) return false;
  const exec = data as unknown as WsTurnExecutionEvent;
  const type = resolveFrameType(exec.turn_id, exec.type);
  if (type == null) {
    console.warn('[driftDispatcher] turn_execution frame for turn', exec.turn_id, 'has no resolvable type — dropped');
    return true;
  }
  const h = hooks();

  // ANY execution frame proves the send's turn is now tracked by its own
  // working/settle signals — release the POST-scoped busy hold (see
  // session.sendMessage's `_pendingByTurn` comment).
  h.releasePendingSend(exec.turn_id, type);

  if (exec.state === 'working') {
    setTurnWorking(exec.turn_id, type, true);
    void _refetchAndUpsert(exec.turn_id, type);
    return true;
  }

  if (exec.state === 'cancelled') {
    _dispatchCancelled(exec.turn_id, type);
    return true;
  }

  setTurnWorking(exec.turn_id, type, false);
  clearLiveTurn(type, exec.turn_id);
  if (exec.state === 'crashed') {
    h.setErrorMessage('Turn failed unexpectedly');
  }
  if (!isOpenPanelTurn(exec.turn_id, type)) {
    setTurnDone(exec.turn_id, type, true);
  }
  void h.finishTurn(exec.turn_id, type);
  return true;
}

/**
 * Fetch a turn block once and fan it into every accepting surface. The block
 * carries its own authoritative `type` (turn_id is only unique PER TYPE) —
 * that's trusted over whatever type this call was made with, loudly warning
 * on disagreement rather than silently upserting under the wrong channel.
 */
async function _refetchAndUpsert(
  turnId: number,
  type: string,
  options: { force?: boolean } = {},
): Promise<void> {
  const block = await convoApi.thread(turnId, type);
  if (block.type !== type) {
    console.warn(
      '[driftDispatcher] refetched turn', turnId, 'came back as type', block.type,
      'but was requested as', type, '— trusting the fetched type',
    );
  }
  upsertTurnToSurfaces(block, block.type, options);
}

/**
 * Cancelled turn — the DELETE-triggered terminal state. DOM reconciliation
 * (the force-upserted refetch / `removeTurn` for an emptied turn) is the ONE
 * shared `reconcileCancelledTurn` fetch (see `utils/cancelReconcile.ts`) —
 * deduped against whatever `session.requestStop`'s own post-interrupt
 * reconcile already kicked off for the same turn, since this frame can also
 * be the ONLY signal for a cancel initiated from another tab/device. Queues
 * are drained via the session hook once the shared reconcile settles.
 */
function _dispatchCancelled(turnId: number, type: string): void {
  void reconcileCancelledTurn(turnId, type).then(() => hooks().drainQueues());
}

/** Route content-free push event types (keyed on `type`); returns true when
 *  handled. Verbatim relocation of session.ts's former `_routeSimpleEvent`. */
function _dispatchSimpleEvent(data: WsPushEvent): boolean {
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
    default:
      return false;
  }
}

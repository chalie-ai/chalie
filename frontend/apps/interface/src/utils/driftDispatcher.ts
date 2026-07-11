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
import { ConfigType } from '@chalie/shared';
import type { WsPushEvent, WsTurnExecutionEvent } from '@chalie/shared';
import { conversation as convoApi } from '../api/conversation';
import { showToast } from './toast';
import { clearLiveTurn, finishLiveTool, startLiveTool } from './liveActTrail';
import { reconcileCancelledTurn } from './cancelReconcile';
import { setTurnDone, setTurnWorking, upsertTurnToSurfaces } from './turnDom';
import { useNotificationsStore } from '../stores/notifications';
import type { TipState, UpdateState } from '../stores/notifications';
import { useTasksStore } from '../stores/tasks';
import { usePermissionsStore } from '../stores/permissions';

/** The session-owned side effects a turn_execution frame still needs — see
 *  module docblock for why this is dependency-injected rather than a static
 *  import of `stores/session.ts`. */
export interface SessionHooks {
  /** Release a send's POST-scoped busy hold — called for EVERY execution frame. */
  releasePendingSend(turnId: number, type: string): void;
  /** turn_id currently open in the slide-over panel, or null (D16 gate). */
  getPanelThreadId(): number | null;
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
 */
function _dispatchToolCall(data: WsPushEvent): boolean {
  if (!isToolCallEvent(data)) return false;
  const turnId = (data as { turn_id?: number | null }).turn_id ?? null;
  if (turnId == null) return true;
  const type = (data as { type?: string }).type ?? ConfigType.USER;
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
 * turn stays in flight, no error bubble). Returns true when handled.
 */
function _dispatchTurnSignal(data: WsPushEvent): boolean {
  if (!isTurnSignal(data)) return false;
  const status = (data as { status?: string }).status;
  const type = (data as { type?: string }).type ?? ConfigType.USER;
  const turnId = (data as { turn_id?: number | null }).turn_id ?? null;
  switch (status) {
    case 'updated':
      if (turnId != null) void _refetchAndUpsert(turnId, type);
      return true;
    case 'provider_retry':
      showToast((data as { message?: string }).message ?? 'The AI provider had a problem — retrying…');
      return true;
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
 */
function _dispatchTurnExecution(data: WsPushEvent): boolean {
  if (!isTurnExecutionEvent(data)) return false;
  const exec = data as unknown as WsTurnExecutionEvent;
  const type = exec.type ?? ConfigType.USER;
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
  if (exec.turn_id !== h.getPanelThreadId()) {
    setTurnDone(exec.turn_id, type, true);
  }
  void h.finishTurn(exec.turn_id, type);
  return true;
}

/** Fetch a turn block once and fan it into every accepting surface. */
async function _refetchAndUpsert(
  turnId: number,
  type: string,
  options: { force?: boolean } = {},
): Promise<void> {
  const block = await convoApi.thread(turnId, type);
  upsertTurnToSurfaces(block, type, options);
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
    case 'quick_tip':
      useNotificationsStore().handleTip(data as unknown as TipState);
      return true;
    default:
      return false;
  }
}

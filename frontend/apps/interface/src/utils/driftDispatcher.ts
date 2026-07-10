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
 * Interim seam: lane lifecycle (single-flight bookkeeping, queue draining,
 * background notifications) still lives on the session store — `_finishTurn`
 * and `_handleCancelled` are called here for THAT side effect only. Nothing
 * renders from what they write any more; that buffer write dies in a later
 * slice once lanes/sendMessage move off it too.
 */
import { ConfigType } from '@chalie/shared';
import type { WsPushEvent, WsTurnExecutionEvent } from '@chalie/shared';
import { useSessionStore } from '../stores/session';
import { conversation as convoApi } from '../api/conversation';
import { showToast } from './toast';
import { clearLiveTurn, finishLiveTool, startLiveTool } from './liveActTrail';
import { useConversationFeed } from '../composables/useConversationFeed';
import { removeTurn, setTurnDone, setTurnWorking, upsertTurnToSurfaces } from './turnDom';
import { useNotificationsStore } from '../stores/notifications';
import type { TipState, UpdateState } from '../stores/notifications';
import { useTasksStore } from '../stores/tasks';
import { usePermissionsStore } from '../stores/permissions';

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
 * hands lane bookkeeping to session._finishTurn. `crashed` does the same but
 * additionally raises the 'turn failed' toast. `cancelled` is its own branch
 * (see `_dispatchCancelled`) — a normal settle would leave a fully-rendered,
 * discarded response looking exactly like a completed one, when the whole
 * point of cancel is that it never counted.
 */
function _dispatchTurnExecution(data: WsPushEvent): boolean {
  if (!isTurnExecutionEvent(data)) return false;
  const exec = data as unknown as WsTurnExecutionEvent;
  const type = exec.type ?? ConfigType.USER;
  const session = useSessionStore();

  // ANY execution frame proves the send's turn is now tracked by its own
  // working/settle signals — release the POST-scoped busy hold (see
  // session.sendMessage's `_pendingByTurn` comment).
  session._releasePendingSend(exec.turn_id, type);

  if (exec.state === 'working') {
    setTurnWorking(exec.turn_id, type, true);
    // Interim seam: tasks.ts / SchedulerDock still read the buffer's working
    // flag for the Activity dock's live phase; the settle paths (_finishTurn /
    // _handleCancelled) clear it. Dies with the buffer in a later slice.
    useConversationFeed(type).setWorking(exec.turn_id, true);
    void _refetchAndUpsert(exec.turn_id, type);
    return true;
  }

  if (exec.state === 'cancelled') {
    void _dispatchCancelled(exec.turn_id, type);
    return true;
  }

  setTurnWorking(exec.turn_id, type, false);
  clearLiveTurn(type, exec.turn_id);
  if (exec.state === 'crashed') {
    session.errorMessage = 'Turn failed unexpectedly';
  }
  if (exec.turn_id !== session.panelThreadId) {
    setTurnDone(exec.turn_id, type, true);
  }
  void session._finishTurn(exec.turn_id, type);
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
 * Cancelled turn — the DELETE-triggered terminal state. The backend's
 * `serialize_turn` strips a cancelled turn's trailing, reply-less user row(s)
 * before this refetch ever runs (real content a turn already wrote before the
 * cancel landed always stays), so re-fetching now IS the authoritative
 * post-cancel state. Upserted with `force: true` (D13) — this is the one case
 * where the freshest read can legitimately be SMALLER than what's already
 * rendered (a fork turn's earlier-settled content already stamped a higher
 * version; the now-stripped orphan reply never raises it), and the version
 * guard would otherwise reject the write, leaving a phantom orphan bubble.
 * Empty messages → the turn never happened from the surface's perspective,
 * removed from every surface exactly like an aborted send. `session.
 * _handleCancelled` runs alongside for lane/buffer bookkeeping (the `requestStop`
 * reconcile calls it too — idempotent either order).
 */
async function _dispatchCancelled(turnId: number, type: string): Promise<void> {
  const session = useSessionStore();
  void session._handleCancelled(turnId, type);
  // A cancel is terminal regardless of what the refetch returns (or whether
  // it fails), so clear working FIRST — this frame can originate from
  // another tab/device (any WS push, not just this client's stop button),
  // and clearing only on some branches would leave a live-working record
  // behind that permanently gates the thread's sends.
  setTurnWorking(turnId, type, false);
  try {
    const block = await convoApi.thread(turnId, type);
    if (block.messages.length) {
      upsertTurnToSurfaces(block, type, { force: true });
    } else {
      removeTurn(turnId, type);
    }
  } catch {
    // Best-effort — leave whatever is rendered; a later refetch (panel open,
    // page refresh) reconciles against the DB regardless.
  }
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

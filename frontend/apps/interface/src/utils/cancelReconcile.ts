/**
 * Cancel reconcile — the ONE shared fetch for a cancelled turn's post-cancel
 * DOM state, deduped across its two triggers: `session.requestStop`'s
 * post-interrupt reconcile (the user's own stop button) and the WS
 * 'cancelled' frame (`driftDispatcher`'s turn_execution branch — the same
 * cancel confirmed server-side, or a cancel initiated from another
 * tab/device). Both call `reconcileCancelledTurn` for the same `type:turnId`
 * key within the same round-trip window; an in-flight promise cache collapses
 * whichever call lands second onto the first's already-running fetch instead
 * of firing a second one.
 */
import { conversation as convoApi } from '../api/conversation';
import { clearLiveTurn } from './liveActTrail';
import { removeTurn, setTurnWorking, upsertTurnToSurfaces } from './turnDom';

const _inFlight = new Map<string, Promise<void>>();

function key(turnId: number, type: string): string {
  return `${type}:${turnId}`;
}

/**
 * Reconcile a cancelled turn against the backend's authoritative post-cancel
 * content. The backend's `serialize_turn` strips a cancelled turn's
 * trailing, reply-less user row(s) before this refetch ever runs (real
 * content a turn already wrote before the cancel landed always stays), so
 * re-fetching now IS the authoritative post-cancel state. Upserted with
 * `force: true` (D13) — this is the one case where the freshest read can
 * legitimately be SMALLER than what's already rendered (a fork turn's
 * earlier-settled content already stamped a higher version; the
 * now-stripped orphan reply never raises it), and the version guard would
 * otherwise reject the write, leaving a phantom orphan bubble. Empty
 * messages → the turn never happened from the surface's perspective,
 * removed from every surface exactly like an aborted send.
 *
 * `working`/live-trail state is cleared FIRST, synchronously, ahead of the
 * fetch — a cancel is terminal regardless of what the refetch returns (or
 * whether it fails), and this frame can originate from another tab/device
 * (any WS push, not just this client's stop button), so clearing only on
 * some branches would leave a live-working record behind that permanently
 * gates the thread's sends.
 */
export function reconcileCancelledTurn(turnId: number, type: string): Promise<void> {
  const k = key(turnId, type);
  const existing = _inFlight.get(k);
  if (existing) return existing;

  setTurnWorking(turnId, type, false);
  clearLiveTurn(type, turnId);

  const promise = _reconcile(turnId, type).finally(() => {
    _inFlight.delete(k);
  });
  _inFlight.set(k, promise);
  return promise;
}

async function _reconcile(turnId: number, type: string): Promise<void> {
  try {
    const block = await convoApi.thread(turnId, type);
    if (block.type !== type) {
      console.warn(
        '[cancelReconcile] refetched turn', turnId, 'came back as type', block.type,
        'but was requested as', type, '— trusting the fetched type',
      );
    }
    if (block.messages.length) {
      upsertTurnToSurfaces(block, block.type, { force: true });
    } else {
      removeTurn(turnId, block.type);
    }
  } catch {
    // Best-effort — leave whatever is rendered; a later refetch (panel open,
    // page refresh) reconciles against the DB regardless.
  }
}

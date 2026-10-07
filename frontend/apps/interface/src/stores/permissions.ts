/**
 * Permissions store — the user-review queue of pending permission gates.
 *
 * Two feeds, one queue: the drift dispatcher routes every WS
 * `permission_request` frame here via `enqueue(data)` and every
 * `permission_resolved` frame via `remove(id)`; the session store calls
 * `refreshPending()` on every WS connect so a reload or a dropped socket
 * restores the asks the frames would have queued and drops the ones whose
 * resolution was missed (the backend thread keeps waiting on the gate
 * regardless — the frame is only the visual trigger).
 * `enqueue` dedupes on `request_id`, so the live frame and the REST listing
 * overlapping is harmless — and drops any frame without a well-formed
 * origin, loudly: the backend denies an originless ask before parking it,
 * so such a frame is malformed and must not surface as an ask nobody can
 * open.
 *
 * An ask is shown and answered on the live activity line of its origin's
 * turn. Each mounted live line calls `markShown(requestId)` on mount and
 * `unmarkShown(requestId)` on unmount — a count, since the same turn can
 * render twice, on the main chat and in the thread panel — and the input
 * dock lists `offScreen`: the queued asks no mounted line is showing (a
 * scheduled task, or a thread whose panel is closed).
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import { policies } from '../api';
import type { PendingPermission, PermissionOrigin } from '../api/policies';

export type { PermissionOrigin } from '../api/policies';

/** One queued ask — the `permission_request` frame / pending listing item as kept here. */
export interface PermissionRequest {
  /** Opaque ID; resolves the gate on /api/policies/respond. */
  request_id: string;
  /** Permission key the policy gated, e.g. "pim" or "email.send". */
  action_id: string;
  /** The model's one-line summary of the gated action; empty when the backend sent none. */
  summary: string;
  /** The instant the ask parked (ISO-8601 UTC with offset); empty when the backend sent none. */
  asked_at: string;
  /** The turn the gate belongs to — its live activity line shows and answers the ask. */
  origin: PermissionOrigin;
}

/** The `origin` object as sent, or null when absent/malformed — never a guessed one. */
function readOrigin(raw: unknown): PermissionOrigin | null {
  if (raw == null || typeof raw !== 'object') return null;
  const o = raw as Partial<PermissionOrigin>;
  if (typeof o.type !== 'string' || typeof o.turn_id !== 'number') return null;
  return { type: o.type, turn_id: o.turn_id, forked: o.forked === true };
}

export const usePermissionsStore = defineStore('permissions', {
  state: () => ({
    queue: [] as PermissionRequest[],
    /** request_id → how many mounted live lines show that ask. */
    shown: {} as Record<string, number>,
  }),

  getters: {
    /** The queued asks no mounted live line is showing, in queue order. */
    offScreen(state): PermissionRequest[] {
      return state.queue.filter((r) => (state.shown[r.request_id] ?? 0) <= 0);
    },
  },

  actions: {
    /**
     * Enqueue a pending gate for user review — from a live `permission_request`
     * frame or a `GET /api/policies/pending` item (same fields). Silently drops
     * entries missing `request_id`/`action_id` or already queued; drops one
     * without a well-formed origin loudly.
     */
    enqueue(data: WsPushEvent | PendingPermission): void {
      const payload = data as Partial<PendingPermission>;
      if (!payload.request_id || !payload.action_id) return;

      if (this.queue.some((r) => r.request_id === payload.request_id)) return;

      const origin = readOrigin(payload.origin);
      if (origin == null) {
        console.error('[permissions] ask without an origin dropped:', payload.request_id);
        return;
      }

      this.queue.push({
        request_id: payload.request_id,
        action_id: payload.action_id,
        summary: typeof payload.summary === 'string' ? payload.summary : '',
        asked_at: typeof payload.asked_at === 'string' ? payload.asked_at : '',
        origin,
      });
    },

    /** A live line showing this ask mounted. */
    markShown(requestId: string): void {
      this.shown[requestId] = (this.shown[requestId] ?? 0) + 1;
    },

    /** A live line showing this ask unmounted. */
    unmarkShown(requestId: string): void {
      const n = (this.shown[requestId] ?? 0) - 1;
      if (n <= 0) delete this.shown[requestId];
      else this.shown[requestId] = n;
    },

    /** Drop an ask — the gate was answered, cancelled, or failed elsewhere (a
     *  `permission_resolved` frame, or another tab's answer). No-op if absent. */
    remove(requestId: string): void {
      this.queue = this.queue.filter((r) => r.request_id !== requestId);
    },

    /**
     * Reconcile the queue with the pending gates over REST — called on every
     * WS connect (initial load and reconnect) and after a failed `respond`.
     * An ask queued before the fetch that the listing no longer has is dropped
     * (its `permission_resolved` frame was missed while the socket was down);
     * a frame that arrives while the fetch is in flight is not in that
     * snapshot, so it is never pruned; the listing's gates are enqueued.
     * Best-effort: a failed fetch leaves the queue as it is; the next connect
     * retries.
     */
    async refreshPending(): Promise<void> {
      const queuedBefore = new Set(this.queue.map((r) => r.request_id));
      let pending: PendingPermission[];
      try {
        pending = await policies.pending();
      } catch (err) {
        console.warn('[permissions] pending gates fetch failed:', err);
        return;
      }
      const listed = new Set(pending.map((item) => item.request_id));
      this.queue = this.queue.filter(
        (r) => listed.has(r.request_id) || !queuedBefore.has(r.request_id),
      );
      for (const item of pending) this.enqueue(item);
    },

    /**
     * Optimistic removal: dismiss the ask before the network round-trip. If
     * the POST fails the gate is still parked on the backend, so the listing is
     * re-read to bring the ask back — it only returns gates that are still
     * open, so one answered elsewhere in the meantime stays gone.
     */
    async respond(requestId: string, approved: boolean): Promise<void> {
      this.remove(requestId);

      try {
        await policies.respond({ request_id: requestId, approved });
      } catch (err) {
        console.warn('[permissions] respond failed:', err);
        void this.refreshPending();
      }
    },
  },
});

/**
 * Permissions store — the user-review queue of pending permission gates.
 *
 * Two feeds, one queue: the drift dispatcher routes every WS
 * `permission_request` frame here via `enqueue(data)` and every
 * `permission_resolved` frame via `remove(id)`; the session store calls
 * `refreshPending()` on every WS connect so a reload or a dropped socket
 * restores the cards the frames would have painted (the backend thread keeps
 * waiting on the gate regardless — the frame is only the visual trigger).
 * `enqueue` dedupes on `request_id`, so the live frame and the REST listing
 * overlapping is harmless. PermissionStack.vue renders the queue and calls
 * `respond()` when the user taps Allow or Deny.
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import { policies } from '../api';
import type { PendingPermission, PermissionOrigin } from '../api/policies';

export type { PermissionOrigin } from '../api/policies';

/** One queued card — the `permission_request` frame / pending listing item as kept here. */
export interface PermissionRequest {
  /** Opaque ID; resolves the gate on /api/policies/respond. */
  request_id: string;
  /** Permission key the policy gated, e.g. "pim" or "email.send". */
  action_id: string;
  /** The model's one-line summary of the gated action; empty when the backend sent none. */
  summary: string;
  /** The turn the gate belongs to — decides the lane the card renders in. Null
   *  only for a frame that arrived without one (routed to the spine stack). */
  origin: PermissionOrigin | null;
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
  }),

  actions: {
    /**
     * Enqueue a pending gate for user review — from a live `permission_request`
     * frame or a `GET /api/policies/pending` item (same fields). Silently drops
     * entries missing `request_id`/`action_id` or already queued.
     */
    enqueue(data: WsPushEvent | PendingPermission): void {
      const payload = data as Partial<PendingPermission>;
      if (!payload.request_id || !payload.action_id) return;

      if (this.queue.some((r) => r.request_id === payload.request_id)) return;

      this.queue.push({
        request_id: payload.request_id,
        action_id: payload.action_id,
        summary: typeof payload.summary === 'string' ? payload.summary : '',
        origin: readOrigin(payload.origin),
      });
    },

    /** Drop a card — the gate was answered, cancelled, or failed elsewhere (a
     *  `permission_resolved` frame, or another tab's answer). No-op if absent. */
    remove(requestId: string): void {
      this.queue = this.queue.filter((r) => r.request_id !== requestId);
    },

    /**
     * Re-read the pending gates over REST and enqueue whatever is missing —
     * called on every WS connect (initial load and reconnect). Best-effort:
     * a failed fetch leaves the queue as it is; the next connect retries.
     */
    async refreshPending(): Promise<void> {
      let pending: PendingPermission[];
      try {
        pending = await policies.pending();
      } catch (err) {
        console.warn('[permissions] pending gates fetch failed:', err);
        return;
      }
      for (const item of pending) this.enqueue(item);
    },

    /** Optimistic removal: dismiss the card before the network round-trip. */
    async respond(requestId: string, approved: boolean): Promise<void> {
      this.remove(requestId);

      try {
        await policies.respond({ request_id: requestId, approved });
      } catch (err) {
        // Don't re-surface the card — the user already decided; if the gate is
        // in fact still open, the next connect's refreshPending brings it back.
        console.warn('[permissions] respond failed:', err);
      }
    },
  },
});

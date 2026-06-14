/**
 * Permissions store — permission_request WS push events → user-review queue.
 *
 * The session store routes every WS `permission_request` frame here via
 * `enqueue(data)`. PermissionStack.vue renders the queue and calls `respond()`
 * when the user taps Allow or Deny.
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import { policies } from '../api';

/** Shape of a single permission_request WS frame. */
export interface PermissionRequest {
  /** Backend-assigned opaque ID; used to resolve the gate on /api/policies/respond. */
  request_id: string;
  /** Action/capability key, e.g. "email.send", "browser.render". */
  action_id: string;
  /** Optional one-line human-readable description of what Chalie wants to do. */
  description?: string;
  /** Originating skill name, if provided by the backend. */
  skill?: string;
  /** Channel context (e.g. "scheduled", "user"). */
  channel?: string;
}

export const usePermissionsStore = defineStore('permissions', {
  state: () => ({
    queue: [] as PermissionRequest[],
  }),

  actions: {
    /**
     * Enqueue a `permission_request` push event for user review.
     * Called by the session/drift controller — do NOT call ws.* here.
     * Silently drops frames missing `request_id` or already in the queue.
     */
    enqueue(data: WsPushEvent): void {
      const payload = data as unknown as PermissionRequest;
      if (!payload.request_id || !payload.action_id) return;

      // Deduplicate — if already queued, ignore the duplicate frame.
      if (this.queue.some((r) => r.request_id === payload.request_id)) return;

      this.queue.push({
        request_id: payload.request_id,
        action_id: payload.action_id,
        description: payload.description,
        skill: payload.skill,
        channel: payload.channel,
      });
    },

    /**
     * Respond to a queued permission gate and remove the card immediately
     * (optimistic removal — matches legacy behaviour where the card dismisses
     * on click before the network round-trip completes).
     */
    async respond(requestId: string, approved: boolean): Promise<void> {
      // Optimistic removal: dismiss the card right away so the UI feels instant.
      this.queue = this.queue.filter((r) => r.request_id !== requestId);

      try {
        await policies.respond({ request_id: requestId, approved });
      } catch (err) {
        // Log but don't re-surface the card — the backend has a 1-hour safety
        // net and the user has already made their decision.
        console.warn('[permissions] respond failed:', err);
      }
    },
  },
});

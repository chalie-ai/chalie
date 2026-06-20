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
  /** Opaque ID; resolves the gate on /api/policies/respond. */
  request_id: string;
  /** Action/capability key, e.g. "email.send". */
  action_id: string;
  description?: string;
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
     * Enqueue a `permission_request` for user review. Silently drops frames
     * missing `request_id`/`action_id` or already queued.
     */
    enqueue(data: WsPushEvent): void {
      const payload = data as unknown as PermissionRequest;
      if (!payload.request_id || !payload.action_id) return;

      if (this.queue.some((r) => r.request_id === payload.request_id)) return;

      this.queue.push({
        request_id: payload.request_id,
        action_id: payload.action_id,
        description: payload.description,
        skill: payload.skill,
        channel: payload.channel,
      });
    },

    /** Optimistic removal: dismiss the card before the network round-trip. */
    async respond(requestId: string, approved: boolean): Promise<void> {
      this.queue = this.queue.filter((r) => r.request_id !== requestId);

      try {
        await policies.respond({ request_id: requestId, approved });
      } catch (err) {
        // Don't re-surface the card — backend has a 1-hour safety net and the
        // user already decided.
        console.warn('[permissions] respond failed:', err);
      }
    },
  },
});

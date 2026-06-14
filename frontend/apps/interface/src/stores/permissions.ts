/**
 * Permissions store — minimal stub for drift event routing.
 * P1c: fleshed out in Task C7 (permission-request notification UI).
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';

export interface PermissionRequest {
  id?: string;
  [key: string]: unknown;
}

export const usePermissionsStore = defineStore('permissions', {
  state: () => ({
    queue: [] as PermissionRequest[],
  }),
  actions: {
    /**
     * Enqueue a permission_request drift event for user review.
     * P1c: fleshed out in Task C7.
     */
    enqueue(data: WsPushEvent): void {
      // Minimal: push raw event. P1c: surface in-app permission dialog.
      this.queue.push(data as PermissionRequest);
    },
  },
});

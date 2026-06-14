/**
 * Notifications store — minimal stub for drift event routing.
 * P1c: fleshed out in Task C10 (scheduler notification strip).
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';

export interface NotificationItem {
  id?: string;
  [key: string]: unknown;
}

export const useNotificationsStore = defineStore('notifications', {
  state: () => ({
    notifications: [] as NotificationItem[],
  }),
  actions: {
    /**
     * Push a background (tab-unfocused) text notification.
     * P1c: fleshed out in Task C10.
     */
    pushBackground(text: string): void {
      // Minimal: log for now. P1c: show system notification / play sound.
      if (typeof text === 'string' && text.length > 0) {
        // no-op until P1c
      }
    },

    /**
     * Handle a 'notification' drift event.
     * P1c: fleshed out in Task C10.
     */
    applyDriftEvent(data: WsPushEvent): void {
      // Minimal: store raw event. P1c: refresh task strip, play sound.
      void data;
    },
  },
});

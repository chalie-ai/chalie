/**
 * Tasks store — the Activity drawer's open/close state only.
 *
 * The drawer's content is NOT held here. Its thread rows are DOM-derived —
 * see `utils/threadActivity.ts`'s `useThreadActivity` — rather than a store
 * getter: a Pinia getter/Vue computed can't reactively track a raw
 * `querySelectorAll` read. Its agent rows come from the agents store. The
 * two are combined at the component level (PresenceBar.vue, TaskDrawer.vue).
 */
import { defineStore } from 'pinia';

export const useTasksStore = defineStore('tasks', {
  state: () => ({
    /** The Activity drawer's open/closed state. */
    isOpen: false,
  }),

  actions: {
    open(): void {
      this.isOpen = true;
    },

    close(): void {
      this.isOpen = false;
    },
  },
});

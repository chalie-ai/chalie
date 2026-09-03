/**
 * Tasks store — the Activity drawer's open/close state only.
 *
 * The drawer's content is NOT held here: forked-thread activity (the drawer's
 * row family) is DOM-derived — see `utils/threadActivity.ts`'s
 * `useThreadActivity` — rather than a store getter: a Pinia getter/Vue
 * computed can't reactively track a raw `querySelectorAll` read, so
 * `totalCount` is combined at the component level (PresenceBar.vue,
 * TaskDrawer.vue) instead.
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

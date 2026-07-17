/**
 * Tasks store — active delegates (GET /api/subagents/all, each a full
 * snapshot an Activity row renders). Live updates arrive via
 * applyDriftEvent() from the session store. Recurring prompt-schedules live
 * exclusively in the scheduler dock, not here.
 *
 * Forked-thread Activity (the drawer's other row family) is DOM-derived —
 * see `utils/threadActivity.ts`'s `useThreadActivity` — rather than a store
 * getter: a Pinia getter/Vue computed can't reactively track a raw
 * `querySelectorAll` read, so `totalCount` (subagents + forked threads) is
 * combined at the component level (PresenceBar.vue, TaskDrawer.vue) instead.
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import type { ActiveSubagent } from '../api/scheduler';
import { scheduler } from '../api/scheduler';

export interface DriftTask {
  id?: string;
  [key: string]: unknown;
}

export const useTasksStore = defineStore('tasks', {
  state: () => ({
    /** Active delegates keyed by sub_id, each a full ActiveSubagent snapshot. */
    subagents: new Map<string, ActiveSubagent>() as Map<string, ActiveSubagent>,
    isOpen: false,
  }),

  actions: {
    open(): void {
      this.isOpen = true;
    },

    close(): void {
      this.isOpen = false;
    },

    /** Refresh the active-delegate snapshot (GET /api/subagents/all); leaves prior state intact on failure. */
    async loadActiveTasks(): Promise<void> {
      let result;
      try {
        result = await scheduler.subagentsActive();
      } catch {
        return;
      }

      const fresh = new Map<string, ActiveSubagent>();
      for (const sa of result.subagents ?? []) {
        fresh.set(sa.sub_id, sa);
      }
      for (const [id, sa] of this.subagents) {
        if (!fresh.has(id)) {
          fresh.set(id, sa);
        }
      }
      this.subagents = fresh;
    },

    /**
     * Handle a task-category drift event.
     */
    applyDriftEvent(data: WsPushEvent): void {
      const type = data.type as string;

      if (type === 'subagent_start') {
        const sa = data as unknown as ActiveSubagent;
        if (sa.sub_id) {
          this.subagents.set(sa.sub_id, {
            sub_id: sa.sub_id,
            tool_name: sa.tool_name,
            summary: sa.summary ?? null,
            started_at: sa.started_at,
          });
        }
        return;
      }

      if (type === 'subagent_end') {
        const sub_id = (data as { sub_id?: string }).sub_id;
        if (sub_id) {
          this.subagents.delete(sub_id);
        }
      }
    },
  },
});

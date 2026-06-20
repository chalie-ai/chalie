/**
 * Tasks store — reminders (GET /scheduler?status=pending, filtered to items with
 * a due_at) + active subagents (GET /chat/subagents/active, only { sub_id } per
 * entry). Live updates arrive via applyDriftEvent() from the session store.
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import { scheduler } from '../api/scheduler';
import type { ScheduledItem, ActiveSubagent } from '../api/scheduler';

export type { ScheduledItem };

export interface DriftTask {
  id?: string;
  [key: string]: unknown;
}

export const useTasksStore = defineStore('tasks', {
  state: () => ({
    /** Pending reminders with a due_at, cached from the last poll. */
    reminders: [] as ScheduledItem[],
    /** Active subagents keyed by sub_id (backend returns only { sub_id }). */
    subagents: new Map<string, ActiveSubagent>() as Map<string, ActiveSubagent>,
    isOpen: false,
  }),

  getters: {
    totalCount(state): number {
      return state.reminders.length + state.subagents.size;
    },
  },

  actions: {
    open(): void {
      this.isOpen = true;
    },

    close(): void {
      this.isOpen = false;
    },

    /**
     * Fetch reminders and subagents in parallel. On partial failure the
     * successful half still updates its slice of state.
     */
    async loadActiveTasks(): Promise<void> {
      const [schedResult, subagentResult] = await Promise.allSettled([
        scheduler.pending(),
        scheduler.subagentsActive(),
      ]);

      if (schedResult.status === 'fulfilled') {
        this.reminders = (schedResult.value.items ?? []).filter(
          (r) => r.status === 'pending' && r.due_at != null,
        );
      }

      if (subagentResult.status === 'fulfilled') {
        // Rebuild from the server snapshot, but keep WS-received entries the
        // server hasn't listed yet — they may have started after the request
        // was dispatched.
        const fresh = new Map<string, ActiveSubagent>();
        for (const sa of subagentResult.value.subagents ?? []) {
          fresh.set(sa.sub_id, sa);
        }
        for (const [id, sa] of this.subagents) {
          if (!fresh.has(id)) {
            fresh.set(id, sa);
          }
        }
        this.subagents = fresh;
      }
    },

    /**
     * Handle a task-category drift event. task/reminder refetch reminders (server
     * is the scheduling source of truth); subagent_start/_end add/remove the
     * sub_id from the active map.
     */
    applyDriftEvent(data: WsPushEvent): void {
      const type = data.type as string;

      if (type === 'task' || type === 'reminder') {
        // Fire-and-forget refresh; leave existing reminders intact on error.
        void scheduler.pending().then((res) => {
          this.reminders = (res.items ?? []).filter(
            (r) => r.status === 'pending' && r.due_at != null,
          );
        }).catch(() => {
        });
        return;
      }

      if (type === 'subagent_start') {
        const sub_id = (data as { sub_id?: string }).sub_id;
        if (sub_id) {
          this.subagents.set(sub_id, { sub_id });
        }
        return;
      }

      if (type === 'subagent_end') {
        const sub_id = (data as { sub_id?: string }).sub_id;
        if (sub_id) {
          this.subagents.delete(sub_id);
        }
        return;
      }
    },
  },
});

/**
 * Tasks store — reminders + active subagents.
 *
 * Reminders are fetched from GET /scheduler?status=pending (filtered to items
 * with a due_at).  Active subagents come from GET /chat/subagents/active which
 * returns only { sub_id } per entry.
 *
 * Live updates arrive via applyDriftEvent(), called by the session store
 * whenever a task/reminder/subagent_start/subagent_end push event arrives.
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
    /**
     * Active subagents keyed by sub_id.
     * The backend only returns { sub_id } — no agent_type or description.
     */
    subagents: new Map<string, ActiveSubagent>() as Map<string, ActiveSubagent>,
    /** Whether the task drawer panel is currently open. */
    isOpen: false,
  }),

  getters: {
    /** Total number of visible items (reminders + subagents). */
    totalCount(state): number {
      return state.reminders.length + state.subagents.size;
    },
  },

  actions: {
    /** Open the task drawer panel. */
    open(): void {
      this.isOpen = true;
    },

    /** Close the task drawer panel. */
    close(): void {
      this.isOpen = false;
    },

    /**
     * Fetch reminders and active subagents in parallel, then update state.
     * On partial failure, the successful half still updates its slice of state.
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
        // Replace the full subagents map from the server snapshot.
        // Preserves entries that appeared via WS but aren't in the server list
        // because they may have started after the request was dispatched.
        const fresh = new Map<string, ActiveSubagent>();
        for (const sa of subagentResult.value.subagents ?? []) {
          fresh.set(sa.sub_id, sa);
        }
        // Merge: keep existing WS-received entries not yet cleared by the server.
        for (const [id, sa] of this.subagents) {
          if (!fresh.has(id)) {
            fresh.set(id, sa);
          }
        }
        this.subagents = fresh;
      }
    },

    /**
     * Handle a task-category drift event from the WS push channel.
     *
     * Types handled:
     *   task / reminder  — reload reminders (the server is the source of truth for
     *                      scheduling changes; we refetch rather than mutate in place).
     *   subagent_start   — add the sub_id to the active map.
     *   subagent_end     — remove the sub_id from the active map.
     */
    applyDriftEvent(data: WsPushEvent): void {
      const type = data.type as string;

      if (type === 'task' || type === 'reminder') {
        // A scheduling change occurred; refresh the reminders list.
        // We intentionally do not await — fire-and-forget update.
        void scheduler.pending().then((res) => {
          this.reminders = (res.items ?? []).filter(
            (r) => r.status === 'pending' && r.due_at != null,
          );
        }).catch(() => {
          // Leave existing reminders intact on transient errors.
        });
        return;
      }

      if (type === 'subagent_start') {
        const sub_id = (data as { sub_id?: string }).sub_id;
        if (sub_id) {
          // Backend only guarantees sub_id; store exactly that.
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

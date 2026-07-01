/**
 * Tasks store — reminders (GET /scheduler?status=pending, filtered to items with
 * a due_at) + active delegates (GET /api/subagents, each a full snapshot
 * an Activity row renders). Live updates arrive via applyDriftEvent() from the
 * session store.
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import type { ActiveSubagent, ScheduledItem } from '../api/scheduler';
import { scheduler } from '../api/scheduler';
import { useConversationFeed } from '../composables/useConversationFeed';

export type { ScheduledItem };

export interface DriftTask {
  id?: string;
  [key: string]: unknown;
}

/** A forked thread surfaced in the Activity drawer: its reply is streaming right
 *  now (`working`, pink) or has settled unseen (`done`, blue). */
export interface ThreadActivityItem {
  turn_id: number;
  label: string;
  snippet: string;
  kind: 'working' | 'done';
}

export const useTasksStore = defineStore('tasks', {
  state: () => ({
    /** Pending reminders with a due_at, cached from the last poll. */
    reminders: [] as ScheduledItem[],
    /** Active delegates keyed by sub_id, each a full ActiveSubagent snapshot. */
    subagents: new Map<string, ActiveSubagent>() as Map<string, ActiveSubagent>,
    isOpen: false,
  }),

  getters: {
    /** Forked threads with live Activity as dock rows — most recently active first. */
    threadActivity(): ThreadActivityItem[] {
      const convo = useConversationFeed();
      return convo.threadList()
        .filter(
          (t) =>
            t.turn_id != null &&
            convo.isForkedThread(t.turn_id) &&
            convo.threadPhase(t.turn_id) != null,
        )
        .sort((a, b) => (b.last_activity_at ?? '').localeCompare(a.last_activity_at ?? ''))
        .map((t) => ({
          turn_id: t.turn_id as number,
          label: t.gist || t.preview,
          snippet: t.preview,
          kind: convo.threadPhase(t.turn_id as number) as ThreadActivityItem['kind'],
        }));
    },

    totalCount(state): number {
      return state.reminders.length + state.subagents.size + this.threadActivity.length;
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
        this.reminders = schedResult.value.filter(
          (r) => r.status === 'pending' && r.due_at != null,
        );
      }

      if (subagentResult.status === 'fulfilled') {
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
     * Handle a task-category drift event.
     */
    applyDriftEvent(data: WsPushEvent): void {
      const type = data.type as string;

      if (type === 'task' || type === 'reminder') {
        void scheduler.pending().then((res) => {
          this.reminders = res.filter(
            (r) => r.status === 'pending' && r.due_at != null,
          );
        }).catch(() => {});
        return;
      }

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
        return;
      }
    },
  },
});

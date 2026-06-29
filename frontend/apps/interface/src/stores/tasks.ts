/**
 * Tasks store — reminders (GET /scheduler?status=pending, filtered to items with
 * a due_at) + active delegates (GET /chat/subagents/active, each a full snapshot
 * an Activity row renders). Live updates arrive via applyDriftEvent() from the
 * session store.
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import { scheduler } from '../api/scheduler';
import type { ScheduledItem, ActiveSubagent } from '../api/scheduler';
import { useConversationStore } from './conversation';

export type { ScheduledItem };

export interface DriftTask {
  id?: string;
  [key: string]: unknown;
}

/** A live thread surfaced in the Activity drawer: an unread reply (`new`) or a
 *  thread whose reply is streaming right now (`thinking`). */
export interface ThreadActivityItem {
  turn_id: number;
  label: string;
  snippet: string;
  kind: 'new' | 'thinking';
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
    /** Live/unread threads as Activity rows — most recently active first; threads
     *  with no turn_id (not yet bound) are skipped. The `thinking` pulse is the
     *  real per-thread streaming signal (`isTurnWorking`, driven by the turn's
     *  `working`/`done` WS signals), not a time window — it lights only while a
     *  reply is actually in flight. `.filter` yields a fresh array, so the
     *  `last_activity_at` sort never mutates the store. */
    threadActivity(): ThreadActivityItem[] {
      const convo = useConversationStore();
      return convo.threads
        .filter(
          (t) =>
            t.turn_id != null &&
            ((t.unread ?? 0) > 0 || convo.isTurnWorking(t.turn_id)),
        )
        .sort((a, b) => (b.last_activity_at ?? '').localeCompare(a.last_activity_at ?? ''))
        .map((t) => ({
          turn_id: t.turn_id as number,
          label: t.gist || t.preview,
          snippet: t.preview,
          kind: ((t.unread ?? 0) > 0 ? 'new' : 'thinking') as ThreadActivityItem['kind'],
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
     * delegate snapshot from the active map.
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

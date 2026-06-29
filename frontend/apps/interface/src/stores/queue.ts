/**
 * Message-queue store — frontend-only deferred sends, scoped per conversation.
 *
 * When a scope (the main spine or a specific thread) is working, a new send is
 * held here instead of interrupting the in-flight turn. Each scope keeps its own
 * ordered list of texts; they render faded at the scope's tail and dispatch as
 * ONE newline-joined message the moment that scope next settles (session store).
 */
import { defineStore } from 'pinia';

/** Scope/lane key: the main spine, or a thread by its turn_id. */
export function laneKey(threadId: number | null): string {
  return threadId == null ? 'main' : `t${threadId}`;
}

export const useQueueStore = defineStore('queue', {
  state: () => ({
    /** Queued texts per scope key, oldest first. */
    byScope: {} as Record<string, string[]>,
  }),

  getters: {
    /** The queued texts for a scope, in send order. */
    queuedFor(state): (threadId: number | null) => string[] {
      return (threadId) => state.byScope[laneKey(threadId)] ?? [];
    },
    /** Scope keys that hold at least one queued message — drives draining. */
    pendingScopes(state): string[] {
      return Object.keys(state.byScope).filter((k) => state.byScope[k].length > 0);
    },
  },

  actions: {
    enqueue(threadId: number | null, text: string): void {
      (this.byScope[laneKey(threadId)] ??= []).push(text);
    },
    removeAt(threadId: number | null, index: number): void {
      this.byScope[laneKey(threadId)]?.splice(index, 1);
    },
    /** Take the whole scope's queue as ONE newline-joined message, clearing it. */
    take(threadId: number | null): string {
      const k = laneKey(threadId);
      const joined = (this.byScope[k] ?? []).join('\n');
      delete this.byScope[k];
      return joined;
    },
  },
});

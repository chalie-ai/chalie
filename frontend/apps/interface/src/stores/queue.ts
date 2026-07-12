/**
 * Message-queue store — frontend-only deferred sends, scoped per conversation.
 *
 * When a scope (the main spine or a specific thread) is working, a new send is
 * held here instead of interrupting the in-flight turn. Each scope keeps its own
 * ordered list of texts; they render faded at the scope's tail and dispatch as
 * ONE newline-joined message the moment that scope next settles (session store).
 */
import { defineStore } from 'pinia';
import { ConfigType } from '@chalie/shared';

/** Scope/lane key: the main spine, or a thread by its turn_id. */
export function laneKey(threadId: number | null): string {
  return threadId == null ? 'main' : `t${threadId}`;
}

/** One deferred send: its text plus any attachments that were pending when the
 *  lane was busy. Files ride alongside the text so a flush replays the whole
 *  original send, not just the words. `thinkingLevel` is the per-thread
 *  thinking level chosen by the user; null means "omit the field downstream". */
export interface QueuedMessage {
  text: string;
  files: File[];
  thinkingLevel: string | null;
}

export const useQueueStore = defineStore('queue', {
  state: () => ({
    /** Queued entries per scope key, oldest first. */
    byScope: {} as Record<string, QueuedMessage[]>,
    /** ConfigType each scope belongs to, so a drain re-sends on the right surface. */
    typeByScope: {} as Record<string, string>,
  }),

  getters: {
    /** The queued entries for a scope, in send order. */
    queuedFor(state): (threadId: number | null) => QueuedMessage[] {
      return (threadId) => state.byScope[laneKey(threadId)] ?? [];
    },
    /** Scope keys that hold at least one queued message — drives draining. */
    pendingScopes(state): string[] {
      return Object.keys(state.byScope).filter((k) => state.byScope[k].length > 0);
    },
  },

  actions: {
    enqueue(threadId: number | null, text: string, type: string = ConfigType.USER, files: File[] = [], thinkingLevel: string | null = null): void {
      const key = laneKey(threadId);
      this.byScope[key] ??= [];
      this.byScope[key].push({ text, files, thinkingLevel });
      this.typeByScope[key] = type;
    },
    removeAt(threadId: number | null, index: number): void {
      this.byScope[laneKey(threadId)]?.splice(index, 1);
    },
    /** The ConfigType a queued scope must drain on (defaults to the user spine). */
    typeFor(threadId: number | null): string {
      return this.typeByScope[laneKey(threadId)] ?? ConfigType.USER;
    },
    /** Take the whole scope's queue, clearing it. Texts join into ONE
     *  newline-joined message (unchanged shape for the send API); every queued
     *  entry's files concatenate, in order, into one attachment batch. The
     *  thinkingLevel is the last queued entry's level (most recent send wins);
     *  null means no level was set on that entry. */
    take(threadId: number | null): { text: string; files: File[]; thinkingLevel: string | null } {
      const k = laneKey(threadId);
      const entries = this.byScope[k] ?? [];
      const text = entries.map((e) => e.text).join('\n');
      const files = entries.flatMap((e) => e.files);
      const thinkingLevel = entries.length > 0 ? entries[entries.length - 1].thinkingLevel : null;
      delete this.byScope[k];
      delete this.typeByScope[k];
      return { text, files, thinkingLevel };
    },
  },
});

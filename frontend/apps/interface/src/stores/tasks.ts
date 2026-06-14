/**
 * Tasks store — minimal stub for drift event routing.
 * P1c: fleshed out in Task C9 (task strip component).
 */
import { defineStore } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';

export interface DriftTask {
  id?: string;
  [key: string]: unknown;
}

export const useTasksStore = defineStore('tasks', {
  state: () => ({
    tasks: [] as DriftTask[],
  }),
  actions: {
    /**
     * Handle a task-category drift event from the WS push channel.
     * P1c: fleshed out in Task C9.
     */
    applyDriftEvent(data: WsPushEvent): void {
      // Minimal: store raw event for future task-strip component.
      // P1c: parse into structured DriftTask, update strip, play sound.
      void data;
    },
  },
});

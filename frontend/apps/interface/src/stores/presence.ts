/**
 * Presence state machine — port of frontend/interface/presence.js.
 * CSS drives all visual animation via [data-state] selectors on the dot element.
 */
import { defineStore } from 'pinia';

const LABELS: Record<string, string> = {
  processing: 'Working...',
  thinking: 'Thinking...',
  narrating: 'Researching...',
  retrieving_memory: 'Remembering...',
  planning: 'Planning...',
  responding: 'Speaking...',
  still_working: 'Still working...',
  resting: 'Chalie',
  error: 'Connection lost',
};

export const usePresenceStore = defineStore('presence', {
  state: () => ({
    state: 'resting' as string,
    label: LABELS['resting'] as string,
  }),
  actions: {
    /**
     * Transition to a new state.
     * Unknown states are silently ignored (port of presence.js setState guard).
     */
    setState(newState: string): void {
      if (!LABELS[newState]) return;
      this.state = newState;
      this.label = LABELS[newState];
    },
  },
  getters: {
    labels: () => LABELS,
  },
});

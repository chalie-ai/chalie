/**
 * Typed CustomEvent bus for Chalie interface events.
 *
 * All events are dispatched on / listened from `document`.  The 7 event types
 * exactly match §9.3 of the parity map.
 *
 * Usage:
 *   const unbind = on('chalie:action', (detail) => { ... });
 *   emit('chalie:theme-changed', { theme: 'dark' });
 *   unbind(); // inside onBeforeUnmount
 */

export interface ChalieEventMap {
  'chalie:theme-changed': { theme: 'dark' | 'light' };
  'chalie:pin-moment': { content?: string };
  'chalie:speak-message': { text: string };
  'chalie:voice-transcript': { text: string };
  'chalie:action': Record<string, unknown>;
  'chalie:silent-action': Record<string, unknown>;
  'chalie:attention': Record<string, unknown>;
  /** Recall button → App.vue opens the moment-search dialog (no payload). */
  'chalie:open-recall': Record<string, never>;
}

/**
 * Dispatch a typed Chalie event on `document`.
 */
export function emit<K extends keyof ChalieEventMap>(
  type: K,
  detail: ChalieEventMap[K],
): void {
  document.dispatchEvent(new CustomEvent(type, { detail }));
}

/**
 * Subscribe to a typed Chalie event on `document`.
 * Returns an unbind function — call it in `onBeforeUnmount`.
 */
export function on<K extends keyof ChalieEventMap>(
  type: K,
  handler: (detail: ChalieEventMap[K]) => void,
): () => void {
  const listener = (e: Event) => {
    handler((e as CustomEvent<ChalieEventMap[K]>).detail);
  };
  document.addEventListener(type, listener);
  return () => document.removeEventListener(type, listener);
}

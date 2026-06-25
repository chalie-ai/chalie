/**
 * Typed CustomEvent bus for Chalie interface events — all dispatched on and
 * listened from `document`.
 */

export interface ChalieEventMap {
  'chalie:theme-changed': { theme: 'dark' | 'light' };
  'chalie:speak-message': { text: string };
  'chalie:voice-transcript': { text: string };
  'chalie:action': Record<string, unknown>;
  'chalie:silent-action': Record<string, unknown>;
  'chalie:attention': Record<string, unknown>;
  'chalie:open-thread-search': void;
}

export function emit<K extends keyof ChalieEventMap>(
  type: K,
  detail: ChalieEventMap[K],
): void {
  document.dispatchEvent(new CustomEvent(type, { detail }));
}

/** Subscribe to an event; returns an unbind for `onBeforeUnmount`. */
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

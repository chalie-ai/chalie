/**
 * Context-usage store — keyed per (type, turnId).
 *
 * One writer: `record`, fed by the `context_usage` turn signal the backend
 * pushes once per CHAT provider call (see services/provider_service.py), routed
 * here by `utils/driftDispatcher.ts::_dispatchTurnSignal`. A dock reads its own
 * thread via `usageDisplayFor(type, turnId)`; two docks on different threads
 * never interfere, since every entry is keyed by `type:turnId`.
 *
 * A `turnId` of null means "whatever this channel spent last" — the spine dock,
 * which is not bound to a thread. That resolves through `latestByType`, written
 * on every `record`, rather than a synthetic key: frames only ever carry real
 * turn ids. A key with no entry renders no meter, which is the intended state
 * for a thread that has spent nothing yet.
 */
import { computed, ref } from 'vue';
import { defineStore } from 'pinia';

/** `type:turnId` — the cache key. */
function keyOf(type: string, turnId: number): string {
  return `${type}:${turnId}`;
}

export const useContextUsageStore = defineStore('contextUsage', () => {
  const byKey = ref<Record<string, { tokens: number; window: number }>>({});
  /** ConfigType → the turn that most recently reported usage on that channel. */
  const latestByType = ref<Record<string, number>>({});

  /**
   * The reading a dock should paint: its own thread's when it has a `turnId`,
   * otherwise the channel's most recent one (the spine dock's semantic).
   */
  function entryFor(type: string, turnId: number | null): { tokens: number; window: number } | undefined {
    const id = turnId ?? latestByType.value[type];
    if (id == null) return undefined;
    return byKey.value[keyOf(type, id)];
  }

  /**
   * Returns a function that, given a (type, turnId), yields the display string
   * for that dock's thread (or '' until the first reading lands).
   */
  const usageDisplayFor = computed<
    (type: string, turnId: number | null) => string
  >(() => (type: string, turnId: number | null): string => {
    const entry = entryFor(type, turnId);
    if (!entry || (entry.tokens === 0 && entry.window === 0)) return '';
    return `${(entry.tokens / 1000).toFixed(1)}/${(entry.window / 1000).toFixed(1)}k`;
  });

  /**
   * Clamped 0..1 ratio of tokens used vs. context window. Mirrors
   * `usageDisplayFor` but yields a number suitable for a meter-bar width.
   * Returns 0 when there is no entry or the window is zero (divide-by-zero).
   */
  function usageRatioFor(type: string, turnId: number | null): number {
    const entry = entryFor(type, turnId);
    if (!entry || entry.window === 0) return 0;
    return Math.min(1, Math.max(0, entry.tokens / entry.window));
  }

  /**
   * Write the reading carried by a `context_usage` turn signal. The backend
   * sends this only for a CHAT call that came back with a real token count, so
   * an entry appearing here always means something was actually spent.
   */
  function record(type: string, turnId: number, tokens: number, window: number): void {
    byKey.value[keyOf(type, turnId)] = { tokens, window };
    latestByType.value[type] = turnId;
  }

  return {
    byKey,
    latestByType,
    usageDisplayFor,
    usageRatioFor,
    record,
  };
});

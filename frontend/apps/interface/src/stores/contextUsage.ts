/**
 * Context-usage store — keyed per (type, turnId).
 *
 * One writer: `record`, fed by the `context_usage` turn signal the backend
 * pushes once per CHAT provider call (see services/provider_service.py), routed
 * here by `utils/driftDispatcher.ts::_dispatchTurnSignal`. A dock reads its own
 * thread via `percentUsedFor(type, turnId, modelWindow)`; two docks on
 * different threads never interfere, since every entry is keyed by
 * `type:turnId`.
 *
 * A `turnId` of null means "whatever this channel spent last" — the spine dock,
 * which is not bound to a thread. That resolves through `latestByType`, written
 * on every `record`, rather than a synthetic key: frames only ever carry real
 * turn ids. A key with no entry renders no meter, which is the intended state
 * for a thread that has spent nothing yet.
 */
import { ref } from 'vue';
import { defineStore } from 'pinia';

export const useContextUsageStore = defineStore('contextUsage', () => {
  const byKey = ref<Record<string, { tokens: number; window: number }>>({});
  /** ConfigType → the turn that most recently reported usage on that channel. */
  const latestByType = ref<Record<string, number>>({});

  /** `type:turnId` — the cache key. */
  function keyOf(type: string, turnId: number): string {
    return `${type}:${turnId}`;
  }

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
   * The whole percentage of the context window that dock's thread has used,
   * 0..100, or null until a reading with a real window lands. Measured against
   * `modelWindow` when the dock knows it — the window of the model its next
   * message goes to — else against the window the reading was taken on.
   */
  function percentUsedFor(type: string, turnId: number | null, modelWindow: number | null): number | null {
    const entry = entryFor(type, turnId);
    if (!entry || entry.window === 0) return null;
    const against = modelWindow || entry.window;
    return Math.round(Math.min(1, Math.max(0, entry.tokens / against)) * 100);
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
    percentUsedFor,
    record,
  };
});

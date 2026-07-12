/**
 * Context-usage store — keyed per (type, turnId).
 *
 * Each InputDock reads its own thread's token count via the store's
 * `usageDisplayFor(type, turnId)` getter; the cache is populated by
 * coalesced `refresh(type, turnId)` calls. Two docks on different threads
 * never interfere: in-flight and queued state are keyed by `type:turnId`.
 */
import { computed, ref } from 'vue';
import { defineStore } from 'pinia';
import { system } from '../api';

export const useContextUsageStore = defineStore('contextUsage', () => {
  const byKey = ref<Record<string, { tokens: number; window: number }>>({});

  /** `type:turnId` — the cache key used for both the store and the coalescing maps. */
  function keyOf(type: string, turnId: number): string {
    return `${type}:${turnId}`;
  }

  /**
   * Returns a function that, given a (type, turnId), yields the display
   * string for that dock's own thread (or '' until the first real fetch).
   */
  const usageDisplayFor = computed<
    (type: string, turnId: number) => string
  >(() => (type: string, turnId: number): string => {
    const entry = byKey.value[keyOf(type, turnId)];
    if (!entry || (entry.tokens === 0 && entry.window === 0)) return '';
    return `${(entry.tokens / 1000).toFixed(1)}/${(entry.window / 1000).toFixed(1)}k`;
  });

  /**
   * Returns a clamped 0..1 ratio of tokens used vs. context window for a
   * (type, turnId). Mirrors `usageDisplayFor` but yields a number suitable for
   * a meter-bar width. Returns 0 when the cache entry is missing or the window
   * is zero (avoids divide-by-zero).
   */
  function usageRatioFor(type: string, turnId: number): number {
    const entry = byKey.value[keyOf(type, turnId)];
    if (!entry || entry.window === 0) return 0;
    return Math.min(1, Math.max(0, entry.tokens / entry.window));
  }

  // Coalescing maps — keyed by keyOf(type, turnId); kept in module closure so
  // two docks refreshing concurrently do not collapse into one in-flight flag.
  const _refreshing: Record<string, boolean> = {};
  const _queued: Record<string, boolean> = {};

  /**
   * Fetch context usage for a specific (type, turnId). Coalesced per key: a
   * fetch in flight for that key sets _queued[k] and returns; on settle, a
   * queued request for the same key drains with exactly one trailing call.
   * Transient/auth errors leave the last painted value in place (no reset).
   * Null tokens or window are returned early and do not overwrite the cache.
   */
  async function refresh(type = 'user', turnId = -1): Promise<void> {
    const k = keyOf(type, turnId);
    if (_refreshing[k]) {
      _queued[k] = true;
      return;
    }
    _refreshing[k] = true;
    try {
      let data: Awaited<ReturnType<typeof system.contextUsage>>;
      try {
        data = await system.contextUsage(type, turnId);
      } catch {
        // Transient / auth miss — leave last value in place.
        return;
      }
      const tokens = data?.last_request_tokens;
      const window = data?.context_window;
      if (tokens == null || window == null) return;
      byKey.value[k] = { tokens, window };
    } finally {
      _refreshing[k] = false;
      if (_queued[k]) {
        _queued[k] = false;
        void refresh(type, turnId);
      }
    }
  }

  return {
    byKey,
    usageDisplayFor,
    usageRatioFor,
    refresh,
  };
});

/**
 * Context-usage + thinking-level store.
 *
 * Thinking-level slice: persists the 3-value override (auto / medium / high)
 * with optimistic update and rollback on failure.
 * Context-usage slice: fetches last-request tokens + context window; coalesced
 * refresh prevents request storms when the caller (session ws.onAny) fires on
 * every inbound WS frame.
 */
import { ref, computed } from 'vue';
import { defineStore } from 'pinia';
import { system } from '../api';

/** Order is load-bearing: cycleLevel() advances through this verbatim sequence. */
const LEVELS = ['auto', 'medium', 'high'] as const;

type ThinkingLevel = (typeof LEVELS)[number];

export const useContextUsageStore = defineStore('contextUsage', () => {
  const level = ref<ThinkingLevel>('auto');

  const levelLabel = computed<string>(
    () => level.value.charAt(0).toUpperCase() + level.value.slice(1),
  );

  /** Load persisted level; empty / absent / unknown collapse to 'auto'. */
  async function loadLevel(): Promise<void> {
    try {
      const data = await system.thinkingLevel();
      const v = data?.value;
      level.value = (LEVELS as readonly string[]).includes(v ?? '') ? (v as ThinkingLevel) : 'auto';
    } catch {
      level.value = 'auto';
    }
  }

  /** Persist a level; optimistic, reverts on error. */
  async function setLevel(next: ThinkingLevel): Promise<void> {
    const previous = level.value;
    level.value = next;
    try {
      await system.setThinkingLevel(next === 'auto' ? '' : next);
    } catch {
      level.value = previous;
    }
  }

  /** Advance through LEVELS in order, wrapping around. */
  async function cycleLevel(): Promise<void> {
    await setLevel(LEVELS[(LEVELS.indexOf(level.value) + 1) % LEVELS.length]);
  }

  const lastRequestTokens = ref<number>(0);
  const contextWindow = ref<number>(0);

  /** Empty string until the first successful fetch (mirrors legacy hidden state). */
  const usageDisplay = computed<string>(() => {
    if (lastRequestTokens.value === 0 && contextWindow.value === 0) return '';
    return `${(lastRequestTokens.value / 1000).toFixed(1)}/${(contextWindow.value / 1000).toFixed(1)}k`;
  });

  let _refreshing = false;
  let _refreshQueued = false;

  /**
   * Fetch context usage. Coalesced: a fetch in flight sets _refreshQueued and
   * returns; on settle, a queued request drains with exactly one trailing call.
   * Transient/auth errors leave the last painted value in place (no reset).
   */
  async function refresh(): Promise<void> {
    if (_refreshing) {
      _refreshQueued = true;
      return;
    }
    _refreshing = true;
    try {
      let data: Awaited<ReturnType<typeof system.contextUsage>>;
      try {
        data = await system.contextUsage();
      } catch {
        // Transient / auth miss — leave last value in place.
        return;
      }
      const tokens = data?.last_request_tokens;
      const window = data?.context_window;
      if (tokens == null || window == null) return;
      lastRequestTokens.value = tokens;
      contextWindow.value = window;
    } finally {
      _refreshing = false;
      if (_refreshQueued) {
        _refreshQueued = false;
        void refresh();
      }
    }
  }

  return {
    level,
    levelLabel,
    loadLevel,
    setLevel,
    cycleLevel,
    lastRequestTokens,
    contextWindow,
    usageDisplay,
    refresh,
  };
});

/**
 * Context-usage + thinking-level store — port of frontend/interface/chat_controls.js.
 *
 * Thinking-level slice: persists the 3-value override (auto / medium / high) to the
 * backend `settings` table. Optimistic update with rollback on failure.
 *
 * Context-usage slice: fetches last-request tokens + context window and derives a
 * formatted display string. Coalesced refresh prevents request storms when the caller
 * (session store's ws.onAny) fires on every inbound WS frame.
 */
import { ref, computed } from 'vue';
import { defineStore } from 'pinia';
import { system } from '../api';

// ── Constants ─────────────────────────────────────────────────────────────────

/**
 * Ordered cycle used by cycleLevel().
 * Port of chat_controls.js line 10 — VERBATIM order matters.
 */
const LEVELS = ['auto', 'medium', 'high'] as const;

type ThinkingLevel = (typeof LEVELS)[number];

// ── Store ──────────────────────────────────────────────────────────────────────

export const useContextUsageStore = defineStore('contextUsage', () => {
  // ── Thinking-level state ────────────────────────────────────────────────────

  const level = ref<ThinkingLevel>('auto');

  /**
   * Capitalised label for display — port of chat_controls.js _setLabel:
   *   `level.charAt(0).toUpperCase() + level.slice(1)`
   */
  const levelLabel = computed<string>(
    () => level.value.charAt(0).toUpperCase() + level.value.slice(1),
  );

  /**
   * Load the persisted level from the backend setting.
   * Empty / absent / unknown values collapse to 'auto'.
   * Port of chat_controls.js _loadThinking.
   */
  async function loadLevel(): Promise<void> {
    try {
      const data = await system.thinkingLevel();
      const v = data?.value;
      level.value = (LEVELS as readonly string[]).includes(v ?? '') ? (v as ThinkingLevel) : 'auto';
    } catch {
      level.value = 'auto';
    }
  }

  /**
   * Persist a specific level.
   * Optimistic: updates state immediately, reverts on error.
   * Port of chat_controls.js _select (without the menu-close/guard).
   */
  async function setLevel(next: ThinkingLevel): Promise<void> {
    const previous = level.value;
    level.value = next;
    try {
      await system.setThinkingLevel(next === 'auto' ? '' : next);
    } catch {
      level.value = previous;
    }
  }

  /**
   * Advance through LEVELS in order, wrapping around.
   * Port of chat_controls.js btn click → _select(btn.dataset.level),
   * where the buttons are rendered in LEVELS order.
   */
  async function cycleLevel(): Promise<void> {
    await setLevel(LEVELS[(LEVELS.indexOf(level.value) + 1) % LEVELS.length]);
  }

  // ── Context-usage state ─────────────────────────────────────────────────────

  const lastRequestTokens = ref<number>(0);
  const contextWindow = ref<number>(0);

  /**
   * Formatted display string.
   * Port of chat_controls.js refreshContext lines 116-117:
   *   `${(tokens / 1000).toFixed(1)}/${(contextWindow / 1000).toFixed(1)}k`
   *
   * Empty string when no data has arrived yet (mirrors the legacy hidden state
   * before the first successful fetch).
   */
  const usageDisplay = computed<string>(() => {
    if (lastRequestTokens.value === 0 && contextWindow.value === 0) return '';
    return `${(lastRequestTokens.value / 1000).toFixed(1)}/${(contextWindow.value / 1000).toFixed(1)}k`;
  });

  // ── Coalescer internals (reactive enough for strict TS, not for templates) ──

  let _refreshing = false;
  let _refreshQueued = false;

  /**
   * Fetch context usage and update state.
   *
   * Coalesced — port of chat_controls.js refreshContext:
   *   • If a fetch is already in flight, set _refreshQueued and return immediately.
   *   • When the in-flight fetch settles, if _refreshQueued was set, drain it with
   *     exactly one trailing call.
   *
   * On transient/auth errors the last painted value is left in place (no reset).
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
        // Transient / auth miss — leave last value in place (port of chat_controls.js line 111)
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

  // ── Public surface ──────────────────────────────────────────────────────────

  return {
    // Thinking-level
    level,
    levelLabel,
    loadLevel,
    setLevel,
    cycleLevel,

    // Context-usage
    lastRequestTokens,
    contextWindow,
    usageDisplay,
    refresh,
  };
});

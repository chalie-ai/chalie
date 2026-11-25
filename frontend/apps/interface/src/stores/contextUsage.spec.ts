// @vitest-environment happy-dom
/**
 * Context-usage store — feature specs for the meter's single live feed.
 *
 * Drives the REAL router (`utils/driftDispatcher.ts::dispatchDrift`) with the
 * exact `context_usage` wire shape `models/turn_signal.py` emits, into the REAL
 * Pinia store, and reads back through the REAL getters an InputDock renders
 * with. Zero mocks of in-process code — the only stub is `@chalie/shared`'s
 * network boundary, which this path never calls.
 *
 * The contract under test is the one the frame has to satisfy: a frame carries
 * a real `turn_id` (never a synthetic -1), yet BOTH docks must paint from it —
 * the thread panel's dock, which knows its turnId, and the footer/spine dock,
 * which has none and means "whatever this channel spent last".
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';

// The network boundary — imported by the dispatcher's module graph, never
// reached by a context_usage frame (it triggers no refetch).
vi.mock('@chalie/shared', () => ({
  api: { get: vi.fn(), post: vi.fn() },
  getWebSocket: () => ({}),
  ConfigType: { USER: 'user', SCHEDULED: 'scheduled' },
  AuthError: class extends Error {},
}));

import { dispatchDrift, registerSessionHooks } from '../utils/driftDispatcher';
import { useContextUsageStore } from './contextUsage';

/** The frame ProviderService pushes once per CHAT call (turn_signal.py). */
function contextUsageFrame(type: string, turnId: number, tokens: number, window: number) {
  return { status: 'context_usage', type, turn_id: turnId, tokens_input: tokens, context_window: window };
}

describe('context_usage frame → meter', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    // The dispatcher demands hooks; a context_usage frame uses none of them.
    registerSessionHooks({
      releasePendingSend: () => { /* unused */ },
      getPanelThreadId: () => null,
      getPanelType: () => 'user',
      setErrorMessage: () => { /* unused */ },
      finishTurn: async () => { /* unused */ },
      drainQueues: () => { /* unused */ },
    });
  });

  it('paints the footer/spine dock, which carries no turnId, from a frame carrying a real one', () => {
    const store = useContextUsageStore();

    dispatchDrift(contextUsageFrame('user', 42, 12_000, 128_000) as never);

    // The spine dock reads (type, null) — InputDock.vue's `props.turnId` default.
    expect(store.percentUsedFor('user', null, null)).toBe(9);
  });

  it('paints a thread dock from its own turn, and keeps two threads independent', () => {
    const store = useContextUsageStore();

    dispatchDrift(contextUsageFrame('user', 10, 1_000, 128_000) as never);
    dispatchDrift(contextUsageFrame('user', 20, 90_000, 128_000) as never);

    expect(store.percentUsedFor('user', 10, null)).toBe(1);
    expect(store.percentUsedFor('user', 20, null)).toBe(70);
    // The spine follows the latest reading, not the highest turn id.
    expect(store.percentUsedFor('user', null, null)).toBe(70);
  });

  it('keeps each channel on its own latest — a scheduled turn never repaints the user spine', () => {
    const store = useContextUsageStore();

    dispatchDrift(contextUsageFrame('user', 7, 5_000, 128_000) as never);
    dispatchDrift(contextUsageFrame('scheduled', 8, 99_000, 128_000) as never);

    expect(store.percentUsedFor('user', null, null)).toBe(4);
    expect(store.percentUsedFor('scheduled', null, null)).toBe(77);
  });

  it('renders no meter for a thread that has spent nothing, rather than a fabricated zero', () => {
    const store = useContextUsageStore();

    expect(store.percentUsedFor('user', null, null)).toBeNull();
    expect(store.percentUsedFor('user', 99, null)).toBeNull();
  });

  it('drops a frame missing its reading instead of painting a zeroed meter', () => {
    const store = useContextUsageStore();
    dispatchDrift(contextUsageFrame('user', 42, 12_000, 128_000) as never);

    // provider_service.py stays silent when the provider reports no count; a
    // malformed frame must not erase the last real reading.
    dispatchDrift({ status: 'context_usage', type: 'user', turn_id: 42 } as never);

    expect(store.percentUsedFor('user', null, null)).toBe(9);
  });

  it('caps an overrun at a full ring and shows no reading for a call with no window', () => {
    const store = useContextUsageStore();

    dispatchDrift(contextUsageFrame('user', 5, 140_000, 128_000) as never);
    dispatchDrift(contextUsageFrame('user', 6, 3_000, 0) as never);

    expect(store.percentUsedFor('user', 5, null)).toBe(100);
    expect(store.percentUsedFor('user', 6, null)).toBeNull();
  });
});

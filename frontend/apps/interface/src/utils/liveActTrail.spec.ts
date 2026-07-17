/**
 * liveActTrail — unit spec.
 *
 * Exercises the transient live-pill island directly. State is module-level
 * and keyed by feed type, so each test isolates itself with a unique type key.
 */
import { describe, expect, it } from 'vitest';
import {
  clearLiveTurn,
  clearLiveTurnsForToolCallsResolved,
  finishLiveTool,
  liveTrailsFor,
  startLiveTool,
} from './liveActTrail';

describe('startLiveTool / finishLiveTool', () => {
  it('adds an unresolved pill on startLiveTool', () => {
    startLiveTool('t-start', 1, 42, 'calendar', 'checking calendar');
    const pills = liveTrailsFor('t-start', 1)[0]?.pills;
    expect(pills).toHaveLength(1);
    expect(pills?.[0]).toMatchObject({ id: '42', name: 'calendar', resolved: false });
  });

  it('resolves the correct pill on finishLiveTool, preserving others', () => {
    startLiveTool('t-finish', 1, 42, 'calendar');
    startLiveTool('t-finish', 1, 43, 'weather');
    finishLiveTool('t-finish', 1, 42, true);
    const pills = liveTrailsFor('t-finish', 1)[0]?.pills ?? [];
    expect(pills.find((p) => p.id === '42')?.resolved).toBe(true);
    expect(pills.find((p) => p.id === '43')?.resolved).toBe(false);
  });

  it('is a no-op for null callId', () => {
    expect(() => finishLiveTool('t-null', 1, null, true)).not.toThrow();
    expect(liveTrailsFor('t-null', 1)).toEqual([]);
  });
});

describe('clearing', () => {
  it('clearLiveTurn drops the whole trail for a turn', () => {
    startLiveTool('t-clear', 7, 1, 'search');
    clearLiveTurn('t-clear', 7);
    expect(liveTrailsFor('t-clear', 7)).toEqual([]);
  });

  it('clearLiveTurnsForToolCallsResolved drops resolved pills, keeps unresolved', () => {
    startLiveTool('t-reconcile', 1, 42, 'calendar');
    startLiveTool('t-reconcile', 1, 43, 'weather');
    finishLiveTool('t-reconcile', 1, 42, true);
    clearLiveTurnsForToolCallsResolved('t-reconcile', 1, true);
    const pills = liveTrailsFor('t-reconcile', 1)[0]?.pills ?? [];
    expect(pills.map((p) => p.id)).toEqual(['43']);
  });

  it('clearLiveTurnsForToolCallsResolved is a no-op without persisted tool_calls', () => {
    startLiveTool('t-noop', 1, 42, 'calendar');
    finishLiveTool('t-noop', 1, 42, true);
    clearLiveTurnsForToolCallsResolved('t-noop', 1, false);
    expect(liveTrailsFor('t-noop', 1)[0]?.pills).toHaveLength(1);
  });
});

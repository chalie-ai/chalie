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
  setLiveToolDelegate,
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

describe('a delegate call\'s pill', () => {
  const ref = { channel: 'delegate:web_search', turn_id: 12 };

  it('is the same pill once the repeated start frame arrives: it gains the delegate link and is not duplicated', () => {
    startLiveTool('t-link', 5, 31, 'web_search', 'museum hours', 50);
    expect(liveTrailsFor('t-link', 5)[0]?.pills[0]?.delegate).toBeNull();

    // The backend re-sends the call's start once the child turn exists.
    startLiveTool('t-link', 5, 31, 'web_search', 'museum hours', 50);
    setLiveToolDelegate('t-link', 5, 31, ref);

    const pills = liveTrailsFor('t-link', 5)[0]?.pills ?? [];
    expect(pills).toHaveLength(1);
    expect(pills[0]).toMatchObject({ id: '31', name: 'web_search', resolved: false, delegate: ref });
    finishLiveTool('t-link', 5, 31, true);
  });

  it('keeps the same delegate link after the call finishes, successfully or not', () => {
    startLiveTool('t-keep', 5, 31, 'web_search');
    startLiveTool('t-keep', 5, 32, 'web_search');
    setLiveToolDelegate('t-keep', 5, 31, ref);
    setLiveToolDelegate('t-keep', 5, 32, { channel: 'delegate:web_search', turn_id: 13 });

    finishLiveTool('t-keep', 5, 31, true);
    finishLiveTool('t-keep', 5, 32, false);

    const pills = liveTrailsFor('t-keep', 5)[0]?.pills ?? [];
    expect(pills.find((p) => p.id === '31')).toMatchObject({ resolved: true, ok: true, delegate: ref });
    expect(pills.find((p) => p.id === '32')).toMatchObject({
      resolved: true, ok: false, delegate: { channel: 'delegate:web_search', turn_id: 13 },
    });
  });

  it('links only the call it was told about; a pill without a delegate stays a plain pill', () => {
    startLiveTool('t-only', 5, 31, 'web_search');
    startLiveTool('t-only', 5, 40, 'calendar');
    setLiveToolDelegate('t-only', 5, 31, ref);

    const pills = liveTrailsFor('t-only', 5)[0]?.pills ?? [];
    expect(pills.find((p) => p.id === '31')?.delegate).toEqual(ref);
    expect(pills.find((p) => p.id === '40')?.delegate).toBeNull();
    finishLiveTool('t-only', 5, 31, true);
    finishLiveTool('t-only', 5, 40, true);
  });

  it('is not invented by a link for a call or turn that has no live pill, or for a null call id', () => {
    startLiveTool('t-ghost', 5, 31, 'web_search');

    setLiveToolDelegate('t-ghost', 5, 99, ref);
    setLiveToolDelegate('t-ghost', 6, 31, ref);
    setLiveToolDelegate('t-ghost', 5, null, ref);

    expect(liveTrailsFor('t-ghost', 5)[0]?.pills).toHaveLength(1);
    expect(liveTrailsFor('t-ghost', 5)[0]?.pills[0]?.delegate).toBeNull();
    expect(liveTrailsFor('t-ghost', 6)).toEqual([]);
    finishLiveTool('t-ghost', 5, 31, true);
  });

  it('lives on its own feed: a delegate channel feed and the user feed never share pills for the same turn id', () => {
    startLiveTool('user-feed', 5, 31, 'web_search');
    startLiveTool('delegate:web_search', 5, 31, 'fetch_url');
    setLiveToolDelegate('user-feed', 5, 31, ref);

    expect(liveTrailsFor('user-feed', 5)[0]?.pills[0]).toMatchObject({ name: 'web_search', delegate: ref });
    expect(liveTrailsFor('delegate:web_search', 5)[0]?.pills[0]).toMatchObject({ name: 'fetch_url', delegate: null });

    clearLiveTurn('delegate:web_search', 5);
    expect(liveTrailsFor('delegate:web_search', 5)).toEqual([]);
    expect(liveTrailsFor('user-feed', 5)[0]?.pills).toHaveLength(1);
    finishLiveTool('user-feed', 5, 31, true);
  });
});

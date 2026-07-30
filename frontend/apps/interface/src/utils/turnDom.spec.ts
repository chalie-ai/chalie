// @vitest-environment happy-dom
/**
 * turnDom — feature spec for the DOM render substrate (Weave DOM contract).
 *
 * Real DOM (happy-dom), real Vue render()/createVNode, a plain render-function
 * stub component in place of TurnView (TurnView's own rendering is covered by
 * components/conversation/TurnView.spec.ts) so these specs stay scoped to the
 * substrate's own contract: the surface registry, the data-version monotonic
 * guard, ordering, removal, and the working/done attribute + event effects.
 *
 * The surface registry and `appContext` are module-level singletons, so each
 * test re-imports a fresh module via vi.resetModules() — otherwise a
 * surface registered in one test would leak into the next.
 */
import { describe, expect, it, vi } from 'vitest';
import { h } from 'vue';
import type { Component } from 'vue';
import type { ConversationTurnBlock } from '../api/conversation';

// A minimal render-function stub standing in for TurnView — renders its
// identity (turn_id/type) as data-attributes on its own root node, which is
// all upsertTurn/getTurnEl need to find and version it.
const StubComponent: Component = {
  props: ['block', 'type'],
  render(this: { block: ConversationTurnBlock; type: string }) {
    return h(
      'div',
      { 'data-turn-id': this.block.turn_id, 'data-type': this.type },
      `stub-${this.block.turn_id}`,
    );
  },
};

function block(
  turnId: number,
  messageIds: number[],
  opts: { working?: boolean; forked?: boolean } = {},
): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working: opts.working ?? false,
    duration_ms: 0,
    type: 'user',
    messages: messageIds.map((id) => ({
      id: String(id),
      role: 'user',
      content: `msg ${id}`,
      timestamp: '2026-01-01 00:00:00',
      day: '2026-01-01',
      turn_id: turnId,
      // A thread reply row is what makes a turn forked — and so the owner of
      // its own lane rather than the spine's.
      ...(opts.forked ? { thread_message: true } : {}),
    })),
  };
}

async function freshTurnDom() {
  vi.resetModules();
  return import('./turnDom');
}

describe('upsertTurn — creation and ordering', () => {
  it('creates a host with data-version = max message id, inserted in ascending turn_id order', async () => {
    const { upsertTurn } = await freshTurnDom();
    const container = document.createElement('div');
    const h1 = upsertTurn(block(1, [10, 11]), 'user', container, StubComponent)!;
    const h3 = upsertTurn(block(3, [30]), 'user', container, StubComponent)!;
    const h2 = upsertTurn(block(2, [20, 25]), 'user', container, StubComponent)!;

    expect(h2.dataset.version).toBe('25');
    expect(Array.from(container.children)).toEqual([h1, h2, h3]);
  });
});

describe('upsertTurn — data-version monotonic guard', () => {
  it('drops a strictly-lower version re-upsert, leaving the DOM at the prior version', async () => {
    const { upsertTurn } = await freshTurnDom();
    const container = document.createElement('div');
    upsertTurn(block(1, [10, 11]), 'user', container, StubComponent);

    const result = upsertTurn(block(1, [10]), 'user', container, StubComponent);

    expect(result).toBeNull();
    const host = container.firstElementChild as HTMLElement;
    expect(host.dataset.version).toBe('11');
  });

  it('re-applies an equal-version upsert (idempotent self-heal)', async () => {
    const { upsertTurn } = await freshTurnDom();
    const container = document.createElement('div');
    upsertTurn(block(1, [10, 11]), 'user', container, StubComponent);

    const result = upsertTurn(block(1, [10, 11]), 'user', container, StubComponent);

    expect(result).not.toBeNull();
    expect(result!.dataset.version).toBe('11');
  });

  it('{force:true} applies a LOWER version (post-cancel shrink)', async () => {
    const { upsertTurn } = await freshTurnDom();
    const container = document.createElement('div');
    upsertTurn(block(1, [10, 11]), 'user', container, StubComponent);

    const result = upsertTurn(block(1, [10]), 'user', container, StubComponent, {}, { force: true });

    expect(result).not.toBeNull();
    expect(result!.dataset.version).toBe('10');
  });
});

describe('upsertTurnToSurfaces — fan-out', () => {
  it('reaches every registered surface of the matching type whose accepts() passes, skips other types and rejecting surfaces', async () => {
    const { registerSurface, upsertTurnToSurfaces, getTurnEl } = await freshTurnDom();
    const accepted = document.createElement('div');
    const filtered = document.createElement('div');
    const rejected = document.createElement('div');
    const otherType = document.createElement('div');

    registerSurface({ id: 'accepted', type: 'user', container: accepted, component: StubComponent });
    registerSurface({
      id: 'filtered', type: 'user', container: filtered, component: StubComponent,
      accepts: (turnId) => turnId === 1,
    });
    registerSurface({
      id: 'rejected', type: 'user', container: rejected, component: StubComponent,
      accepts: () => false,
    });
    registerSurface({ id: 'other-type', type: 'scheduled', container: otherType, component: StubComponent });

    upsertTurnToSurfaces(block(1, [10]), 'user');

    expect(getTurnEl(1, 'user', accepted)).not.toBeNull();
    expect(getTurnEl(1, 'user', filtered)).not.toBeNull();
    expect(getTurnEl(1, 'user', rejected)).toBeNull();
    expect(getTurnEl(1, 'scheduled', otherType)).toBeNull();
  });

  it('§6.5 step 4 — a block carrying persisted tool_calls drops the turn\'s RESOLVED live pills, keeps unresolved ones', async () => {
    vi.resetModules();
    const { registerSurface, upsertTurnToSurfaces } = await import('./turnDom');
    const { startLiveTool, finishLiveTool, liveTrailsFor } = await import('./liveActTrail');
    const container = document.createElement('div');
    registerSurface({ id: 'x', type: 'user', container, component: StubComponent });

    startLiveTool('user', 1, 100, 'search');
    finishLiveTool('user', 1, 100, true); // resolved — superseded by the chip
    startLiveTool('user', 1, 101, 'browse'); // still running — must survive

    const b = block(1, [10]);
    b.messages[0].tool_calls = [
      { tool_name: 'search', summary: 's', state: 'done', ended_at: null },
    ];
    upsertTurnToSurfaces(b, 'user');

    const pills = liveTrailsFor('user', 1)[0]?.pills ?? [];
    expect(pills.map((p) => p.id)).toEqual(['101']);

    // A block WITHOUT persisted tool_calls leaves the trail untouched.
    upsertTurnToSurfaces(block(1, [10, 11]), 'user');
    expect((liveTrailsFor('user', 1)[0]?.pills ?? []).map((p) => p.id)).toEqual(['101']);
  });

  it('unregisterSurface stops further delivery to that surface', async () => {
    const { registerSurface, unregisterSurface, upsertTurnToSurfaces, getTurnEl } = await freshTurnDom();
    const container = document.createElement('div');
    registerSurface({ id: 'x', type: 'user', container, component: StubComponent });

    upsertTurnToSurfaces(block(1, [10]), 'user');
    expect(getTurnEl(1, 'user', container)).not.toBeNull();

    unregisterSurface('x');
    upsertTurnToSurfaces(block(2, [20]), 'user');
    expect(getTurnEl(2, 'user', container)).toBeNull();
  });
});

describe('upsertTurnToSurfaces — land hook gating on the version guard', () => {
  it('does not fire the land hook when a strictly-stale block is rejected by every surface, but does fire once a version actually applies', async () => {
    const { registerSurface, upsertTurnToSurfaces, onTurnLanded, SPINE_SURFACE_ID } = await freshTurnDom();
    const container = document.createElement('div');
    // Registered as the spine so the FIRST mount below is a genuine new
    // top-level turn (isNewTopLevelTurn — see upsertTurnToSurfaces's own
    // doc comment) — not load-bearing for this test's actual assertion
    // (whether the hook fires at all), just realistic setup.
    registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container, component: StubComponent });

    const landed: Array<[number, string, boolean]> = [];
    onTurnLanded((turnId, type, isNewTopLevelTurn) => landed.push([turnId, type, isNewTopLevelTurn]));

    // First-ever mount at version 11 — applies, fires the hook.
    upsertTurnToSurfaces(block(1, [10, 11]), 'user');
    expect(landed).toEqual([[1, 'user', true]]);

    // Strictly-stale re-upsert (version 10 < already-rendered 11) — every
    // surface rejects it (upsertTurn returns null), so `applied` stays
    // false and the hook must NOT fire again — a version-rejected no-op
    // wrote nothing, so signalling a "landing" here would incorrectly clear
    // a live send echo for content the DOM never actually received.
    upsertTurnToSurfaces(block(1, [10]), 'user');
    expect(landed).toEqual([[1, 'user', true]]);

    // An equal-or-higher version applies — the hook fires again.
    upsertTurnToSurfaces(block(1, [10, 11, 12]), 'user');
    expect(landed).toEqual([[1, 'user', true], [1, 'user', false]]);
  });
});

describe('removeTurn', () => {
  it('unmounts and removes the turn host from every surface', async () => {
    const { registerSurface, upsertTurnToSurfaces, removeTurn } = await freshTurnDom();
    // getAllTurnEls (backing removeTurn/setTurnWorking/setTurnDone) queries
    // document.querySelectorAll — containers must be attached to the document.
    const containerA = document.body.appendChild(document.createElement('div'));
    const containerB = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: 'a', type: 'user', container: containerA, component: StubComponent });
    registerSurface({ id: 'b', type: 'user', container: containerB, component: StubComponent });
    upsertTurnToSurfaces(block(1, [10]), 'user');
    expect(containerA.children.length).toBe(1);
    expect(containerB.children.length).toBe(1);

    removeTurn(1, 'user');

    expect(containerA.children.length).toBe(0);
    expect(containerB.children.length).toBe(0);
    containerA.remove();
    containerB.remove();
  });

  it('drops the live-working record for the removed turn and announces the change', async () => {
    const { setTurnWorking, removeTurn, isTurnWorking } = await freshTurnDom();
    setTurnWorking(1, 'user', true); // live signal only — never rendered
    expect(isTurnWorking(1, 'user')).toBe(true);

    const listener = vi.fn();
    document.addEventListener('turn-state-changed', listener);
    removeTurn(1, 'user');
    document.removeEventListener('turn-state-changed', listener);

    // A removed turn can't stay "working" — a leaked record would gate the
    // thread's sends (isTurnWorking checks the record before the DOM).
    expect(isTurnWorking(1, 'user')).toBe(false);
    expect(listener).toHaveBeenCalledOnce();

    // No record, no announcement: removing an unknown turn stays silent.
    document.addEventListener('turn-state-changed', listener);
    removeTurn(2, 'user');
    document.removeEventListener('turn-state-changed', listener);
    expect(listener).toHaveBeenCalledOnce();
  });
});

describe('evictOldestAbove — bounded DOM window (Ticket B)', () => {
  // happy-dom performs no layout, so getBoundingClientRect().bottom is 0 for
  // every node — stub each host's rect to place it above (bottom<0) or inside
  // (bottom>=0) the viewport.
  function stubBottom(host: HTMLElement, bottom: number): void {
    host.getBoundingClientRect = (): DOMRect =>
      ({ bottom, top: 0, left: 0, right: 0, width: 0, height: 0, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  }

  it('evicts the oldest hosts scrolled above the viewport, keeping the newest `keep` as a contiguous block', async () => {
    const { upsertTurn, evictOldestAbove } = await freshTurnDom();
    const container = document.createElement('div');
    const hosts = Array.from({ length: 40 }, (_, i) =>
      upsertTurn(block(i + 1, [(i + 1) * 10]), 'user', container, StubComponent)!,
    );
    // Oldest 10 (indices 0-9) sit above the viewport; the rest are visible.
    hosts.forEach((h, i) => stubBottom(h, i < 10 ? -100 : 500));

    evictOldestAbove(container, 30);

    const remaining = Array.from(container.querySelectorAll<HTMLElement>(':scope > [data-version]'));
    expect(remaining).toHaveLength(30);
    // Newest 30, still ordered/contiguous: turn_ids 11..40.
    expect(remaining.map((h) => h.querySelector('[data-turn-id]')!.getAttribute('data-turn-id'))).toEqual(
      Array.from({ length: 30 }, (_, i) => String(i + 11)),
    );
  });

  it('stops at the first host still in the viewport — never evicts a visible turn', async () => {
    const { upsertTurn, evictOldestAbove } = await freshTurnDom();
    const container = document.createElement('div');
    const hosts = Array.from({ length: 40 }, (_, i) =>
      upsertTurn(block(i + 1, [(i + 1) * 10]), 'user', container, StubComponent)!,
    );
    // Within the oldest-10 candidate slice only the first 4 are above the fold;
    // the 5th is already in view — eviction must halt there (contiguity).
    hosts.forEach((h, i) => stubBottom(h, i < 4 ? -100 : 500));

    evictOldestAbove(container, 30);

    expect(container.querySelectorAll(':scope > [data-version]')).toHaveLength(36);
  });

  it('ignores daymark dividers — only [data-version] turn hosts are counted or evicted', async () => {
    const { upsertTurn, evictOldestAbove } = await freshTurnDom();
    const container = document.createElement('div');
    const hosts = Array.from({ length: 35 }, (_, i) =>
      upsertTurn(block(i + 1, [(i + 1) * 10]), 'user', container, StubComponent)!,
    );
    // A day divider (no data-version) sits at the very top, as syncDaymarks inserts it.
    const daymark = document.createElement('div');
    daymark.className = 'daymark';
    container.insertBefore(daymark, container.firstChild);
    hosts.forEach((h, i) => stubBottom(h, i < 5 ? -100 : 500));

    evictOldestAbove(container, 30);

    // 5 oldest hosts evicted → 30 hosts; the daymark is untouched.
    expect(container.querySelectorAll(':scope > [data-version]')).toHaveLength(30);
    expect(container.querySelector('.daymark')).not.toBeNull();
  });
});

describe('setTurnWorking / setTurnDone — attribute flips and events', () => {
  it('setTurnWorking flips data-working on every rendered copy and dispatches turn-state-changed', async () => {
    const { registerSurface, upsertTurnToSurfaces, setTurnWorking } = await freshTurnDom();
    const containerA = document.body.appendChild(document.createElement('div'));
    const containerB = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: 'a', type: 'user', container: containerA, component: StubComponent });
    registerSurface({ id: 'b', type: 'user', container: containerB, component: StubComponent });
    upsertTurnToSurfaces(block(1, [10]), 'user');

    const events: CustomEvent[] = [];
    const listener = (e: Event) => events.push(e as CustomEvent);
    document.addEventListener('turn-state-changed', listener);

    setTurnWorking(1, 'user', true);
    expect(containerA.querySelector('[data-turn-id="1"]')?.getAttribute('data-working')).toBe('true');
    expect(containerB.querySelector('[data-turn-id="1"]')?.getAttribute('data-working')).toBe('true');
    expect(events).toHaveLength(1);
    expect(events[0].detail).toMatchObject({ turnId: 1, type: 'user', working: true });

    setTurnWorking(1, 'user', false);
    expect(containerA.querySelector('[data-turn-id="1"]')?.hasAttribute('data-working')).toBe(false);
    expect(events).toHaveLength(2);

    document.removeEventListener('turn-state-changed', listener);
    containerA.remove();
    containerB.remove();
  });

  // D-mount-race fix: a WS 'working' frame can arrive before the turn's
  // first fetch-driven mount. setTurnWorking records the signal into a
  // module-level live-working set regardless of whether any element is
  // rendered yet, and ALWAYS dispatches 'turn-state-changed' — consumers
  // like useDockBusy must hear about the change even with zero DOM copies.
  it('with no element rendered: mutates no attribute anywhere, but still dispatches the event and records isTurnWorking as true', async () => {
    const { setTurnWorking, isTurnWorking } = await freshTurnDom();
    const events: CustomEvent[] = [];
    const listener = (e: Event) => events.push(e as CustomEvent);
    document.addEventListener('turn-state-changed', listener);

    setTurnWorking(999, 'user', true);

    expect(events).toHaveLength(1);
    expect(events[0].detail).toMatchObject({ turnId: 999, type: 'user', working: true });
    expect(document.querySelector('[data-working]')).toBeNull();
    expect(isTurnWorking(999, 'user')).toBe(true);
    document.removeEventListener('turn-state-changed', listener);
  });

  it('a later upsertTurn of a live-recorded turn stamps data-working at mount time, even though its own block.working is false', async () => {
    const { registerSurface, upsertTurnToSurfaces, setTurnWorking, getTurnEl } = await freshTurnDom();
    const container = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: 'a', type: 'user', container, component: StubComponent });

    // The WS 'working' frame for turn 999 arrives before any fetch-driven
    // mount exists — recorded live-only, same as the test above.
    setTurnWorking(999, 'user', true);

    // block.working itself is false (a stale/plain snapshot) — the live
    // signal recorded above must still win at mount time.
    upsertTurnToSurfaces(block(999, [9990]), 'user');

    expect(getTurnEl(999, 'user', container)?.hasAttribute('data-working')).toBe(true);
    container.remove();
  });

  it('setTurnDone flips data-done on every rendered copy', async () => {
    const { registerSurface, upsertTurnToSurfaces, setTurnDone } = await freshTurnDom();
    const container = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: 'a', type: 'user', container, component: StubComponent });
    upsertTurnToSurfaces(block(5, [50]), 'user');

    setTurnDone(5, 'user', true);
    expect(container.querySelector('[data-turn-id="5"]')?.getAttribute('data-done')).toBe('true');

    setTurnDone(5, 'user', false);
    expect(container.querySelector('[data-turn-id="5"]')?.hasAttribute('data-done')).toBe(false);
    container.remove();
  });

  it('setTurnDone dispatches turn-state-changed even when no element is rendered (asymmetric vs setTurnWorking)', async () => {
    const { setTurnDone } = await freshTurnDom();
    const events: CustomEvent[] = [];
    const listener = (e: Event) => events.push(e as CustomEvent);
    document.addEventListener('turn-state-changed', listener);

    setTurnDone(999, 'user', true);

    expect(events).toHaveLength(1);
    expect(events[0].detail).toMatchObject({ turnId: 999, type: 'user', done: true });
    document.removeEventListener('turn-state-changed', listener);
  });

  // D16 — `_liveDone` mirrors `_liveWorking`: a settle frame routinely
  // arrives for a turn NO surface renders (a scheduled turn finishing with
  // the dock closed), where `setTurnDone` would stamp zero elements and the
  // state would otherwise be lost.
  it('isTurnDone is true from the live record alone, with no rendered element at all', async () => {
    const { setTurnDone, isTurnDone } = await freshTurnDom();
    setTurnDone(42, 'user', true);
    expect(isTurnDone(42, 'user')).toBe(true);
  });

  it('a later upsertTurn of a live-done-recorded turn stamps data-done at mount time', async () => {
    const { registerSurface, upsertTurnToSurfaces, setTurnDone, getTurnEl } = await freshTurnDom();
    const container = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: 'a', type: 'user', container, component: StubComponent });

    // The settle frame for turn 42 arrives before any fetch-driven mount
    // exists — recorded live-only, exactly like the analogous working case.
    setTurnDone(42, 'user', true);

    upsertTurnToSurfaces(block(42, [420]), 'user');

    expect(getTurnEl(42, 'user', container)?.hasAttribute('data-done')).toBe(true);
    container.remove();
  });

  it('setTurnDone(false) clears the live record, so isTurnDone reverts to false', async () => {
    const { setTurnDone, isTurnDone } = await freshTurnDom();
    setTurnDone(42, 'user', true);
    expect(isTurnDone(42, 'user')).toBe(true);

    setTurnDone(42, 'user', false);
    expect(isTurnDone(42, 'user')).toBe(false);
  });

  it('removeTurn clears the live-done record for the removed turn', async () => {
    const { setTurnDone, removeTurn, isTurnDone } = await freshTurnDom();
    setTurnDone(43, 'user', true);
    expect(isTurnDone(43, 'user')).toBe(true);

    removeTurn(43, 'user');

    expect(isTurnDone(43, 'user')).toBe(false);
  });
});

describe("'turn-upserted' event", () => {
  it('fires on every upsertTurn call, carrying the turnId in detail', async () => {
    const { upsertTurn } = await freshTurnDom();
    const container = document.createElement('div');
    const events: CustomEvent[] = [];
    const listener = (e: Event) => events.push(e as CustomEvent);
    document.addEventListener('turn-upserted', listener);

    upsertTurn(block(7, [70]), 'user', container, StubComponent);
    upsertTurn(block(7, [70, 71]), 'user', container, StubComponent);

    expect(events).toHaveLength(2);
    expect(events.every((e) => e.detail.turnId === 7)).toBe(true);
    document.removeEventListener('turn-upserted', listener);
  });
});

describe('stampWorking (applied by upsertTurn at mount/patch time)', () => {
  it('stamps data-working when the block itself reports working:true', async () => {
    const { upsertTurn, getTurnEl } = await freshTurnDom();
    const container = document.createElement('div');
    upsertTurn(block(1, [10], { working: true }), 'user', container, StubComponent);
    expect(getTurnEl(1, 'user', container)?.hasAttribute('data-working')).toBe(true);
  });

  it('leaves no data-working attribute when block.working is false and there is no live signal', async () => {
    const { upsertTurn, getTurnEl } = await freshTurnDom();
    const container = document.createElement('div');
    upsertTurn(block(1, [10], { working: false }), 'user', container, StubComponent);
    expect(getTurnEl(1, 'user', container)?.hasAttribute('data-working')).toBe(false);
  });
});

describe('isTurnWorking — set-based (live signal) and attribute-based (rendered) paths', () => {
  it('is true from a live signal alone, with no rendered element at all', async () => {
    const { setTurnWorking, isTurnWorking } = await freshTurnDom();
    setTurnWorking(42, 'user', true);
    expect(isTurnWorking(42, 'user')).toBe(true);
  });

  it('is true from a rendered data-working attribute alone, with no live signal ever recorded', async () => {
    const { upsertTurn, isTurnWorking } = await freshTurnDom();
    const container = document.body.appendChild(document.createElement('div'));
    // block.working:true stamps the attribute directly at mount — no
    // setTurnWorking call, so no live signal is ever recorded for turn 1.
    upsertTurn(block(1, [10], { working: true }), 'user', container, StubComponent);
    expect(isTurnWorking(1, 'user')).toBe(true);
    container.remove();
  });

  it('is false once neither a live signal nor a rendered attribute exists for the turn', async () => {
    const { isTurnWorking } = await freshTurnDom();
    expect(isTurnWorking(777, 'user')).toBe(false);
  });
});

describe('isLaneWorking — every lane independent, the spine included', () => {
  it('leaves the spine lane free while a FORKED turn rendered on it works, and busy for its own', async () => {
    const { registerSurface, upsertTurnToSurfaces, isLaneWorking, SPINE_SURFACE_ID, SPINE_LANE_TYPE, SPINE_LANE_TURN_ID } =
      await freshTurnDom();
    const spine = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container: spine, component: StubComponent });

    // A thread reply is working: same turn_id as its spine opener, rendered on
    // the spine — the exact shape that used to freeze the whole spine.
    upsertTurnToSurfaces(block(42, [1, 2], { working: true, forked: true }), 'user');
    expect(isLaneWorking('user', 42)).toBe(true);
    expect(isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID)).toBe(false);

    // The spine's OWN turn working is still the spine's business.
    upsertTurnToSurfaces(block(43, [3], { working: true }), 'user');
    expect(isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID)).toBe(true);

    spine.remove();
  });

  it('re-derives the claim on a repeated refetch of a growing reply', async () => {
    const { registerSurface, upsertTurnToSurfaces, isLaneWorking, SPINE_SURFACE_ID, SPINE_LANE_TYPE, SPINE_LANE_TURN_ID } =
      await freshTurnDom();
    const spine = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container: spine, component: StubComponent });

    upsertTurnToSurfaces(block(42, [1, 2], { working: true, forked: true }), 'user');
    upsertTurnToSurfaces(block(42, [1, 2, 3], { working: true, forked: true }), 'user');

    expect(isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID)).toBe(false);
    expect(isLaneWorking('user', 42)).toBe(true);
    spine.remove();
  });

  it('rebuilds the claim on an element mounted from scratch, which markThreadLane never reached', async () => {
    const { registerSurface, upsertTurnToSurfaces, clearSurfaceContainer, isLaneWorking, SPINE_SURFACE_ID, SPINE_LANE_TYPE, SPINE_LANE_TURN_ID } =
      await freshTurnDom();
    const spine = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container: spine, component: StubComponent });

    upsertTurnToSurfaces(block(42, [1, 2], { working: true, forked: true }), 'user');
    // Destroy the rendered copy outright — an eviction, or a surface tearing
    // down and re-registering. The next upsert builds a brand-new element that
    // no eager `markThreadLane` call ever touched, so the claim can only come
    // back by being re-derived from the block itself.
    clearSurfaceContainer(spine);
    upsertTurnToSurfaces(block(42, [1, 2], { working: true, forked: true }), 'user');

    expect(spine.querySelector('[data-lane-turn-id="42"]')).not.toBeNull();
    expect(isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID)).toBe(false);
    expect(isLaneWorking('user', 42)).toBe(true);
    spine.remove();
  });

  it('reads the spine busy for working markers carrying no turn identity at all (ACT cycles)', async () => {
    const { registerSurface, isLaneWorking, SPINE_SURFACE_ID, SPINE_LANE_TYPE, SPINE_LANE_TURN_ID } =
      await freshTurnDom();
    const spine = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container: spine, component: StubComponent });

    spine.innerHTML = '<div data-working></div>';
    expect(isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID)).toBe(true);
    spine.remove();
  });

  it('is false for the spine lane when no spine surface is registered', async () => {
    const { isLaneWorking, SPINE_LANE_TYPE, SPINE_LANE_TURN_ID } = await freshTurnDom();
    expect(isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID)).toBe(false);
  });

  it('markThreadLane claims a turn before any refetch re-derives the fork', async () => {
    const { registerSurface, upsertTurnToSurfaces, markThreadLane, isLaneWorking, setTurnWorking, SPINE_SURFACE_ID, SPINE_LANE_TYPE, SPINE_LANE_TURN_ID } =
      await freshTurnDom();
    const spine = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container: spine, component: StubComponent });

    // A settled, not-yet-forked spine turn — the FIRST reply into it has no
    // thread_message row anywhere yet.
    upsertTurnToSurfaces(block(42, [1]), 'user');
    markThreadLane(42, 'user');
    setTurnWorking(42, 'user', true);

    expect(isLaneWorking(SPINE_LANE_TYPE, SPINE_LANE_TURN_ID)).toBe(false);
    expect(isLaneWorking('user', 42)).toBe(true);
    spine.remove();
  });

  it('keeps two thread lanes independent of each other', async () => {
    const { setTurnWorking, isLaneWorking } = await freshTurnDom();
    setTurnWorking(42, 'user', true);
    expect(isLaneWorking('user', 42)).toBe(true);
    expect(isLaneWorking('user', 43)).toBe(false);
  });
});

describe('lastUserText', () => {
  it('reads the LAST [data-user-text] row within the supplied host, ignoring earlier rows', async () => {
    const { lastUserText } = await freshTurnDom();
    const host = document.createElement('div');
    host.innerHTML =
      '<div data-user-text="first message"></div>'
      + '<div data-user-text="second message"></div>';

    expect(lastUserText(host)).toBe('second message');
  });

  it('returns empty string for a host with no user-text rows', async () => {
    const { lastUserText } = await freshTurnDom();
    expect(lastUserText(document.createElement('div'))).toBe('');
  });
});

describe('liveWorkingKeys', () => {
  it('snapshots every type:turnId key currently recorded as live-working, updating as signals clear', async () => {
    const { setTurnWorking, liveWorkingKeys } = await freshTurnDom();
    setTurnWorking(1, 'user', true);
    setTurnWorking(2, 'scheduled', true);

    expect(liveWorkingKeys().sort()).toEqual(['scheduled:2', 'user:1']);

    setTurnWorking(1, 'user', false);
    expect(liveWorkingKeys()).toEqual(['scheduled:2']);
  });
});

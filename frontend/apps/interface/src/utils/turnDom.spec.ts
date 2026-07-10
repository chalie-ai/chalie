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
 * test re-imports a fresh module via vi.resetModules() (see
 * composables/useConversationFeed.spec.ts for the established pattern) —
 * otherwise a surface registered in one test would leak into the next.
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

function block(turnId: number, messageIds: number[], opts: { working?: boolean } = {}): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working: opts.working ?? false,
    duration_ms: 0,
    messages: messageIds.map((id) => ({
      id: String(id),
      role: 'user',
      content: `msg ${id}`,
      timestamp: '2026-01-01 00:00:00',
      turn_id: turnId,
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

describe('isSurfaceWorking — scoped to its own registered container only', () => {
  it('is true only for the surface whose OWN container holds the working element', async () => {
    const { registerSurface, isSurfaceWorking } = await freshTurnDom();
    const containerA = document.body.appendChild(document.createElement('div'));
    const containerB = document.body.appendChild(document.createElement('div'));
    registerSurface({ id: 'a', type: 'user', container: containerA, component: StubComponent });
    registerSurface({ id: 'b', type: 'user', container: containerB, component: StubComponent });

    containerB.innerHTML = '<div data-working></div>';

    expect(isSurfaceWorking('a')).toBe(false);
    expect(isSurfaceWorking('b')).toBe(true);
    expect(isSurfaceWorking('never-registered')).toBe(false);
    containerA.remove();
    containerB.remove();
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

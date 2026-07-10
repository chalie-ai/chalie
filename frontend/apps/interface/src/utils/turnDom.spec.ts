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

function block(turnId: number, messageIds: number[]): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working: false,
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

  // KNOWN QUIRK, encoded as-is (not a bug to fix here): setTurnWorking
  // early-returns without dispatching when NO element is rendered for the
  // turn, while setTurnDone (below) dispatches unconditionally — asymmetric
  // by design of the current implementation.
  it('setTurnWorking is a silent no-op (no event) when no element is rendered for the turn', async () => {
    const { setTurnWorking } = await freshTurnDom();
    const events: CustomEvent[] = [];
    const listener = (e: Event) => events.push(e as CustomEvent);
    document.addEventListener('turn-state-changed', listener);

    setTurnWorking(999, 'user', true);

    expect(events).toHaveLength(0);
    document.removeEventListener('turn-state-changed', listener);
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

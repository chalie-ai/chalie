// @vitest-environment happy-dom
/**
 * cancelReconcile — feature spec for the ONE shared post-cancel fetch, deduped
 * across `session.requestStop`'s own trigger and the WS 'cancelled' frame
 * (driftDispatcher's turn_execution branch) — see the module's own docblock.
 *
 * Only the network boundary (`api/conversation`) is mocked. `turnDom` and
 * `liveActTrail` are real — this asserts against the actual rendered DOM
 * (real Vue render()) and the real live-pill state, not a spy on the calls
 * `reconcileCancelledTurn` makes into them.
 */
import { describe, expect, it, vi } from 'vitest';
import { h } from 'vue';
import type { Component } from 'vue';
import type { ConversationTurnBlock } from '../api/conversation';

const threadMock = vi.fn();
vi.mock('../api/conversation', () => ({
  conversation: { thread: (...args: unknown[]) => threadMock(...args) },
}));

// A minimal render-function stub standing in for TurnView (same pattern as
// utils/turnDom.spec.ts) — renders its identity as data-attributes so
// upsertTurn/getTurnEl can find and version it.
const StubComponent: Component = {
  props: ['block', 'type'],
  render(this: { block: ConversationTurnBlock; type: string }) {
    return h('div', { 'data-turn-id': this.block.turn_id, 'data-type': this.type }, `stub-${this.block.turn_id}`);
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
    type: 'user',
    messages: messageIds.map((id) => ({
      id: String(id), role: 'user', content: `msg ${id}`, timestamp: '2026-01-01 00:00:00', day: '2026-01-01', turn_id: turnId,
    })),
  };
}

async function fresh() {
  vi.resetModules();
  threadMock.mockReset();
  const turnDom = await import('./turnDom');
  const { reconcileCancelledTurn } = await import('./cancelReconcile');
  return { turnDom, reconcileCancelledTurn };
}

describe('reconcileCancelledTurn', () => {
  it('clears working synchronously, before the fetch resolves', async () => {
    const { turnDom, reconcileCancelledTurn } = await fresh();
    turnDom.setTurnWorking(1, 'user', true);
    expect(turnDom.isTurnWorking(1, 'user')).toBe(true);

    let resolveFetch: (v: ConversationTurnBlock) => void = () => { /* replaced below */ };
    threadMock.mockReturnValue(new Promise((resolve) => { resolveFetch = resolve; }));

    const p = reconcileCancelledTurn(1, 'user');
    // Cleared BEFORE the in-flight fetch has settled at all.
    expect(turnDom.isTurnWorking(1, 'user')).toBe(false);

    resolveFetch(block(1, [10]));
    await p;
  });

  it('a non-empty fetched block force-upserts into every registered surface', async () => {
    const { turnDom, reconcileCancelledTurn } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: 'a', type: 'user', container, component: StubComponent });
    // A version already rendered HIGHER than the post-cancel block will carry
    // — only `force: true` lets the strip-down apply.
    turnDom.upsertTurn(block(2, [20, 21, 22]), 'user', container, StubComponent);

    threadMock.mockResolvedValue(block(2, [20]));
    await reconcileCancelledTurn(2, 'user');

    expect(turnDom.getTurnEl(2, 'user', container)).not.toBeNull();
    expect((container.firstElementChild as HTMLElement).dataset.version).toBe('20');
    container.remove();
  });

  it('an empty-messages fetched block removes the turn from every surface', async () => {
    const { turnDom, reconcileCancelledTurn } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: 'a', type: 'user', container, component: StubComponent });
    turnDom.upsertTurn(block(3, [30]), 'user', container, StubComponent);
    expect(turnDom.getTurnEl(3, 'user', container)).not.toBeNull();

    threadMock.mockResolvedValue(block(3, []));
    await reconcileCancelledTurn(3, 'user');

    expect(turnDom.getTurnEl(3, 'user', container)).toBeNull();
    container.remove();
  });

  it('concurrent calls for the same type:turnId collapse onto ONE fetch', async () => {
    const { reconcileCancelledTurn } = await fresh();
    let resolveFetch: (v: ConversationTurnBlock) => void = () => { /* replaced below */ };
    threadMock.mockReturnValue(new Promise((resolve) => { resolveFetch = resolve; }));

    const p1 = reconcileCancelledTurn(4, 'user');
    const p2 = reconcileCancelledTurn(4, 'user');

    resolveFetch(block(4, []));
    await Promise.all([p1, p2]);

    expect(threadMock).toHaveBeenCalledTimes(1);
  });

  it('trusts the fetched block\'s own type over the requested one on divergence, upserting under the authoritative type and warning about the mismatch', async () => {
    const { turnDom, reconcileCancelledTurn } = await fresh();
    const userContainer = document.body.appendChild(document.createElement('div'));
    const scheduledContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: 'user-surface', type: 'user', container: userContainer, component: StubComponent });
    turnDom.registerSurface({ id: 'scheduled-surface', type: 'scheduled', container: scheduledContainer, component: StubComponent });

    // Requested (and cancelled) as 'user', but the post-cancel fetch's
    // authoritative block says this turn actually belongs to 'scheduled' —
    // turn_id alone is only unique PER TYPE, so the requested type must
    // never be trusted over what the fetch actually returned.
    threadMock.mockResolvedValue({
      turn_id: 6, gist: null, preview: 'x', last_activity_at: null, working: false, duration_ms: 0, type: 'scheduled',
      messages: [{ id: '60', role: 'user', content: 'hi', timestamp: '2026-01-01 00:00:00', day: '2026-01-01', turn_id: 6 }],
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* silence */ });

    await reconcileCancelledTurn(6, 'user');

    expect(turnDom.getTurnEl(6, 'scheduled', scheduledContainer)).not.toBeNull();
    expect(turnDom.getTurnEl(6, 'user', userContainer)).toBeNull();
    expect(warnSpy).toHaveBeenCalled();

    warnSpy.mockRestore();
    userContainer.remove();
    scheduledContainer.remove();
  });

  it('a fetch failure never throws, and clears the in-flight cache so the next call fetches again', async () => {
    const { reconcileCancelledTurn } = await fresh();
    threadMock.mockRejectedValueOnce(new Error('network down'));

    await expect(reconcileCancelledTurn(5, 'user')).resolves.toBeUndefined();
    expect(threadMock).toHaveBeenCalledTimes(1);

    threadMock.mockResolvedValueOnce(block(5, []));
    await reconcileCancelledTurn(5, 'user');
    expect(threadMock).toHaveBeenCalledTimes(2);
  });
});

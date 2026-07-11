// @vitest-environment happy-dom
/**
 * driftDispatcher — feature spec for `resolveFrameType`'s fallback chain and
 * the authoritative-block.type trust rule, both exercised through the real
 * `turnDom` DOM contract rather than the mocked one `driftDispatcher.spec.ts`
 * uses for its discrimination-order tests.
 *
 * Split into its own file (rather than a describe block inside
 * driftDispatcher.spec.ts) because `vi.mock` is hoisted per FILE: that spec
 * mocks `./turnDom` wholesale to keep its routing assertions cheap, but
 * `resolveFrameType`'s DOM-lookup branch (`findTurnType`) and the
 * fetched-block-wins-on-divergence branch (`_refetchAndUpsert`) can only be
 * proven against the real substrate — a real rendered node, a real surface
 * fan-out. `resolveFrameType`/`isOpenPanelTurn` are private to the module, so
 * both are exercised only through the public `dispatchDrift` entry point,
 * same as every other spec of this module.
 *
 * Only the network boundary (`api/conversation`) is mocked, mirroring
 * `cancelReconcile.spec.ts`'s established convention — everything downstream
 * (turnDom, liveActTrail) runs for real.
 */
import { describe, expect, it, vi } from 'vitest';
import { h } from 'vue';
import type { Component } from 'vue';
import type { ConversationTurnBlock, ConversationMessage } from '../api/conversation';
import type { WsPushEvent } from '@chalie/shared';

const threadMock = vi.fn();
vi.mock('../api/conversation', () => ({
  conversation: { thread: (...args: unknown[]) => threadMock(...args) },
}));

// Same minimal render-function stub as turnDom.spec.ts/cancelReconcile.spec.ts
// — renders its identity as data-attributes so getTurnEl can find it, which
// is also exactly what `findTurnType` reads to resolve a frame's channel.
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

function block(turnId: number, type: string, messages: ConversationMessage[] = []): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    type,
    messages,
  };
}

function frame(data: Record<string, unknown>): WsPushEvent {
  return data as unknown as WsPushEvent;
}

async function fresh() {
  vi.resetModules();
  threadMock.mockReset();
  const turnDom = await import('./turnDom');
  const { dispatchDrift, registerSessionHooks } = await import('./driftDispatcher');
  // A closed-panel hooks double — none of these tests rely on the panel
  // fallback (that pairing is pinned in driftDispatcher.spec.ts against the
  // mocked turnDom); registration is required purely because `hooks()`
  // throws if nothing was ever registered (see the module's own guard).
  registerSessionHooks({
    releasePendingSend: () => { /* not under test */ },
    getPanelThreadId: () => null,
    getPanelType: () => 'user',
    setErrorMessage: () => { /* not under test */ },
    finishTurn: async () => { /* not under test */ },
    drainQueues: () => { /* not under test */ },
  });
  return { turnDom, dispatchDrift };
}

describe('resolveFrameType — DOM fallback (findTurnType)', () => {
  it('a type-less frame for an already-rendered turn resolves via the DOM\'s own stamped data-type', async () => {
    const { turnDom, dispatchDrift } = await fresh();
    const container = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: 'spine', type: 'user', container, component: StubComponent });
    turnDom.upsertTurnToSurfaces(block(7, 'user'), 'user');
    expect(turnDom.getTurnEl(7, 'user', container)).not.toBeNull();

    threadMock.mockResolvedValue(block(7, 'user'));

    // No `type` field at all — the ONLY way this can resolve is by reading
    // the already-rendered node's own data-type.
    dispatchDrift(frame({ status: 'updated', turn_id: 7 }));
    await Promise.resolve();
    await Promise.resolve();

    expect(threadMock).toHaveBeenCalledWith(7, 'user');
    container.remove();
  });

  it('a type-less frame for a turn with no rendered copy and no open panel is dropped with a warning, and never triggers a refetch', async () => {
    const { dispatchDrift } = await fresh();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* silence */ });

    dispatchDrift(frame({ status: 'updated', turn_id: 999 }));
    await Promise.resolve();
    await Promise.resolve();

    expect(threadMock).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});

describe('_refetchAndUpsert — the fetched block\'s own type wins over the requested type', () => {
  it('a refetch that comes back under a DIFFERENT type than requested renders under the FETCHED type, not the requested one, and warns about the mismatch', async () => {
    const { turnDom, dispatchDrift } = await fresh();
    const userContainer = document.body.appendChild(document.createElement('div'));
    const scheduledContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: 'user-surface', type: 'user', container: userContainer, component: StubComponent });
    turnDom.registerSurface({ id: 'scheduled-surface', type: 'scheduled', container: scheduledContainer, component: StubComponent });

    // Requested as 'user' (the frame's own type), but the backend's
    // authoritative fetch says this turn actually belongs to 'scheduled' —
    // turn_id alone is only unique PER TYPE, so trusting the request over
    // the fetch would risk painting turn 8's scheduled content into the
    // user channel.
    threadMock.mockResolvedValue(block(8, 'scheduled', [
      { id: '80', role: 'assistant', content: 'a scheduled reply', timestamp: '2026-01-01 00:00:00', turn_id: 8 },
    ]));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* silence */ });

    dispatchDrift(frame({ status: 'updated', turn_id: 8, type: 'user' }));
    await Promise.resolve();
    await Promise.resolve();

    expect(turnDom.getTurnEl(8, 'scheduled', scheduledContainer)).not.toBeNull();
    expect(turnDom.getTurnEl(8, 'user', userContainer)).toBeNull();
    expect(warnSpy).toHaveBeenCalled();

    warnSpy.mockRestore();
    userContainer.remove();
    scheduledContainer.remove();
  });
});

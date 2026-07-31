// @vitest-environment happy-dom
/**
 * Session store — feature specs for the DOM-held busy contract (D3): no lane
 * model, no `isSending` flag. Busy/working state for every independent
 * conversation surface (the main spine + each open thread) is derived from
 * the DOM (`utils/turnDom.ts`'s `data-working` attribute + its live-signal
 * bookkeeping), not a store-held record.
 *
 * Real DOM (happy-dom — the project's established Vue-mounting environment,
 * see turnDom.spec.ts), real Pinia, real turnDom/queue modules. Only the
 * WS/network boundary is mocked: `getWebSocket` (send/abort are the only
 * stubbed calls, per convention), `getHost`, and the REST `conversation` API
 * (`api/conversation.ts`) — the actual `fetch`/XHR transport this app would
 * otherwise hit.
 *
 * The session store, turnDom, and queue all carry module-level singleton
 * state, so each test re-imports a fresh module graph via
 * `vi.resetModules()` (the pattern established by `utils/turnDom.spec.ts`) —
 * otherwise a surface/turn registered in one test would leak into the next.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { h } from 'vue';
import type { Component } from 'vue';

// ── The WS/network boundary — the only thing this spec mocks. `getWebSocket`
// captures whatever onConnect/onDisconnect callbacks `session.init()`
// registers so reconnect tests can fire them directly, the same way the real
// WebSocketService would invoke them on an actual drop/restore.
const { fakeWs, sendMock, wsCallbacks } = vi.hoisted(() => {
  const sendMock = vi.fn();
  const wsCallbacks: { onConnect: () => void; onDisconnect: () => void } = {
    onConnect: () => { /* replaced by session.init() */ },
    onDisconnect: () => { /* replaced by session.init() */ },
  };
  return {
    sendMock,
    wsCallbacks,
    fakeWs: {
      send: sendMock,
      onConnect: (cb: () => void) => { wsCallbacks.onConnect = cb; },
      onDisconnect: (cb: () => void) => { wsCallbacks.onDisconnect = cb; },
      onDrift: () => { /* not under test */ },
      onAny: () => { /* not under test */ },
      connect: () => { /* not under test */ },
      ensureAlive: () => { /* not under test */ },
    },
  };
});

vi.mock('@chalie/shared', () => ({
  ConfigType: { USER: 'user', SCHEDULED: 'scheduled', DISCOVERY: 'discovery' },
  AuthError: class AuthError extends Error {},
  getWebSocket: () => fakeWs,
  useConnectionStore: () => ({ setConnected: () => { /* not under test */ } }),
  platform: {},
  api: {},
  getHost: () => '',
}));

// The REST boundary `_reconcileWorking`/`reconcileCancelledTurn`/`_finishTurn`
// hit (via api/conversation's `thread()`) — mocked at the network edge only,
// per the boundary rule (everything downstream of the response, including
// the real turnDom/liveActTrail DOM effects, runs unmocked).
const threadMock = vi.fn();
vi.mock('../api/conversation', () => ({
  conversation: {
    thread: (...args: unknown[]) => threadMock(...args),
    threads: vi.fn(),
    batch: vi.fn(),
  },
}));

import { ConfigType } from '@chalie/shared';

/** A minimal but well-formed ConversationTurnBlock for the mocked thread() calls. */
function stubBlock(turnId: number, working: boolean): unknown {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working,
    duration_ms: 0,
    messages: [],
  };
}

// A minimal render-function stub standing in for a turn's real render
// component (same pattern as turnDom.spec.ts/sendEcho.spec.ts) — renders its
// identity as data-attributes so a settled turn's content landing in the DOM
// is directly observable, not just inferred from a mock call.
const StubComponent: Component = {
  props: ['block', 'type'],
  render(this: { block: { turn_id: number }; type: string }) {
    return h(
      'div',
      { 'data-turn-id': this.block.turn_id, 'data-type': this.type },
      `stub-${this.block.turn_id}`,
    );
  },
};

/** Fresh module graph per test — session, turnDom, queue, and sendEcho all
 *  share the SAME instances within one test (imported in the same epoch,
 *  before the next resetModules() call), but never leak into the next test. */
async function freshSession() {
  vi.resetModules();
  setActivePinia(createPinia());
  const turnDom = await import('../utils/turnDom');
  const { threadPhase } = await import('../utils/threadActivity');
  const { useSessionStore } = await import('./session');
  const { useQueueStore } = await import('./queue');
  const sendEcho = await import('../utils/sendEcho');
  return {
    session: useSessionStore(),
    queue: useQueueStore(),
    turnDom,
    threadPhase,
    sendEcho,
  };
}

beforeEach(() => {
  sendMock.mockReset();
  threadMock.mockReset();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true } as Response));
  document.body.innerHTML = '';
});

describe('isSurfaceBusy', () => {
  it('reads a thread\'s busy state from a rendered [data-working][data-turn-id][data-type] element', async () => {
    const { session } = await freshSession();
    const container = document.body.appendChild(document.createElement('div'));
    container.innerHTML = '<div data-working data-turn-id="42" data-type="user"></div>';

    expect(session.isSurfaceBusy(42, ConfigType.USER)).toBe(true);
    expect(session.isSurfaceBusy(43, ConfigType.USER)).toBe(false);
  });

  it('is busy via _pendingSends while a POST is in flight, before any element has rendered', async () => {
    const { session } = await freshSession();
    let resolveSend: (v: unknown) => void = () => { /* replaced below */ };
    sendMock.mockImplementationOnce(
      () => new Promise((resolve) => { resolveSend = resolve; }),
    );

    const pending = session.sendMessage('hello', [], null, ConfigType.USER);
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(true);

    resolveSend(null);
    await pending;
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(false);
  });

  it('stays busy after the POST resolves with a turn_id, until that turn\'s first execution frame releases the hold', async () => {
    const { session } = await freshSession();
    sendMock.mockResolvedValueOnce({ turn_id: 42, type: ConfigType.USER });

    await session.sendMessage('hello', [], null, ConfigType.USER);
    // POST resolved, but execution runs in the background — no WS 'working'
    // frame has been observed yet, so the surface must still gate sends.
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(true);

    session._releasePendingSend(42, ConfigType.USER);
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(false);
  });

  it('treats offline-snapshotted turns (and an offline spine) as busy so drafts queue instead of dropping', async () => {
    const { session, turnDom, queue } = await freshSession();
    session.init();

    const spineContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: ConfigType.USER,
      container: spineContainer,
      component: {},
    });
    spineContainer.innerHTML = '<div data-working data-turn-id="7" data-type="user"></div>';

    wsCallbacks.onDisconnect();

    // Visual markers are cleared, but the backend may still be mid-turn
    // behind the dead socket — both the thread and the spine stay busy.
    expect(turnDom.isTurnWorking(7, ConfigType.USER)).toBe(false);
    expect(session.isSurfaceBusy(7, ConfigType.USER)).toBe(true);
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(true);

    // A send while offline into the mid-turn thread queues rather than
    // hitting the dead transport and losing the draft.
    await session.sendMessage('typed while offline', [], 7, ConfigType.USER);
    expect(sendMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(7)).toEqual([{ text: 'typed while offline', files: [], thinkingLevel: null }]);

    // Reconcile settles the turn on reconnect and drops the blanket flags.
    threadMock.mockResolvedValue(stubBlock(7, false));
    sendMock.mockResolvedValue({ turn_id: 7, type: ConfigType.USER });
    await session._reconcileWorking();
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(false);
  });

  it('the main spine reads busy off its registered SPINE_SURFACE_ID container, not a stable turn_id', async () => {
    const { session, turnDom } = await freshSession();
    const spineContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: ConfigType.USER,
      container: spineContainer,
      component: {},
    });

    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(false);

    spineContainer.innerHTML = '<div data-working></div>';
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(true);
  });
});

describe('sendMessage — surface-scoped busy gate', () => {
  it('enqueues onto a busy surface instead of posting, while a different idle surface still sends over the wire', async () => {
    const { session, queue } = await freshSession();

    // First send on the main spine never settles during this test — the
    // surface stays busy for its whole duration.
    let resolveFirst: (v: unknown) => void = () => { /* replaced below */ };
    sendMock.mockImplementationOnce(
      () => new Promise((resolve) => { resolveFirst = resolve; }),
    );
    const p1 = session.sendMessage('first message', [], null, ConfigType.USER);
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(true); // registered synchronously before the await

    // A second send on the SAME (main) surface while it's busy must defer to
    // the queue, not touch the network again.
    await session.sendMessage('second message while busy', [], null, ConfigType.USER);
    expect(sendMock).toHaveBeenCalledTimes(1);
    expect(queue.queuedFor(null)).toEqual([{ text: 'second message while busy', files: [], thinkingLevel: null }]);

    // A send on a DIFFERENT scope (a thread, not busy) must post immediately.
    sendMock.mockResolvedValueOnce({ turn_id: 909, type: ConfigType.USER });
    await session.sendMessage('thread message', [], 555, ConfigType.USER);
    expect(sendMock).toHaveBeenCalledTimes(2);
    expect(sendMock).toHaveBeenLastCalledWith(
      'thread message', expect.any(Function), [], 555, ConfigType.USER, null,
    );
    expect(queue.queuedFor(555)).toEqual([]);

    // Cleanup: settle the still-pending first send so no dangling promise
    // leaks across tests.
    resolveFirst({ turn_id: 42, type: ConfigType.USER });
    await p1;
  });

  it('leaves the spine and every other thread free to post while one thread works', async () => {
    const { session, turnDom, queue } = await freshSession();

    const spineContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: ConfigType.USER,
      container: spineContainer,
      component: {},
    });
    // Both threads' openers render on the spine — a thread reply continues the
    // SAME turn_id as its opener, so its work shows up here as well as in the
    // panel. That shared rendering is what used to freeze the spine.
    spineContainer.innerHTML =
      '<div data-turn-id="42" data-type="user"></div><div data-turn-id="43" data-type="user"></div>';

    // Reply into thread 42, then let its first turn_execution frame land.
    sendMock.mockResolvedValueOnce({ turn_id: 42, type: ConfigType.USER });
    await session.sendMessage('reply in thread 42', [], 42, ConfigType.USER);
    turnDom.setTurnWorking(42, ConfigType.USER, true);
    session._releasePendingSend(42, ConfigType.USER);

    expect(session.isSurfaceBusy(42, ConfigType.USER)).toBe(true);
    expect(session.isSurfaceBusy(null, ConfigType.USER)).toBe(false);
    expect(session.isSurfaceBusy(43, ConfigType.USER)).toBe(false);

    // The spine posts over the wire rather than queueing behind thread 42.
    sendMock.mockResolvedValueOnce({ turn_id: 44, type: ConfigType.USER });
    await session.sendMessage('a new top-level message', [], null, ConfigType.USER);
    expect(queue.queuedFor(null)).toEqual([]);
    expect(sendMock).toHaveBeenLastCalledWith(
      'a new top-level message', expect.any(Function), [], null, ConfigType.USER, null,
    );

    // ...and so does a second thread, in parallel with the first.
    sendMock.mockResolvedValueOnce({ turn_id: 43, type: ConfigType.USER });
    await session.sendMessage('reply in thread 43', [], 43, ConfigType.USER);
    expect(queue.queuedFor(43)).toEqual([]);
    expect(sendMock).toHaveBeenCalledTimes(3);

    // A second reply into the STILL-working thread 42 does queue — its own
    // lane is the one thing that is genuinely busy.
    await session.sendMessage('second reply in thread 42', [], 42, ConfigType.USER);
    expect(sendMock).toHaveBeenCalledTimes(3);
    expect(queue.queuedFor(42)).toEqual([
      { text: 'second reply in thread 42', files: [], thinkingLevel: null },
    ]);

    spineContainer.remove();
  });
});

describe('_drainLane — queued sends replay their files, not just their text', () => {
  it('replays a queued message\'s text AND its attached files into the resumed send call', async () => {
    const { session, queue } = await freshSession();

    const file = new File(['contents'], 'photo.png', { type: 'image/png' });
    queue.enqueue(77, 'queued while the thread was busy', ConfigType.USER, [file]);

    sendMock.mockResolvedValueOnce({ turn_id: 77, type: ConfigType.USER });
    session._drainQueues();
    // sendMessage's own network call is fire-and-forget from _drainLane
    // (`void`) — flush microtasks so the underlying send() call has
    // actually happened.
    await Promise.resolve();
    await Promise.resolve();

    expect(sendMock).toHaveBeenCalledTimes(1);
    expect(sendMock).toHaveBeenCalledWith(
      'queued while the thread was busy', expect.any(Function), [file], 77, ConfigType.USER, null,
    );
  });
});

describe('requestStop — undo event', () => {
  it('dispatches session:turn-interrupted with {text: restoreText, turnId: dockScope}, turning the FILE_PLACEHOLDER back into empty text', async () => {
    const { session } = await freshSession();
    const received: Array<{ text: string; turnId: number | null }> = [];
    document.addEventListener('session:turn-interrupted', (e) => {
      received.push((e as CustomEvent<{ text: string; turnId: number | null }>).detail);
    });

    await session.requestStop(null, ConfigType.USER, 100, 'draft text');
    expect(received[0]).toEqual({ text: 'draft text', turnId: 100 });

    // '[File attached]' is the file-only placeholder the InputDock echoes as
    // restoreText when the interrupted turn carried no typed text — it must
    // never be handed back to the user as literal draft content.
    await session.requestStop(null, ConfigType.USER, null, '[File attached]');
    expect(received[1]).toEqual({ text: '', turnId: null });
  });
});

describe('requestStop — DELETE only fires for a confirmed in-flight turn', () => {
  it('skips the DELETE for a turnId with no rendered element and no live signal, but fires it once the turn is confirmed working', async () => {
    const { session, turnDom } = await freshSession();
    threadMock.mockResolvedValue(stubBlock(11, false));

    // Turn 10 was never confirmed working (stale/late click on an
    // already-settled turn) — must not hit the network.
    await session.requestStop(10, ConfigType.USER, null, '');
    expect(fetch).not.toHaveBeenCalled();

    // Turn 11 is confirmed working via a live setTurnWorking signal alone
    // (no rendered element) — the DELETE must fire.
    turnDom.setTurnWorking(11, ConfigType.USER, true);
    await session.requestStop(11, ConfigType.USER, null, '');
    expect(fetch).toHaveBeenCalledWith(
      '/api/threads/11?type=user',
      expect.objectContaining({ method: 'DELETE' }),
    );
  });

  it('clears working optimistically but starts the content refetch only AFTER the DELETE resolves (stale pre-cancel content must never be fetched ahead of the cancel)', async () => {
    const { session, turnDom } = await freshSession();
    threadMock.mockResolvedValue(stubBlock(12, false));

    let resolveDelete: (v: unknown) => void = () => { /* replaced below */ };
    vi.stubGlobal('fetch', vi.fn().mockImplementationOnce(
      () => new Promise((resolve) => { resolveDelete = resolve; }),
    ));

    turnDom.setTurnWorking(12, ConfigType.USER, true);
    const stopping = session.requestStop(12, ConfigType.USER, null, '');

    // Spinner drops immediately (optimistic), but the reconcile fetch is
    // held while the DELETE round-trip is still in flight.
    expect(turnDom.isTurnWorking(12, ConfigType.USER)).toBe(false);
    await Promise.resolve();
    expect(threadMock).not.toHaveBeenCalled();

    resolveDelete({ ok: true });
    await stopping;
    expect(threadMock).toHaveBeenCalledWith(12, ConfigType.USER);
  });
});

describe('reconnect reconcile', () => {
  it('onDisconnect snapshots every in-flight turn (rendered AND live-only) into _offlineWorking, then clears all visual working state', async () => {
    const { session, turnDom } = await freshSession();
    session.init();

    const container = document.body.appendChild(document.createElement('div'));
    container.innerHTML = '<div data-working data-turn-id="5" data-type="user"></div>';
    turnDom.setTurnWorking(6, ConfigType.USER, true); // live signal only, never rendered

    wsCallbacks.onDisconnect();

    expect(session._offlineWorking).toEqual(new Set(['user:5', 'user:6']));
    expect(container.querySelector('[data-turn-id="5"]')?.hasAttribute('data-working')).toBe(false);
    expect(turnDom.isTurnWorking(5, ConfigType.USER)).toBe(false);
    expect(turnDom.isTurnWorking(6, ConfigType.USER)).toBe(false);
  });

  it('_reconcileWorking settles a snapshotted turn whose refetch says it is no longer working (drains queues), and restores the spinner for one still working', async () => {
    const { session, turnDom, threadPhase, queue } = await freshSession();
    session._offlineWorking.add('user:5'); // will refetch as settled
    session._offlineWorking.add('user:6'); // will refetch as still working
    threadMock.mockImplementation(async (turnId: number) => {
      if (turnId === 5) return stubBlock(5, false);
      if (turnId === 6) return stubBlock(6, true);
      throw new Error(`unexpected turnId ${turnId}`);
    });

    // A queued send on an unrelated, idle scope proves _finishTurn's
    // _drainQueues actually ran as part of settling turn 5.
    sendMock.mockResolvedValueOnce({ turn_id: 909, type: ConfigType.USER });
    queue.enqueue(909, 'queued while offline', ConfigType.USER, []);

    await session._reconcileWorking();
    await Promise.resolve(); // flush _finishTurn's fire-and-forget _drainQueues -> sendMessage

    expect(session._offlineWorking.size).toBe(0);
    expect(turnDom.isTurnWorking(5, ConfigType.USER)).toBe(false); // settled, no longer working
    // D16: _reconcileWorking stamps setTurnDone(5, ..., true) for any settled
    // turn the panel doesn't have open (see session.ts _reconcileWorking) —
    // turnDom's `_liveDone` record makes that observable via threadPhase even
    // though nothing re-rendered turn 5's element in this test.
    expect(threadPhase(5, ConfigType.USER)).toBe('done');
    expect(turnDom.isTurnWorking(6, ConfigType.USER)).toBe(true); // restored
    expect(threadPhase(6, ConfigType.USER)).toBe('working');
    expect(sendMock).toHaveBeenCalledWith(
      'queued while offline', expect.any(Function), [], 909, ConfigType.USER, null,
    );
  });

  it('keeps a key snapshotted when its refetch fails, so the next reconnect retries it', async () => {
    const { session } = await freshSession();
    session._offlineWorking.add('user:9');
    threadMock.mockRejectedValueOnce(new Error('network down'));

    await session._reconcileWorking();

    expect(session._offlineWorking.has('user:9')).toBe(true);
  });

  it('the settled branch renders the turn\'s real content into the spine surface AND clears a send echo stranded by the outage', async () => {
    const { session, turnDom, sendEcho } = await freshSession();
    const spineContainer = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: turnDom.SPINE_SURFACE_ID,
      type: ConfigType.USER,
      container: spineContainer,
      component: StubComponent,
    });

    // A send echo was standing in for turn 21's content when the WS dropped
    // mid-turn — the outage swallowed its updated/completed frames (no
    // replay), so nothing else will EVER upsert this turn or clear the echo
    // except this reconcile.
    sendEcho.mountSendEcho('typed right before the drop', null, ConfigType.USER);
    expect(spineContainer.querySelector('[data-send-echo]')).not.toBeNull();

    session._offlineWorking.add(`${ConfigType.USER}:21`);
    threadMock.mockResolvedValue({
      turn_id: 21, gist: null, preview: 'reply', last_activity_at: null,
      working: false, duration_ms: 0, type: ConfigType.USER,
      messages: [{ id: '210', role: 'assistant', content: 'the real reply', timestamp: '2026-01-01 00:00:00', turn_id: 21 }],
    });

    await session._reconcileWorking();

    // The real content actually rendered — without this, a turn whose
    // frames were lost during the outage would never appear on reconnect.
    expect(turnDom.getTurnEl(21, ConfigType.USER, spineContainer)).not.toBeNull();
    // And the stranded echo is gone — the land hook fired as part of that
    // same upsert, exactly as it would for a live 'updated' frame.
    expect(spineContainer.querySelector('[data-send-echo]')).toBeNull();

    spineContainer.remove();
  });
});

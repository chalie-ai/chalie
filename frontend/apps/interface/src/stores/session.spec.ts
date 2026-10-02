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
 * stubbed calls, per convention), `api.upload` (the spine join's POST),
 * `getHost`, and the REST `conversation` API
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
const { fakeWs, sendMock, uploadMock, wsCallbacks } = vi.hoisted(() => {
  const sendMock = vi.fn();
  // `api.upload` is the multipart POST a spine follow-up uses to join its
  // working turn (the same network edge as the WS service's own POST).
  const uploadMock = vi.fn();
  const wsCallbacks: { onConnect: () => void; onDisconnect: () => void } = {
    onConnect: () => { /* replaced by session.init() */ },
    onDisconnect: () => { /* replaced by session.init() */ },
  };
  return {
    sendMock,
    uploadMock,
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
  api: { upload: (...args: unknown[]) => uploadMock(...args) },
  getHost: () => '',
}));

// The REST boundary `_reconcileWorking`/`reconcileCancelledTurn`/`_finishTurn`
// hit (via api/conversation's `thread()`) — mocked at the network edge only,
// per the boundary rule (everything downstream of the response, including
// the real turnDom/liveActTrail DOM effects, runs unmocked).
const threadMock = vi.fn();
// The stop's DELETE (`requestStop`), at the same edge.
const stopMock = vi.fn();
vi.mock('../api/conversation', () => ({
  conversation: {
    thread: (...args: unknown[]) => threadMock(...args),
    stop: (...args: unknown[]) => stopMock(...args),
    threads: vi.fn(),
    batch: vi.fn(),
  },
}));

// The pending permission-gate listing `refreshPending()` re-reads on every
// connect (`api/policies.ts`) — the same network edge, mocked the same way.
const pendingMock = vi.fn();
vi.mock('../api/policies', () => ({
  policies: {
    pending: () => pendingMock(),
    respond: vi.fn(),
  },
}));

import { ConfigType } from '@chalie/shared';
import type { ConversationTurnBlock } from '../api/conversation';

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
  uploadMock.mockReset();
  threadMock.mockReset();
  stopMock.mockReset();
  stopMock.mockResolvedValue({ cancelled: true, reason: null });
  pendingMock.mockReset();
  pendingMock.mockResolvedValue([]);
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

    // A second reply into the STILL-working thread 42 posts to that thread
    // too — a text sent while its lane's turn works joins the turn.
    await session.sendMessage('second reply in thread 42', [], 42, ConfigType.USER);
    expect(sendMock).toHaveBeenCalledTimes(4);
    expect(sendMock).toHaveBeenLastCalledWith(
      'second reply in thread 42', expect.any(Function), [], 42, ConfigType.USER, null,
    );
    expect(queue.queuedFor(42)).toEqual([]);

    spineContainer.remove();
  });
});

describe('sendMessage — a text sent while the lane\'s turn works joins it', () => {
  /** A registered spine whose rendered DOM is exactly `html`. */
  async function spineWith(html: string) {
    const ctx = await freshSession();
    const spine = document.body.appendChild(document.createElement('div'));
    ctx.turnDom.registerSurface({
      id: ctx.turnDom.SPINE_SURFACE_ID,
      type: ConfigType.USER,
      container: spine,
      component: {},
    });
    spine.innerHTML = html;
    return { ...ctx, spine };
  }

  const WORKING_7 = '<div data-working data-turn-id="7" data-type="user"></div>';
  const JOINED_OK = { result: { turn_id: 7, type: ConfigType.USER } };

  it('a spine follow-up POSTs join=1 to the unclaimed working turn: no lane claim, no echo, nothing queued, never the WS send', async () => {
    // Turn 5 is a forked thread's work rendered on the spine (lane-claimed);
    // turn 7 is the spine's own working turn. The join must pick 7.
    const { session, queue, spine } = await spineWith(
      '<div data-working data-turn-id="5" data-type="user" data-lane-type="user" data-lane-turn-id="5"></div>'
      + WORKING_7,
    );
    uploadMock.mockResolvedValue(JOINED_OK);

    await session.sendMessage('and add the chart too', [], null, ConfigType.USER, 'high');

    expect(uploadMock).toHaveBeenCalledTimes(1);
    const [path, form] = uploadMock.mock.calls[0] as [string, FormData];
    expect(path).toBe('/api/threads/7');
    expect(form.get('text')).toBe('and add the chart too');
    expect(form.get('type')).toBe(ConfigType.USER);
    expect(form.get('join')).toBe('1');
    // Honoured by the backend only if the turn finished first and the text
    // starts a new one instead — it must still ride along.
    expect(form.get('thinking_level')).toBe('high');
    expect(sendMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(null)).toEqual([]);
    // The joined turn stays the spine's: only the pre-existing claim on 5.
    expect(Array.from(spine.querySelectorAll('[data-lane-turn-id]')).map((e) => e.getAttribute('data-turn-id')))
      .toEqual(['5']);
    // No echo — the real row paints with the `updated` refetch the join triggers.
    expect(spine.querySelector('[data-send-echo]')).toBeNull();
  });

  it('a failed join POST surfaces the error and leaves the lane free to join again (no stuck hold, nothing queued)', async () => {
    const { session, queue } = await spineWith(WORKING_7);
    uploadMock.mockRejectedValueOnce(new Error('network down'));

    await session.sendMessage('lost in transit', [], null, ConfigType.USER);

    expect(session.errorMessage).toBe('Chat request failed.');
    expect(queue.queuedFor(null)).toEqual([]);

    uploadMock.mockResolvedValue(JOINED_OK);
    await session.sendMessage('try again', [], null, ConfigType.USER);
    expect(uploadMock).toHaveBeenCalledTimes(2);
    expect(queue.queuedFor(null)).toEqual([]);
  });

  it('two texts sent one after the other both join the working turn — the first join does not hold the lane', async () => {
    const { session, queue } = await spineWith(WORKING_7);
    uploadMock.mockResolvedValue(JOINED_OK);

    await session.sendMessage('first follow-up', [], null, ConfigType.USER);
    await session.sendMessage('second follow-up', [], null, ConfigType.USER);

    expect(uploadMock).toHaveBeenCalledTimes(2);
    expect((uploadMock.mock.calls[1] as [string, FormData])[1].get('text')).toBe('second follow-up');
    expect(queue.queuedFor(null)).toEqual([]);
  });

  it('a text typed while the lane\'s own join POST is still in flight queues, even though a working turn is on screen', async () => {
    const { session, queue } = await spineWith(WORKING_7);
    let resolveFirst: (v: unknown) => void = () => { /* replaced below */ };
    uploadMock.mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }));

    const first = session.sendMessage('first follow-up', [], null, ConfigType.USER);
    await session.sendMessage('typed before the first POST resolved', [], null, ConfigType.USER);

    expect(uploadMock).toHaveBeenCalledTimes(1);
    expect(sendMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(null)).toEqual([
      { text: 'typed before the first POST resolved', files: [], thinkingLevel: null },
    ]);

    resolveFirst(JOINED_OK);
    await first;
  });

  it('a spine text sent offline queues, even if a refetch repainted the working turn before the reconnect reconcile', async () => {
    const { session, queue, spine } = await spineWith(WORKING_7);
    session.init();
    wsCallbacks.onDisconnect();
    // The reconnect's first `updated` refetch re-renders the still-working
    // turn before `_reconcileWorking` has dropped the offline flags.
    spine.querySelector('[data-turn-id="7"]')?.setAttribute('data-working', 'true');

    await session.sendMessage('typed while offline', [], null, ConfigType.USER);

    expect(uploadMock).not.toHaveBeenCalled();
    expect(sendMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(null)).toEqual([{ text: 'typed while offline', files: [], thinkingLevel: null }]);
  });

  it('a busy lane queues a message with files — a join refuses attachments — on the spine and in a thread alike', async () => {
    const { session, queue } = await spineWith(
      WORKING_7 + '<div data-working data-turn-id="42" data-type="user" data-lane-turn-id="42"></div>',
    );
    const file = new File(['bytes'], 'chart.png', { type: 'image/png' });

    await session.sendMessage('see attached', [file], null, ConfigType.USER);
    await session.sendMessage('and here, in the thread', [file], 42, ConfigType.USER);

    expect(uploadMock).not.toHaveBeenCalled();
    expect(sendMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(null)).toEqual([{ text: 'see attached', files: [file], thinkingLevel: null }]);
    expect(queue.queuedFor(42)).toEqual([{ text: 'and here, in the thread', files: [file], thinkingLevel: null }]);
  });

  it('a spine whose only working marker names no turn has nothing to join, so the text queues', async () => {
    const { session, queue } = await spineWith('<div data-working></div>');

    await session.sendMessage('nothing to join yet', [], null, ConfigType.USER);

    expect(uploadMock).not.toHaveBeenCalled();
    expect(sendMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(null)).toEqual([{ text: 'nothing to join yet', files: [], thinkingLevel: null }]);
  });

  /** A thread panel registered over the spine, rendering turn 42 as `html`. */
  async function panelOver(spineHtml: string, panelHtml: string) {
    const ctx = await spineWith(spineHtml);
    const panel = document.body.appendChild(document.createElement('div'));
    panel.innerHTML = panelHtml;
    ctx.turnDom.registerSurface({
      id: 'panel', type: ConfigType.USER, container: panel, component: {}, accepts: (id) => id === 42,
    });
    return { ...ctx, panel };
  }

  it('a reply into a working thread goes to that thread over the wire — no join POST, no echo, nothing queued', async () => {
    const claimed = '<div data-working data-turn-id="42" data-type="user" data-lane-type="user" data-lane-turn-id="42"></div>';
    const { session, queue, panel } = await panelOver(claimed, claimed);
    sendMock.mockResolvedValueOnce({ turn_id: 42, type: ConfigType.USER });

    await session.sendMessage('one more thing for the thread', [], 42, ConfigType.USER);

    expect(sendMock).toHaveBeenCalledTimes(1);
    expect(sendMock).toHaveBeenCalledWith(
      'one more thing for the thread', expect.any(Function), [], 42, ConfigType.USER, null,
    );
    expect(uploadMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(42)).toEqual([]);
    expect(panel.querySelector('[data-send-echo]')).toBeNull();
  });

  it('a reply into a turn still working its opener joins that exchange: no lane claim, so the spine still reads the work as its own', async () => {
    const working = '<div data-working data-turn-id="42" data-type="user"></div>';
    const { session, turnDom, spine, panel } = await panelOver(working, working);
    sendMock.mockResolvedValueOnce({ turn_id: 42, type: ConfigType.USER });

    await session.sendMessage('a reply typed in the panel', [], 42, ConfigType.USER);

    expect(sendMock).toHaveBeenCalledTimes(1);
    expect(uploadMock).not.toHaveBeenCalled();
    expect(panel.querySelector('[data-send-echo]')).toBeNull();
    // A join is no fork: neither copy is claimed for a thread lane.
    expect(spine.querySelector('[data-turn-id="42"]')?.hasAttribute('data-lane-turn-id')).toBe(false);
    expect(panel.querySelector('[data-turn-id="42"]')?.hasAttribute('data-lane-turn-id')).toBe(false);
    expect(turnDom.isLaneWorking(turnDom.SPINE_LANE_TYPE, turnDom.SPINE_LANE_TURN_ID)).toBe(true);
  });

  it('idle lanes start as before — a new spine turn and a thread reply go over the wire with an echo, never as a join', async () => {
    const { session, queue, turnDom, spine } = await spineWith('');
    const panel = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: 'panel', type: ConfigType.USER, container: panel, component: {}, accepts: (id) => id === 555,
    });
    sendMock.mockResolvedValue({ turn_id: 20, type: ConfigType.USER });

    await session.sendMessage('a fresh question', [], null, ConfigType.USER);
    await session.sendMessage('reply into a settled thread', [], 555, ConfigType.USER);

    expect(sendMock).toHaveBeenNthCalledWith(
      1, 'a fresh question', expect.any(Function), [], null, ConfigType.USER, null,
    );
    expect(sendMock).toHaveBeenNthCalledWith(
      2, 'reply into a settled thread', expect.any(Function), [], 555, ConfigType.USER, null,
    );
    expect(uploadMock).not.toHaveBeenCalled();
    expect(queue.queuedFor(null)).toEqual([]);
    expect(queue.queuedFor(555)).toEqual([]);
    expect(spine.querySelector('[data-send-echo]')).not.toBeNull();
    expect(panel.querySelector('[data-send-echo]')).not.toBeNull();
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

describe('requestStop — only a message the cancel removes goes back to the dock', () => {
  // The real TurnView renders every copy, so the rows a cancel would drop carry
  // the same `data-dropped-on-cancel` marker the app paints: the working turn's
  // trailing user rows nothing has answered, stopping at one that joined it.
  async function spineRendering(block: ConversationTurnBlock) {
    const ctx = await freshSession();
    const { default: TurnView } = await import('../components/conversation/TurnView.vue');
    const spine = document.body.appendChild(document.createElement('div'));
    ctx.turnDom.registerSurface({
      id: ctx.turnDom.SPINE_SURFACE_ID, type: ConfigType.USER, container: spine, component: TurnView,
    });
    ctx.turnDom.upsertTurnToSurfaces(block, ConfigType.USER);
    return { ...ctx, spine, TurnView };
  }

  function row(
    id: number,
    role: 'user' | 'assistant',
    content: string,
    extra: Partial<ConversationTurnBlock['messages'][number]> = {},
  ): ConversationTurnBlock['messages'][number] {
    return {
      id: String(id), role, content, timestamp: '2026-01-01 00:00:00', day: '2026-01-01', turn_id: 12, ...extra,
    };
  }

  function turn12(messages: ConversationTurnBlock['messages'], working: boolean): ConversationTurnBlock {
    return {
      turn_id: 12, gist: null, preview: 'turn 12', last_activity_at: null, working, duration_ms: 0,
      type: ConfigType.USER, messages,
    };
  }

  /** Collects every `session:turn-interrupted` the stop dispatches. */
  function listenForRestores(): { received: Array<{ text: string; turnId: number | null }>; stop: () => void } {
    const received: Array<{ text: string; turnId: number | null }> = [];
    const onInterrupted = (e: Event): void => {
      received.push((e as CustomEvent<{ text: string; turnId: number | null }>).detail);
    };
    document.addEventListener('session:turn-interrupted', onInterrupted);
    return { received, stop: () => document.removeEventListener('session:turn-interrupted', onInterrupted) };
  }

  it('a message joined into the turn stays in the transcript after Stop, so it is NOT handed back to the dock', async () => {
    const rows = [row(1, 'user', 'plan my trip to Rome'), row(2, 'user', 'make it three days', { joined: true })];
    const { session, turnDom, spine } = await spineRendering(turn12(rows, true));
    // What the dock hands over: the last user row, read off the DOM before the stop.
    const restoreText = turnDom.lastUserText(spine);
    expect(restoreText).toBe('make it three days');
    // The cancel strands the joined row unread but keeps it, and the opener before it.
    threadMock.mockResolvedValue(turn12(rows, false));
    const { received, stop } = listenForRestores();

    await session.requestStop(12, ConfigType.USER, null, restoreText);
    stop();

    expect(received).toEqual([]);
    expect(Array.from(spine.querySelectorAll('[data-user-text]')).map((r) => (r as HTMLElement).dataset.userText))
      .toEqual(['plan my trip to Rome', 'make it three days']);
  });

  it('a message the turn already answered stays too: a stop after an interim reply hands nothing back', async () => {
    const rows = [row(1, 'user', 'plan my trip to Rome'), row(2, 'assistant', 'looking at flights first')];
    const { session, turnDom, spine } = await spineRendering(turn12(rows, true));
    threadMock.mockResolvedValue(turn12(rows, false));
    const { received, stop } = listenForRestores();

    await session.requestStop(12, ConfigType.USER, null, turnDom.lastUserText(spine));
    stop();

    expect(received).toEqual([]);
  });

  it('a cancelled opener the backend strips (nothing after it) IS handed back — once the server accepts the stop, never while it could still refuse it', async () => {
    const { session, turnDom, spine } = await spineRendering(turn12([row(1, 'user', 'plan my trip to Rome')], true));
    const restoreText = turnDom.lastUserText(spine);
    // Nothing answered or joined after the opener: the cancel strips it, and a
    // turn with no rows left is removed from every surface.
    threadMock.mockResolvedValue(turn12([], false));
    let accept: (ack: unknown) => void = () => { /* replaced below */ };
    stopMock.mockImplementationOnce(() => new Promise((resolve) => { accept = resolve; }));
    const { received, stop } = listenForRestores();

    const stopping = session.requestStop(12, ConfigType.USER, null, restoreText);
    await Promise.resolve();
    expect(received).toEqual([]);
    accept({ cancelled: true, reason: null });
    await stopping;
    stop();

    expect(received).toEqual([{ text: 'plan my trip to Rome', turnId: null }]);
    expect(turnDom.getTurnEl(12, ConfigType.USER, spine)).toBeNull();
  });

  it('what the cancel strips is judged from the copy as it stood before the stop: the cancelled frame that redraws the copy without it lands before the server answers', async () => {
    const opener = [row(1, 'user', 'plan my trip to Rome'), row(2, 'assistant', 'Rome it is', { settled: true })];
    const { session, turnDom, TurnView } = await spineRendering(turn12(opener, false));
    const panel = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: 'panel', type: ConfigType.USER, container: panel, component: TurnView,
      props: { fullThread: true }, accepts: (id) => id === 12,
    });
    turnDom.upsertTurnToSurfaces(
      turn12([...opener, row(3, 'user', 'what about Florence?', { thread_message: true })], true), ConfigType.USER,
    );
    const restoreText = turnDom.lastUserText(panel);
    // The cancel strips the unanswered reply but keeps the opener, so the
    // turn's copy survives the redraw — just without the reply.
    threadMock.mockResolvedValue(turn12(opener, false));
    const { reconcileCancelledTurn } = await import('../utils/cancelReconcile');
    stopMock.mockImplementationOnce(async () => {
      await reconcileCancelledTurn(12, ConfigType.USER);
      expect(panel.querySelector('[data-user-text="what about Florence?"]')).toBeNull();
      return { cancelled: true, reason: null };
    });
    const { received, stop } = listenForRestores();

    await session.requestStop(12, ConfigType.USER, 12, restoreText);
    stop();

    expect(received).toEqual([{ text: 'what about Florence?', turnId: 12 }]);
  });

  it('a failed refetch after the cancel still leaves the opener handed back to the dock', async () => {
    const { session, turnDom, spine } = await spineRendering(turn12([row(1, 'user', 'plan my trip to Rome')], true));
    threadMock.mockRejectedValue(new Error('network down'));
    const { received, stop } = listenForRestores();

    await session.requestStop(12, ConfigType.USER, null, turnDom.lastUserText(spine));
    stop();

    expect(received).toEqual([{ text: 'plan my trip to Rome', turnId: null }]);
  });

  it('a thread reply\'s joined follow-up is kept by judging the thread panel\'s own copy, since the spine\'s copy hides thread rows', async () => {
    const opener = [row(1, 'user', 'plan my trip to Rome'), row(2, 'assistant', 'Rome it is', { settled: true })];
    const { session, turnDom, spine, TurnView } = await spineRendering(turn12(opener, false));
    const panel = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: 'panel', type: ConfigType.USER, container: panel, component: TurnView,
      props: { fullThread: true }, accepts: (id) => id === 12,
    });
    const rows = [
      ...opener,
      row(3, 'user', 'what about Florence?', { thread_message: true }),
      row(4, 'user', 'and Venice?', { thread_message: true, joined: true }),
    ];
    turnDom.upsertTurnToSurfaces(turn12(rows, true), ConfigType.USER);
    // Only the panel shows the thread rows.
    const restoreText = turnDom.lastUserText(panel);
    expect(restoreText).toBe('and Venice?');
    expect(spine.querySelectorAll('[data-user-text]')).toHaveLength(1);
    // The reply's cancel keeps the joined 'and Venice?' row and what precedes it.
    threadMock.mockResolvedValue(turn12(rows, false));
    const { received, stop } = listenForRestores();

    await session.requestStop(12, ConfigType.USER, 12, restoreText);
    stop();

    expect(received).toEqual([]);
  });

  it('a thread reply the turn has not answered is handed back from the panel\'s copy', async () => {
    const opener = [row(1, 'user', 'plan my trip to Rome'), row(2, 'assistant', 'Rome it is', { settled: true })];
    const { session, turnDom, TurnView } = await spineRendering(turn12(opener, false));
    const panel = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({
      id: 'panel', type: ConfigType.USER, container: panel, component: TurnView,
      props: { fullThread: true }, accepts: (id) => id === 12,
    });
    const rows = [...opener, row(3, 'user', 'what about Florence?', { thread_message: true })];
    turnDom.upsertTurnToSurfaces(turn12(rows, true), ConfigType.USER);
    threadMock.mockResolvedValue(turn12(opener, false));
    const { received, stop } = listenForRestores();

    await session.requestStop(12, ConfigType.USER, 12, turnDom.lastUserText(panel));
    stop();

    expect(received).toEqual([{ text: 'what about Florence?', turnId: 12 }]);
  });
});

describe('requestStop — DELETE only fires for a confirmed in-flight turn', () => {
  it('skips the DELETE for a turnId with no rendered element and no live signal, but fires it once the turn is confirmed working', async () => {
    const { session, turnDom } = await freshSession();
    threadMock.mockResolvedValue(stubBlock(11, false));

    // Turn 10 was never confirmed working (stale/late click on an
    // already-settled turn) — must not hit the network.
    await session.requestStop(10, ConfigType.USER, null, '');
    expect(stopMock).not.toHaveBeenCalled();

    // Turn 11 is confirmed working via a live setTurnWorking signal alone
    // (no rendered element) — the DELETE must fire.
    turnDom.setTurnWorking(11, ConfigType.USER, true);
    await session.requestStop(11, ConfigType.USER, null, '');
    expect(stopMock).toHaveBeenCalledWith(11, ConfigType.USER);
  });

  it('undoes nothing while the DELETE is in flight, then clears working and refetches (stale pre-cancel content must never be fetched ahead of the cancel)', async () => {
    const { session, turnDom } = await freshSession();
    threadMock.mockResolvedValue(stubBlock(12, false));
    const handBacks: unknown[] = [];
    document.addEventListener('session:turn-interrupted', (e) => { handBacks.push((e as CustomEvent).detail); });

    let resolveDelete: (v: unknown) => void = () => { /* replaced below */ };
    stopMock.mockImplementationOnce(() => new Promise((resolve) => { resolveDelete = resolve; }));

    turnDom.setTurnWorking(12, ConfigType.USER, true);
    const stopping = session.requestStop(12, ConfigType.USER, 12, 'draft');

    // Until the server accepts the stop the turn may still be running, so its
    // spinner (and stop control), its draft and its content all stay put.
    await Promise.resolve();
    expect(turnDom.isTurnWorking(12, ConfigType.USER)).toBe(true);
    expect(handBacks).toEqual([]);
    expect(threadMock).not.toHaveBeenCalled();

    resolveDelete({ cancelled: true, reason: null });
    await stopping;
    expect(turnDom.isTurnWorking(12, ConfigType.USER)).toBe(false);
    expect(handBacks).toEqual([{ text: 'draft', turnId: 12 }]);
    expect(threadMock).toHaveBeenCalledWith(12, ConfigType.USER);
  });
});

describe('requestStop — a stop that fails', () => {
  it('says so in the dock banner and undoes nothing, leaving the turn running and stoppable; a retry then stops it and takes the banner down', async () => {
    const { session, turnDom } = await freshSession();
    threadMock.mockResolvedValue(stubBlock(14, false));
    const logged = vi.spyOn(console, 'error').mockImplementation(() => { /* asserted below */ });
    const handBacks: unknown[] = [];
    document.addEventListener('session:turn-interrupted', (e) => { handBacks.push((e as CustomEvent).detail); });
    stopMock.mockRejectedValueOnce(new Error('HTTP 500'));

    turnDom.setTurnWorking(14, ConfigType.USER, true);
    await session.requestStop(14, ConfigType.USER, 14, 'draft');

    expect(session.errorMessage).toBe("Couldn't stop and undo. Try again.");
    expect(logged).toHaveBeenCalled();
    expect(turnDom.isTurnWorking(14, ConfigType.USER)).toBe(true);
    expect(handBacks).toEqual([]);
    expect(threadMock).not.toHaveBeenCalled();

    await session.requestStop(14, ConfigType.USER, 14, 'draft');

    expect(stopMock).toHaveBeenCalledTimes(2);
    expect(session.errorMessage).toBeNull();
    expect(turnDom.isTurnWorking(14, ConfigType.USER)).toBe(false);
    expect(handBacks).toEqual([{ text: 'draft', turnId: 14 }]);
  });

  it('stays quiet when the turn had already ended (no_active_turn), and leaves an unrelated error on the banner alone', async () => {
    const { session, turnDom } = await freshSession();
    threadMock.mockResolvedValue(stubBlock(15, false));
    stopMock.mockResolvedValueOnce({ cancelled: null, reason: 'no_active_turn' });
    session.errorMessage = 'Provider quota exceeded';

    turnDom.setTurnWorking(15, ConfigType.USER, true);
    await session.requestStop(15, ConfigType.USER, null, '');

    expect(session.errorMessage).toBe('Provider quota exceeded');
    expect(turnDom.isTurnWorking(15, ConfigType.USER)).toBe(false);
    expect(threadMock).toHaveBeenCalledWith(15, ConfigType.USER);
  });
});

describe('reconnect reconcile', () => {
  it('onConnect re-reads the pending permission gates, so a gate still open across a reload or a dropped socket gets its card back', async () => {
    const { session } = await freshSession();
    const { usePermissionsStore } = await import('./permissions');
    const permissions = usePermissionsStore();
    pendingMock.mockResolvedValue([
      {
        request_id: 'gate-1',
        action_id: 'pim',
        summary: 'Read the inbox',
        origin: { type: ConfigType.USER, turn_id: 7, forked: true },
      },
    ]);

    session.init();
    expect(permissions.queue).toEqual([]);
    wsCallbacks.onConnect();
    await new Promise((resolve) => setTimeout(resolve, 0)); // flush pending() -> enqueue

    expect(pendingMock).toHaveBeenCalledTimes(1);
    expect(permissions.queue).toEqual([
      {
        request_id: 'gate-1',
        action_id: 'pim',
        summary: 'Read the inbox',
        origin: { type: 'user', turn_id: 7, forked: true },
      },
    ]);

    // A second connect (reconnect) re-reads and dedupes — the same gate is not queued twice.
    wsCallbacks.onConnect();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(pendingMock).toHaveBeenCalledTimes(2);
    expect(permissions.queue.map((r) => r.request_id)).toEqual(['gate-1']);
  });

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

  it('onDisconnect drops every delegate call\'s live pills, whose settling frame died with the socket, and leaves a user turn\'s pills for the reconcile', async () => {
    const { session } = await freshSession();
    const { liveTrailsFor, startLiveTool } = await import('../utils/liveActTrail');
    session.init();
    startLiveTool('delegate:web_search', 12, 31, 'fetch_url', 'museum.example/hours');
    startLiveTool('delegate:code_agent', 4, 32, 'read_file');
    startLiveTool(ConfigType.USER, 12, 33, 'web_search', 'museum hours');

    wsCallbacks.onDisconnect();

    expect(liveTrailsFor('delegate:web_search', 12)).toEqual([]);
    expect(liveTrailsFor('delegate:code_agent', 4)).toEqual([]);
    // Same turn id on the user channel: a different turn, not swept up.
    expect(liveTrailsFor(ConfigType.USER, 12)[0]?.pills.map((p) => p.id)).toEqual(['33']);
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

describe('the slide-over panel target — a thread (turn id + type) or a delegate transcript (channel + turn id)', () => {
  const SEARCH = { channel: 'delegate:web_search', turn_id: 12 };
  const AGENT = { channel: 'delegate:code_agent', turn_id: 3 };

  it('opening a delegate transcript replaces an open thread: the delegate is the target and no thread id is left behind', async () => {
    const { session } = await freshSession();
    session.openThreadPanel(5, ConfigType.USER);
    expect(session.panelThreadId).toBe(5);
    expect(session.panelType).toBe(ConfigType.USER);
    expect(session.panelDelegate).toBeNull();

    session.openDelegatePanel(SEARCH);

    expect(session.panelDelegate).toEqual(SEARCH);
    expect(session.panelThreadId).toBeNull();
  });

  it('opening a thread replaces an open delegate transcript and takes the thread\'s own type', async () => {
    const { session } = await freshSession();
    session.openDelegatePanel(SEARCH);

    session.openThreadPanel(9, ConfigType.SCHEDULED);

    expect(session.panelDelegate).toBeNull();
    expect(session.panelThreadId).toBe(9);
    expect(session.panelType).toBe(ConfigType.SCHEDULED);
  });

  it('a second delegate click replaces the first, even for the same turn id on another channel', async () => {
    const { session } = await freshSession();
    session.openDelegatePanel(SEARCH);
    session.openDelegatePanel(AGENT);
    expect(session.panelDelegate).toEqual(AGENT);

    session.openDelegatePanel({ channel: 'delegate:code_agent', turn_id: SEARCH.turn_id });
    expect(session.panelDelegate).toEqual({ channel: 'delegate:code_agent', turn_id: 12 });
    expect(session.panelThreadId).toBeNull();
  });

  it('closing clears whichever target was open, thread or delegate', async () => {
    const { session } = await freshSession();

    session.openThreadPanel(5, ConfigType.USER);
    session.closeThreadPanel();
    expect(session.panelThreadId).toBeNull();
    expect(session.panelDelegate).toBeNull();

    session.openDelegatePanel(SEARCH);
    session.closeThreadPanel();
    expect(session.panelThreadId).toBeNull();
    expect(session.panelDelegate).toBeNull();
  });

  it('opening a thread clears its standing "done" marker, but opening a delegate transcript with the same turn id leaves the user turn\'s marker alone', async () => {
    const { session, turnDom } = await freshSession();
    turnDom.setTurnDone(12, ConfigType.USER, true);
    expect(turnDom.isTurnDone(12, ConfigType.USER)).toBe(true);

    // A delegate's turn 12 is not the user's turn 12.
    session.openDelegatePanel(SEARCH);
    expect(turnDom.isTurnDone(12, ConfigType.USER)).toBe(true);

    session.openThreadPanel(12, ConfigType.USER);
    expect(turnDom.isTurnDone(12, ConfigType.USER)).toBe(false);
  });
});

/**
 * Session store — feature specs for two just-fixed defects:
 *
 *  1. `requestStop` undo restores the draft to the CORRECT dock: the dispatched
 *     'session:turn-interrupted' event now carries the owning lane's scope id
 *     (null for the main spine, the thread's root turn_id for a thread lane —
 *     NOT necessarily the specific in-flight turn_id, which can differ once a
 *     thread has run more than one turn) plus the original draft text.
 *  2. `sendMessage` is scoped per lane: a send onto a BUSY lane defers to the
 *     queue store instead of hitting the network, while a different, non-busy
 *     lane still posts immediately — pinning turn_id-scoping (not a single
 *     global send-in-flight flag).
 *
 * Real Pinia stores, real WebSocketService call shape (only `send`/`abort` are
 * stubbed — the network/ws boundary, per convention). `document`/`window`/
 * `navigator` are stubbed with real EventTarget instances so the module's
 * ambient-sensor and event-bus code (both real, unmocked) can run under
 * vitest's `environment: 'node'` — this is DOM scaffolding, not a mock of
 * behaviour under test.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';

// ── Hoisted DOM scaffolding — vitest hoists vi.hoisted() above the module's
// own static imports, so `document`/`window`/`navigator` exist before
// `./session` (and its transitive `useAmbientSensor` import) ever evaluates.
vi.hoisted(() => {
  class FakeEventTarget extends EventTarget {}
  const fakeDocument = Object.assign(new FakeEventTarget(), {
    hasFocus: () => true,
    visibilityState: 'visible' as DocumentVisibilityState,
    documentElement: { dataset: {} as Record<string, string> },
    // @vue/runtime-dom probes `document.createElement('template')` once at
    // import time (client-hydration detection) even though nothing in this
    // spec ever mounts a component — a stub element is enough to satisfy it.
    createElement: () => ({}) as unknown as HTMLElement,
  });
  // Node's own `navigator` global is a read-only getter — plain assignment
  // throws. `vi.stubGlobal` overrides it (and `document`/`window`, absent in
  // vitest's `environment: 'node'`) safely.
  vi.stubGlobal('document', fakeDocument);
  vi.stubGlobal('window', new FakeEventTarget());
  vi.stubGlobal('navigator', { onLine: true });
});

// ── The WS/network boundary — the only thing this spec mocks. `getWebSocket`,
// `useConnectionStore`, `platform`, `api`, `getHost` are re-exported here
// (rather than importing the real `@chalie/shared` barrel) purely to dodge an
// unrelated vitest config gap: the interface app's vitest.config.ts has no Vue
// SFC plugin, so the real barrel's `.vue` re-exports fail to parse. `ConfigType`
// carries its real production values.
const { fakeWs, sendMock, abortMock } = vi.hoisted(() => {
  const sendMock = vi.fn();
  const abortMock = vi.fn();
  return {
    sendMock,
    abortMock,
    fakeWs: {
      send: sendMock,
      abort: abortMock,
      onConnect: () => {},
      onDisconnect: () => {},
      onDrift: () => {},
      onAny: () => {},
      connect: () => {},
      ensureAlive: () => {},
      sendAction: () => {},
    },
  };
});

vi.mock('@chalie/shared', () => ({
  ConfigType: { USER: 'user', SCHEDULED: 'scheduled', DISCOVERY: 'discovery' },
  AuthError: class AuthError extends Error {},
  getWebSocket: () => fakeWs,
  useConnectionStore: () => ({ setConnected: () => {} }),
  platform: {},
  api: {},
  getHost: () => '',
}));

import { ConfigType } from '@chalie/shared';
import { useSessionStore } from './session';
import { useQueueStore } from './queue';

beforeEach(() => {
  setActivePinia(createPinia());
  sendMock.mockReset();
  abortMock.mockReset();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true } as Response));
});

describe('requestStop — undo restores the draft to the correct dock', () => {
  it('for a thread lane, dispatches the THREAD ROOT turn_id (the lane key), not the specific ACT turn_id', async () => {
    const session = useSessionStore();
    // The lane is keyed by the thread's root turn_id (100) but the currently
    // in-flight ACT turn within that thread has its own, different id (105) —
    // exactly the case that would restore to the wrong dock before the fix.
    session.lanes['t100'] = { liveTurnId: 105, userText: 'draft in the thread', type: ConfigType.USER };

    const received: Array<{ text: string; turnId: number | null }> = [];
    document.addEventListener('session:turn-interrupted', (e) => {
      received.push((e as CustomEvent<{ text: string; turnId: number | null }>).detail);
    });

    await session.requestStop(105, ConfigType.USER);

    expect(received).toHaveLength(1);
    expect(received[0].turnId).toBe(100);
    expect(received[0].text).toBe('draft in the thread');
  });

  it('for the main spine, dispatches turnId: null and the original draft text', async () => {
    const session = useSessionStore();
    session.lanes['main'] = { liveTurnId: 55, userText: 'draft on the spine', type: ConfigType.USER };

    const received: Array<{ text: string; turnId: number | null }> = [];
    document.addEventListener('session:turn-interrupted', (e) => {
      received.push((e as CustomEvent<{ text: string; turnId: number | null }>).detail);
    });

    await session.requestStop(55, ConfigType.USER);

    expect(received).toHaveLength(1);
    expect(received[0].turnId).toBeNull();
    expect(received[0].text).toBe('draft on the spine');
  });
});

describe('sendMessage — turn_id-scoped busy gate', () => {
  it('enqueues onto a busy lane instead of posting, while a different, non-busy lane still sends over the wire', async () => {
    const session = useSessionStore();
    const queue = useQueueStore();

    // First send on the main spine never settles during this test — the lane
    // stays busy for its whole duration.
    let resolveFirst: (v: { turn_id: number; type: string } | null) => void = () => {};
    sendMock.mockImplementationOnce(
      () => new Promise((resolve) => { resolveFirst = resolve; }),
    );
    const p1 = session.sendMessage('first message', [], null, ConfigType.USER);
    expect(session.isSending).toBe(true); // lane registered synchronously before the await

    // A second send on the SAME (main) lane while it's busy must defer to the
    // queue, not touch the network again.
    await session.sendMessage('second message while busy', [], null, ConfigType.USER);
    expect(sendMock).toHaveBeenCalledTimes(1);
    expect(queue.queuedFor(null)).toEqual([{ text: 'second message while busy', files: [] }]);

    // A send on a DIFFERENT scope (a thread, not busy) must post immediately —
    // pinning turn_id-scoping over a single global "sending" flag.
    sendMock.mockResolvedValueOnce({ turn_id: 909, type: ConfigType.USER });
    await session.sendMessage('thread message', [], 555, ConfigType.USER);
    expect(sendMock).toHaveBeenCalledTimes(2);
    expect(sendMock).toHaveBeenLastCalledWith(
      'thread message', expect.any(Function), [], 555, ConfigType.USER,
    );
    expect(queue.queuedFor(555)).toEqual([]);

    // Cleanup: settle the still-pending first send so no dangling promise leaks
    // across tests.
    resolveFirst({ turn_id: 42, type: ConfigType.USER });
    await p1;
  });
});

describe('_drainLane — queued sends replay their files, not just their text', () => {
  it('replays a queued message\'s text AND its attached files into the resumed send call', async () => {
    const session = useSessionStore();
    const queue = useQueueStore();

    const file = new File(['contents'], 'photo.png', { type: 'image/png' });
    queue.enqueue(77, 'queued while the thread was busy', ConfigType.USER, [file]);

    sendMock.mockResolvedValueOnce({ turn_id: 77, type: ConfigType.USER });
    session._drainQueues();
    // sendMessage's own network call is fire-and-forget from _drainLane (`void`)
    // — flush microtasks so the underlying send() call has actually happened.
    await Promise.resolve();
    await Promise.resolve();

    expect(sendMock).toHaveBeenCalledTimes(1);
    expect(sendMock).toHaveBeenCalledWith(
      'queued while the thread was busy', expect.any(Function), [file], 77, ConfigType.USER,
    );
  });
});

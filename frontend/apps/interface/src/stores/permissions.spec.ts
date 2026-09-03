// @vitest-environment happy-dom
/**
 * Permissions store — feature specs for the queue's two feeds and the card
 * survival contract: the live `permission_request` frame and the
 * `GET /api/policies/pending` listing (re-read on every WS connect) overlap on
 * purpose, and `enqueue` dedupes on `request_id` so a card is never doubled;
 * `remove` is what a `permission_resolved` frame drives; `respond` stays
 * optimistic and re-reads the listing when the POST fails.
 *
 * Real Pinia, real store. Only the REST boundary (`api/policies.ts`) is mocked
 * — the transport this store would otherwise hit.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';

const { pendingMock, respondMock } = vi.hoisted(() => ({ pendingMock: vi.fn(), respondMock: vi.fn() }));
vi.mock('../api/policies', () => ({
  policies: {
    pending: () => pendingMock(),
    respond: (payload: unknown) => respondMock(payload),
  },
}));

import { usePermissionsStore } from './permissions';

const liveFrame = {
  type: 'permission_request',
  request_id: 'r1',
  action_id: 'pim',
  summary: 'Read the inbox',
  origin: { type: 'user', turn_id: 7, forked: true },
};

/** Drain the microtask queue — `respond` re-reads the listing without awaiting it. */
const flush = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  setActivePinia(createPinia());
  pendingMock.mockReset();
  respondMock.mockReset();
  respondMock.mockResolvedValue(undefined);
});

describe('enqueue', () => {
  it('keeps a live frame\'s request_id, action_id, summary and origin', () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    expect(store.queue).toEqual([
      { request_id: 'r1', action_id: 'pim', summary: 'Read the inbox', origin: { type: 'user', turn_id: 7, forked: true } },
    ]);
  });

  it('a frame without an origin (or a malformed one) queues with origin null — routed to the spine, never a guessed turn', () => {
    const store = usePermissionsStore();
    store.enqueue({ type: 'permission_request', request_id: 'a', action_id: 'search' });
    store.enqueue({ type: 'permission_request', request_id: 'b', action_id: 'search', origin: { type: 'user' } });
    store.enqueue({ type: 'permission_request', request_id: 'c', action_id: 'search', origin: 'nope' });
    expect(store.queue.map((r) => [r.request_id, r.origin, r.summary])).toEqual([
      ['a', null, ''],
      ['b', null, ''],
      ['c', null, ''],
    ]);
  });

  it('drops a frame missing request_id or action_id', () => {
    const store = usePermissionsStore();
    store.enqueue({ type: 'permission_request', action_id: 'pim' });
    store.enqueue({ type: 'permission_request', request_id: 'r9' });
    expect(store.queue).toEqual([]);
  });

  it('dedupes on request_id — the same gate pushed twice is one card', () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    store.enqueue(liveFrame);
    expect(store.queue).toHaveLength(1);
  });
});

describe('refreshPending', () => {
  it('restores the listed gates, skipping the one a live frame already queued, in listing order', async () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame); // arrived over the socket before the fetch answered
    pendingMock.mockResolvedValue([
      { request_id: 'r1', action_id: 'pim', summary: 'Read the inbox', origin: { type: 'user', turn_id: 7, forked: true } },
      { request_id: 'r2', action_id: 'email.send', summary: 'Send the reply', origin: { type: 'scheduled', turn_id: 3, forked: false } },
    ]);

    await store.refreshPending();

    expect(store.queue.map((r) => r.request_id)).toEqual(['r1', 'r2']);
    expect(store.queue[1]).toEqual({
      request_id: 'r2',
      action_id: 'email.send',
      summary: 'Send the reply',
      origin: { type: 'scheduled', turn_id: 3, forked: false },
    });
  });

  it('a failed fetch leaves the queue exactly as it was', async () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    pendingMock.mockRejectedValue(new Error('network down'));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    await store.refreshPending();

    expect(store.queue.map((r) => r.request_id)).toEqual(['r1']);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('drops a queued card the listing no longer has — its permission_resolved frame was missed while the socket was down', async () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    store.enqueue({ ...liveFrame, request_id: 'r2' });
    pendingMock.mockResolvedValue([{ ...liveFrame, request_id: 'r2' }]);

    await store.refreshPending();

    expect(store.queue.map((r) => r.request_id)).toEqual(['r2']);
  });

  it('a frame that arrives while the fetch is in flight survives the reconcile — only cards queued before the fetch can be pruned', async () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame); // queued before the fetch and absent from the listing: pruned
    let deliver!: (listing: unknown[]) => void;
    pendingMock.mockReturnValue(new Promise<unknown[]>((resolve) => { deliver = resolve; }));

    const refreshing = store.refreshPending();
    store.enqueue({ ...liveFrame, request_id: 'r3' }); // live frame, mid-fetch
    deliver([{ ...liveFrame, request_id: 'r2' }]);
    await refreshing;

    expect(store.queue.map((r) => r.request_id)).toEqual(['r3', 'r2']);
  });
});

describe('remove / respond', () => {
  it('remove drops that gate\'s card and nothing else; an unknown id is a no-op', () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    store.enqueue({ ...liveFrame, request_id: 'r2' });
    store.remove('r1');
    expect(store.queue.map((r) => r.request_id)).toEqual(['r2']);
    store.remove('nope');
    expect(store.queue.map((r) => r.request_id)).toEqual(['r2']);
  });

  it('respond dismisses the card before the round-trip and posts the decision', async () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    let settle!: () => void;
    respondMock.mockReturnValue(new Promise<void>((resolve) => { settle = resolve; }));

    const responding = store.respond('r1', true);
    expect(store.queue).toEqual([]); // gone before the network answers
    settle();
    await responding;

    expect(respondMock).toHaveBeenCalledWith({ request_id: 'r1', approved: true });
  });

  it('a failed respond brings the card back when the listing still has the gate — it is still parked on the backend', async () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    respondMock.mockRejectedValue(new Error('500'));
    pendingMock.mockResolvedValue([liveFrame]);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    await store.respond('r1', false);
    await flush();

    expect(store.queue.map((r) => r.request_id)).toEqual(['r1']);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('a failed respond leaves the card gone when the listing no longer has the gate — it was answered or cancelled elsewhere', async () => {
    const store = usePermissionsStore();
    store.enqueue(liveFrame);
    respondMock.mockRejectedValue(new Error('500'));
    pendingMock.mockResolvedValue([]);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    await store.respond('r1', true);
    await flush();

    expect(pendingMock).toHaveBeenCalledTimes(1); // re-read, not a blind re-add
    expect(store.queue).toEqual([]);
    warn.mockRestore();
  });
});

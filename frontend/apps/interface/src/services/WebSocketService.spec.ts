// @vitest-environment happy-dom
/**
 * WebSocketService — 401 auth-expiry detection on the two HTTP send paths
 * (POST /api/thread via send/postChat, POST /api/action via sendAction).
 *
 * Regression spec for #1878 part 1: a 401 on either send path used to surface
 * as a generic 'Chat request failed.' with no auth-expiry signal, so a message
 * sent after session expiry looked accepted while the spinner hung — the
 * heartbeat only caught up 5 minutes later. Now a 401 fires the registered
 * onAuthError handler (mirrors ApiClient.fail401), closing the gap.
 *
 * Only the transport edge is stubbed (`fetch`, the `WebSocket` constructor) —
 * the real WebSocketService runs, the real send/sendAction/postChat execute,
 * and the real callback registration drives the assertion.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { WebSocketService } from '@chalie/shared';

/** Minimal fake WebSocket — send/postChat only need open/onopen to pass the
 *  isConnected gate; this spec never exercises frames. The real connect()
 *  assigns ``ws.onopen`` synchronously after ``new WebSocket()``; onopen fires
 *  on the next microtask so that handler is in place before it runs (mirrors
 *  the browser, which fires onopen after the handshake, never inline). */
class FakeWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  readyState = FakeWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  send(): void { /* not under test */ }
  close(): void { /* not under test */ }
  constructor() {
    queueMicrotask(() => this.onopen?.());
  }
}

/** A fetch Response stub. status defaults to 200; tests override per-call. */
function fakeResponse(status: number, body: unknown = {}): Response {
  return {
    status,
    ok: status >= 200 && status < 300,
    json: async () => body,
  } as Response;
}

/** Drain the microtask queue enough times to settle a fetch().then(async ...)
 *  chain: FakeWebSocket.onopen (1), fetch resolve (1), resp.json() (1). */
async function flushMicrotasks(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

beforeEach(() => {
  vi.stubGlobal('WebSocket', FakeWebSocket);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('WebSocketService — 401 auth-expiry on send paths', () => {
  it('fires onAuthError when POST /api/thread returns 401 (send)', async () => {
    const ws = new WebSocketService(() => '', () => '');
    const authError = vi.fn();
    ws.onAuthError(authError);
    // Force isConnected true so send() reaches postChat's fetch. connect()
    // assigns onopen synchronously; the FakeWebSocket fires it on the next
    // microtask, which sets `connected = true`.
    ws.connect();
    await flushMicrotasks();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fakeResponse(401)));
    const sendFailure = vi.fn();

    await ws.send('hello', sendFailure);

    expect(authError).toHaveBeenCalledTimes(1);
    // The user still gets a failure signal so the spinner/echo releases.
    expect(sendFailure).toHaveBeenCalledTimes(1);
  });

  it('does NOT fire onAuthError on a non-401 failure (send)', async () => {
    const ws = new WebSocketService(() => '', () => '');
    const authError = vi.fn();
    ws.onAuthError(authError);
    ws.connect();
    await flushMicrotasks();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fakeResponse(500)));
    const sendFailure = vi.fn();

    await ws.send('hello', sendFailure);

    expect(authError).not.toHaveBeenCalled();
    expect(sendFailure).toHaveBeenCalledTimes(1);
  });

  it('fires onAuthError when POST /api/action returns 401 (sendAction)', async () => {
    const ws = new WebSocketService(() => '', () => '');
    const authError = vi.fn();
    ws.onAuthError(authError);
    ws.connect();
    await flushMicrotasks();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fakeResponse(401)));
    const onError = vi.fn();
    const onDone = vi.fn();

    ws.sendAction({ foo: 'bar' }, { onError, onDone });
    // sendAction's .then awaits resp.json() before invoking callbacks — flush
    // the fetch resolution + the json() resolution.
    await flushMicrotasks();

    expect(authError).toHaveBeenCalledTimes(1);
    // The card still releases its loading state (onDone fires) — no hang.
    expect(onDone).toHaveBeenCalledTimes(1);
  });
});

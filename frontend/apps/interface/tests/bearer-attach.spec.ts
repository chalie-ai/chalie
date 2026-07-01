import { expect, test } from '@playwright/test';

// Real-hot-path feature test for the shared bearer-attach. Drives the
// genuine ApiClient / WebSocketService singletons against the live backend and
// observes the ACTUAL outbound HTTP headers + the ACTUAL /ws handshake. Zero
// mocks: localStorage is the real token store, the clients are the real ones.
//
// Security invariant: the bearer token must NEVER travel in the /ws URL — a
// query-string credential leaks into reverse-proxy/access logs, Referer, and
// history. The native client sends it as the FIRST WS frame
// ({"type":"auth","token":...}) instead.

const TEST_TOKEN = 'feature-test-bearer-token-xyz';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#loadingOverlay')).toBeHidden({ timeout: 15_000 });
  // Start every case from the unpaired (web) state.
  await page.evaluate(() => localStorage.removeItem('chalie_access_token'));
});

test('no token => web path unchanged: no Authorization header, no token on /ws', async ({
  page,
}) => {
  // 1) HTTP: drive the REAL ApiClient singleton and capture the outbound headers.
  const reqHeadersPromise = page.waitForRequest((r) => r.url().includes('/ready'));
  await page.evaluate(async () => {
    const mod = await import('/src/index.ts');
    await mod.api.ready();
  });
  const req = await reqHeadersPromise;
  expect(req.headers()['authorization']).toBeUndefined();

  // 2) WS: the live socket opened on load (or reconnect) carries no token — not
  // in the URL, and no auth frame is sent on the cookie path.
  const wsPromise = page.waitForEvent('websocket');
  await page.evaluate(async () => {
    const mod = await import('/src/index.ts');
    const ws = mod.getWebSocket();
    ws.close();
    ws.connect();
  });
  const ws = await wsPromise;
  expect(ws.url()).toContain('/ws');
  expect(ws.url()).not.toContain('token=');
});

test('token present => Authorization: Bearer on HTTP and auth frame on /ws (never in the URL)', async ({
  page,
}) => {
  // Write the token through the REAL accessor, then drive the REAL singletons.
  await page.evaluate(async (t) => {
    const mod = await import('/src/index.ts');
    mod.setToken(t);
  }, TEST_TOKEN);

  // 1) HTTP: the real ApiClient now spreads the bearer header.
  const reqHeadersPromise = page.waitForRequest((r) => r.url().includes('/ready'));
  await page.evaluate(async () => {
    const mod = await import('/src/index.ts');
    await mod.api.ready();
  });
  const req = await reqHeadersPromise;
  expect(req.headers()['authorization']).toBe(`Bearer ${TEST_TOKEN}`);

  // 2) WS: a reconnect opens a bare /ws (NO token in the URL) and sends the
  // token as the first WS frame. Capture the first sent frame.
  const wsPromise = page.waitForEvent('websocket');
  await page.evaluate(async () => {
    const mod = await import('/src/index.ts');
    const ws = mod.getWebSocket();
    ws.close();
    ws.connect();
  });
  const ws = await wsPromise;
  expect(ws.url()).toContain('/ws');
  expect(ws.url()).not.toContain('token=');
  // The first client→server frame must be the auth handshake carrying the token.
  const frame = await ws.waitForEvent('framesent', { timeout: 5_000 });
  const payload =
    typeof frame.data === 'string' ? frame.data : frame.data?.toString() ?? '';
  const parsed = JSON.parse(payload) as { type?: string; token?: string };
  expect(parsed.type).toBe('auth');
  expect(parsed.token).toBe(TEST_TOKEN);

  // Cleanup so a later spec/run starts unpaired.
  await page.evaluate(async () => {
    const mod = await import('/src/index.ts');
    mod.setToken('');
  });
});

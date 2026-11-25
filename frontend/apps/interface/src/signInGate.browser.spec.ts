/**
 * The chat loads only for a signed-in user with a provider — feature spec, in
 * a real browser.
 *
 * Before the chat mounts it asks the server who is signed in. Nothing mounts,
 * and nothing else is asked, until the answer comes. A signed-in user with a
 * provider gets the chat; anyone else is sent on: with no account, or when the
 * server answers the check with an error, to the on-boarding; signed out, to
 * the sign-in page, which brings them back to where they were; with no
 * provider, to the Brain to set one up. A server that cannot be reached at all
 * lets the chat load.
 *
 * Real throughout: the chat as vite builds it with its own config, run in
 * Chrome. The fakes are the backend, answered at the network, and the server
 * end of the socket. The pages a user is sent on to are stand-ins. Every other
 * host is refused.
 */
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import type { Rollup } from 'vite';
import { chromium } from '@playwright/test';
import type { Browser, BrowserContext, Page, Route } from '@playwright/test';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const ORIGIN = 'http://chalie.test';

const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.woff2': 'font/woff2',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

const SIGNED_IN = {
  has_master_account: true,
  has_session: true,
  has_providers: true,
  vault_state: 'unlocked',
  has_vision_provider: true,
};

/** The pages a user is sent on to: each a stand-in that says where it is. */
const ELSEWHERE = ['/login/', '/on-boarding/', '/brain/'];

type Reply = (route: Route) => Promise<void>;

/** How the sign-in check is answered; every check after the first is answered signed in. */
let firstCheck: Reply;
let checks: number;
/** Every API path the page asked for, in order, and each socket it opened. */
let asked: string[];
let sockets: number;

let files: Map<string, { type: string; body: string | Uint8Array }>;
let browser: Browser;
let context: BrowserContext;
let page: Page;

/** Build the chat in memory, keyed by the path the browser asks for. */
async function bundle(): Promise<Map<string, { type: string; body: string | Uint8Array }>> {
  const result = (await build({
    root: ROOT,
    configFile: fileURLToPath(new URL('../vite.config.ts', import.meta.url)),
    logLevel: 'silent',
    build: { write: false },
  })) as Rollup.RollupOutput;
  const out = new Map<string, { type: string; body: string | Uint8Array }>();
  for (const f of result.output) {
    const type = CONTENT_TYPES[f.fileName.slice(f.fileName.lastIndexOf('.'))];
    if (!type) throw new Error(`No content type for the bundled ${f.fileName}`);
    out.set(`/${f.fileName}`, { type, body: f.type === 'chunk' ? f.code : f.source });
  }
  return out;
}

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

const answer = (body: unknown, status = 200): Reply => (route) => json(route, body, status);

/** The backend: the chat's page and files, the sign-in check, and a 404 for anything else. */
function serve(route: Route): Promise<void> {
  const url = new URL(route.request().url());
  if (url.origin !== ORIGIN) return route.abort();
  if (url.pathname === '/') {
    const index = files.get('/index.html')!;
    return route.fulfill({ contentType: index.type, body: Buffer.from(index.body) });
  }
  if (ELSEWHERE.includes(url.pathname)) {
    return route.fulfill({ contentType: 'text/html', body: '<p id="elsewhere"></p>' });
  }
  const file = files.get(url.pathname);
  if (file) return route.fulfill({ contentType: file.type, body: Buffer.from(file.body) });
  if (url.pathname.startsWith('/api/')) asked.push(url.pathname);
  if (url.pathname === '/api/auth/status') {
    checks += 1;
    return checks === 1 ? firstCheck(route) : json(route, SIGNED_IN);
  }
  return json(route, { error: 'unrouted' }, 404);
}

const mounted = () => page.locator('#overlayRoot');

describe('the chat loads only for a signed-in user with a provider', () => {
  beforeAll(async () => {
    files = await bundle();
    browser = await chromium.launch({ channel: 'chrome' });
  }, 120_000);

  afterAll(async () => {
    await browser?.close();
  });

  beforeEach(async () => {
    checks = 0;
    asked = [];
    sockets = 0;
    context = await browser.newContext();
    page = await context.newPage();
    await page.route('**/*', serve);
    await page.routeWebSocket(`${ORIGIN.replace(/^http/, 'ws')}/ws`, () => {
      sockets += 1;
    });
  });

  afterEach(async () => {
    await context?.close();
  });

  it('nothing mounts and nothing else is asked until the check answers; signed in, the chat then mounts and connects', async () => {
    let release!: () => void;
    const answered = new Promise<void>((resolve) => {
      release = resolve;
    });
    firstCheck = async (route) => {
      await answered;
      await json(route, SIGNED_IN);
    };

    // The page has loaded and its script has run: all that waits is the check.
    await page.goto(`${ORIGIN}/`);
    await expect.poll(() => checks).toBe(1);
    expect(await mounted().count()).toBe(0);
    expect(asked).toEqual(['/api/auth/status']);
    expect(sockets).toBe(0);

    release();
    await mounted().waitFor({ state: 'attached' });
    await expect.poll(() => sockets).toBe(1);
    expect(asked.length).toBeGreaterThan(1);
  });

  it.each([
    ['signed out, to the sign-in page, which brings them back', { ...SIGNED_IN, has_session: false }, 200, '/login/?next=%2F'],
    ['with no account, to the on-boarding', { ...SIGNED_IN, has_master_account: false, has_session: false }, 200, '/on-boarding/'],
    ['with no provider, to the Brain', { ...SIGNED_IN, has_providers: false }, 200, '/brain/'],
    ['when the server fails the check, to the on-boarding', { error: 'Internal error' }, 500, '/on-boarding/'],
    ['when the server refuses the check, to the on-boarding', { error: 'Unauthorized' }, 401, '/on-boarding/'],
  ])('a user is sent on %s, and the chat neither mounts nor connects', async (_, body, status, to) => {
    firstCheck = answer(body, status);

    await page.goto(`${ORIGIN}/`);
    await page.waitForURL(`${ORIGIN}${to}`);
    await page.locator('#elsewhere').waitFor({ state: 'attached' });

    expect(asked).toEqual(['/api/auth/status']);
    expect(sockets).toBe(0);
  });

  it('a server that cannot be reached lets the chat load', async () => {
    firstCheck = (route) => route.abort('connectionrefused');

    await page.goto(`${ORIGIN}/`);
    await mounted().waitFor({ state: 'attached' });
    await expect.poll(() => sockets).toBe(1);
    expect(page.url()).toBe(`${ORIGIN}/`);
  });
});

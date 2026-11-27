/**
 * The Brain heals a dead live connection when its window is focused again —
 * feature spec, in a real browser.
 *
 * After the machine sleeps, the socket can be dead without ever closing: no
 * close event comes, so nothing reconnects, and the watchdog's own timers did
 * not run while it slept. Focusing the window checks the connection at once: a
 * socket silent past the stale threshold is dropped and a new one opened, the
 * open asks are read again, and an ask that arrives on the new socket shows in
 * the ask dock. A focus while the socket is live keeps it.
 *
 * Real throughout: the Brain as vite builds it with its own config, run in
 * Chrome. The fakes are the backend, answered at the network, the server end
 * of the socket, and the clock, moved past the stale threshold without running
 * a timer, as a sleep moves it. Every other host is refused.
 */
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import type { Rollup } from 'vite';
import { chromium } from '@playwright/test';
import type { Browser, BrowserContext, Page, Route, WebSocketRoute } from '@playwright/test';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const ORIGIN = 'http://chalie.test';
const BASE = '/brain/';
/** Longer than the socket's stale threshold. */
const SLEEP_MS = 2 * 60 * 1000;

const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.woff2': 'font/woff2',
};

const SIGNED_IN = {
  has_master_account: true,
  has_session: true,
  has_providers: true,
  vault_state: 'unlocked',
  has_vision_provider: true,
};

const QUESTION = {
  type: 'ask_opened',
  ask_id: 'q1',
  kind: 'question',
  asked_at: new Date().toISOString(),
  data: {
    question: 'Which ferry should I book?',
    answers: [
      { label: 'The 08:15', description: '' },
      { label: 'The 17:40', description: '' },
    ],
    multi_select: false,
    origin: { type: 'user', turn_id: 7, forked: false },
  },
};

/** Each socket the page opened, and whether the page has closed it. */
let sockets: { route: WebSocketRoute; closed: boolean }[];
let openReads: number;

let files: Map<string, { type: string; body: string | Uint8Array }>;
let browser: Browser;
let context: BrowserContext;
let page: Page;

/** Build the Brain in memory, keyed by the path the browser asks for. */
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
    out.set(`${BASE}${f.fileName}`, { type, body: f.type === 'chunk' ? f.code : f.source });
  }
  return out;
}

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

/** The backend: the built files, the sign-in check and the open-asks listing. */
function serve(route: Route): Promise<void> {
  const url = new URL(route.request().url());
  if (url.origin !== ORIGIN) return route.abort();
  const file = files.get(url.pathname) ?? (url.pathname.startsWith(BASE) ? files.get(`${BASE}index.html`) : undefined);
  if (file) return route.fulfill({ contentType: file.type, body: Buffer.from(file.body) });
  if (url.pathname === '/api/auth/status') return json(route, SIGNED_IN);
  if (url.pathname === '/api/asks/open') {
    openReads += 1;
    return json(route, { success: true, result: [], pagination: { page: 1, limit: 50, total: 0 } });
  }
  return json(route, { error: 'unrouted' }, 404);
}

/** Open the Brain with the clock under the spec's hand, and wait for the first connect's read of the open asks. */
async function load(): Promise<void> {
  await page.clock.install();
  await page.goto(`${ORIGIN}${BASE}lists`);
  await expect.poll(() => openReads).toBe(1);
}

const refocus = (): Promise<void> => page.evaluate(() => void globalThis.dispatchEvent(new Event('focus')));

const questionTitle = () => page.locator('.ask-dock .qcard__title');

describe('the Brain heals a dead live connection when its window is focused again', () => {
  beforeAll(async () => {
    files = await bundle();
    browser = await chromium.launch({ channel: 'chrome' });
  }, 120_000);

  afterAll(async () => {
    await browser?.close();
  });

  beforeEach(async () => {
    sockets = [];
    openReads = 0;
    context = await browser.newContext();
    page = await context.newPage();
    await page.route('**/*', serve);
    await page.routeWebSocket(`${ORIGIN.replace(/^http/, 'ws')}/ws`, (route) => {
      const socket = { route, closed: false };
      sockets.push(socket);
      route.onClose(() => {
        socket.closed = true;
      });
    });
  });

  afterEach(async () => {
    await context?.close();
  });

  it('a focus after a sleep drops the silent socket, opens a new one, reads the open asks again, and an ask that arrives on it shows', async () => {
    await load();
    expect(sockets).toHaveLength(1);

    const now = await page.evaluate(() => Date.now());
    await page.clock.setSystemTime(now + SLEEP_MS);
    await refocus();

    await expect.poll(() => sockets.length).toBe(2);
    await expect.poll(() => sockets[0].closed).toBe(true);
    await expect.poll(() => openReads).toBe(2);

    sockets[1].route.send(JSON.stringify(QUESTION));
    await questionTitle().waitFor();
    expect(await questionTitle().textContent()).toBe('Which ferry should I book?');
  });

  it('a focus while the socket is live keeps it', async () => {
    await load();

    await refocus();
    sockets[0].route.send(JSON.stringify(QUESTION));
    await questionTitle().waitFor();

    expect(sockets).toHaveLength(1);
    expect(sockets[0].closed).toBe(false);
    expect(openReads).toBe(1);
  });
});

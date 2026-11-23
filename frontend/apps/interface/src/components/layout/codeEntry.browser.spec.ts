/**
 * A sign-in code entered the way an authenticator shows it — feature spec, in
 * a real browser.
 *
 * Many authenticator apps show a code as two groups, "048 217". A person
 * types it key by key, or pastes it whole. The field keeps only its digits, so
 * either way it must end up holding all six, with Sign in on. A paste lands
 * in one insertion, and a browser cuts an insertion short to any length limit
 * on the field before the page sees it: a limit of six would keep "048 21",
 * five digits, and Sign in would stay off. A value set from script is never
 * cut short, which is why this runs in a browser and not in happy-dom.
 *
 * Real throughout: the input dock's asks, Pinia and the api layer, bundled by
 * vite with the app's own config and run in Chrome, typed into through
 * Playwright's keyboard. The pasted code is inserted as one piece of text,
 * without touching the machine's clipboard. The only fake is the backend,
 * answered at the network; every other host is refused.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import type { Rollup } from 'vite';
import { chromium } from '@playwright/test';
import type { Browser, Page, Route } from '@playwright/test';

const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const ORIGIN = 'http://chalie.test';
const ENTRY = 'virtual:code-entry-page';

/** The code as an authenticator shows it, and the six digits it stands for. */
const SHOWN = '048 217';
const CODE = '048217';

/** The off-screen asks of the input dock on their own, loaded from the open asks as on connect. */
const PAGE_SCRIPT = `
import '@chalie/shared/styles/main.scss';
import '/src/styles/interface.scss';
import { createApp, h } from 'vue';
import { createPinia } from 'pinia';
import { useAsksStore } from '@chalie/shared';
import OffScreenAsks from '/src/components/layout/OffScreenAsks.vue';
const pinia = createPinia();
createApp({ render: () => h(OffScreenAsks) }).use(pinia).mount('#app');
void useAsksStore(pinia).refreshOpen();
`;

const CONTENT_TYPES: Record<string, string> = { '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2' };

/** A UniFi sign-in from a scheduled run, waiting on its code. */
const SIGN_IN = {
  ask_id: 'c1',
  kind: 'code',
  asked_at: '2026-11-20T09:00:00+00:00',
  data: { service: 'unifi', origin: null },
};

let files: Map<string, { type: string; body: string | Uint8Array }>;
let browser: Browser;
let page: Page;

/** Bundle the page in memory, keyed by the path the browser asks for. */
async function bundle(): Promise<Map<string, { type: string; body: string | Uint8Array }>> {
  const result = (await build({
    root: ROOT,
    configFile: fileURLToPath(new URL('../../../vite.config.ts', import.meta.url)),
    logLevel: 'silent',
    plugins: [
      {
        name: 'code-entry-page',
        resolveId: (id) => (id === ENTRY ? `\0${ENTRY}` : null),
        load: (id) => (id === `\0${ENTRY}` ? PAGE_SCRIPT : null),
      },
    ],
    build: { write: false, cssCodeSplit: false, rollupOptions: { input: ENTRY, output: { entryFileNames: 'page.js' } } },
  })) as Rollup.RollupOutput;
  const out = new Map<string, { type: string; body: string | Uint8Array }>();
  for (const f of result.output) {
    const type = CONTENT_TYPES[f.fileName.slice(f.fileName.lastIndexOf('.'))];
    if (!type) throw new Error(`No content type for the bundled ${f.fileName}`);
    out.set(`/${f.fileName}`, { type, body: f.type === 'chunk' ? f.code : f.source });
  }
  const styles = [...out.keys()].filter((p) => p.endsWith('.css'));
  out.set('/', {
    type: 'text/html',
    body:
      '<!doctype html><html><head>' +
      styles.map((p) => `<link rel="stylesheet" href="${p}">`).join('') +
      '</head><body><div id="app"></div><script type="module" src="/page.js"></script></body></html>',
  });
  return out;
}

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

/** The backend: the open asks, and an answer that closes the sign-in. */
function serve(route: Route): Promise<void> {
  const request = route.request();
  const url = new URL(request.url());
  if (url.origin !== ORIGIN) return route.abort();
  const file = files.get(url.pathname);
  if (file) return route.fulfill({ contentType: file.type, body: Buffer.from(file.body) });
  if (url.pathname === '/api/asks/open' && request.method() === 'GET') {
    return json(route, { success: true, result: [SIGN_IN], pagination: { page: 1, limit: 50, total: 1 } });
  }
  if (url.pathname === `/api/asks/reply/${SIGN_IN.ask_id}` && request.method() === 'POST') {
    return route.fulfill({ status: 204 });
  }
  return json(route, { error: 'unrouted' }, 404);
}

const row = () => page.locator('.offscreen-asks > .activity');
const field = () => page.getByLabel('6-digit code');
const signIn = () => page.getByRole('button', { name: 'Sign in' });

describe('a sign-in code entered the way an authenticator shows it', () => {
  beforeAll(async () => {
    files = await bundle();
    browser = await chromium.launch({ channel: 'chrome' });
    page = await browser.newPage();
    await page.route('**/*', serve);
  }, 120_000);

  afterAll(async () => {
    await browser?.close();
  });

  beforeEach(async () => {
    await page.goto(`${ORIGIN}/`);
    await row().waitFor();
  });

  it('typed key by key, the field keeps its six digits and Sign in comes on', async () => {
    await field().focus();
    await page.keyboard.type(SHOWN);

    expect([await field().inputValue(), await signIn().isEnabled()]).toEqual([CODE, true]);
  });

  it('pasted whole, the field keeps all six digits, Sign in comes on, and the code it sends is those six', async () => {
    await field().focus();
    await page.keyboard.insertText(SHOWN);

    expect([await field().inputValue(), await signIn().isEnabled()]).toEqual([CODE, true]);

    const answer = page.waitForRequest((r) => r.method() === 'POST' && new URL(r.url()).pathname.startsWith('/api/asks/reply/'));
    await signIn().click();
    const sent = await answer;
    expect([new URL(sent.url()).pathname, sent.postDataJSON()]).toEqual([`/api/asks/reply/${SIGN_IN.ask_id}`, { value: { code: CODE } }]);
    await row().waitFor({ state: 'detached' });
  });
});

/**
 * Quoting a reply under a real mouse — feature spec, in a real browser.
 *
 * Selecting text in one of Chalie's replies floats a Quote button over it;
 * Quote opens a one-line reply field, and Enter hands the quote and the reply
 * to the composer as a block. The composer sends nothing until the user does,
 * and then sends one ordinary message: the blocks, a blank line, the typed
 * text. A selection made by dragging the mouse, and the button pressed without
 * losing it, only happen in a real browser — which is why this is not in
 * happy-dom.
 *
 * Real throughout: a Chalie reply, the InputDock, the selection bar, Pinia,
 * the session send path and the shared socket service, bundled by vite with
 * the app's own config and run in Chrome, driven through Playwright's mouse
 * and keyboard. The only fakes are at the network: the backend's answers and
 * an open socket; every other host is refused.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import type { Rollup } from 'vite';
import { chromium } from '@playwright/test';
import type { Browser, Page, Route } from '@playwright/test';

const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const ORIGIN = 'http://chalie.test';
const ENTRY = 'virtual:quote-selection-page';

/** A chat with one reply of Chalie's and a line that is not one, its dock and
 *  the selection bar, styled and mounted as the app mounts them. */
const PAGE_SCRIPT = `
import '@chalie/shared/styles/main.scss';
import '/src/styles/interface.scss';
import { createApp, h } from 'vue';
import { createPinia } from 'pinia';
import { getWebSocket } from '@chalie/shared';
import ChalieBubble from '/src/components/conversation/ChalieBubble.vue';
import InputDock from '/src/components/layout/InputDock.vue';
import QuoteSelection from '/src/components/conversation/QuoteSelection.vue';
const message = {
  id: '501',
  role: 'assistant',
  content: '<p>Paris is the capital of France.</p><p>Lyon is the third-largest city.</p>',
  timestamp: '2026-01-01 00:00:00',
  day: '2026-01-01',
  created_at: '2026-01-01T00:00:00Z',
  turn_id: 5,
};
createApp({
  render: () => [
    h('main', { 'data-quote-dock': 'main_spine_dock' }, [
      h('p', { class: 'not-a-reply' }, 'A line that is not a reply of Chalie.'),
      h(ChalieBubble, { message }),
    ]),
    h(InputDock, { dockId: 'main_spine_dock' }),
    h(QuoteSelection),
  ],
}).use(createPinia()).mount('#app');
getWebSocket().connect();
`;

const CONTENT_TYPES: Record<string, string> = { '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2' };

const MODELS = [{ id: 3, name: 'Opus', model: 'claude-opus-4', context_window: 200_000 }];

/** Every chat send's text, in order. */
const sent: string[] = [];

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
        name: 'quote-selection-page',
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

/** The backend: the model listing, the thread settings read, and the send. */
async function serve(route: Route): Promise<void> {
  const request = route.request();
  const url = new URL(request.url());
  if (url.origin !== ORIGIN) return route.abort();
  const file = files.get(url.pathname);
  if (file) return route.fulfill({ contentType: file.type, body: Buffer.from(file.body) });
  if (url.pathname === '/api/providers/all') {
    return json(route, { success: true, result: MODELS, pagination: { page: 1, limit: 100, total: MODELS.length } });
  }
  if (url.pathname === '/api/threads/settings/-1') {
    return json(route, { success: true, result: { provider_id: 3, thinking_level: 'auto', turn_id: null } });
  }
  if (request.method() === 'POST' && url.pathname === '/api/threads/-1') {
    const form = await new Response(request.postData(), {
      headers: { 'Content-Type': request.headers()['content-type'] },
    }).formData();
    sent.push(String(form.get('text')));
    return json(route, { success: true, result: { turn_id: 7, type: 'user' } });
  }
  return json(route, { error: 'unrouted' }, 404);
}

/** Drag the mouse across `text` where it first appears inside `scope`. */
async function select(text: string, scope: string): Promise<void> {
  const at = await page.evaluate(
    ([needle, root]) => {
      const walker = document.createTreeWalker(document.querySelector(root)!, NodeFilter.SHOW_TEXT);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const i = node.textContent!.indexOf(needle);
        if (i === -1) continue;
        const r = document.createRange();
        r.setStart(node, i);
        r.setEnd(node, i + 1);
        const first = r.getBoundingClientRect();
        r.setStart(node, i + needle.length - 1);
        r.setEnd(node, i + needle.length);
        const last = r.getBoundingClientRect();
        return { from: first.left + 1, to: last.right - 1, y: (first.top + first.bottom) / 2 };
      }
      throw new Error(`"${needle}" is not on the page`);
    },
    [text, scope] as const,
  );
  await page.mouse.move(at.from, at.y);
  await page.mouse.down();
  await page.mouse.move(at.to, at.y, { steps: 8 });
  await page.mouse.up();
}

const rows = async (): Promise<string[]> =>
  (await page.locator('.quote-rows__row').allTextContents()).map((t) => t.trim());

describe('quoting a reply under a real mouse', () => {
  beforeAll(async () => {
    files = await bundle();
    browser = await chromium.launch({ channel: 'chrome' });
    page = await browser.newPage();
    await page.route('**/*', serve);
    // An open socket, as a healthy server holds one; it says nothing.
    await page.routeWebSocket(`${ORIGIN.replace('http', 'ws')}/ws`, () => {});
  }, 120_000);

  afterAll(async () => {
    await browser?.close();
  });

  beforeEach(async () => {
    sent.length = 0;
    await page.goto(`${ORIGIN}/`);
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.locator('#chatInput').waitFor();
  });

  it('quotes a dragged selection with a reply, another without one, and sends both with the typed text', async () => {
    await select('Paris is the capital', '.speech-form__text');
    await page.locator('.quote-bar__btn', { hasText: 'Quote' }).click();
    // The reply field opens over the quote, focused, the quote shown above it.
    expect(await page.locator('.quote-pop__quote').textContent()).toBe('Paris is the capital');
    await page.keyboard.type('and Lyon?');
    await page.keyboard.press('Enter');

    expect(await page.locator('.quote-pop').count()).toBe(0);
    expect(await rows()).toEqual(['“Paris is the capital”With reply']);
    // Nothing went out: the quote waits for the user to send.
    expect(sent).toEqual([]);
    // The message field has the focus back, to type alongside the quote.
    expect(await page.evaluate(() => document.activeElement?.id)).toBe('chatInput');

    // A triple-click takes the whole paragraph; Enter with no reply adds it bare.
    await page.locator('.speech-form__text p', { hasText: 'Lyon' }).click({ clickCount: 3 });
    await page.locator('.quote-bar__btn').click();
    await page.keyboard.press('Enter');
    expect(await rows()).toEqual(['“Paris is the capital”With reply', '“Lyon is the third-largest city.”']);

    await page.locator('#chatInput').fill('Which is bigger?');
    await page.locator('button[aria-label="Send message"]').click();
    await expect.poll(() => sent.length).toBe(1);
    // The browser's form encoding carries each line break as CR LF, as it does
    // for any message typed over several lines.
    expect(sent[0]).toBe(
      '[quote_block]{"quote": "Paris is the capital", "reply": "and Lyon?"}[/quote_block]\r\n' +
        '[quote_block]{"quote": "Lyon is the third-largest city.", "reply": ""}[/quote_block]\r\n' +
        '\r\n' +
        'Which is bigger?',
    );
    expect(await rows()).toEqual([]);
  });

  it('offers no Quote outside a reply, and Escape or a click away drops it', async () => {
    await select('not a reply', '.not-a-reply');
    // The bar follows a selection on the next tick; give it more than that.
    await page.waitForTimeout(300);
    expect(await page.locator('.quote-bar').count()).toBe(0);

    await select('the capital', '.speech-form__text');
    await page.locator('.quote-bar__btn').waitFor();
    await page.mouse.click(5, 5);
    expect(await page.locator('.quote-bar').count()).toBe(0);

    await select('the capital', '.speech-form__text');
    await page.locator('.quote-bar__btn').click();
    await page.keyboard.type('never mind');
    await page.keyboard.press('Escape');
    expect(await page.locator('.quote-pop').count()).toBe(0);
    expect(await rows()).toEqual([]);
  });
});

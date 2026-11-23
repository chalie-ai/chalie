/**
 * The model menu under a real mouse — feature spec, in a real browser.
 *
 * While a person's click is dispatched, a browser runs pending microtasks
 * between one listener and the next, so Vue re-renders the menu before the
 * click reaches the document. Choosing a model swaps the model rows out for its
 * levels, and the back row swaps them in again: the row that was clicked has
 * left the page by the time the document hears the click, and the menu must
 * still count that click as its own. A click dispatched from script runs every
 * listener first and re-renders after, which is why this runs in a browser and
 * not in happy-dom.
 *
 * Real throughout: the InputDock, Pinia and the api layer, bundled by vite with
 * the app's own config and run in Chrome, clicked through Playwright's mouse.
 * The only fake is the backend, answered at the network; every other host is
 * refused.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import type { Rollup } from 'vite';
import { chromium } from '@playwright/test';
import type { Browser, Page, Route } from '@playwright/test';

const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const ORIGIN = 'http://chalie.test';
const ENTRY = 'virtual:model-menu-page';

/** The footer dock on its own, styled and mounted as the app mounts it. */
const PAGE_SCRIPT = `
import '@chalie/shared/styles/main.scss';
import '/src/styles/interface.scss';
import { createApp, h } from 'vue';
import { createPinia } from 'pinia';
import InputDock from '/src/components/layout/InputDock.vue';
createApp({ render: () => h(InputDock, { dockId: 'main_spine_dock' }) }).use(createPinia()).mount('#app');
`;

const CONTENT_TYPES: Record<string, string> = { '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2' };

const MODELS = [
  { id: 3, name: 'Opus', model: 'claude-opus-4', context_window: 200_000 },
  { id: 4, name: 'Sonnet', model: 'claude-sonnet-4', context_window: 100_000 },
];

interface Settings {
  provider_id: number;
  thinking_level: string;
}

/** The channel's one thread, which -1 names. */
const THREAD = 12;
let thread: Settings;
const saves: { path: string; body: Settings }[] = [];

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
        name: 'model-menu-page',
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

/** The backend: the model listing and the thread settings read and save. */
function serve(route: Route): Promise<void> {
  const request = route.request();
  const url = new URL(request.url());
  if (url.origin !== ORIGIN) return route.abort();
  const file = files.get(url.pathname);
  if (file) return route.fulfill({ contentType: file.type, body: Buffer.from(file.body) });
  if (url.pathname === '/api/providers/all') {
    return json(route, { success: true, result: MODELS, pagination: { page: 1, limit: 100, total: MODELS.length } });
  }
  if (url.pathname === '/api/threads/settings/-1') {
    if (request.method() === 'POST') {
      thread = request.postDataJSON() as Settings;
      saves.push({ path: url.pathname + url.search, body: thread });
    }
    return json(route, { success: true, result: { ...thread, turn_id: THREAD } });
  }
  return json(route, { error: 'unrouted' }, 404);
}

const levels = async (): Promise<string[]> =>
  (await page.locator('#modelMenu [data-level]').allTextContents()).map((t) => t.trim());
const modelRows = async (): Promise<string[]> =>
  (await page.locator('#modelMenu [data-model]').allTextContents()).map((t) => t.trim());
const label = async (): Promise<string | null> => page.locator('#modelLabel').textContent();

describe('the model menu under a real mouse', () => {
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
    thread = { provider_id: 3, thinking_level: 'high' };
    saves.length = 0;
    await page.goto(`${ORIGIN}/`);
    await page.locator('#modelTrigger:not([disabled])').waitFor();
    await page.locator('#modelLabel', { hasText: 'Opus · High' }).waitFor();
  });

  it('a clicked model opens its levels, the back row returns to the models, and the level picked is saved and shown', async () => {
    await page.click('#modelTrigger');
    expect(await modelRows()).toEqual(['Opus', 'Sonnet']);

    await page.click('#modelMenu [data-model="4"]');
    expect(await levels()).toEqual(['Auto', 'Medium', 'High']);

    await page.click('#modelMenu .model-select__item--back');
    expect(await modelRows()).toEqual(['Opus', 'Sonnet']);

    await page.click('#modelMenu [data-model="4"]');
    await page.click('#modelMenu [data-level="medium"]');
    expect(await page.locator('#modelMenu').count()).toBe(0);

    // The label changes once the save has answered.
    await page.locator('#modelLabel', { hasText: 'Sonnet · Medium' }).waitFor({ timeout: 5_000 });
    expect(await label()).toBe('Sonnet · Medium');
    expect(saves).toEqual([
      { path: '/api/threads/settings/-1?type=user', body: { provider_id: 4, thinking_level: 'medium' } },
    ]);
  });

  it('a click outside the menu still closes it without saving', async () => {
    await page.click('#modelTrigger');
    await page.click('#modelMenu [data-model="4"]');
    expect(await levels()).toEqual(['Auto', 'Medium', 'High']);

    await page.click('#chatInput');

    expect(await page.locator('#modelMenu').count()).toBe(0);
    expect(saves).toEqual([]);
    expect(await label()).toBe('Opus · High');
  });
});

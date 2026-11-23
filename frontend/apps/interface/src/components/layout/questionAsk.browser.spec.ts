/**
 * A question in the input dock under a real mouse and keyboard — feature
 * spec, in a real browser.
 *
 * Chalie asks one question with 2 to 4 answers; no live line shows its turn,
 * so it waits in the dock above the composer. On a single-select question a
 * click on an answer sends it; on a multi-select one clicks pick and unpick,
 * and Send sends the answers picked in the order they are listed. The field
 * for the user's own answer is always there, and Enter sends it. The row goes
 * as the answer leaves; an answer the server says came too late stays gone
 * without the open asks being read again or anything told to the user, and one
 * that fails for another reason is told to the user in a toast and brings the
 * row back, since the question still waits, with the user's own answer still
 * typed in it.
 *
 * Real throughout: the InputDock, Pinia and the api layer, bundled by vite with
 * the app's own config and run in Chrome, clicked and typed into through
 * Playwright. The only fake is the backend, answered at the network; every
 * other host is refused.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import type { Rollup } from 'vite';
import { chromium } from '@playwright/test';
import type { Browser, Page, Route } from '@playwright/test';

const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const ORIGIN = 'http://chalie.test';
const ENTRY = 'virtual:question-ask-page';

/** The footer dock on its own, styled and mounted as the app mounts it, with
 *  the app's toast told of a failed answer and the open asks read as every
 *  socket connect reads them. */
const PAGE_SCRIPT = `
import '@chalie/shared/styles/main.scss';
import '/src/styles/interface.scss';
import { createApp, h } from 'vue';
import { createPinia } from 'pinia';
import { registerReplyFailureReporter, useAsksStore } from '@chalie/shared';
import { showToast } from '/src/utils/toast';
import InputDock from '/src/components/layout/InputDock.vue';
const pinia = createPinia();
registerReplyFailureReporter((message) => showToast(message));
createApp({ render: () => h(InputDock, { dockId: 'main_spine_dock' }) }).use(pinia).mount('#app');
void useAsksStore(pinia).refreshOpen();
`;

const CONTENT_TYPES: Record<string, string> = { '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2' };

const ON_CHAT = { type: 'user', turn_id: 7, forked: false };

function question(ask_id: string, text: string, labels: string[], multi_select = false) {
  return {
    ask_id,
    kind: 'question',
    asked_at: new Date().toISOString(),
    data: {
      question: text,
      answers: labels.map((label, i) => ({ label, description: i === 0 ? 'The usual' : '' })),
      multi_select,
      origin: ON_CHAT,
    },
  };
}

/** What the backend holds open, how it answers a reply, and every reply and open-asks read it saw. */
let open: object[];
let replyStatus: number;
const replies: { path: string; body: unknown }[] = [];
let openReads = 0;

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
        name: 'question-ask-page',
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

/** The backend: the open-asks listing and the reply to an ask. */
function serve(route: Route): Promise<void> {
  const request = route.request();
  const url = new URL(request.url());
  if (url.origin !== ORIGIN) return route.abort();
  const file = files.get(url.pathname);
  if (file) return route.fulfill({ contentType: file.type, body: Buffer.from(file.body) });
  if (url.pathname === '/api/asks/open') {
    openReads += 1;
    return json(route, { success: true, result: open, pagination: { page: 1, limit: 50, total: open.length } });
  }
  if (url.pathname.startsWith('/api/asks/reply/') && request.method() === 'POST') {
    replies.push({ path: url.pathname, body: request.postDataJSON() });
    if (replyStatus === 500) return json(route, { error: 'Could not answer the ask' }, 500);
    const answered = decodeURIComponent(url.pathname.slice('/api/asks/reply/'.length));
    open = open.filter((a) => (a as { ask_id: string }).ask_id !== answered);
    if (replyStatus === 404) return json(route, { error: 'That ask was already answered' }, 404);
    return route.fulfill({ status: 204 });
  }
  return json(route, { error: 'unrouted' }, 404);
}

const answer = (label: string) => page.locator('.question__answer', { hasText: label });
const pressed = async (): Promise<(string | null)[]> =>
  page.locator('.question__answer').evaluateAll((bs) => bs.map((b) => b.getAttribute('aria-pressed')));
const sendPicked = () => page.locator('.question__answers .question__send');
const sendOwn = () => page.locator('.question__other .question__send');
/** Wait for the browser to hear back from its reply. */
const replied = () => page.waitForResponse((r) => r.url().includes('/api/asks/reply/'));

describe('a question in the input dock under a real mouse and keyboard', () => {
  beforeAll(async () => {
    files = await bundle();
    browser = await chromium.launch({ channel: 'chrome' });
    page = await browser.newPage();
    await page.route('**/*', serve);
  }, 120_000);

  afterAll(async () => {
    await browser?.close();
  });

  beforeEach(() => {
    replyStatus = 204;
    replies.length = 0;
    openReads = 0;
  });

  async function load(...asks: object[]): Promise<void> {
    open = asks;
    await page.goto(`${ORIGIN}/`);
    await page.locator('.question').waitFor();
  }

  it('a click on an answer of a single-select question sends that answer, and the row goes', async () => {
    await load(question('q1', 'Which ferry should I book?', ['The 08:15', 'The 17:40']));
    expect(await page.locator('.question__text').textContent()).toBe('Which ferry should I book?');
    expect(await pressed()).toEqual([null, null]);

    const reply = replied();
    await answer('The 17:40').click();
    await reply;

    expect(replies).toEqual([{ path: '/api/asks/reply/q1', body: { value: { labels: ['The 17:40'] } } }]);
    expect(await page.locator('.question').count()).toBe(0);
  });

  it('on a multi-select question clicks pick and unpick, and Send sends the answers picked in the order they are listed', async () => {
    await load(question('q2', 'What should I add to the list?', ['Bread', 'Milk', 'Eggs', 'Butter'], true));
    expect(await sendPicked().isDisabled()).toBe(true);

    await answer('Eggs').click();
    await answer('Bread').click();
    await answer('Butter').click();
    await answer('Butter').click();
    expect(await pressed()).toEqual(['true', 'false', 'true', 'false']);
    expect(replies).toEqual([]);

    const reply = replied();
    await sendPicked().click();
    await reply;

    expect(replies).toEqual([{ path: '/api/asks/reply/q2', body: { value: { labels: ['Bread', 'Eggs'] } } }]);
    expect(await page.locator('.question').count()).toBe(0);
  });

  it("the user's own answer cannot be sent blank, and Enter sends it typed", async () => {
    await load(question('q3', 'Which day suits you?', ['Tuesday', 'Friday']));
    const field = page.locator('.question__field');
    expect(await field.getAttribute('placeholder')).toBe('Other…');
    expect(await sendOwn().isDisabled()).toBe(true);

    await field.click();
    await page.keyboard.type('   ');
    expect(await sendOwn().isDisabled()).toBe(true);
    await page.keyboard.press('Enter');
    expect(replies).toEqual([]);

    await page.keyboard.type('Any evening next week');
    expect(await sendOwn().isDisabled()).toBe(false);
    const reply = replied();
    await page.keyboard.press('Enter');
    await reply;

    expect(replies).toEqual([{ path: '/api/asks/reply/q3', body: { value: { other: 'Any evening next week' } } }]);
    expect(await page.locator('.question').count()).toBe(0);
  });

  it('an answer the server says came too late leaves the row gone, with nothing told to the user, and the open asks are not read again', async () => {
    await load(question('q4', 'Which day suits you?', ['Tuesday', 'Friday']));
    replyStatus = 404;

    const reply = replied();
    await answer('Friday').click();
    await reply;
    await page.waitForTimeout(250);

    expect(replies).toEqual([{ path: '/api/asks/reply/q4', body: { value: { labels: ['Friday'] } } }]);
    expect(await page.locator('.question').count()).toBe(0);
    expect(await page.locator('.chalie-toast').count()).toBe(0);
    expect(openReads).toBe(1);
  });

  it("an answer that fails to send tells the user the server's reason and brings the question back with their own answer still typed, still answerable, since it still waits", async () => {
    await load(question('q5', 'Which day suits you?', ['Tuesday', 'Friday']));
    replyStatus = 500;

    await page.locator('.question__field').click();
    await page.keyboard.type('Any evening next week');
    const failed = replied();
    await sendOwn().click();
    await failed;
    await page.locator('.question').waitFor();

    expect(replies).toEqual([{ path: '/api/asks/reply/q5', body: { value: { other: 'Any evening next week' } } }]);
    expect(await page.locator('.chalie-toast').textContent()).toBe('Could not answer the ask');
    expect(openReads).toBe(2);
    expect(await page.locator('.question__field').inputValue()).toBe('Any evening next week');

    replyStatus = 204;
    const reply = replied();
    await answer('Tuesday').click();
    await reply;

    expect(replies.at(-1)).toEqual({ path: '/api/asks/reply/q5', body: { value: { labels: ['Tuesday'] } } });
    expect(await page.locator('.question').count()).toBe(0);
  });
});

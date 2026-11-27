/**
 * The question card in the input dock under a real mouse and keyboard —
 * feature spec, in a real browser.
 *
 * Chalie asks a question with 2 to 4 answers; no live line shows its turn, so
 * it waits on the card above the composer, one page per open question. On a
 * single-select question a click, its number key or Enter on it sends an
 * answer; on a multi-select one those pick and unpick, and one button sends
 * the picks in the order they are listed. The "Something else" field is always
 * there and Enter in it sends what is typed. Skip, the close button and Escape
 * dismiss the question unanswered. Each closes its page and the next question
 * shows, with the keyboard still on the card. An answer the server says came
 * too late stays gone without anything told to the user; one that fails for
 * another reason is told to the user in a toast and brings the question back
 * with the user's own answer still typed in it.
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

/** What the backend holds open, how it answers a reply or dismiss, and every one of those and open-asks read it saw. */
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

/** The backend: the open-asks listing, and the reply to or dismiss of an ask. */
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
  const settled = /^\/api\/asks\/(?:reply|dismiss)\/(.+)$/.exec(url.pathname);
  if (settled && request.method() === 'POST') {
    replies.push({ path: url.pathname, body: request.postDataJSON() });
    if (replyStatus === 500) return json(route, { error: 'Could not answer the ask' }, 500);
    const answered = decodeURIComponent(settled[1]);
    open = open.filter((a) => (a as { ask_id: string }).ask_id !== answered);
    if (replyStatus === 404) return json(route, { error: 'That ask was already answered' }, 404);
    return route.fulfill({ status: 204 });
  }
  return json(route, { error: 'unrouted' }, 404);
}

const card = () => page.locator('.qcard');
const title = () => page.locator('.qcard__title').textContent();
const answer = (label: string) => page.locator('.qcard__answer', { hasText: label });
const pressed = async (): Promise<(string | null)[]> =>
  page.locator('.qcard__answer').evaluateAll((bs) => bs.map((b) => b.getAttribute('aria-pressed')));
const field = () => page.locator('.qcard__field');
const sendButton = () => page.locator('.qcard__other .btn');
/** The text of whatever has the keyboard. */
const focused = () => page.evaluate(() => (document.activeElement as HTMLElement | null)?.innerText ?? null);
/** Wait for the browser to hear back from its reply or dismiss. */
const replied = () => page.waitForResponse((r) => /\/api\/asks\/(reply|dismiss)\//.test(r.url()));

describe('the question card in the input dock under a real mouse and keyboard', () => {
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
    await card().waitFor();
  }

  it("reads in the app's own type and sizes: a 20px title, 15px answers over 12px mono descriptions, 32px badges, rows parted by fill alone", async () => {
    await load(question('q0', 'Which ferry should I book?', ['The 08:15', 'The 17:40']));
    const look = await page.evaluate(() => {
      const css = (sel: string) => getComputedStyle(document.querySelector(sel)!);
      const badge = document.querySelector('.qcard__badge')!.getBoundingClientRect();
      return {
        title: [css('.qcard__title').fontSize, css('.qcard__title').fontWeight, css('.qcard__title').fontFamily.split(',')[0]],
        label: css('.qcard__label').fontSize,
        description: [css('.qcard__description').fontSize, css('.qcard__description').fontFamily.split(',')[0]],
        badge: [badge.width, badge.height],
        rowBorder: css('.qcard__answer').borderTopStyle,
        rowHeight: document.querySelector('.qcard__answer')!.getBoundingClientRect().height,
        sendHeight: document.querySelector('.qcard__other .btn')!.getBoundingClientRect().height,
      };
    });
    expect(look).toEqual({
      title: ['20px', '600', expect.stringMatching(/Inter/)],
      label: '15px',
      description: ['12px', expect.stringMatching(/mono/i)],
      badge: [32, 32],
      rowBorder: 'none',
      rowHeight: expect.any(Number),
      sendHeight: 32,
    });
    expect(look.rowHeight).toBeGreaterThanOrEqual(48);
  });

  it('a click on an answer of a single-select question sends that answer, and the card goes', async () => {
    await load(question('q1', 'Which ferry should I book?', ['The 08:15', 'The 17:40']));
    expect(await title()).toBe('Which ferry should I book?');
    expect(await pressed()).toEqual([null, null]);

    const reply = replied();
    await answer('The 17:40').click();
    await reply;

    expect(replies).toEqual([{ path: '/api/asks/reply/q1', body: { value: { labels: ['The 17:40'] } } }]);
    expect(await card().count()).toBe(0);
  });

  it('on a multi-select question clicks pick and unpick, Skip turns into Submit, and one Submit sends the picks in the order they are listed', async () => {
    await load(question('q2', 'What should I add to the list?', ['Bread', 'Milk', 'Eggs', 'Butter'], true));
    expect(await sendButton().textContent()).toContain('Skip');

    await answer('Eggs').click();
    await answer('Bread').click();
    await answer('Butter').click();
    await answer('Butter').click();
    expect(await pressed()).toEqual(['true', 'false', 'true', 'false']);
    expect(replies).toEqual([]);
    expect(await sendButton().textContent()).toContain('Submit');

    const reply = replied();
    await sendButton().click();
    await reply;

    expect(replies).toEqual([{ path: '/api/asks/reply/q2', body: { value: { labels: ['Bread', 'Eggs'] } } }]);
    expect(await card().count()).toBe(0);
  });

  it('Something else cannot be sent blank, and Enter sends it typed', async () => {
    await load(question('q3', 'Which day suits you?', ['Tuesday', 'Friday']));
    expect(await field().getAttribute('placeholder')).toBe('Something else');

    await field().click();
    await page.keyboard.type('   ');
    expect(await sendButton().textContent()).toContain('Skip');
    await page.keyboard.press('Enter');
    expect(replies).toEqual([]);

    await page.keyboard.type('Any evening next week');
    expect(await sendButton().textContent()).toContain('Submit');
    const reply = replied();
    await page.keyboard.press('Enter');
    await reply;

    expect(replies).toEqual([{ path: '/api/asks/reply/q3', body: { value: { other: 'Any evening next week' } } }]);
    expect(await card().count()).toBe(0);
  });

  it('Skip and the close button each dismiss the question they show, and the next question shows', async () => {
    await load(question('q4', 'Which day suits you?', ['Tuesday', 'Friday']), question('q5', 'Window or aisle?', ['Window', 'Aisle']));

    let reply = replied();
    await sendButton().click();
    await reply;
    expect(await title()).toBe('Window or aisle?');
    reply = replied();
    await page.getByRole('button', { name: 'Close' }).click();
    await reply;

    expect(replies).toEqual([
      { path: '/api/asks/dismiss/q4', body: {} },
      { path: '/api/asks/dismiss/q5', body: {} },
    ]);
    expect(await card().count()).toBe(0);
  });

  it('pages through every open question, the pager shut at its ends, and answering a page shows the one after it', async () => {
    await load(
      question('q6', 'Which ferry should I book?', ['The 08:15', 'The 17:40']),
      question('q7', 'Which days suit you?', ['Monday', 'Friday'], true),
      question('q8', 'Window or aisle?', ['Window', 'Aisle']),
    );
    const prev = page.getByRole('button', { name: 'Previous question' });
    const next = page.getByRole('button', { name: 'Next question' });
    expect(await page.getByRole('group', { name: 'Question 1 of 3' }).locator('span').textContent()).toBe('1 of 3');
    expect([await prev.isDisabled(), await next.isDisabled()]).toEqual([true, false]);
    expect(await sendButton().textContent()).toContain('Skip');

    await next.click();
    expect(await title()).toBe('Which days suit you?');
    await answer('Friday').click();
    expect(await sendButton().textContent()).toContain('Next');
    const reply = replied();
    await sendButton().click();
    await reply;

    expect(await title()).toBe('Window or aisle?');
    expect(await page.getByRole('group', { name: 'Question 2 of 2' }).count()).toBe(1);
    expect(await next.isDisabled()).toBe(true);
    await prev.click();
    expect(await title()).toBe('Which ferry should I book?');
    expect(replies).toEqual([{ path: '/api/asks/reply/q7', body: { value: { labels: ['Friday'] } } }]);
  });

  it('the keyboard alone answers every page: arrows walk the answers and the field, Enter and number keys pick, Enter in the field sends, Escape skips', async () => {
    await load(
      question('q9', 'Which ferry should I book?', ['The 08:15', 'The 12:00', 'The 17:40']),
      question('q10', 'What should I add to the list?', ['Bread', 'Milk', 'Eggs'], true),
      question('q11', 'Window or aisle?', ['Window', 'Aisle']),
    );

    await field().focus();
    await page.keyboard.press('ArrowUp');
    expect(await focused()).toContain('The 17:40');
    await page.keyboard.press('ArrowUp');
    expect(await focused()).toContain('The 12:00');
    let reply = replied();
    await page.keyboard.press('Enter');
    await reply;

    expect(await title()).toBe('What should I add to the list?');
    expect(await focused()).toContain('Bread');
    await page.keyboard.press('3');
    await page.keyboard.press('9');
    await page.keyboard.press('1');
    expect(await pressed()).toEqual(['true', 'false', 'true']);
    for (let i = 0; i < 4; i++) await page.keyboard.press('ArrowDown');
    expect(await page.evaluate(() => document.activeElement?.classList.contains('qcard__field'))).toBe(true);
    reply = replied();
    await page.keyboard.press('Enter');
    await reply;

    expect(await title()).toBe('Window or aisle?');
    reply = replied();
    await page.keyboard.press('Escape');
    await reply;

    expect(replies).toEqual([
      { path: '/api/asks/reply/q9', body: { value: { labels: ['The 12:00'] } } },
      { path: '/api/asks/reply/q10', body: { value: { labels: ['Bread', 'Eggs'] } } },
      { path: '/api/asks/dismiss/q11', body: {} },
    ]);
    expect(await card().count()).toBe(0);
  });

  it('an answer the server says came too late leaves the card gone, with nothing told to the user, and the open asks are not read again', async () => {
    await load(question('q12', 'Which day suits you?', ['Tuesday', 'Friday']));
    replyStatus = 404;

    const reply = replied();
    await answer('Friday').click();
    await reply;
    await page.waitForTimeout(250);

    expect(replies).toEqual([{ path: '/api/asks/reply/q12', body: { value: { labels: ['Friday'] } } }]);
    expect(await card().count()).toBe(0);
    expect(await page.locator('.chalie-toast').count()).toBe(0);
    expect(openReads).toBe(1);
  });

  it("an answer that fails to send tells the user the server's reason and brings the question back with their own answer still typed, still answerable, since it still waits", async () => {
    await load(question('q13', 'Which day suits you?', ['Tuesday', 'Friday']));
    replyStatus = 500;

    await field().click();
    await page.keyboard.type('Any evening next week');
    const failed = replied();
    await sendButton().click();
    await failed;
    await card().waitFor();

    expect(replies).toEqual([{ path: '/api/asks/reply/q13', body: { value: { other: 'Any evening next week' } } }]);
    expect(await page.locator('.chalie-toast').textContent()).toBe('Could not answer the ask');
    expect(openReads).toBe(2);
    expect(await field().inputValue()).toBe('Any evening next week');

    replyStatus = 204;
    const reply = replied();
    await answer('Tuesday').click();
    await reply;

    expect(replies.at(-1)).toEqual({ path: '/api/asks/reply/q13', body: { value: { labels: ['Tuesday'] } } });
    expect(await card().count()).toBe(0);
  });
});

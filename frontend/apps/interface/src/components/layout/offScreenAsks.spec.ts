// @vitest-environment happy-dom
/**
 * The input dock's off-screen asks — feature spec on the real dock.
 *
 * A subagent the chat started can ask for permission after the chat turn that
 * started it has ended: no live line shows that turn any more, so the ask
 * waits in the input dock. The spec loads the open asks the way the app does
 * on every connect, from the backend's listing, mounts the real OffScreenAsks
 * on the real asks, browser and session stores, and asserts what a person
 * reads and can press:
 *
 *   - an ask no live line shows gets one row naming its turn and what it asks,
 *     with Deny, Allow and Open;
 *   - each row says where its ask came from: the chat, a thread or a scheduled
 *     task — a browser hand-off's row too;
 *   - Allow and Deny each send their own answer for that ask alone, and its
 *     row goes;
 *   - an answer the server fails on tells the user why and brings its row
 *     back, since the ask still waits; one that never reaches the server says
 *     it did not go through; one the server says came after the ask was
 *     already answered elsewhere leaves the row gone, with nothing to tell;
 *   - Open shows the ask's turn in the thread panel and answers nothing;
 *   - every open question waits on one card under the rows, a page each: the
 *     question as its title, where it came from with Open, one numbered row
 *     per answer over its description, and a "Something else" field with
 *     Skip; a single-select pick is sent as its label, a multi-select's picks
 *     are sent together by one button, the user's own words are sent as
 *     theirs, and Skip or close dismisses the question; each closes its page
 *     and the next question shows;
 *   - an answer or skip the server fails on tells the user why and brings the
 *     question back as the user left it;
 *   - an ask from a run no chat turn shows has no origin before its words; its
 *     row still offers Open when it is a hand-off, whose tab can be shown, and
 *     not when it is a permission ask, which has no turn to open.
 *
 * The network is the one stand-in: the spec serves the open-asks listing and
 * the browser's tab list, and reads each answer the dock sent. The chat app's
 * toast is registered for failed answers as the app does at startup.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import OffScreenAsks from './OffScreenAsks.vue';
import { registerReplyFailureReporter, useAsksStore } from '@chalie/shared';
import type { AskOrigin } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import { useBrowserStore } from '../../stores/browser';
import type { BrowserTab } from '../../api/browser';
import { showToast } from '../../utils/toast';

/** The chat turn that started the subagent; it ended before the ask came. */
const CHAT_TURN = 7;
const ON_CHAT: AskOrigin = { type: 'user', turn_id: CHAT_TURN, forked: false };

const backend = {
  open: [] as object[],
  tabs: [] as BrowserTab[],
  answers: [] as { path: string; body: unknown }[],
  /** How the server answers each answer: a status, or `offline` for a request that never reaches it. */
  replyAnswer: 204 as number | 'offline',
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname === '/api/asks/open' && (init?.method ?? 'GET') === 'GET') {
    return json({ success: true, result: backend.open, pagination: { page: 1, limit: 50, total: backend.open.length } });
  }
  if (url.pathname === '/api/browser/tabs' && (init?.method ?? 'GET') === 'GET') {
    return json({ success: true, result: { tabs: backend.tabs } });
  }
  const settled = /^\/api\/asks\/(?:reply|dismiss)\/(.+)$/.exec(url.pathname);
  if (settled && init?.method === 'POST') {
    backend.answers.push({ path: url.pathname, body: JSON.parse(String(init.body)) });
    const answer = backend.replyAnswer;
    if (answer === 'offline') throw new TypeError('Failed to fetch');
    if (answer !== 204 && answer !== 404) return json({ error: 'Could not answer the ask' }, answer);
    const answered = decodeURIComponent(settled[1]);
    backend.open = backend.open.filter((a) => (a as { ask_id: string }).ask_id !== answered);
    if (answer === 404) return json({ error: 'That ask was already answered' }, 404);
    return new Response(null, { status: 204 });
  }
  return json({ error: 'unrouted' }, 404);
}

/** A subagent's ask on the turn that started it — the chat turn unless given — as the backend lists it. */
function ask(ask_id: string, action_id: string, summary: string, origin: AskOrigin | null = ON_CHAT) {
  return { ask_id, kind: 'permission', asked_at: new Date().toISOString(), data: { action_id, summary, origin } };
}

/** A question the chat asked, as the backend lists it. */
function question(ask_id: string, text: string, answers: { label: string; description: string }[], multi_select = false) {
  return {
    ask_id,
    kind: 'question',
    asked_at: new Date().toISOString(),
    data: { question: text, answers, multi_select, origin: ON_CHAT },
  };
}

const FERRIES = [
  { label: 'The 08:15', description: 'Arrives before noon' },
  { label: 'The 17:40', description: '' },
];
const DAYS = [
  { label: 'Monday', description: '' },
  { label: 'Tuesday', description: '' },
  { label: 'Friday', description: '' },
];

/** A hand-off of a browser tab Chalie opened, as the backend lists it. */
function handoff(ask_id: string, tab: BrowserTab, reason: string) {
  return { ask_id, kind: 'handoff', asked_at: new Date().toISOString(), data: { tab_id: tab.id, reason, origin: tab.origin } };
}

function chaliesTab(id: string, origin: AskOrigin | null): BrowserTab {
  return { id, url: 'https://example.com/login', title: 'Sign in', opener: 'chalie', origin, certificate_error: null };
}

let wrapper: VueWrapper | null = null;

/** The dock with the backend's open asks and tabs loaded, as on connect, the browser panel closed. */
async function mountDock(...asks: object[]): Promise<VueWrapper> {
  backend.open = asks;
  await useAsksStore().refreshOpen();
  const browser = useBrowserStore();
  await browser.refresh();
  browser.closePanel();
  wrapper = mount(OffScreenAsks);
  await flushPromises();
  return wrapper;
}

const rows = (w: VueWrapper) => w.findAll('.offscreen-asks > .activity');
const labels = (w: VueWrapper): string[] => rows(w).map((r) => r.find('.activity__label').text());
const origins = (w: VueWrapper): string[] => rows(w).map((r) => r.find('.activity__origin').text());
const rowNamed = (w: VueWrapper, label: string) => {
  const found = rows(w).find((r) => r.find('.activity__label').text() === label);
  if (!found) throw new Error(`no dock row for "${label}"`);
  return found;
};
const card = (w: VueWrapper) => w.find('.offscreen-asks > .qcard');
const title = (w: VueWrapper) => card(w).find('.qcard__title').text();
const answerButtons = (w: VueWrapper) => card(w).findAll('.qcard__answer');
const sendButton = (w: VueWrapper) => card(w).find('.qcard__other .btn');
const toast = () => document.querySelector('.chalie-toast')?.textContent ?? null;

beforeEach(() => {
  setActivePinia(createPinia());
  backend.open = [];
  backend.tabs = [];
  backend.answers = [];
  backend.replyAnswer = 204;
  registerReplyFailureReporter((message) => showToast(message));
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('an ask no live line shows', () => {
  it('gets one row naming its turn and what it asks, with Deny, Allow and Open', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));

    expect(rows(w)).toHaveLength(1);
    const row = rows(w)[0];
    expect(row.find('.activity__origin').text()).toBe(`Chat · #${CHAT_TURN} ·`);
    expect(row.find('.activity__label').text()).toBe('Read the ferry booking email');
    expect(row.find('.activity__ask').text()).toBe('Read Email');
    expect(row.text()).toContain('waiting for you');
    expect(row.findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow', 'Open']);
  });

  it('says where each ask came from: the chat, a thread or a scheduled task', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script', { type: 'user', turn_id: 12, forked: true }),
      ask('r3', 'email.search', 'Check the timetable inbox', { type: 'scheduled', turn_id: 31, forked: false }),
    );

    expect(origins(w)).toEqual([`Chat · #${CHAT_TURN} ·`, 'Thread · #12 ·', 'Scheduled task · #31 ·']);
  });

  it('a browser hand-off the chat asked for says Chat once the person closes the browser panel', async () => {
    const run = chaliesTab('run', ON_CHAT);
    backend.tabs = [run];
    const w = await mountDock(handoff('h1', run, 'Please sign in'));

    expect(labels(w)).toEqual(['Please sign in']);
    expect(origins(w)).toEqual([`Chat · #${CHAT_TURN} ·`]);
  });

  it('Allow sends approval for that ask alone, and its row goes', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script'),
    );

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: true } }]);
    expect(labels(w)).toEqual(['Write the fare table script']);
  });

  it('Deny sends a refusal for that ask alone, and its row goes', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script'),
    );

    await rowNamed(w, 'Write the fare table script').find('.activity__answer--deny').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r2', body: { value: false } }]);
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it('an Allow the server fails on tells the user why, and its row comes back since the ask still waits', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script'),
    );
    backend.replyAnswer = 500;

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: true } }]);
    expect(toast()).toBe('Could not answer the ask');
    expect(labels(w)).toEqual(['Write the fare table script', 'Read the ferry booking email']);
    expect(rowNamed(w, 'Read the ferry booking email').findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow', 'Open']);
  });

  it('a Deny the server fails on tells the user why, and its row comes back too', async () => {
    const w = await mountDock(ask('r1', 'email.send', 'Send the weekly digest'));
    backend.replyAnswer = 500;

    await rowNamed(w, 'Send the weekly digest').find('.activity__answer--deny').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: false } }]);
    expect(toast()).toBe('Could not answer the ask');
    expect(labels(w)).toEqual(['Send the weekly digest']);
  });

  it('an answer that never reaches the server tells the user it did not go through, and its row comes back', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));
    backend.replyAnswer = 'offline';

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(toast()).toBe("Couldn't send your answer");
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it('an answer the server says came after the ask was already answered leaves the row gone, with nothing to tell the user', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));
    backend.replyAnswer = 404;

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: true } }]);
    expect(toast()).toBeNull();
    expect(labels(w)).toEqual([]);
  });

  it('Open shows its turn in the thread panel and answers nothing', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));
    const session = useSessionStore();

    await rowNamed(w, 'Read the ferry booking email').find('.activity__stop').trigger('click');
    await flushPromises();

    expect([session.panelThreadId, session.panelType]).toEqual([CHAT_TURN, 'user']);
    expect(backend.answers).toEqual([]);
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it('a question waits on a card under the rows: its title, where it came from with Open, one numbered row per answer over its description, and a Something else field with Skip', async () => {
    const w = await mountDock(question('q1', 'Which ferry should I book?', FERRIES), ask('r1', 'email.read', 'Read the ferry booking email'));

    expect(labels(w)).toEqual(['Read the ferry booking email']);
    expect(w.find('.offscreen-asks').element.lastElementChild?.classList.contains('qcard')).toBe(true);
    expect(title(w)).toBe('Which ferry should I book?');
    expect(card(w).findAll('.qcard__meta > *').map((x) => x.text())).toEqual([`Chat · #${CHAT_TURN} ·`, 'waiting for you · 0s', 'Open']);
    expect(answerButtons(w).map((b) => [b.find('.qcard__badge').text(), ...b.findAll('.qcard__words > span').map((x) => x.text())])).toEqual([
      ['1', 'The 08:15', 'Arrives before noon'],
      ['2', 'The 17:40'],
    ]);
    expect(card(w).find('.qcard__answers').attributes('role')).toBe('group');
    expect(card(w).find(`#${card(w).find('.qcard__answers').attributes('aria-describedby')}`).text()).toBe('Choose one of 2 options.');
    expect(card(w).find('.qcard__field').attributes('placeholder')).toBe('Something else');
    expect(sendButton(w).text()).toBe('Skip');
    expect(card(w).find('.qcard__pager').exists()).toBe(false);
    expect(card(w).find('.qcard__hint').text()).toBe('↑↓ to navigate · Enter to select · Esc to skip');
  });

  it('a single-select pick is sent as its label at once, and the card goes while the rows stay', async () => {
    const w = await mountDock(question('q1', 'Which ferry should I book?', FERRIES), ask('r1', 'email.read', 'Read the ferry booking email'));

    await answerButtons(w)[1].trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/q1', body: { value: { labels: ['The 17:40'] } } }]);
    expect(card(w).exists()).toBe(false);
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it("a multi-select's answers pick and unpick without sending, and one Submit sends the picks in the answers' order", async () => {
    const w = await mountDock(question('q1', 'Which days suit you?', DAYS, true));
    expect(card(w).find(`#${card(w).find('.qcard__answers').attributes('aria-describedby')}`).text()).toBe('Choose one or more of 3 options.');
    expect(sendButton(w).attributes('title')).toBe('Select an option to send');

    for (const i of [2, 0, 1, 1]) await answerButtons(w)[i].trigger('click');

    expect(backend.answers).toEqual([]);
    expect(answerButtons(w).map((b) => b.attributes('aria-pressed'))).toEqual(['true', 'false', 'true']);
    expect(sendButton(w).text()).toBe('Submit');

    await sendButton(w).trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/q1', body: { value: { labels: ['Monday', 'Friday'] } } }]);
    expect(card(w).exists()).toBe(false);
  });

  it("the user's own words turn Skip into Submit and are sent as theirs, by the button or Enter", async () => {
    const w = await mountDock(question('q1', 'Which ferry should I book?', FERRIES), question('q2', 'Which days suit you?', DAYS, true));
    expect(sendButton(w).attributes('title')).toBe('Type an answer to send');

    await card(w).find('.qcard__field').setValue('  The late one on Sunday  ');
    expect(sendButton(w).text()).toBe('Next');
    await sendButton(w).trigger('click');
    await flushPromises();

    await card(w).find('.qcard__field').setValue('Whichever is cheapest');
    expect(sendButton(w).text()).toBe('Submit');
    await card(w).find('.qcard__field').trigger('keydown', { key: 'Enter' });
    await flushPromises();

    expect(backend.answers).toEqual([
      { path: '/api/asks/reply/q1', body: { value: { other: 'The late one on Sunday' } } },
      { path: '/api/asks/reply/q2', body: { value: { other: 'Whichever is cheapest' } } },
    ]);
    expect(card(w).exists()).toBe(false);
  });

  it('Skip and the close button each dismiss the question they show, unanswered', async () => {
    const w = await mountDock(question('q1', 'Which ferry should I book?', FERRIES), question('q2', 'Which days suit you?', DAYS, true));

    await sendButton(w).trigger('click');
    await flushPromises();
    expect(title(w)).toBe('Which days suit you?');
    await card(w).find('[aria-label="Close"]').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([
      { path: '/api/asks/dismiss/q1', body: {} },
      { path: '/api/asks/dismiss/q2', body: {} },
    ]);
    expect(card(w).exists()).toBe(false);
  });

  it('several questions page on the one card: the pager reads where it is, its ends are shut, and the page stays put while another question closes', async () => {
    const w = await mountDock(
      question('q1', 'Which ferry should I book?', FERRIES),
      question('q2', 'Which days suit you?', DAYS, true),
      question('q3', 'Window or aisle?', [{ label: 'Window', description: '' }, { label: 'Aisle', description: '' }]),
    );
    const pager = () => card(w).find('.qcard__pager');
    const prev = () => pager().find('[aria-label="Previous question"]');
    const next = () => pager().find('[aria-label="Next question"]');

    expect([pager().attributes('aria-label'), pager().find('span').text()]).toEqual(['Question 1 of 3', '1 of 3']);
    expect([prev().attributes('disabled'), next().attributes('disabled')]).toEqual(['', undefined]);

    await next().trigger('click');
    await next().trigger('click');
    expect([title(w), pager().find('span').text(), next().attributes('disabled')]).toEqual(['Window or aisle?', '3 of 3', '']);
    await prev().trigger('click');
    expect(title(w)).toBe('Which days suit you?');

    backend.open = backend.open.filter((a) => (a as { ask_id: string }).ask_id !== 'q1');
    useAsksStore().remove('q1');
    await flushPromises();
    expect([title(w), pager().find('span').text()]).toEqual(['Which days suit you?', '1 of 2']);

    await card(w).find('.qcard__field').setValue('Wednesday');
    await sendButton(w).trigger('click');
    await flushPromises();
    expect(title(w)).toBe('Window or aisle?');
    expect(pager().exists()).toBe(false);
    expect(sendButton(w).text()).toBe('Skip');
  });

  it('a number key picks its answer and Escape skips, from anywhere on the card', async () => {
    const w = await mountDock(question('q1', 'Which ferry should I book?', FERRIES), question('q2', 'Which days suit you?', DAYS, true));

    await answerButtons(w)[0].trigger('keydown', { key: '2' });
    await flushPromises();
    await answerButtons(w)[0].trigger('keydown', { key: '9' });
    await answerButtons(w)[0].trigger('keydown', { key: '3' });
    expect(answerButtons(w).map((b) => b.attributes('aria-pressed'))).toEqual(['false', 'false', 'true']);
    await card(w).find('.qcard__field').trigger('keydown', { key: 'Escape' });
    await flushPromises();

    expect(backend.answers).toEqual([
      { path: '/api/asks/reply/q1', body: { value: { labels: ['The 17:40'] } } },
      { path: '/api/asks/dismiss/q2', body: {} },
    ]);
  });

  it('an answer the server fails on tells the user why and brings the question back with their words still typed', async () => {
    const w = await mountDock(question('q1', 'Which ferry should I book?', FERRIES));
    backend.replyAnswer = 500;

    await card(w).find('.qcard__field').setValue('The late one on Sunday');
    await sendButton(w).trigger('click');
    await flushPromises();

    expect(toast()).toBe('Could not answer the ask');
    expect(title(w)).toBe('Which ferry should I book?');
    expect((card(w).find('.qcard__field').element as HTMLInputElement).value).toBe('The late one on Sunday');
    expect(sendButton(w).text()).toBe('Submit');
  });

  it("a skip the server fails on tells the user why and brings the question back with its picks", async () => {
    const w = await mountDock(question('q1', 'Which days suit you?', DAYS, true));
    await answerButtons(w)[1].trigger('click');
    backend.replyAnswer = 500;

    await card(w).find('[aria-label="Close"]').trigger('click');
    await flushPromises();

    expect(toast()).toBe('Could not answer the ask');
    expect(answerButtons(w).map((b) => b.attributes('aria-pressed'))).toEqual(['false', 'true', 'false']);
  });

  it("a question's Open shows its turn in the thread panel and answers nothing", async () => {
    const w = await mountDock(question('q1', 'Which ferry should I book?', FERRIES));
    const session = useSessionStore();

    await card(w).find('.qcard__meta .activity__stop').trigger('click');

    expect([session.panelThreadId, session.panelType]).toEqual([CHAT_TURN, 'user']);
    expect(backend.answers).toEqual([]);
    expect(title(w)).toBe('Which ferry should I book?');
  });

  it('a hand-off from a run no chat turn shows has no origin before its words, and Open still shows its tab', async () => {
    const run = chaliesTab('run', null);
    backend.tabs = [run];
    const w = await mountDock(handoff('h1', run, 'Please solve the captcha'));
    const row = rowNamed(w, 'Please solve the captcha');

    expect(row.find('.activity__origin').exists()).toBe(false);
    expect(row.findAll('button').map((b) => b.text())).toEqual(['Open']);

    await row.find('.activity__stop').trigger('click');
    const browser = useBrowserStore();
    expect([browser.open, browser.selectedTabId]).toEqual([true, run.id]);
    expect(backend.answers).toEqual([]);
  });

  it('a permission ask from a run no chat turn shows has no origin before its words and no Open, only Deny and Allow', async () => {
    const w = await mountDock(ask('r1', 'email.send', 'Send the weekly digest', null));
    const row = rowNamed(w, 'Send the weekly digest');

    expect(row.find('.activity__origin').exists()).toBe(false);
    expect(row.findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow']);
  });
});

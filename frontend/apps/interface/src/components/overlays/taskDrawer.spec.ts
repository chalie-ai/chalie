// @vitest-environment happy-dom
/**
 * The Activity drawer and its header button — feature spec.
 *
 * The drawer holds two lists, each under its own title and shown only when it
 * has rows: "Threads", the forked threads whose reply settled unseen, newest
 * first, and under it "Agents", every subagent still running or waiting for
 * its caller's reply, oldest first. The header draws its Activity button — the
 * bot icon, pink — only while either list has a row; with neither, there is no
 * button at all.
 *
 * A subagent call (`subtask`, `web_search`, `web_browse`, `pim`, `code_agent`)
 * writes one `agents` row, and the row keeps running after the call — often
 * after the whole turn that started it settled. Its row in the drawer is its
 * type's icon and the one-line summary of the call that started it — its
 * type's plain name when the call has none, never the tool name or a number —
 * then the working dot while it runs, or the word "waiting" while it waits for
 * a reply. Picking a row opens its transcript in the thread panel, headed by the row's
 * title, and closes the drawer; the drawer closes itself once the last row of
 * both lists clears.
 *
 * The agents list is the agents table: read in full over `GET /api/agents/all`
 * on load, then kept current by each `agent_changed` frame, driven here
 * through the real dispatcher (`dispatchDrift`), the same entry point the
 * WebSocket feeds. A running or waiting row is added or updated; an ended one
 * is removed.
 * The threads list is read off the chat's rendered turns, stamped done through
 * the real `setTurnDone`, as a settled reply stamps them.
 *
 * Real components throughout (PresenceBar, TaskDrawer, SideDrawer,
 * ThreadPanel), the real Pinia stores, the real turnDom registry and the real
 * API client. The only fake is the backend at the network boundary: `fetch`,
 * answering the agents read with the rows served and a subagent's transcript
 * read with its task.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import ThreadPanel from '../conversation/ThreadPanel.vue';
import PresenceBar from '../layout/PresenceBar.vue';
import TaskDrawer from './TaskDrawer.vue';
import { useAgentsStore } from '../../stores/agents';
import { useSessionStore } from '../../stores/session';
import { useTasksStore } from '../../stores/tasks';
import { dispatchDrift } from '../../utils/driftDispatcher';
import { registerSurface, setTurnDone, SPINE_SURFACE_ID, unregisterSurface } from '../../utils/turnDom';
import type { Agent, AgentStatus } from '../../api/agents';
import type { DelegateTurnBlock } from '../../api/conversation';

const KINDS = ['subtask', 'web_search', 'web_browse', 'pim', 'code_agent'] as const;
type Kind = (typeof KINDS)[number];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** One agents row as the server sends it. Its `turn_id` never equals its
 *  `agent_id`, so a row keyed or opened by the wrong one goes red. */
function agent(agentId: number, type: Kind, summary: string, extra: Partial<Agent> = {}): Agent {
  return {
    agent_id: agentId, spawned_by_channel: 'user', spawned_in_turn: 900, type, prompt: `The task for ${summary}`,
    status: 'running', response_id: null, turn_id: agentId + 100, channel: `delegate:${type}`, summary, ...extra,
  };
}

/** The running rows the backend serves; a test sets them before the read. */
let served: Agent[] = [];
/** Every agents read the page sent, by its query. */
let agentReads: string[] = [];
/** When set, the agents read waits for this before it answers. */
let holdAgentsRead: Promise<void> | null = null;

/** The backend at the network boundary: the agents listing, paged as the
 *  endpoint pages it, and a subagent's transcript read with its task. */
async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname === '/api/agents/all') {
    agentReads.push(url.search);
    const rows = served;
    if (holdAgentsRead) await holdAgentsRead;
    const page = Number(url.searchParams.get('page'));
    const limit = Number(url.searchParams.get('limit'));
    const result = rows.slice((page - 1) * limit, page * limit);
    return json({ success: true, result, pagination: { page, limit, total: rows.length } });
  }
  const match = /^\/api\/threads\/(\d+)$/.exec(url.pathname);
  const channel = url.searchParams.get('channel');
  if (!match || channel === null) return json({ error: 'unrouted' }, 404);
  const turnId = Number(match[1]);
  const transcript: DelegateTurnBlock = {
    turn_id: turnId, gist: null, preview: 'The task', last_activity_at: null,
    working: true, duration_ms: 0, type: null, channel,
    messages: [{
      id: `${turnId}0`, role: 'user', content: 'The task', timestamp: '2026-01-01 00:00:00',
      day: '2026-01-01', created_at: '2026-01-01T00:00:00Z', turn_id: turnId,
    }],
  };
  return json({ success: true, result: transcript });
}

let mounted: VueWrapper[] = [];
/** The chat's rendered turns: where a forked thread's opener lives. */
let chat: HTMLElement;
/** Every thread a test stamped done, so the next test starts with none. */
let stamped: number[] = [];

beforeEach(() => {
  setActivePinia(createPinia());
  served = [];
  agentReads = [];
  holdAgentsRead = null;
  stamped = [];
  vi.stubGlobal('fetch', fakeBackend);
  chat = document.body.appendChild(document.createElement('div'));
  registerSurface({ id: SPINE_SURFACE_ID, type: 'user', container: chat, component: {} });
});

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  for (const turnId of stamped) setTurnDone(turnId, 'user', false);
  unregisterSurface(SPINE_SURFACE_ID);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

/** The app's header and its Activity drawer, as the app mounts them, then the
 *  agents read the page makes when its socket connects. */
async function loadApp(): Promise<void> {
  mounted.push(mount(PresenceBar, { attachTo: document.body }));
  mounted.push(mount(TaskDrawer, { attachTo: document.body }));
  await useAgentsStore().refresh();
  await flushPromises();
}

/** A forked thread's opener on the chat, stamped as TurnView stamps it, whose
 *  reply then settles unseen. */
async function threadSettles(turnId: number, gist: string, lastActivity: string): Promise<void> {
  const el = chat.appendChild(document.createElement('div'));
  el.setAttribute('data-turn-id', String(turnId));
  el.setAttribute('data-type', 'user');
  el.setAttribute('data-forked', 'true');
  el.setAttribute('data-gist', gist);
  el.setAttribute('data-preview', `${gist}, in full`);
  el.setAttribute('data-last-activity', lastActivity);
  stamped.push(turnId);
  setTurnDone(turnId, 'user', true);
  await flushPromises();
}

/** One `agent_changed` frame, as the WebSocket delivers it. */
async function send(row: Agent): Promise<void> {
  dispatchDrift({ type: 'agent_changed', agent: row } as unknown as WsPushEvent);
  await flushPromises();
}

const ended = (row: Agent, status: Exclude<AgentStatus, 'running' | 'waiting'>): Agent =>
  ({ ...row, status, response_id: status === 'done' ? 5000 + row.agent_id : null });

/** `row` as its turn left it when that turn ended on a message to its caller:
 *  waiting for the reply, keeping the row the turn ended on. */
const waiting = (row: Agent): Agent => ({ ...row, status: 'waiting', response_id: 5000 + row.agent_id });

/** `row` running again, on the next turn its caller's reply opened. */
const resumed = (row: Agent): Agent => ({ ...row, status: 'running', response_id: null, turn_id: row.turn_id + 1 });

const headerButton = (): HTMLButtonElement | null => document.querySelector<HTMLButtonElement>('#taskDrawerBtn');
const drawer = (): HTMLElement => document.querySelector<HTMLElement>('aside[aria-label="Activity"]')!;
/** The click-catcher behind the drawer: shown only while the drawer is open. */
const drawerScrim = (): HTMLElement => document.querySelector<HTMLElement>('.side-drawer__scrim')!;
const rowButtons = (): HTMLButtonElement[] => [...drawer().querySelectorAll<HTMLButtonElement>('.drawer-row')];

/** The lucide icon an element draws, by its class (`lucide-search`). */
const iconOf = (el: Element | null): string | undefined =>
  el ? [...el.classList].find((k) => /^lucide-(?!icon$).*(?<!-icon)$/.test(k)) : undefined;

/** The drawer's list top to bottom: each section title, each thread by its
 *  label, each agent by its title. */
function layout(): string[] {
  return [...drawer().querySelector('.side-drawer__list')!.children].map((el) => {
    if (el.matches('.task-drawer__section')) return `# ${el.textContent!.trim()}`;
    if (el.matches('.task-drawer__thread')) return `thread: ${el.querySelector('.task-drawer__thread-label')!.textContent!.trim()}`;
    if (el.matches('.drawer-row')) return `agent: ${el.querySelector('.drawer-row__label')!.textContent!.trim()}`;
    return `other: ${el.textContent!.trim()}`;
  });
}

/** Each agent row, in order, as [its title, its type's icon]. A row has no
 *  line under its title. */
function agentRows(): (string | undefined)[][] {
  return rowButtons().map((r) => {
    expect(r.querySelector('.drawer-row__sub')).toBeNull();
    return [r.querySelector('.drawer-row__label')!.textContent!.trim(), iconOf(r.querySelector('.drawer-row__icon'))];
  });
}

/** Each agent row, in order, as [its title, what stands after it]: `waiting`
 *  when it shows that word, `working` when it shows the working dot. A row
 *  shows exactly one of them. */
function agentStates(): string[][] {
  return rowButtons().map((r) => {
    const word = r.querySelector('.task-drawer__status')?.textContent?.trim();
    const dot = r.querySelector('.thread-activity-dot.working') !== null;
    expect(word === undefined ? dot : !dot).toBe(true);
    return [r.querySelector('.drawer-row__label')!.textContent!.trim(), word ?? 'working'];
  });
}

function rowTitled(title: string): HTMLButtonElement {
  const found = rowButtons().filter((r) => r.querySelector('.drawer-row__label')!.textContent!.trim() === title);
  expect(found).toHaveLength(1);
  return found[0]!;
}

async function openDrawer(): Promise<void> {
  headerButton()!.click();
  await flushPromises();
  expect(useTasksStore().isOpen).toBe(true);
  expect(drawer().classList.contains('hidden')).toBe(false);
  expect(drawerScrim().classList.contains('hidden')).toBe(false);
}

/** The open transcript panel's heading. */
const panelHeading = (): string | null => document.querySelector('.thread-panel__title')?.textContent ?? null;

const HOURS = agent(11, 'subtask', 'compare museum hours');
const PRICES = agent(12, 'web_search', 'museum prices');

describe('the header\'s Activity button', () => {
  it('is not drawn when no thread reply waits unseen and no agent runs', async () => {
    await loadApp();
    expect(agentReads).toEqual(['?page=1&limit=100']);
    expect(headerButton()).toBeNull();
  });

  it('is drawn while any agent runs — the bot icon, pink, titled Activity — and goes once the last one ends, whichever way it ended', async () => {
    served = [HOURS, PRICES];
    await loadApp();
    const button = headerButton()!;
    expect(iconOf(button.querySelector('svg'))).toBe('lucide-bot');
    expect(button.classList.contains('has-activity')).toBe(true);
    expect(button.title).toBe('Activity');
    expect(button.getAttribute('aria-label')).toBe('Activity');
    // The header has one button for both lists; there is no Subagents button.
    expect(document.querySelector('header [aria-label="Subagents"]')).toBeNull();

    await send(ended(HOURS, 'done'));
    expect(headerButton()).not.toBeNull();

    await send(ended(PRICES, 'failed'));
    expect(headerButton()).toBeNull();

    await send(agent(13, 'pim', 'check my calendar'));
    expect(headerButton()?.classList.contains('has-activity')).toBe(true);
    await send(ended(agent(13, 'pim', 'check my calendar'), 'cancelled'));
    expect(headerButton()).toBeNull();
  });

  it('is drawn, pink, while a thread\'s reply waits unseen, and goes once it is seen', async () => {
    await loadApp();
    await threadSettles(7, 'plan the trip', '2026-01-01 10:00:00');
    expect(iconOf(headerButton()!.querySelector('svg'))).toBe('lucide-bot');
    expect(headerButton()!.classList.contains('has-activity')).toBe(true);

    setTurnDone(7, 'user', false);
    await flushPromises();
    expect(headerButton()).toBeNull();
  });
});

describe('the Activity drawer', () => {
  it('with only agents running shows only the Agents list, under its title', async () => {
    served = [HOURS, PRICES];
    await loadApp();
    await openDrawer();

    expect(layout()).toEqual(['# Agents', 'agent: compare museum hours', 'agent: museum prices']);
  });

  it('with only threads waiting shows only the Threads list, under its title, newest first', async () => {
    await loadApp();
    await threadSettles(7, 'plan the trip', '2026-01-01 10:00:00');
    await threadSettles(8, 'pack the bags', '2026-01-01 11:00:00');
    await openDrawer();

    expect(layout()).toEqual(['# Threads', 'thread: pack the bags', 'thread: plan the trip']);
  });

  it('with both shows Threads on top and Agents under, each under its own title', async () => {
    served = [HOURS, PRICES];
    await loadApp();
    await threadSettles(7, 'plan the trip', '2026-01-01 10:00:00');
    await threadSettles(8, 'pack the bags', '2026-01-01 11:00:00');
    await openDrawer();

    expect(layout()).toEqual([
      '# Threads', 'thread: pack the bags', 'thread: plan the trip',
      '# Agents', 'agent: compare museum hours', 'agent: museum prices',
    ]);
  });

  it('closes itself once the last row clears, counting threads and agents together', async () => {
    served = [HOURS];
    await loadApp();
    await threadSettles(7, 'plan the trip', '2026-01-01 10:00:00');
    await openDrawer();

    await send(ended(HOURS, 'done'));
    expect(useTasksStore().isOpen).toBe(true);
    expect(layout()).toEqual(['# Threads', 'thread: plan the trip']);

    setTurnDone(7, 'user', false);
    await flushPromises();
    expect(useTasksStore().isOpen).toBe(false);
    expect(drawerScrim().classList.contains('hidden')).toBe(true);
    expect(headerButton()).toBeNull();

    // Agents alone: the last one ending closes it too.
    await send(PRICES);
    await openDrawer();
    await send(ended(PRICES, 'cancelled'));
    expect(useTasksStore().isOpen).toBe(false);
    expect(drawerScrim().classList.contains('hidden')).toBe(true);
    expect(headerButton()).toBeNull();
  });

  it('a thread row opens that thread and closes the drawer, though an agent still runs', async () => {
    served = [HOURS];
    await loadApp();
    await threadSettles(7, 'plan the trip', '2026-01-01 10:00:00');
    const session = useSessionStore();
    await openDrawer();

    drawer().querySelector<HTMLButtonElement>('.task-drawer__thread')!.click();
    await flushPromises();
    expect(session.panelThreadId).toBe(7);
    expect(session.panelType).toBe('user');
    expect(useTasksStore().isOpen).toBe(false);
    expect(layout()).toEqual(['# Agents', 'agent: compare museum hours']);
  });
});

describe('the Activity drawer\'s agents', () => {
  it('lists each running agent from the first read as its type\'s icon and its call\'s summary, oldest first, never a tool name or a number', async () => {
    served = [PRICES, HOURS, agent(14, 'web_search', '')];
    await loadApp();

    expect(agentRows()).toEqual([
      ['compare museum hours', 'lucide-bot'],
      ['museum prices', 'lucide-search'],
      ['Web search', 'lucide-search'],
    ]);
    const listed = drawer().querySelector('.side-drawer__list')!.textContent!;
    expect(listed).not.toMatch(/\d/);
    expect(listed).not.toMatch(/subtask|web_search/);
  });

  it('lists an agent whatever spawned it — the chat, a schedule, or another subagent', async () => {
    served = [
      agent(21, 'subtask', 'from the chat'),
      agent(22, 'web_search', 'from a schedule', { spawned_by_channel: 'schedule' }),
      agent(23, 'web_browse', 'from a subtask', { spawned_by_channel: 'delegate:subtask', spawned_in_turn: 121 }),
    ];
    await loadApp();

    expect(agentRows()).toEqual([
      ['from the chat', 'lucide-bot'],
      ['from a schedule', 'lucide-search'],
      ['from a subtask', 'lucide-globe'],
    ]);
  });

  it('reads every page of running agents, not just the first', async () => {
    served = Array.from({ length: 101 }, (_, i) => agent(1000 + i, 'subtask', `task ${String.fromCharCode(65 + (i % 26))}`));
    await loadApp();

    expect(agentReads).toEqual(['?page=1&limit=100', '?page=2&limit=100']);
    expect(rowButtons()).toHaveLength(101);
  });

  it('adds an agent the moment its frame says it started, and takes it out when its frame says it ended', async () => {
    served = [PRICES];
    await loadApp();
    await openDrawer();

    await send(HOURS);
    expect(agentRows()).toEqual([['compare museum hours', 'lucide-bot'], ['museum prices', 'lucide-search']]);

    await send(ended(HOURS, 'done'));
    expect(agentRows()).toEqual([['museum prices', 'lucide-search']]);
  });

  it.each(['done', 'cancelled', 'failed'] as const)('takes out an agent whose frame says it ended %s, and keeps the rest', async (status) => {
    served = [HOURS, PRICES];
    await loadApp();

    await send(ended(HOURS, status));
    expect(agentRows()).toEqual([['museum prices', 'lucide-search']]);
  });

  it('keeps an agent whose frame says it waits for a reply, the word "waiting" in place of its working dot, until its next turn runs', async () => {
    served = [HOURS, PRICES];
    await loadApp();
    expect(agentStates()).toEqual([['compare museum hours', 'working'], ['museum prices', 'working']]);

    await send(waiting(HOURS));
    expect(agentStates()).toEqual([['compare museum hours', 'waiting'], ['museum prices', 'working']]);

    await send(resumed(HOURS));
    expect(agentStates()).toEqual([['compare museum hours', 'working'], ['museum prices', 'working']]);
  });

  it.each([
    ['stopped while it waits', [ended(waiting(HOURS), 'cancelled')]],
    ['done once the turn its reply opened ends', [resumed(HOURS), ended(resumed(HOURS), 'done')]],
  ] as const)('takes out a waiting agent once it ended — %s — and keeps the rest', async (_how, frames) => {
    served = [HOURS, PRICES];
    await loadApp();
    await send(waiting(HOURS));
    expect(agentStates()).toEqual([['compare museum hours', 'waiting'], ['museum prices', 'working']]);

    for (const frame of frames) await send(frame);
    expect(agentStates()).toEqual([['museum prices', 'working']]);
  });

  it('keeps the Activity button, and the agent in the drawer, while the only agent waits for a reply', async () => {
    served = [HOURS];
    await loadApp();

    await send(waiting(HOURS));
    expect(headerButton()?.classList.contains('has-activity')).toBe(true);
    await openDrawer();
    expect(layout()).toEqual(['# Agents', 'agent: compare museum hours']);
    expect(agentStates()).toEqual([['compare museum hours', 'waiting']]);
  });

  it('updates a listed agent in place from a later frame for the same agent, never as a second row', async () => {
    served = [HOURS, PRICES];
    await loadApp();

    await send({ ...HOURS, summary: 'compare museum and gallery hours' });
    expect(agentRows()).toEqual([
      ['compare museum and gallery hours', 'lucide-bot'],
      ['museum prices', 'lucide-search'],
    ]);
  });

  it('a pick opens that agent\'s own transcript headed by the row\'s title — the call\'s summary, else its type\'s name — and closes the drawer', async () => {
    served = [HOURS, agent(14, 'web_search', '')];
    await loadApp();
    mounted.push(mount(ThreadPanel, { attachTo: document.body }));
    const session = useSessionStore();

    await openDrawer();
    rowTitled('compare museum hours').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:subtask', turn_id: 111 });
    expect(panelHeading()).toBe('compare museum hours');
    expect(useTasksStore().isOpen).toBe(false);
    expect(drawerScrim().classList.contains('hidden')).toBe(true);

    await openDrawer();
    rowTitled('Web search').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 114 });
    expect(panelHeading()).toBe('Web search');
    expect(useTasksStore().isOpen).toBe(false);
  });

  it('Escape or a click outside closes the drawer without opening anything', async () => {
    served = [HOURS];
    await loadApp();
    const session = useSessionStore();

    await openDrawer();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await flushPromises();
    expect(useTasksStore().isOpen).toBe(false);
    expect(drawerScrim().classList.contains('hidden')).toBe(true);

    await openDrawer();
    drawerScrim().click();
    await flushPromises();
    expect(useTasksStore().isOpen).toBe(false);
    expect(session.panelDelegate).toBeNull();
  });

  it('lists every kind of agent, each its own row though they share a turn number, each opening its own transcript', async () => {
    served = [
      agent(31, 'subtask', 'compare museum hours', { turn_id: 3 }),
      agent(32, 'web_search', 'search museum prices', { turn_id: 3 }),
      agent(33, 'web_browse', 'read the museum site', { turn_id: 3 }),
      agent(34, 'pim', 'check my calendar', { turn_id: 3 }),
      agent(35, 'code_agent', 'fix the itinerary script', { turn_id: 3 }),
    ];
    await loadApp();
    const session = useSessionStore();

    expect(agentRows()).toEqual([
      ['compare museum hours', 'lucide-bot'],
      ['search museum prices', 'lucide-search'],
      ['read the museum site', 'lucide-globe'],
      ['check my calendar', 'lucide-mail'],
      ['fix the itinerary script', 'lucide-square-terminal'],
    ]);

    await openDrawer();
    rowTitled('compare museum hours').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:subtask', turn_id: 3 });
    await openDrawer();
    rowTitled('fix the itinerary script').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:code_agent', turn_id: 3 });
  });

  it('titles every kind of agent whose call has no summary by its type\'s plain name under its type\'s icon', async () => {
    served = KINDS.map((kind, i) => agent(41 + i, kind, kind === 'web_search' ? ' ' : ''));
    await loadApp();

    expect(agentRows()).toEqual([
      ['Subtask', 'lucide-bot'],
      ['Web search', 'lucide-search'],
      ['Web browsing', 'lucide-globe'],
      ['Mail & calendar', 'lucide-mail'],
      ['Coding', 'lucide-square-terminal'],
    ]);
    const listed = drawer().querySelector('.side-drawer__list')!.textContent!;
    expect(listed).not.toMatch(/\d|subtask|web_search|web_browse|pim|code_agent/);
  });
});

describe('a read and frames that cross it', () => {
  it('a frame that lands while a read is in flight wins over that read: an agent that ended stays out, one that started stays in', async () => {
    served = [HOURS];
    await loadApp();
    expect(agentRows()).toEqual([['compare museum hours', 'lucide-bot']]);

    // A reconnect's read is answered with the rows as they stood when it was
    // sent — HOURS still running — but answers only after both frames land.
    let release!: () => void;
    holdAgentsRead = new Promise<void>((resolve) => { release = resolve; });
    const read = useAgentsStore().refresh();
    await flushPromises();
    await send(ended(HOURS, 'done'));
    await send(PRICES);
    release();
    await read;
    await flushPromises();

    expect(agentRows()).toEqual([['museum prices', 'lucide-search']]);
    expect(headerButton()).not.toBeNull();
  });

  it('only the newest read applies: an older read that answers last is dropped', async () => {
    await loadApp();

    let releaseOld!: () => void;
    served = [HOURS];
    holdAgentsRead = new Promise<void>((resolve) => { releaseOld = resolve; });
    const older = useAgentsStore().refresh();
    await flushPromises();

    served = [PRICES];
    holdAgentsRead = null;
    await useAgentsStore().refresh();
    await flushPromises();
    expect(agentRows()).toEqual([['museum prices', 'lucide-search']]);

    releaseOld();
    await older;
    await flushPromises();
    expect(agentRows()).toEqual([['museum prices', 'lucide-search']]);
  });

  it('a read that fails leaves the list as it was, and says so in the console', async () => {
    served = [HOURS];
    await loadApp();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    vi.stubGlobal('fetch', async () => json({ success: false, error: 'boom' }, 500));
    await useAgentsStore().refresh();
    await flushPromises();

    expect(agentRows()).toEqual([['compare museum hours', 'lucide-bot']]);
    expect(warn).toHaveBeenCalledWith('[agents] active agents fetch failed:', expect.anything());
  });
});

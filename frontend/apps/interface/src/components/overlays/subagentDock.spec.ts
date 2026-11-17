// @vitest-environment happy-dom
/**
 * The Subagents dock and its header button — feature spec.
 *
 * A subagent call (`subtask`, `web_search`, `web_browse`, `pim`, `code_agent`)
 * writes one `agents` row, and the row keeps running after the call — often
 * after the whole turn that started it settled. While any agent runs, the
 * header's Subagents button turns pink. Clicking it slides out the Subagents
 * dock, which lists every running agent once, whichever turn or channel
 * spawned it: its type's icon and the one-line summary of the call that
 * started it — its type's plain name when the call has none, never the tool
 * name or a number. Picking a row opens its transcript in the thread panel,
 * headed by the row's title, and closes the dock.
 *
 * The list is the agents table: read in full over `GET /api/agents/all` on
 * load, then kept current by each `agent_changed` frame, driven here through
 * the real dispatcher (`dispatchDrift`), the same entry point the WebSocket
 * feeds. A running row is added or updated; an ended one is removed.
 *
 * Real components throughout (PresenceBar, SubagentDock, SideDrawer,
 * ThreadPanel), the real Pinia stores and the real API client. The only fake
 * is the backend at the network boundary: `fetch`, answering the agents read
 * with the rows served and a subagent's transcript read with its task.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import ThreadPanel from '../conversation/ThreadPanel.vue';
import PresenceBar from '../layout/PresenceBar.vue';
import SubagentDock from './SubagentDock.vue';
import { useAgentsStore } from '../../stores/agents';
import { useSessionStore } from '../../stores/session';
import { dispatchDrift } from '../../utils/driftDispatcher';
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

beforeEach(() => {
  setActivePinia(createPinia());
  served = [];
  agentReads = [];
  holdAgentsRead = null;
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

/** The app's header and its Subagents dock, as the app mounts them, then the
 *  agents read the page makes when its socket connects. */
async function loadApp(): Promise<void> {
  mounted.push(mount(PresenceBar, { attachTo: document.body }));
  mounted.push(mount(SubagentDock, { attachTo: document.body }));
  await useAgentsStore().refresh();
  await flushPromises();
}

/** One `agent_changed` frame, as the WebSocket delivers it. */
async function send(row: Agent): Promise<void> {
  dispatchDrift({ type: 'agent_changed', agent: row } as unknown as WsPushEvent);
  await flushPromises();
}

const ended = (row: Agent, status: Exclude<AgentStatus, 'running'>): Agent =>
  ({ ...row, status, response_id: status === 'done' ? 5000 + row.agent_id : null });

const headerButton = (): HTMLButtonElement => document.querySelector<HTMLButtonElement>('#subagentDockBtn')!;
const isPink = (): boolean => headerButton().classList.contains('has-activity');
const dock = (): HTMLElement => document.querySelector<HTMLElement>('aside[aria-label="Subagents"]')!;
/** The click-catcher behind the dock: shown only while the dock is open. */
const dockScrim = (): HTMLElement => document.querySelector<HTMLElement>('.side-drawer__scrim')!;
const rowButtons = (): HTMLButtonElement[] => [...dock().querySelectorAll<HTMLButtonElement>('.drawer-row')];
const emptyText = (): string | null => dock().querySelector('.drawer-empty')?.textContent ?? null;

/** The lucide icon an element draws, by its class (`lucide-search`). */
const iconOf = (el: Element | null): string | undefined =>
  el ? [...el.classList].find((k) => /^lucide-(?!icon$).*(?<!-icon)$/.test(k)) : undefined;

/** Each dock row, in order, as [its title, its type's icon]. A row has no line
 *  under its title. */
function dockRows(): (string | undefined)[][] {
  return rowButtons().map((r) => {
    expect(r.querySelector('.drawer-row__sub')).toBeNull();
    return [r.querySelector('.drawer-row__label')!.textContent!.trim(), iconOf(r.querySelector('.drawer-row__icon'))];
  });
}

function rowTitled(title: string): HTMLButtonElement {
  const found = rowButtons().filter((r) => r.querySelector('.drawer-row__label')!.textContent!.trim() === title);
  expect(found).toHaveLength(1);
  return found[0]!;
}

async function openDock(): Promise<void> {
  headerButton().click();
  await flushPromises();
  expect(useSessionStore().subagentDockOpen).toBe(true);
  expect(dock().classList.contains('hidden')).toBe(false);
  expect(dockScrim().classList.contains('hidden')).toBe(false);
}

/** The open transcript panel's heading. */
const panelHeading = (): string | null => document.querySelector('.thread-panel__title')?.textContent ?? null;

const HOURS = agent(11, 'subtask', 'compare museum hours');
const PRICES = agent(12, 'web_search', 'museum prices');

describe('the header\'s Subagents button', () => {
  it('is plain when the first read finds no agent running', async () => {
    await loadApp();
    expect(agentReads).toEqual(['?page=1&limit=100']);
    expect(isPink()).toBe(false);
  });

  it('is pink while any agent runs and plain again once the last one ends, whichever way it ended', async () => {
    served = [HOURS, PRICES];
    await loadApp();
    expect(isPink()).toBe(true);

    await send(ended(HOURS, 'done'));
    expect(isPink()).toBe(true);

    await send(ended(PRICES, 'failed'));
    expect(isPink()).toBe(false);

    await send(agent(13, 'pim', 'check my calendar'));
    expect(isPink()).toBe(true);
    await send(ended(agent(13, 'pim', 'check my calendar'), 'cancelled'));
    expect(isPink()).toBe(false);
  });
});

describe('the Subagents dock', () => {
  it('lists each running agent from the first read as its type\'s icon and its call\'s summary, oldest first, never a tool name or a number', async () => {
    served = [PRICES, HOURS, agent(14, 'web_search', '')];
    await loadApp();

    expect(dockRows()).toEqual([
      ['compare museum hours', 'lucide-bot'],
      ['museum prices', 'lucide-search'],
      ['Web search', 'lucide-search'],
    ]);
    const listed = dock().querySelector('.side-drawer__list')!.textContent!;
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

    expect(dockRows()).toEqual([
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

  it('says no subagents run when none does, adds one the moment its frame says it started, and takes it out when its frame says it ended', async () => {
    await loadApp();
    await openDock();
    expect(dockRows()).toEqual([]);
    expect(emptyText()).toBe('No subagents running.');

    await send(HOURS);
    expect(dockRows()).toEqual([['compare museum hours', 'lucide-bot']]);
    expect(emptyText()).toBeNull();

    await send(ended(HOURS, 'done'));
    expect(dockRows()).toEqual([]);
    expect(emptyText()).toBe('No subagents running.');
  });

  it.each(['done', 'cancelled', 'failed'] as const)('takes out an agent whose frame says it ended %s, and keeps the rest', async (status) => {
    served = [HOURS, PRICES];
    await loadApp();

    await send(ended(HOURS, status));
    expect(dockRows()).toEqual([['museum prices', 'lucide-search']]);
  });

  it('updates a listed agent in place from a later frame for the same agent, never as a second row', async () => {
    served = [HOURS, PRICES];
    await loadApp();

    await send({ ...HOURS, summary: 'compare museum and gallery hours' });
    expect(dockRows()).toEqual([
      ['compare museum and gallery hours', 'lucide-bot'],
      ['museum prices', 'lucide-search'],
    ]);
  });

  it('a pick opens that agent\'s own transcript headed by the row\'s title — the call\'s summary, else its type\'s name — and closes the dock', async () => {
    served = [HOURS, agent(14, 'web_search', '')];
    await loadApp();
    mounted.push(mount(ThreadPanel, { attachTo: document.body }));
    const session = useSessionStore();

    await openDock();
    rowTitled('compare museum hours').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:subtask', turn_id: 111 });
    expect(panelHeading()).toBe('compare museum hours');
    expect(session.subagentDockOpen).toBe(false);
    expect(dockScrim().classList.contains('hidden')).toBe(true);

    await openDock();
    rowTitled('Web search').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 114 });
    expect(panelHeading()).toBe('Web search');
    expect(session.subagentDockOpen).toBe(false);
  });

  it('Escape or a click outside closes the dock without opening anything', async () => {
    served = [HOURS];
    await loadApp();
    const session = useSessionStore();

    await openDock();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await flushPromises();
    expect(session.subagentDockOpen).toBe(false);
    expect(dockScrim().classList.contains('hidden')).toBe(true);

    await openDock();
    dockScrim().click();
    await flushPromises();
    expect(session.subagentDockOpen).toBe(false);
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

    expect(dockRows()).toEqual([
      ['compare museum hours', 'lucide-bot'],
      ['search museum prices', 'lucide-search'],
      ['read the museum site', 'lucide-globe'],
      ['check my calendar', 'lucide-mail'],
      ['fix the itinerary script', 'lucide-square-terminal'],
    ]);

    await openDock();
    rowTitled('compare museum hours').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:subtask', turn_id: 3 });
    await openDock();
    rowTitled('fix the itinerary script').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:code_agent', turn_id: 3 });
  });

  it('titles every kind of agent whose call has no summary by its type\'s plain name under its type\'s icon', async () => {
    served = KINDS.map((kind, i) => agent(41 + i, kind, kind === 'web_search' ? ' ' : ''));
    await loadApp();

    expect(dockRows()).toEqual([
      ['Subtask', 'lucide-bot'],
      ['Web search', 'lucide-search'],
      ['Web browsing', 'lucide-globe'],
      ['Mail & calendar', 'lucide-mail'],
      ['Coding', 'lucide-square-terminal'],
    ]);
    const listed = dock().querySelector('.side-drawer__list')!.textContent!;
    expect(listed).not.toMatch(/\d|subtask|web_search|web_browse|pim|code_agent/);
  });
});

describe('a read and frames that cross it', () => {
  it('a frame that lands while a read is in flight wins over that read: an agent that ended stays out, one that started stays in', async () => {
    served = [HOURS];
    await loadApp();
    expect(dockRows()).toEqual([['compare museum hours', 'lucide-bot']]);

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

    expect(dockRows()).toEqual([['museum prices', 'lucide-search']]);
    expect(isPink()).toBe(true);
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
    expect(dockRows()).toEqual([['museum prices', 'lucide-search']]);

    releaseOld();
    await older;
    await flushPromises();
    expect(dockRows()).toEqual([['museum prices', 'lucide-search']]);
  });

  it('a read that fails leaves the list as it was, and says so in the console', async () => {
    served = [HOURS];
    await loadApp();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    vi.stubGlobal('fetch', async () => json({ success: false, error: 'boom' }, 500));
    await useAgentsStore().refresh();
    await flushPromises();

    expect(dockRows()).toEqual([['compare museum hours', 'lucide-bot']]);
    expect(warn).toHaveBeenCalledWith('[agents] running agents fetch failed:', expect.anything());
  });
});

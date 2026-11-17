// @vitest-environment happy-dom
/**
 * The Subagents dock and its header button — feature spec.
 *
 * On the person's chat a subagent call (`subtask`, `web_search`, `web_browse`,
 * `pim`, `code_agent`) starts a background turn on its own `delegate:<name>`
 * channel and returns at once, so the subagent keeps running after the call —
 * often after the whole turn that started it settled. While any subagent of
 * any turn on screen still runs, the header's Subagents button turns pink.
 * Clicking it slides out the Subagents dock, which lists every running
 * subagent once: the one-line summary of the call that started it (the
 * subagent's name when the call has none) over that subagent's name. Picking a
 * row opens its transcript in the thread panel, headed by the row's title, and
 * closes the dock.
 *
 * Whether a subagent still runs is decided by its own execution frames once one
 * was seen, else by the REST read's `working` on its chip. The frames are
 * driven through the real dispatcher (`dispatchDrift`), the same entry point
 * the WebSocket feeds.
 *
 * Real components throughout (TurnView, PresenceBar, SubagentDock, SideDrawer,
 * ThreadPanel), the real Pinia session store, the real live act-trail and the
 * real list of running subagents. The only fake is the backend at the network
 * boundary: `fetch`, answering a subagent's transcript read with the task it
 * was given.
 *
 * The list of running subagents is module-level state that every mounted turn
 * view adds to, so every turn view mounted here is unmounted after its test,
 * and the list is checked empty once they are.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import type { DelegateRef, TurnExecutionState, WsPushEvent } from '@chalie/shared';
import TurnView from '../conversation/TurnView.vue';
import ThreadPanel from '../conversation/ThreadPanel.vue';
import PresenceBar from '../layout/PresenceBar.vue';
import SubagentDock from './SubagentDock.vue';
import { useSessionStore } from '../../stores/session';
import { dispatchDrift } from '../../utils/driftDispatcher';
import {
  clearAll, clearDelegateFeeds, runningSubagents, setDelegateState, setLiveToolDelegate, startLiveTool,
} from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock, ToolCallChip } from '../../api/conversation';

const SUBTASK = 'delegate:subtask';
const WEB_SEARCH = 'delegate:web_search';
const WEB_BROWSE = 'delegate:web_browse';
const PIM = 'delegate:pim';
const CODE_AGENT = 'delegate:code_agent';
const QUIET_TURN = 900;
const TURN = 901;
/** Every subagent turn this spec drives frames for, as [channel, turn]. */
const FRAMED: [string, number][] = [[SUBTASK, 4], [SUBTASK, 5], [WEB_SEARCH, 6], [WEB_SEARCH, 7]];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** The backend at the network boundary: a subagent's transcript read
 *  (`GET /api/threads/<id>?channel=…`) answers with the task it was given. */
async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
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

function track<T extends VueWrapper>(wrapper: T): T {
  mounted.push(wrapper);
  return wrapper;
}

/** Takes one view off the screen before the test ends. */
function unmountNow(wrapper: VueWrapper): void {
  mounted = mounted.filter((w) => w !== wrapper);
  wrapper.unmount();
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  // Every turn view is gone, so none still shows a running subagent.
  expect(runningSubagents()).toEqual([]);
  clearAll(ConfigType.USER);
  // An ended subagent's state survives `clearDelegateFeeds()`, which drops
  // only still-running entries — so mark every subagent this spec drives
  // frames for as running first, and the clear leaves none of them seen.
  for (const [channel, id] of FRAMED) setDelegateState(channel, id, 'working');
  clearDelegateFeeds();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

function chip(id: number, toolName: string, summary: string, delegate: ToolCallChip['delegate']): ToolCallChip {
  return { id, tool_name: toolName, params: null, summary, state: 'done', ended_at: null, delegate };
}

/** A subagent ref as a REST read carries it: whether its turn ran at the read. */
const ref = (channel: string, turnId: number, working: boolean): DelegateRef => ({ channel, turn_id: turnId, working });

/** Subtask #4 still running as of the read; subtask #5 already done. */
const START_4 = chip(91, 'subtask', 'compare museum hours', ref(SUBTASK, 4, true));
const START_5 = chip(92, 'subtask', 'book a table', ref(SUBTASK, 5, false));
const PLAIN = chip(93, 'calendar', 'todays agenda', null);

function row(turnId: number, n: number, role: ConversationMessage['role'], content: string, extra: Partial<ConversationMessage> = {}): ConversationMessage {
  return {
    id: `${turnId}${n}`, role, content, timestamp: '2026-01-01 00:00:00', day: '2026-01-01',
    created_at: '2026-01-01T00:00:00Z', turn_id: turnId, ...extra,
  };
}

/** A settled chat turn: the prompt, then the settled reply that made `toolCalls`. */
function settledBlock(turnId: number, toolCalls: ToolCallChip[]): ConversationTurnBlock {
  const messages = [
    row(turnId, 0, 'user', 'Plan my museum day'),
    row(turnId, 1, 'assistant', 'I started some subagents for that.', { tool_calls: toolCalls, settled: true }),
  ];
  return {
    turn_id: turnId, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working: false, duration_ms: 0, messages, type: 'user',
  };
}

function mountTurn(block: ConversationTurnBlock, fullThread = false): VueWrapper {
  return track(mount(TurnView, {
    props: { block, type: ConfigType.USER, fullThread },
    attachTo: document.body,
  }));
}

/** The app's header and its Subagents dock, as the app mounts them. */
function mountHeaderAndDock(): void {
  track(mount(PresenceBar, { attachTo: document.body }));
  track(mount(SubagentDock, { attachTo: document.body }));
}

/** A subagent turn's execution frame, as the WebSocket delivers it: no `type`,
 *  its channel in its place. */
function executionFrame(channel: string, turnId: number, state: TurnExecutionState): WsPushEvent {
  return {
    id: turnId, channel, type: null, turn_id: turnId,
    started_at: '2026-01-01T00:00:01Z', ended_at: state === 'working' ? null : '2026-01-01T00:00:09Z',
    cancel_requested: false, state, stop_reason: null,
  } as unknown as WsPushEvent;
}

async function send(frame: WsPushEvent): Promise<void> {
  dispatchDrift(frame);
  await flushPromises();
}

const headerButton = (): HTMLButtonElement => document.querySelector<HTMLButtonElement>('#subagentDockBtn')!;
const isPink = (): boolean => headerButton().classList.contains('has-activity');
const dock = (): HTMLElement => document.querySelector<HTMLElement>('aside[aria-label="Subagents"]')!;
/** The click-catcher behind the dock: shown only while the dock is open. */
const dockScrim = (): HTMLElement => document.querySelector<HTMLElement>('.side-drawer__scrim')!;
const rowButtons = (): HTMLButtonElement[] => [...dock().querySelectorAll<HTMLButtonElement>('.drawer-row')];

/** Each dock row, in order, as [its title, the subagent's name under it]. */
function dockRows(): string[][] {
  return rowButtons().map((r) => [
    r.querySelector('.drawer-row__label')!.textContent!.trim(),
    r.querySelector('.drawer-row__sub')!.textContent!.trim(),
  ]);
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

describe('the header\'s Subagents button', () => {
  it('is pink while any subagent on screen runs and plain again once the last one ends; a turn whose subagents all ended leaves it plain', async () => {
    mountHeaderAndDock();
    mountTurn(settledBlock(QUIET_TURN, [START_5, PLAIN]));
    expect(isPink()).toBe(false);

    mountTurn(settledBlock(TURN, [START_4, chip(94, 'web_search', 'museum prices', ref(WEB_SEARCH, 6, true))]));
    await flushPromises();
    expect(isPink()).toBe(true);

    await send(executionFrame(SUBTASK, 4, 'completed'));
    expect(isPink()).toBe(true);

    await send(executionFrame(WEB_SEARCH, 6, 'crashed'));
    expect(isPink()).toBe(false);
  });
});

describe('the Subagents dock', () => {
  it('lists each running subagent as its call\'s summary over the subagent\'s name, never a turn number; a call with no summary is titled by its subagent\'s name', async () => {
    mountHeaderAndDock();
    mountTurn(settledBlock(TURN, [START_4, START_5, chip(94, 'web_search', '', ref(WEB_SEARCH, 6, true)), PLAIN]));
    await flushPromises();

    expect(dockRows()).toEqual([
      ['compare museum hours', 'subtask'],
      ['web_search', 'web_search'],
    ]);
    // Subtask #4 and web_search #6 — neither number shows.
    expect(dock().querySelector('.side-drawer__list')!.textContent).not.toMatch(/\d/);
  });

  it('a start and a follow-up message to the same subtask are one row, under the first call\'s summary', async () => {
    mountHeaderAndDock();
    const followUp = chip(95, 'subtask', 'also check the cafe', ref(SUBTASK, 4, true));
    mountTurn(settledBlock(TURN, [START_4, followUp, PLAIN]));
    await flushPromises();

    expect(dockRows()).toEqual([['compare museum hours', 'subtask']]);
  });

  it('a pick opens that subagent\'s transcript headed by the row\'s title — the call\'s summary, else the subagent\'s name — and closes the dock', async () => {
    mountHeaderAndDock();
    track(mount(ThreadPanel, { attachTo: document.body }));
    mountTurn(settledBlock(TURN, [START_4, chip(94, 'web_search', '', ref(WEB_SEARCH, 6, true))]));
    const session = useSessionStore();

    await openDock();
    rowTitled('compare museum hours').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: SUBTASK, turn_id: 4 });
    expect(panelHeading()).toBe('compare museum hours');
    expect(session.subagentDockOpen).toBe(false);
    expect(dockScrim().classList.contains('hidden')).toBe(true);

    await openDock();
    rowTitled('web_search').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: WEB_SEARCH, turn_id: 6 });
    expect(panelHeading()).toBe('web_search');
    expect(session.subagentDockOpen).toBe(false);
  });

  it('Escape or a click outside closes the dock without opening anything', async () => {
    mountHeaderAndDock();
    mountTurn(settledBlock(TURN, [START_4]));
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

  it('says no subagents run when none does, and lists one the moment its own frame says it runs', async () => {
    mountHeaderAndDock();
    mountTurn(settledBlock(TURN, [START_5, PLAIN]));
    await openDock();

    expect(dockRows()).toEqual([]);
    expect(dock().querySelector('.drawer-empty')?.textContent).toBe('No subagents running.');

    // #5's read said done, but its own frame says it runs — the frame wins.
    await send(executionFrame(SUBTASK, 5, 'working'));
    expect(dockRows()).toEqual([['book a table', 'subtask']]);
    expect(dock().querySelector('.drawer-empty')).toBeNull();
  });

  it('lists every kind of subagent — subtask, web_search, web_browse, pim, code_agent — each its own row though they share a turn number, each opening its own transcript', async () => {
    mountHeaderAndDock();
    mountTurn(settledBlock(TURN, [
      chip(101, 'subtask', 'compare museum hours', ref(SUBTASK, 3, true)),
      chip(102, 'web_search', 'search museum prices', ref(WEB_SEARCH, 3, true)),
      chip(103, 'web_browse', 'read the museum site', ref(WEB_BROWSE, 3, true)),
      chip(104, 'pim', 'check my calendar', ref(PIM, 3, true)),
      chip(105, 'code_agent', 'fix the itinerary script', ref(CODE_AGENT, 3, true)),
      PLAIN,
    ]));
    await flushPromises();
    const session = useSessionStore();

    expect(dockRows()).toEqual([
      ['compare museum hours', 'subtask'],
      ['search museum prices', 'web_search'],
      ['read the museum site', 'web_browse'],
      ['check my calendar', 'pim'],
      ['fix the itinerary script', 'code_agent'],
    ]);

    await openDock();
    rowTitled('compare museum hours').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: SUBTASK, turn_id: 3 });
    await openDock();
    rowTitled('fix the itinerary script').click();
    await flushPromises();
    expect(session.panelDelegate).toEqual({ channel: CODE_AGENT, turn_id: 3 });
  });

  it('lists a running step\'s call once it names its subagent, and drops it when that subagent crashes', async () => {
    mountHeaderAndDock();
    const running: ConversationTurnBlock = { ...settledBlock(TURN, []), working: true, messages: [row(TURN, 0, 'user', 'Plan my museum day')] };
    mountTurn(running);

    startLiveTool(ConfigType.USER, TURN, 81, 'web_search', 'museum hours', { instructions: 'Find the museum opening hours' });
    await flushPromises();
    expect(dockRows()).toEqual([]);

    setLiveToolDelegate(ConfigType.USER, TURN, 81, { channel: WEB_SEARCH, turn_id: 7 });
    await flushPromises();
    expect(dockRows()).toEqual([['museum hours', 'web_search']]);
    expect(isPink()).toBe(true);

    await send(executionFrame(WEB_SEARCH, 7, 'crashed'));
    expect(dockRows()).toEqual([]);
    expect(isPink()).toBe(false);
  });

  it('lists a turn drawn twice once, keeps it while either copy is on screen, and drops it with the last copy', async () => {
    mountHeaderAndDock();
    const block = settledBlock(TURN, [START_4, PLAIN]);
    // The chat's copy, then the thread panel's full-thread copy of the same turn.
    const chatCopy = mountTurn(block);
    const panelCopy = mountTurn(block, true);
    await flushPromises();
    expect(dockRows()).toEqual([['compare museum hours', 'subtask']]);

    unmountNow(chatCopy);
    await flushPromises();
    expect(dockRows()).toEqual([['compare museum hours', 'subtask']]);
    expect(isPink()).toBe(true);

    unmountNow(panelCopy);
    await flushPromises();
    expect(dockRows()).toEqual([]);
    expect(dock().querySelector('.drawer-empty')?.textContent).toBe('No subagents running.');
    expect(isPink()).toBe(false);
  });
});

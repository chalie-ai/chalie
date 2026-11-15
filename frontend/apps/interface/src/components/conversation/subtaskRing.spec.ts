// @vitest-environment happy-dom
/**
 * The running-subagents ring — feature spec.
 *
 * On the person's chat a subagent call (`subtask`, `web_search`, `web_browse`,
 * `pim`, `code_agent`) starts a background turn on its own `delegate:<name>`
 * channel and returns at once, so the subagent keeps running after the call —
 * often after the whole turn that started it settled. While any of a turn's
 * subagents still runs, the turn shows a round button at its bottom right,
 * just above its footer, counting them; clicking it lists them, each named as
 * its transcript's panel names it with its turn number, and picking one opens
 * its transcript in the thread panel, as a delegate pill does. Turn numbers
 * are counted per subagent, so two subagents can share one.
 *
 * Whether a subagent still runs is decided by its own execution frames once one
 * was seen, else by the REST read's `working` on its chip. The frames are
 * driven through the real dispatcher (`dispatchDrift`), the same entry point
 * the WebSocket feeds. No delegate surface is registered, so the dispatcher
 * records the run state and fetches nothing — the network is never reached.
 *
 * Real components throughout (TurnView → SubtaskRing), real Pinia session
 * store, real live act-trail and run map, no mocks. A pick is asserted by what
 * it does to the session (`panelDelegate`), the thing the ThreadPanel reads.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { nextTick } from 'vue';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import type { WsPushEvent } from '@chalie/shared';
import TurnView from './TurnView.vue';
import { useSessionStore } from '../../stores/session';
import { dispatchDrift } from '../../utils/driftDispatcher';
import { clearAll, clearDelegateFeeds, setDelegateRunning } from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock, ToolCallChip } from '../../api/conversation';

const SUBTASK = 'delegate:subtask';
const WEB_SEARCH = 'delegate:web_search';
const TURN = 901;
/** Every subagent turn this spec drives frames for, as [channel, turn]. */
const FRAMED: [string, number][] = [[SUBTASK, 4], [SUBTASK, 5]];

let mounted: VueWrapper[] = [];

function track<T extends VueWrapper>(wrapper: T): T {
  mounted.push(wrapper);
  return wrapper;
}

beforeEach(() => {
  setActivePinia(createPinia());
});

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  clearAll(ConfigType.USER);
  // An ended subagent's state survives `clearDelegateFeeds()`, which drops
  // only still-running entries — so mark every subagent this spec drives
  // frames for as running first, and the clear leaves none of them seen.
  for (const [channel, id] of FRAMED) setDelegateRunning(channel, id, true);
  clearDelegateFeeds();
  document.body.innerHTML = '';
});

function chip(id: number, toolName: string, summary: string, delegate: ToolCallChip['delegate']): ToolCallChip {
  return { id, tool_name: toolName, params: null, summary, state: 'done', ended_at: null, delegate };
}

/** The chips of a turn that started two subtasks — #4 still running and #5
 *  already done as of the read — beside an ordinary call. */
const START_4 = chip(91, 'subtask', 'compare museum hours', { channel: SUBTASK, turn_id: 4, working: true });
const START_5 = chip(92, 'subtask', 'book a table', { channel: SUBTASK, turn_id: 5, working: false });
const PLAIN = chip(93, 'calendar', 'todays agenda', null);

function row(id: string, role: ConversationMessage['role'], content: string, extra: Partial<ConversationMessage> = {}): ConversationMessage {
  return {
    id, role, content, timestamp: '2026-01-01 00:00:00', day: '2026-01-01',
    created_at: '2026-01-01T00:00:00Z', turn_id: TURN, ...extra,
  };
}

/** A settled user-channel block: the prompt, then the settled assistant row
 *  that made `toolCalls` and closes the exchange. */
function settledBlock(toolCalls: ToolCallChip[]): ConversationTurnBlock {
  const messages = [
    row('9010', 'user', 'Plan my museum day'),
    row('9011', 'assistant', 'I started two subtasks for that.', { tool_calls: toolCalls, settled: true }),
  ];
  return {
    turn_id: TURN, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working: false, duration_ms: 0, messages, type: 'user',
  };
}

function mountTurn(toolCalls: ToolCallChip[]): VueWrapper {
  return track(mount(TurnView, {
    props: { block: settledBlock(toolCalls), type: ConfigType.USER },
    attachTo: document.body,
  }));
}

/** A delegate turn's execution frame, as the WebSocket delivers it: no `type`,
 *  its channel in its place. */
function executionFrame(channel: string, turnId: number, state: 'working' | 'completed'): WsPushEvent {
  return {
    id: turnId, channel, type: null, turn_id: turnId,
    started_at: '2026-01-01T00:00:01Z', ended_at: state === 'working' ? null : '2026-01-01T00:00:09Z',
    cancel_requested: false, state, stop_reason: null,
  } as unknown as WsPushEvent;
}

/** The turn's rows, in the order drawn. */
function rowsOf(wrapper: VueWrapper): HTMLElement[] {
  return [...(wrapper.element as HTMLElement).children].filter(
    (el): el is HTMLElement => el instanceof HTMLElement && el.classList.contains('msg-row'),
  );
}

function trigger(wrapper: VueWrapper) {
  return wrapper.get('.subtask-ring .subtask-ring__trigger');
}

function menuItems(wrapper: VueWrapper) {
  return wrapper.findAll('.subtask-ring [role="menuitem"]');
}

describe('a settled turn whose subtask still runs', () => {
  it('shows the ring above its footer counting the running subagent; its menu lists it and a pick opens its transcript', async () => {
    const wrapper = mountTurn([START_4, START_5, PLAIN]);
    const session = useSessionStore();

    const ring = wrapper.get('.subtask-ring');
    expect(ring.get('.subtask-ring__count').text()).toBe('1');
    expect(trigger(wrapper).attributes('aria-label')).toBe('1 running subagent(s)');
    expect(trigger(wrapper).attributes('title')).toBe('Running subagents');

    // Its row stands right before the turn's last footer, in the turn's own
    // row order: the prompt, the step, the ring, the footer.
    const rows = rowsOf(wrapper);
    const ringRow = ring.element.closest('.msg-row') as HTMLElement;
    expect(ringRow.classList.contains('msg-row--subtasks')).toBe(true);
    expect(rows.indexOf(ringRow)).toBe(2);
    expect(rows).toHaveLength(4);
    const footers = rows.filter((r) => r.classList.contains('msg-row--footer'));
    expect(footers).toHaveLength(1);
    expect(ringRow.nextElementSibling).toBe(footers.at(-1));

    // Closed until clicked.
    expect(trigger(wrapper).attributes('aria-expanded')).toBe('false');
    expect(wrapper.find('.subtask-ring [role="menu"]').exists()).toBe(false);

    await trigger(wrapper).trigger('click');
    expect(trigger(wrapper).attributes('aria-expanded')).toBe('true');
    const items = menuItems(wrapper);
    // Only the running subtask: #5 had finished as of the read.
    expect(items).toHaveLength(1);
    expect(items[0]!.get('.subtask-ring__name').text()).toBe('subtask #4');
    expect(items[0]!.get('.subtask-ring__summary').text()).toBe('compare museum hours');

    await items[0]!.trigger('click');
    expect(session.panelDelegate).toEqual({ channel: SUBTASK, turn_id: 4 });
    expect(session.panelThreadId).toBeNull();
    expect(wrapper.find('.subtask-ring [role="menu"]').exists()).toBe(false);
    expect(trigger(wrapper).attributes('aria-expanded')).toBe('false');
  });
});

describe('a working turn whose earlier exchange started the subtask', () => {
  it('in the full thread the ring stands last, under the exchange still running — not above the earlier exchange\'s footer', () => {
    const settled = settledBlock([START_4, START_5, PLAIN]);
    const block: ConversationTurnBlock = {
      ...settled,
      working: true,
      messages: [
        ...settled.messages,
        row('9012', 'user', 'And lunch nearby?', { thread_message: true }),
        row('9013', 'assistant', '', { thread_message: true }),
      ],
    };
    const wrapper = track(mount(TurnView, {
      props: { block, type: ConfigType.USER, fullThread: true },
      attachTo: document.body,
    }));

    const rows = rowsOf(wrapper);
    const ringRow = wrapper.get('.subtask-ring').element.closest('.msg-row') as HTMLElement;
    expect(rows.at(-1)).toBe(ringRow);
    // The earlier exchange keeps its own footer, and the ring is not above it.
    const footer = wrapper.get('.msg-row--footer').element;
    expect(footer.previousElementSibling).not.toBe(ringRow);
  });
});

describe('the subtasks\' own execution frames decide once seen', () => {
  it('a terminal frame for the running subtask removes the ring; a working frame for the other brings it back listing that one', async () => {
    const wrapper = mountTurn([START_4, START_5, PLAIN]);
    expect(wrapper.find('.subtask-ring').exists()).toBe(true);

    dispatchDrift(executionFrame(SUBTASK, 4, 'completed'));
    await nextTick();
    expect(wrapper.find('.subtask-ring').exists()).toBe(false);
    expect(wrapper.find('.msg-row--subtasks').exists()).toBe(false);

    // #5's read said done, but its own frame says it runs — the frame wins.
    dispatchDrift(executionFrame(SUBTASK, 5, 'working'));
    await nextTick();
    expect(wrapper.get('.subtask-ring__count').text()).toBe('1');
    await trigger(wrapper).trigger('click');
    const items = menuItems(wrapper);
    expect(items.map((i) => i.get('.subtask-ring__name').text())).toEqual(['subtask #5']);
    expect(items[0]!.get('.subtask-ring__summary').text()).toBe('book a table');
  });
});

describe('one subtask called twice', () => {
  it('a start and a follow-up message call for the same subtask count once, under the first call\'s summary', async () => {
    const followUp = chip(94, 'subtask', 'also check the cafe', { channel: SUBTASK, turn_id: 4, working: true });
    const wrapper = mountTurn([START_4, followUp, PLAIN]);

    expect(wrapper.get('.subtask-ring__count').text()).toBe('1');
    await trigger(wrapper).trigger('click');
    const items = menuItems(wrapper);
    expect(items).toHaveLength(1);
    expect(items[0]!.get('.subtask-ring__name').text()).toBe('subtask #4');
    expect(items[0]!.get('.subtask-ring__summary').text()).toBe('compare museum hours');
  });
});

describe('closing the open menu', () => {
  it('a click outside the ring closes it; a click inside the menu keeps it open', async () => {
    const wrapper = mountTurn([START_4, START_5, PLAIN]);
    await trigger(wrapper).trigger('click');
    expect(wrapper.find('.subtask-ring [role="menu"]').exists()).toBe(true);

    wrapper.get('.subtask-ring [role="menu"]').element.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await nextTick();
    expect(wrapper.find('.subtask-ring [role="menu"]').exists()).toBe(true);

    document.body.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await nextTick();
    expect(wrapper.find('.subtask-ring [role="menu"]').exists()).toBe(false);
    expect(trigger(wrapper).attributes('aria-expanded')).toBe('false');
    expect(useSessionStore().panelDelegate).toBeNull();
  });

  it('Escape closes it', async () => {
    const wrapper = mountTurn([START_4, START_5, PLAIN]);
    await trigger(wrapper).trigger('click');
    expect(wrapper.find('.subtask-ring [role="menu"]').exists()).toBe(true);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await nextTick();
    expect(wrapper.find('.subtask-ring [role="menu"]').exists()).toBe(false);
    expect(trigger(wrapper).attributes('aria-expanded')).toBe('false');
  });
});

describe('every kind of subagent', () => {
  it('another subagent\'s call still marked working draws the ring, named as its transcript\'s panel names it; a block with no running subagent draws none', async () => {
    const search = chip(95, 'web_search', 'museum hours', { channel: WEB_SEARCH, turn_id: 4, working: true });
    const wrapper = mountTurn([PLAIN, search]);
    const session = useSessionStore();

    expect(wrapper.get('.subtask-ring__count').text()).toBe('1');
    await trigger(wrapper).trigger('click');
    const items = menuItems(wrapper);
    expect(items.map((i) => i.get('.subtask-ring__name').text())).toEqual(['web_search #4']);
    expect(items[0]!.get('.subtask-ring__summary').text()).toBe('museum hours');
    await items[0]!.trigger('click');
    expect(session.panelDelegate).toEqual({ channel: WEB_SEARCH, turn_id: 4 });

    const ended = chip(96, 'web_search', 'museum hours', { channel: WEB_SEARCH, turn_id: 4, working: false });
    const quiet = mountTurn([PLAIN, ended]);
    expect(quiet.find('.subtask-ring').exists()).toBe(false);
    expect(quiet.find('.msg-row--subtasks').exists()).toBe(false);
    // The rest of the turn still draws: prompt, step, footer.
    expect(rowsOf(quiet)).toHaveLength(3);
    expect(quiet.find('.msg-row--footer').exists()).toBe(true);
  });

  it('two subagents sharing a turn number are two entries, each opening its own transcript', async () => {
    const subtask = chip(97, 'subtask', 'compare museum hours', { channel: SUBTASK, turn_id: 3, working: true });
    const search = chip(98, 'web_search', 'search museum prices', { channel: WEB_SEARCH, turn_id: 3, working: true });
    const wrapper = mountTurn([subtask, search, PLAIN]);
    const session = useSessionStore();

    expect(wrapper.get('.subtask-ring__count').text()).toBe('2');
    expect(trigger(wrapper).attributes('aria-label')).toBe('2 running subagent(s)');
    await trigger(wrapper).trigger('click');
    expect(menuItems(wrapper).map((i) => [i.get('.subtask-ring__name').text(), i.get('.subtask-ring__summary').text()])).toEqual([
      ['subtask #3', 'compare museum hours'],
      ['web_search #3', 'search museum prices'],
    ]);

    await menuItems(wrapper)[0]!.trigger('click');
    expect(session.panelDelegate).toEqual({ channel: SUBTASK, turn_id: 3 });
    await trigger(wrapper).trigger('click');
    await menuItems(wrapper)[1]!.trigger('click');
    expect(session.panelDelegate).toEqual({ channel: WEB_SEARCH, turn_id: 3 });
  });
});

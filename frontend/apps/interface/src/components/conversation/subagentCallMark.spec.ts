// @vitest-environment happy-dom
/**
 * The mark on a call that started a subagent — feature spec.
 *
 * On the person's chat a subagent call (`subtask`, `web_search`, `web_browse`,
 * `pim`, `code_agent`) starts a background turn on its own `delegate:<name>`
 * channel and returns at once, so the subagent keeps running after the call —
 * often after the whole turn that started it settled. The call's status mark
 * reads how that subagent stands: running while it works, failed once it
 * crashed or was stopped, succeeded once it ended otherwise. A failed one is
 * counted on its step's head. Only the chat does this: a caller that waits (a
 * scheduled task, a subagent's own transcript) got its subagent's outcome as
 * the call's own, so its mark reads the call alone.
 *
 * Whether a subagent still runs is decided by its own execution frames once one
 * was seen, else by the REST read on its chip. The frames are driven through
 * the real dispatcher (`dispatchDrift`), the same entry point the WebSocket
 * feeds. No delegate surface is registered, so the dispatcher records the run
 * state and fetches nothing — the network is never reached.
 *
 * Real components throughout (TurnView and its step's call list), the real
 * Pinia session store and the real live act-trail, no mocks. Every turn view
 * mounted here is unmounted after its test.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { nextTick } from 'vue';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import type { DelegateRef, TurnExecutionState, WsPushEvent } from '@chalie/shared';
import TurnView from './TurnView.vue';
import { dispatchDrift } from '../../utils/driftDispatcher';
import {
  clearAll, clearDelegateFeeds, finishLiveTool, setDelegateState, setLiveToolDelegate, startLiveTool,
} from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock, ToolCallChip } from '../../api/conversation';

const SUBTASK = 'delegate:subtask';
const WEB_SEARCH = 'delegate:web_search';
const TURN = 901;
/** Every subagent turn this spec drives frames for, as [channel, turn]. */
const FRAMED: [string, number][] = [[SUBTASK, 4], [WEB_SEARCH, 6], [WEB_SEARCH, 7]];

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
  for (const [channel, id] of FRAMED) setDelegateState(channel, id, 'working');
  clearDelegateFeeds();
  document.body.innerHTML = '';
});

function chip(id: number, toolName: string, summary: string, delegate: ToolCallChip['delegate']): ToolCallChip {
  return { id, tool_name: toolName, params: null, summary, state: 'done', ended_at: null, delegate };
}

/** A subtask still running as of the read, beside an ordinary call. */
const START_4 = chip(91, 'subtask', 'compare museum hours', { channel: SUBTASK, turn_id: 4, working: true });
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
    row('9011', 'assistant', 'I started some subagents for that.', { tool_calls: toolCalls, settled: true }),
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
function executionFrame(channel: string, turnId: number, state: TurnExecutionState): WsPushEvent {
  return {
    id: turnId, channel, type: null, turn_id: turnId,
    started_at: '2026-01-01T00:00:01Z', ended_at: state === 'working' ? null : '2026-01-01T00:00:09Z',
    cancel_requested: false, state, stop_reason: null,
  } as unknown as WsPushEvent;
}

/** A subagent ref as a REST read carries it: how its turn stood at the read. */
function readRef(channel: string, turnId: number, state: TurnExecutionState): DelegateRef {
  return {
    channel, turn_id: turnId,
    working: state === 'working', crashed: state === 'crashed', cancelled: state === 'cancelled',
  };
}

/** Each call row, in order, as its summary and what its status mark reads. */
function marks(wrapper: VueWrapper): string[] {
  return wrapper.findAll('.calls .call').map((c) => `${c.get('.call__summary').text()}: ${c.get('.call__mark').attributes('aria-label')}`);
}

describe('a subagent call\'s mark on the chat reads as its subagent stands', () => {
  it('reads running while the subagent works, then succeeded on its completed frame', async () => {
    const wrapper = mountTurn([START_4, PLAIN]);
    expect(marks(wrapper)).toEqual(['compare museum hours: running', 'todays agenda: succeeded']);

    dispatchDrift(executionFrame(SUBTASK, 4, 'completed'));
    await nextTick();
    expect(marks(wrapper)).toEqual(['compare museum hours: succeeded', 'todays agenda: succeeded']);
    expect(wrapper.find('.step__failed').exists()).toBe(false);
  });

  it('a running step\'s call reads running once it names its subagent, and failed once that subagent crashes, counted on the step\'s head', async () => {
    startLiveTool(ConfigType.USER, TURN, 81, 'web_search', 'museum hours', { instructions: 'Find the museum opening hours' });
    setLiveToolDelegate(ConfigType.USER, TURN, 81, { channel: WEB_SEARCH, turn_id: 6 });
    finishLiveTool(ConfigType.USER, TURN, 81, true);
    const running: ConversationTurnBlock = { ...settledBlock([]), working: true, messages: [row('9010', 'user', 'Plan my museum day')] };
    const wrapper = track(mount(TurnView, { props: { block: running, type: ConfigType.USER }, attachTo: document.body }));
    expect(marks(wrapper)).toEqual(['museum hours: running']);

    dispatchDrift(executionFrame(WEB_SEARCH, 6, 'crashed'));
    await nextTick();
    expect(marks(wrapper)).toEqual(['museum hours: failed']);
    expect(wrapper.get('.step__failed').text()).toBe('1 failed');
  });

  it('reads failed once the subagent is stopped', async () => {
    const wrapper = mountTurn([chip(95, 'web_search', 'museum hours', readRef(WEB_SEARCH, 7, 'working'))]);
    dispatchDrift(executionFrame(WEB_SEARCH, 7, 'working'));
    await nextTick();
    expect(marks(wrapper)).toEqual(['museum hours: running']);

    dispatchDrift(executionFrame(WEB_SEARCH, 7, 'cancelled'));
    await nextTick();
    expect(marks(wrapper)).toEqual(['museum hours: failed']);
    expect(wrapper.get('.step__failed').text()).toBe('1 failed');
  });

  it('a reloaded history, no frame seen, reads each call as the read had its subagent; a call that failed itself stays failed, and any other tool reads its own outcome', () => {
    const wrapper = mountTurn([
      chip(101, 'subtask', 'compare museum hours', readRef(SUBTASK, 11, 'working')),
      chip(102, 'subtask', 'book a table', readRef(SUBTASK, 12, 'completed')),
      chip(103, 'web_search', 'search ticket prices', readRef(WEB_SEARCH, 13, 'crashed')),
      chip(104, 'web_search', 'search opening days', readRef(WEB_SEARCH, 14, 'cancelled')),
      { ...chip(105, 'web_search', 'search the cafe menu', readRef(WEB_SEARCH, 15, 'completed')), state: 'error' },
      PLAIN,
    ]);
    expect(marks(wrapper)).toEqual([
      'compare museum hours: running',
      'book a table: succeeded',
      'search ticket prices: failed',
      'search opening days: failed',
      'search the cafe menu: failed',
      'todays agenda: succeeded',
    ]);
    expect(wrapper.get('.step__failed').text()).toBe('3 failed');
  });

  it('a caller that waits keeps each call\'s own mark: a scheduled task, and a subagent\'s own transcript', () => {
    const calls = [
      chip(106, 'web_search', 'search ticket prices', readRef(WEB_SEARCH, 16, 'crashed')),
      chip(107, 'subtask', 'book a table', readRef(SUBTASK, 17, 'cancelled')),
    ];
    const scheduled = track(mount(TurnView, {
      props: { block: { ...settledBlock(calls), type: ConfigType.SCHEDULED }, type: ConfigType.SCHEDULED },
      attachTo: document.body,
    }));
    const transcript: DelegateTurnBlock = { ...settledBlock(calls), type: null, channel: SUBTASK };
    const subagent = track(mount(TurnView, {
      props: { block: transcript, canReply: false, fullThread: true },
      attachTo: document.body,
    }));

    for (const wrapper of [scheduled, subagent]) {
      expect(marks(wrapper)).toEqual(['search ticket prices: succeeded', 'book a table: succeeded']);
      expect(wrapper.find('.step__failed').exists()).toBe(false);
    }
  });
});

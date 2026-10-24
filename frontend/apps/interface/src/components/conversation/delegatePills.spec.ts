// @vitest-environment happy-dom
/**
 * Delegate tool-call pills — feature spec.
 *
 * A tool call that ran as a subagent carries `delegate: {channel, turn_id}`.
 * Wherever it is drawn it must become a real <button> (keyboard-reachable,
 * named, with the Bot glyph) that opens that subagent's transcript in the
 * thread panel; a call without a delegate must stay the inert chip it always
 * was. The places a call is drawn are all covered, each through the real
 * component the conversation uses for it:
 *   - live, while the turn works          (the running block's calls)
 *   - settled, in its row's block         (a finished exchange)
 *   - settled mid-stream, in a block      (a working turn's stored rows)
 *
 * Real components, real Pinia session store, real live act-trail feed. A click
 * is asserted by what it does to the session (`panelDelegate`), the thing the
 * ThreadPanel reads, not by spying on a method. The Bot icon is identified by
 * rendering lucide's own Bot and comparing glyphs, so a restyled class name
 * does not break the spec but a different icon does.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { Bot } from '@lucide/vue';
import { ConfigType } from '@chalie/shared';
import TurnView from './TurnView.vue';
import ActivityLine from './ActivityLine.vue';
import { useSessionStore } from '../../stores/session';
import { clearAll, finishLiveTool, setLiveToolDelegate, startLiveTool } from '../../utils/liveActTrail';
import type { LiveToolPill } from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock } from '../../api/conversation';

const REF = { channel: 'delegate:web_search', turn_id: 12 };
const TURN = 301;

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
  document.body.innerHTML = '';
});

/** The inner glyph markup lucide draws for the Bot icon. */
const BOT_GLYPH = mount(Bot).element.innerHTML;

function hasBotIcon(el: Element): boolean {
  return [...el.querySelectorAll('svg')].some((svg) => svg.innerHTML === BOT_GLYPH);
}

/** Every <button> in the wrapper that carries the Bot glyph — the delegate pills. */
function delegateButtons(wrapper: VueWrapper): HTMLButtonElement[] {
  return [...wrapper.element.querySelectorAll('button')].filter(hasBotIcon);
}

/** A delegate button is named by what it shows (its visible text) and carries a
 *  `title` hint — never a label that replaces the visible text. */
function expectNamedByContent(button: HTMLButtonElement, visible: string): void {
  expect(button.getAttribute('type')).toBe('button');
  expect(button.textContent ?? '').toContain(visible);
  expect(button.getAttribute('title')).toBeTruthy();
  const label = button.getAttribute('aria-label');
  if (label != null) expect(label).toContain(visible);
}

/** Every <button> whose visible text mentions `label`. */
function buttonsMentioning(wrapper: VueWrapper, label: string): HTMLButtonElement[] {
  return [...wrapper.element.querySelectorAll('button')]
    .filter((b) => (b.textContent ?? '').includes(label)) as HTMLButtonElement[];
}

/** The element a plain chip's own text sits in. */
function labelOf(wrapper: { element: Element }, text: string): HTMLElement {
  const holder = [...wrapper.element.querySelectorAll<HTMLElement>('*')].find((el) =>
    [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && (n.textContent ?? '').includes(text)));
  expect(holder).toBeDefined();
  return holder!;
}

function livePill(over: Partial<LiveToolPill>): LiveToolPill {
  return {
    id: '31', name: 'web_search', summary: 'museum hours',
    ok: false, resolved: false, delegate: null, ...over,
  };
}

/** The running block's call rows, in the order they ran. */
function liveCalls(wrapper: VueWrapper): HTMLElement[] {
  return [...(wrapper.element as Element).querySelectorAll<HTMLElement>('.activity .call')];
}

type ToolCalls = NonNullable<ConversationMessage['tool_calls']>;

const DELEGATE_CALL: ToolCalls[number] = {
  id: 51, tool_name: 'web_search', summary: 'museum hours', state: 'done', ended_at: null, delegate: REF,
};
const PLAIN_CALL: ToolCalls[number] = {
  id: 52, tool_name: 'calendar', summary: 'todays agenda', state: 'done', ended_at: null, delegate: null,
};

/** The assistant row carrying `toolCalls` — a row whose work was tools. */
function callsMessage(toolCalls: ToolCalls): ConversationMessage {
  return {
    id: '3011', role: 'assistant', content: '', timestamp: '2026-01-01 00:00:01',
    day: '2026-01-01', created_at: '2026-01-01T00:00:00Z', turn_id: TURN, tool_calls: toolCalls,
  };
}

/** The settled assistant answer that closes the exchange: the same shape as
 *  `callsMessage`'s row, with real content and its own id. */
const ANSWER: ConversationMessage = {
  id: '3012', role: 'assistant', content: 'The museum opens at nine.',
  timestamp: '2026-01-01 00:00:02', day: '2026-01-01', created_at: '2026-01-01T00:00:02Z',
  turn_id: TURN, settled: true,
};

function turnWith(working: boolean, toolCalls: ToolCalls, answer?: ConversationMessage): ConversationTurnBlock {
  const messages: ConversationMessage[] = [
    { id: '3010', role: 'user', content: 'When does the museum open?', timestamp: '2026-01-01 00:00:00', day: '2026-01-01', created_at: '2026-01-01T00:00:00Z', turn_id: TURN },
    { ...callsMessage(toolCalls), content: 'Let me look that up.' },
    ...(answer ? [answer] : []),
  ];
  return {
    turn_id: TURN, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

describe('live calls while the turn works', () => {
  it('a delegate call on the running line is a button named by its visible content, with the Bot icon, running or finished, and a click opens that subagent\'s transcript', async () => {
    const wrapper = track(mount(ActivityLine, {
      props: {
        pills: [
          livePill({ id: '31', delegate: REF }),
          livePill({ id: '32', name: 'web_search', resolved: true, ok: true, delegate: { channel: 'delegate:web_search', turn_id: 13 } }),
          // A call with no summary line is still a delegate pill.
          livePill({ id: '33', name: 'web_search', summary: '', delegate: { channel: 'delegate:web_search', turn_id: 14 } }),
        ],
      },
    }));
    const session = useSessionStore();

    expect(liveCalls(wrapper)).toHaveLength(3);
    const buttons = delegateButtons(wrapper);
    expect(buttons).toHaveLength(3);
    // One sentence per pill: 31 and 32 read by their summary, 33 — with no
    // summary — by its name.
    expectNamedByContent(buttons[0]!, 'museum hours');
    expectNamedByContent(buttons[1]!, 'museum hours');
    expectNamedByContent(buttons[2]!, 'web_search');

    buttons[0]!.click();
    expect(session.panelDelegate).toEqual(REF);
    expect(session.panelThreadId).toBeNull();

    // The finished pill opens its own, different, transcript.
    buttons[1]!.click();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 13 });
    buttons[2]!.click();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 14 });
  });

  it('a live call without a delegate stays a plain row: no button, no Bot icon, and nothing opens', async () => {
    const wrapper = track(mount(ActivityLine, {
      props: {
        pills: [
          livePill({ id: '40', name: 'calendar', summary: 'todays agenda' }),
          livePill({ id: '41', name: 'weather', summary: '', resolved: true, ok: false }),
        ],
      },
    }));
    const session = useSessionStore();

    // One row per call, in order: 40 by its summary, 41 — with no summary — by
    // its name. A plain row is a div, not a button, with no Bot glyph.
    const rows = liveCalls(wrapper);
    expect(rows.map((r) => r.textContent?.trim())).toEqual(['todays agenda', 'weather']);
    for (const r of rows) expect(r.tagName).toBe('DIV');
    expect(delegateButtons(wrapper)).toHaveLength(0);

    // Pressing a plain row does nothing and never lands on a button.
    for (const [i, text] of [[0, 'todays agenda'], [1, 'weather']] as const) {
      const label = labelOf({ element: rows[i]! }, text);
      expect(label.closest('button')).toBeNull();
      label.click();
    }
    expect(session.panelDelegate).toBeNull();
  });

  it('the live feed of a working turn draws a delegate pill as a button once its link lands, and keeps it through completion', async () => {
    startLiveTool(ConfigType.USER, TURN, 31, 'web_search', 'museum hours');
    startLiveTool(ConfigType.USER, TURN, 40, 'calendar', 'todays agenda');
    const wrapper = track(mount(TurnView, {
      props: { block: { ...turnWith(true, []), messages: turnWith(true, []).messages.slice(0, 1) }, type: ConfigType.USER },
    }));
    expect(delegateButtons(wrapper)).toHaveLength(0);

    setLiveToolDelegate(ConfigType.USER, TURN, 31, REF);
    await wrapper.vm.$nextTick();
    expect(delegateButtons(wrapper)).toHaveLength(1);
    // The non-delegate call (40) stays a plain row, named by its summary.
    const plain = liveCalls(wrapper).filter((r) => r.textContent?.includes('todays agenda'));
    expect(plain.map((r) => r.tagName)).toEqual(['DIV']);

    finishLiveTool(ConfigType.USER, TURN, 31, true);
    finishLiveTool(ConfigType.USER, TURN, 40, true);
    await wrapper.vm.$nextTick();
    const [pill] = delegateButtons(wrapper);
    expect(pill).toBeDefined();
    pill!.click();
    expect(useSessionStore().panelDelegate).toEqual(REF);
  });
});

describe('settled step blocks', () => {
  it('a delegate call is a named button with the Bot icon that opens its transcript, beside a plain call that is not a button', async () => {
    // The settled exchange's work stands above its answer as a block: the
    // row that carried the calls, then the settled answer that closes it.
    const wrapper = track(mount(TurnView, {
      props: { block: turnWith(false, [DELEGATE_CALL, PLAIN_CALL], ANSWER), type: ConfigType.USER },
    }));
    const session = useSessionStore();

    expect(wrapper.findAll('.call')).toHaveLength(2);
    const buttons = delegateButtons(wrapper);
    expect(buttons).toHaveLength(1);
    // One sentence per row: the delegate row reads by its summary; the plain
    // row's text stays a plain row, not a button.
    expectNamedByContent(buttons[0]!, 'museum hours');
    expect(wrapper.text()).toContain('todays agenda');
    expect(buttonsMentioning(wrapper, 'todays agenda')).toHaveLength(0);

    buttons[0]!.click();
    expect(session.panelDelegate).toEqual(REF);
    expect(session.panelThreadId).toBeNull();
  });

  it('a block\'s delegate button is in the tab order as drawn — no fold holds a call', () => {
    const wrapper = track(mount(TurnView, {
      props: { block: turnWith(false, [DELEGATE_CALL], ANSWER), type: ConfigType.USER },
    }));
    const [button] = delegateButtons(wrapper);
    expect(button).toBeDefined();
    expect(button!.closest('[inert]')).toBeNull();
  });
});

describe('settled calls on a working turn (its step blocks)', () => {
  it('a step row that only called tools draws its delegate call as a button in its block; a plain call in the same block is not a button', async () => {
    // A working turn whose provider call asked for tools and said nothing: its
    // block is that call's work alone, above the running block.
    const turn = turnWith(true, [DELEGATE_CALL, PLAIN_CALL]);
    turn.messages[1] = { ...turn.messages[1]!, content: '' };
    const wrapper = track(mount(TurnView, { props: { block: turn, type: ConfigType.USER } }));
    const session = useSessionStore();
    expect(wrapper.find('[data-transcript-row-id="3011"] .speech-form__text').exists()).toBe(false);
    const buttons = delegateButtons(wrapper);
    expect(buttons).toHaveLength(1);
    expect(buttons[0]!.closest('[inert]')).toBeNull();
    expectNamedByContent(buttons[0]!, 'museum hours');
    expect(wrapper.text()).toContain('todays agenda');
    expect(buttonsMentioning(wrapper, 'todays agenda')).toHaveLength(0);

    buttons[0]!.click();
    expect(session.panelDelegate).toEqual(REF);
  });
});

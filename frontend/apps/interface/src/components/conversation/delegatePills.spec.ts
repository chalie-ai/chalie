// @vitest-environment happy-dom
/**
 * Delegate tool-call pills — feature spec.
 *
 * A tool call that ran as a subagent carries `delegate: {channel, turn_id}`.
 * Wherever it is drawn it must become a real <button> (keyboard-reachable,
 * named, with the Bot glyph) that opens that subagent's transcript in the
 * thread panel; a call without a delegate must stay the inert chip it always
 * was. The three places a call is drawn are all covered, each through the
 * real TurnView the conversation uses:
 *   - live, while the turn works         (ActCycle)
 *   - settled, folded into the footer    (BubbleFooter trace)
 *   - settled mid-stream, on a step row  (the row's own BubbleFooter strip)
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
import ActCycle from './ActCycle.vue';
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
  return [...wrapper.element.querySelectorAll('button')].filter((b) => (b.textContent ?? '').includes(label));
}

/** The control a person presses to unfold a turn's tool trace: the one
 *  expand/collapse button the rendered turn offers (found by role and state,
 *  not by its wording). */
function traceToggle(wrapper: VueWrapper): HTMLButtonElement {
  const toggles = [...(wrapper.element as Element).querySelectorAll<HTMLButtonElement>('button[aria-expanded]')];
  expect(toggles).toHaveLength(1);
  return toggles[0]!;
}

/** The element a plain chip's own text sits in. */
function labelOf(wrapper: VueWrapper, text: string): HTMLElement {
  const holder = [...(wrapper.element as Element).querySelectorAll<HTMLElement>('*')].find((el) =>
    [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && (n.textContent ?? '').includes(text)));
  expect(holder).toBeDefined();
  return holder!;
}

function livePill(over: Partial<LiveToolPill>): LiveToolPill {
  return {
    id: '31', name: 'web_search', summary: 'museum hours', startedAt: Date.now(),
    ok: false, resolved: false, transcriptRowId: null, delegate: null, ...over,
  };
}

type ToolCalls = NonNullable<ConversationMessage['tool_calls']>;

const DELEGATE_CALL: ToolCalls[number] = {
  tool_name: 'web_search', summary: 'museum hours', state: 'done', ended_at: null, delegate: REF,
};
const PLAIN_CALL: ToolCalls[number] = {
  tool_name: 'calendar', summary: 'todays agenda', state: 'done', ended_at: null, delegate: null,
};

function turnWith(working: boolean, toolCalls: ToolCalls): ConversationTurnBlock {
  const messages: ConversationMessage[] = [
    { id: '3010', role: 'user', content: 'When does the museum open?', timestamp: '2026-01-01 00:00:00', day: '2026-01-01', turn_id: TURN },
    {
      id: '3011', role: 'assistant', content: 'Let me look that up.', timestamp: '2026-01-01 00:00:01',
      day: '2026-01-01', turn_id: TURN, tool_calls: toolCalls,
    },
  ];
  return {
    turn_id: TURN, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

describe('live pills while the turn works (ActCycle)', () => {
  it('a delegate pill is a button named by its visible content, with the Bot icon, running or finished, and a click opens that subagent\'s transcript', async () => {
    const wrapper = track(mount(ActCycle, {
      props: {
        pills: [
          livePill({ id: '31', delegate: REF }),
          livePill({ id: '32', name: 'web_search', resolved: true, ok: true, ms: 1800, delegate: { channel: 'delegate:web_search', turn_id: 13 } }),
          // A call with no summary line is still a delegate pill.
          livePill({ id: '33', name: 'web_search', summary: '', delegate: { channel: 'delegate:web_search', turn_id: 14 } }),
        ],
      },
    }));
    const session = useSessionStore();

    const buttons = delegateButtons(wrapper);
    expect(buttons).toHaveLength(3);
    for (const b of buttons) expectNamedByContent(b, 'web_search');

    buttons[0]!.click();
    expect(session.panelDelegate).toEqual(REF);
    expect(session.panelThreadId).toBeNull();

    // The finished pill opens its own, different, transcript.
    buttons[1]!.click();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 13 });
    buttons[2]!.click();
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 14 });
  });

  it('a pill without a delegate stays a plain chip: no button, no Bot icon, and nothing opens', () => {
    const wrapper = track(mount(ActCycle, {
      props: {
        pills: [
          livePill({ id: '40', name: 'calendar', summary: 'todays agenda' }),
          livePill({ id: '41', name: 'weather', summary: '', resolved: true, ok: false }),
        ],
      },
    }));
    const session = useSessionStore();

    expect(wrapper.text()).toContain('calendar');
    expect(wrapper.text()).toContain('weather');
    expect(buttonsMentioning(wrapper, 'calendar')).toHaveLength(0);
    expect(buttonsMentioning(wrapper, 'weather')).toHaveLength(0);
    expect(delegateButtons(wrapper)).toHaveLength(0);

    // Pressing a plain chip does nothing and never lands on a button.
    for (const text of ['calendar', 'weather']) {
      const label = labelOf(wrapper, text);
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
    expect(buttonsMentioning(wrapper, 'calendar')).toHaveLength(0);

    finishLiveTool(ConfigType.USER, TURN, 31, true);
    finishLiveTool(ConfigType.USER, TURN, 40, true);
    await wrapper.vm.$nextTick();
    const [pill] = delegateButtons(wrapper);
    expect(pill).toBeDefined();
    pill!.click();
    expect(useSessionStore().panelDelegate).toEqual(REF);
  });
});

describe('settled pills on the footer trace (BubbleFooter)', () => {
  it('a delegate call is a named button with the Bot icon that opens its transcript, beside a plain call that is not a button', async () => {
    const wrapper = track(mount(TurnView, {
      props: { block: turnWith(false, [DELEGATE_CALL, PLAIN_CALL]), type: ConfigType.USER },
    }));
    const session = useSessionStore();

    // Open the trace the way a person does.
    traceToggle(wrapper).click();
    await wrapper.vm.$nextTick();

    const buttons = delegateButtons(wrapper);
    expect(buttons).toHaveLength(1);
    expectNamedByContent(buttons[0]!, 'web_search');
    expect(wrapper.text()).toContain('calendar');
    expect(buttonsMentioning(wrapper, 'calendar')).toHaveLength(0);

    buttons[0]!.click();
    expect(session.panelDelegate).toEqual(REF);
    expect(session.panelThreadId).toBeNull();
  });

  it('a folded trace keeps its delegate button out of the tab order, and unfolding brings it back', async () => {
    const wrapper = track(mount(TurnView, {
      props: { block: turnWith(false, [DELEGATE_CALL]), type: ConfigType.USER },
    }));
    const [folded] = delegateButtons(wrapper);
    expect(folded).toBeDefined();
    expect(folded!.closest('[inert]')).not.toBeNull();

    traceToggle(wrapper).click();
    await wrapper.vm.$nextTick();

    expect(delegateButtons(wrapper)[0]!.closest('[inert]')).toBeNull();
  });
});

describe('settled pills on a working turn\'s step row (its own trace strip)', () => {
  it('a step row that only called tools draws its delegate call as a button in its own strip; a plain call in the same strip is not a button', async () => {
    // A working turn whose provider call asked for tools and said nothing: no
    // bubble, just that call's own strip under the question.
    const turn = turnWith(true, [DELEGATE_CALL, PLAIN_CALL]);
    turn.messages[1] = { ...turn.messages[1]!, content: '' };
    const wrapper = track(mount(TurnView, { props: { block: turn, type: ConfigType.USER } }));
    const session = useSessionStore();
    expect(wrapper.find('[data-transcript-row-id="3011"]').exists()).toBe(false);
    const [folded] = delegateButtons(wrapper);
    expect(folded).toBeDefined();
    expect(folded!.closest('[inert]')).not.toBeNull();

    traceToggle(wrapper).click();
    await wrapper.vm.$nextTick();

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

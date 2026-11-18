// @vitest-environment happy-dom
/**
 * A turn in the chat listens for nothing — feature spec.
 *
 * Every turn on the page used to register its own document listener to follow
 * whether it was working or done, so a settled turn kept one for as long as
 * the page stayed open. That state now lives on the turn's own root, where the
 * render contract stamps it and flips it in place, and the thread pill's dot
 * reads it from the stylesheet.
 *
 * Real throughout: the turnDom surface registry rendering SpineTurn (and the
 * TurnView inside it) into an attached container, the real Pinia session
 * store, and the stylesheets the app loads, so the dot is read as the reader
 * sees it. happy-dom keeps an element's first style read even after an
 * ancestor's attribute changes, so each state is set before its dot is read.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import '@chalie/shared/styles/main.scss';
import '../../styles/conversation.scss';
import type { ConversationTurnBlock } from '../../api/conversation';
import {
  registerSurface,
  setTurnDone,
  setTurnWorking,
  unregisterSurface,
  upsertTurnToSurfaces,
} from '../../utils/turnDom';
import SpineTurn from './SpineTurn.vue';

const SURFACE = 'spine-turn-spec';

/** A settled turn whose reply was forked into a thread, last active long ago. */
function forkedTurn(turnId: number): ConversationTurnBlock {
  const at = '2026-11-01 10:00:00';
  const msg = { timestamp: at, day: '2026-11-01', created_at: at, turn_id: turnId };
  return {
    turn_id: turnId, gist: 'museum hours', preview: 'when does it open?', last_activity_at: at,
    working: false, duration_ms: 1000, type: ConfigType.USER,
    messages: [
      { ...msg, id: `${turnId}0`, role: 'user', content: 'when does it open?' },
      { ...msg, id: `${turnId}1`, role: 'assistant', content: 'At nine.', settled: true },
      { ...msg, id: `${turnId}2`, role: 'user', content: 'and on Sundays?', thread_message: true },
    ],
  };
}

let container: HTMLElement;

beforeEach(() => {
  setActivePinia(createPinia());
  container = document.createElement('div');
  document.body.appendChild(container);
  registerSurface({ id: SURFACE, type: ConfigType.USER, container, component: SpineTurn });
});

afterEach(() => {
  unregisterSurface(SURFACE);
  container.remove();
  vi.restoreAllMocks();
});

function dotOf(turnId: number): HTMLElement {
  const dot = container.querySelector<HTMLElement>(`[data-turn-id="${turnId}"] .thread-pill__dot`);
  if (!dot) throw new Error(`turn ${turnId} shows no thread pill`);
  return dot;
}

/** What the reader sees of a dot: its fill, its ring, and whether it pulses. */
function looks(dot: HTMLElement): { fill: string; ring: string; pulse: string } {
  const style = getComputedStyle(dot);
  return { fill: style.backgroundColor, ring: style.borderTopWidth, pulse: style.animation };
}

describe('a turn in the chat', () => {
  it('registers no document listener when it renders, so a settled turn carries none', () => {
    const listen = vi.spyOn(document, 'addEventListener');

    upsertTurnToSurfaces(forkedTurn(301), ConfigType.USER);

    expect(dotOf(301).isConnected).toBe(true);
    expect(listen.mock.calls.map(([name]) => name)).toEqual([]);
  });

  it("shows working and done on its thread pill's dot, flipped in place on the rendered turn, with working winning over done", () => {
    const tokens = getComputedStyle(document.documentElement);
    const pink = tokens.getPropertyValue('--pink').trim();
    const text = tokens.getPropertyValue('--text').trim();
    const states = [
      { turnId: 311, working: false, done: false },
      { turnId: 312, working: true, done: false },
      { turnId: 313, working: false, done: true },
      { turnId: 314, working: true, done: true },
    ];

    const seen = states.map(({ turnId, working, done }) => {
      upsertTurnToSurfaces(forkedTurn(turnId), ConfigType.USER);
      const dot = dotOf(turnId);
      if (working) setTurnWorking(turnId, ConfigType.USER, true);
      if (done) setTurnDone(turnId, ConfigType.USER, true);
      // Flipped in place: the dot the turn rendered is still the one on the page.
      expect(dotOf(turnId)).toBe(dot);
      return looks(dot);
    });

    const pulsing = 'working 1.4s ease-in-out infinite';
    expect(seen).toEqual([
      { fill: 'transparent', ring: '1.5px', pulse: '' },
      { fill: pink, ring: '0px', pulse: pulsing },
      { fill: text, ring: '0px', pulse: '' },
      { fill: pink, ring: '0px', pulse: pulsing },
    ]);
  });
});

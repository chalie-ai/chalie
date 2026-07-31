// @vitest-environment happy-dom
/**
 * daymarks — feature spec for the conversation spine's date dividers.
 *
 * Real DOM (happy-dom), real upsertTurn mounting real hosts, real
 * syncDaymarks reconciliation — no mocks, no hand-built `data-day` stamps.
 * The whole point is to exercise the seam the divider actually depends on:
 * backend block → stampDay → `data-day` → syncDaymarks → rendered label.
 *
 * This seam shipped a defect because nothing covered it. The backend's
 * `timestamp` is a *display* string (`%d %b %H:%M` — "29 Jul 10:35", no
 * year); stampDay used to feed it to `new Date()`, which silently invents a
 * year — 2000 in WebKit, 2001 in V8 — so the divider rendered that year's
 * weekday and `Today ·` could never match. The `day` key exists to keep any
 * date parsing out of the frontend entirely; these specs pin that.
 */
import { describe, expect, it, vi } from 'vitest';
import { h } from 'vue';
import type { Component } from 'vue';
import type { ConversationTurnBlock } from '../api/conversation';
import { syncDaymarks } from './daymarks';
import { localDayKey } from './time';

const StubComponent: Component = {
  props: ['block', 'type'],
  render(this: { block: ConversationTurnBlock; type: string }) {
    return h('div', { 'data-turn-id': this.block.turn_id, 'data-type': this.type });
  },
};

/** A block as the backend actually projects it: a year-less display
 *  `timestamp` alongside the machine-readable `day` key. */
function block(turnId: number, day: string, timestamp = '29 Jul 10:35'): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: `turn ${turnId}`,
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    type: 'user',
    messages: [{ id: String(turnId), role: 'user', content: 'hi', timestamp, day, turn_id: turnId }],
  };
}

async function freshTurnDom() {
  vi.resetModules();
  return import('./turnDom');
}

async function mount(container: HTMLElement, blocks: ConversationTurnBlock[]) {
  const { upsertTurn } = await freshTurnDom();
  for (const b of blocks) upsertTurn(b, 'user', container, StubComponent);
}

describe('day stamping — the backend key is the only source', () => {
  it('stamps the block\'s backend `day` verbatim, never a year invented from the display timestamp', async () => {
    const container = document.createElement('div');
    document.body.appendChild(container);

    await mount(container, [block(1, '2026-07-29')]);

    const host = container.querySelector('[data-version]') as HTMLElement;
    expect(host.dataset.day).toBe('2026-07-29');
    // The regression guard: parsing "29 Jul 10:35" yields 2000 (WebKit) or
    // 2001 (V8). Neither may ever reach the stamp.
    expect(host.dataset.day).not.toMatch(/^(2000|2001)-/);
  });

  it('leaves the stamp unset when a block carries no day, so the turn joins the group above it', async () => {
    const container = document.createElement('div');
    document.body.appendChild(container);

    await mount(container, [block(1, '')]);

    const host = container.querySelector('[data-version]') as HTMLElement;
    expect(host.dataset.day).toBeUndefined();
  });
});

describe('syncDaymarks — divider rendering', () => {
  it('labels the current day "Today ·", the symptom the invented year made unreachable', async () => {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const today = localDayKey(new Date());

    await mount(container, [block(1, today)]);
    syncDaymarks(container);

    const mark = container.querySelector('.daymark') as HTMLElement;
    expect(mark.dataset.daymark).toBe(today);
    expect(mark.textContent).toMatch(/^Today · /);
  });

  it('labels a past day with the weekday that day actually fell on', async () => {
    const container = document.createElement('div');
    document.body.appendChild(container);

    // 27 Jul 2026 was a Monday. Under the old parse the year came from the
    // JS engine, not the data — 27 Jul 2000 was a Thursday, 2001 a Friday —
    // which is the class of defect reported pre-release.
    await mount(container, [block(1, '2026-07-27', '27 Jul 09:15')]);
    syncDaymarks(container);

    const mark = container.querySelector('.daymark') as HTMLElement;
    expect(mark.textContent).toBe('Monday · 27 July');
  });

  it('inserts one divider per day boundary, before the first host of each day', async () => {
    const container = document.createElement('div');
    document.body.appendChild(container);

    await mount(container, [
      block(1, '2026-07-27'),
      block(2, '2026-07-27'),
      block(3, '2026-07-29'),
    ]);
    syncDaymarks(container);

    const marks = Array.from(container.querySelectorAll('.daymark')) as HTMLElement[];
    expect(marks.map((m) => m.dataset.daymark)).toEqual(['2026-07-27', '2026-07-29']);
    // The second divider sits immediately before turn 3, not after it.
    expect(marks[1].nextElementSibling?.getAttribute('data-version')).toBe('3');
  });

  it('is idempotent — re-running does not accumulate duplicate dividers', async () => {
    const container = document.createElement('div');
    document.body.appendChild(container);

    await mount(container, [block(1, '2026-07-27'), block(2, '2026-07-29')]);
    syncDaymarks(container);
    syncDaymarks(container);

    expect(container.querySelectorAll('.daymark')).toHaveLength(2);
  });
});

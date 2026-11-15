// @vitest-environment happy-dom
/**
 * Conversation history paging — feature spec.
 *
 * Opening the chat loads the 10 most recent turns, lands at the bottom, and
 * reaching the top loads the next 10 older ones, keeping the reader's place,
 * until none are left.
 *
 * Real throughout: ConversationFeed, the SpineTurn it registers as its
 * surface, the turnDom surface registry, the real Pinia session store, the
 * real autoscroll, and the real REST wrappers and shared ApiClient. Two things
 * are stood in for. `fetch` is the network boundary, answered with a history of
 * `TOTAL` turns. The page's geometry is modelled, because happy-dom has no
 * layout: each rendered turn is `turnPx` tall, the viewport 800px, and
 * animation frames run when the test pumps them. A scroll moves the page and
 * fires `scroll` only when the position changes; a smooth scroll is modelled
 * by its first animation step (2px toward its target), which is what the
 * browser shows the page before a later scroll replaces it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import ConversationFeed from './ConversationFeed.vue';
import type { ConversationTurnBlock } from '../../api/conversation';

const VIEWPORT = 800;
/** Height above the first turn (history spinner / end-of-history pill). */
const HEAD_PX = 100;

/** One `GET /api/threads/all` the feed sent. */
interface ListRead {
  limit: number;
  page: number;
}

let total = 0;
let turnPx = 400;
let scrollY = 0;
let lists: ListRead[] = [];
let batches: number[][] = [];
let frames: FrameRequestCallback[] = [];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function block(turnId: number): ConversationTurnBlock {
  const at = '2026-11-01T10:00:00Z';
  return {
    turn_id: turnId, gist: null, preview: `Question ${turnId}`, last_activity_at: at,
    working: false, duration_ms: 1000, type: 'user',
    messages: [
      { id: `${turnId}0`, role: 'user', content: `Question ${turnId}`, timestamp: at, day: '2026-11-01', created_at: at, turn_id: turnId },
      { id: `${turnId}1`, role: 'assistant', content: `Answer ${turnId}`, timestamp: at, day: '2026-11-01', created_at: at, turn_id: turnId },
    ],
  };
}

/** The backend, at the network boundary: a history of `total` turns, newest
 *  first, listed a page at a time and read in batches. */
async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname === '/api/threads/all') {
    const limit = Number(url.searchParams.get('limit'));
    const page = Number(url.searchParams.get('page'));
    lists.push({ limit, page });
    const ids = Array.from({ length: total }, (_, i) => total - i).slice((page - 1) * limit, page * limit);
    return json({
      success: true,
      result: ids.map((turn_id) => ({ turn_id, last_activity_at: null, row_count: 2, preview: '', gist: null, type: 'user' })),
      pagination: { page, limit, total },
    });
  }
  if (url.pathname === '/api/threads/batch') {
    const ids = url.searchParams.getAll('id[]').map(Number);
    batches.push(ids);
    return json({ success: true, result: { blocks: ids.map(block) } });
  }
  return json({ error: 'unrouted' }, 404);
}

/** Turn ids in the spine, top to bottom. */
function renderedTurns(): number[] {
  return Array.from(document.querySelectorAll('.conversation-spine__turns > [data-version] [data-turn-id]'))
    .map((el) => Number(el.getAttribute('data-turn-id')))
    .filter((id, i, all) => all.indexOf(id) === i);
}

function pageHeight(): number {
  return HEAD_PX + document.querySelectorAll('.conversation-spine__turns > [data-version]').length * turnPx;
}

function maxScroll(): number {
  return Math.max(0, pageHeight() - VIEWPORT);
}

/** Where a turn's top sits in the viewport. */
function viewportTop(turnId: number): number {
  return HEAD_PX + renderedTurns().indexOf(turnId) * turnPx - scrollY;
}

function moveTo(y: number): void {
  const next = Math.min(Math.max(0, Math.round(y)), maxScroll());
  if (next === scrollY) return;
  scrollY = next;
  window.dispatchEvent(new Event('scroll'));
}

function modelScrollTo(options?: ScrollToOptions | number): void {
  if (typeof options !== 'object') throw new Error('the feed scrolls with an options object');
  const target = options.top ?? scrollY;
  moveTo(options.behavior === 'smooth' ? scrollY + Math.sign(target - scrollY) * 2 : target);
}

/** Let fetches, renders and animation frames run until nothing is pending. */
async function settle(): Promise<void> {
  for (let i = 0; i < 50; i++) {
    await flushPromises();
    if (!frames.length) return;
    const due = frames;
    frames = [];
    for (const cb of due) cb(performance.now());
  }
  throw new Error('the feed never went quiet');
}

/** The reader scrolls to the very top, then waits for what that loads. */
async function readerReachesTop(): Promise<void> {
  moveTo(0);
  await settle();
}

let wrapper: VueWrapper | null = null;

async function openChat(): Promise<VueWrapper> {
  wrapper = mount(ConversationFeed, { attachTo: document.body });
  await settle();
  return wrapper;
}

beforeEach(() => {
  setActivePinia(createPinia());
  total = 0;
  turnPx = 400;
  scrollY = 0;
  lists = [];
  batches = [];
  frames = [];
  vi.stubGlobal('fetch', fakeBackend);
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => frames.push(cb));
  vi.stubGlobal('innerHeight', VIEWPORT);
  vi.spyOn(window, 'scrollY', 'get').mockImplementation(() => scrollY);
  vi.spyOn(window, 'scrollTo').mockImplementation(modelScrollTo as typeof window.scrollTo);
  vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockImplementation(pageHeight);
  vi.spyOn(document.body, 'scrollHeight', 'get').mockImplementation(pageHeight);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('opening the chat', () => {
  it('loads only the 10 most recent turns, renders exactly those, and lands at the bottom', async () => {
    total = 60;

    await openChat();

    expect(lists).toEqual([{ limit: 10, page: 1 }]);
    expect(batches).toEqual([[60, 59, 58, 57, 56, 55, 54, 53, 52, 51]]);
    expect(renderedTurns()).toEqual([51, 52, 53, 54, 55, 56, 57, 58, 59, 60]);
    expect(scrollY).toBe(maxScroll());
  });

  it('does not take its own trip to the bottom, which sets off from the top, for the reader reaching the top', async () => {
    // Every rendered turn asks for a smooth scroll to the bottom, and its first
    // step is a scroll a couple of pixels below the top, before the jump to
    // the bottom lands.
    total = 60;

    await openChat();

    expect(lists).toHaveLength(1);
    expect(renderedTurns()).toHaveLength(10);
    expect(scrollY).toBe(maxScroll());
  });

  it('shows the end of history and asks for nothing more when the whole history fits in one page', async () => {
    total = 8;

    const feed = await openChat();

    expect(lists).toEqual([{ limit: 10, page: 1 }]);
    expect(renderedTurns()).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(feed.text()).toContain('End of thread history');
  });

  it('keeps loading older pages while 10 short turns leave the top in view, since such a page cannot be scrolled', async () => {
    // 10 turns of 30px leave the page shorter than the viewport.
    total = 60;
    turnPx = 30;

    await openChat();

    expect(lists).toEqual([{ limit: 10, page: 1 }, { limit: 10, page: 2 }, { limit: 10, page: 3 }]);
    expect(renderedTurns()).toHaveLength(30);
    expect(scrollY).toBeGreaterThanOrEqual(150);
  });
});

describe('reaching the top', () => {
  it('loads exactly the next 10 older turns each time, keeping the reader\'s place, until none are left', async () => {
    total = 25;
    const feed = await openChat();

    const oldTop = renderedTurns()[0];
    moveTo(0);
    const before = viewportTop(oldTop);
    await settle();

    expect(lists).toEqual([{ limit: 10, page: 1 }, { limit: 10, page: 2 }]);
    expect(batches[1]).toEqual([15, 14, 13, 12, 11, 10, 9, 8, 7, 6]);
    expect(renderedTurns()).toEqual(Array.from({ length: 20 }, (_, i) => i + 6));
    // The turn that was at the top has not moved on screen.
    expect(viewportTop(oldTop)).toBe(before);
    expect(feed.text()).not.toContain('End of thread history');

    await readerReachesTop();

    expect(lists.at(-1)).toEqual({ limit: 10, page: 3 });
    expect(renderedTurns()).toEqual(Array.from({ length: 25 }, (_, i) => i + 1));
    expect(feed.text()).toContain('End of thread history');

    await readerReachesTop();

    expect(lists).toHaveLength(3);
  });

  it('does not load anything while the reader is away from the top', async () => {
    total = 60;
    await openChat();

    moveTo(1000);
    await settle();
    moveTo(150);
    await settle();

    expect(lists).toHaveLength(1);
  });
});

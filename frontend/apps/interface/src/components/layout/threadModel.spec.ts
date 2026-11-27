// @vitest-environment happy-dom
/**
 * A thread's model, thinking level and context ring — feature spec.
 *
 * Each thread keeps its own model and thinking level, picked from one menu:
 * it lists the models, and choosing one lists that model's levels. A pick is
 * saved to the dock's thread at once and no message carries one: a thread
 * panel's dock saves to its own thread; the footer dock shows and saves the
 * channel's latest thread, which a new thread copies, and on a channel with no
 * thread yet shows the default with nothing to change until the first message
 * makes one. Beside the menu, a ring shows how full the thread's context
 * window is, fed by the `context_usage` frame the server pushes per call.
 *
 * Real throughout: the InputDock, the Pinia stores, the session send path, the
 * shared WebSocketService and the drift router a frame goes through. The only
 * fakes are the boundaries: `fetch` (the backend), the global `WebSocket` class
 * (an open socket the test pushes frames into) and the Brain navigation.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { getWebSocket, webPlatformAdapter } from '@chalie/shared';
import InputDock from './InputDock.vue';
import { useSessionStore } from '../../stores/session';
import { dispatchDrift } from '../../utils/driftDispatcher';

/** One multipart chat send: where it went and the settings it carried. */
interface Send {
  path: string;
  provider: string | null;
  level: string | null;
}

const OPUS = { id: 3, name: 'Anthropic', model: 'claude-opus-4', context_window: 200_000 };
const SONNET = { id: 4, name: 'Work', model: 'claude-sonnet-4', context_window: 100_000 };

interface Settings {
  provider_id: number;
  thinking_level: string;
}

/** The model the backend falls back to: the Brain's default. */
const DEFAULT = OPUS;

const sends: Send[] = [];
const settingsReads: string[] = [];
const settingsSaves: { path: string; body: unknown }[] = [];
/** The channel's threads and their saved settings, by turn id. */
let threads = new Map<number, Settings>();
let saveFails = false;
/** Every configured model, served a page at a time; null fails the listing. */
let modelList: { id: number; name: string; model: string; context_window: number | null }[] | null = [];
/** The total the listing claims, when it differs from the rows it holds. */
let claimedTotal: number | null = null;
/** The listing pages read, in order. */
const listingPages: number[] = [];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** The channel's latest thread (highest turn id), which -1 names. */
function latestThread(): number | null {
  return threads.size ? Math.max(...threads.keys()) : null;
}

/** The backend, at the network boundary: the model listing, the settings read
 *  and save, and the chat send. As the server does, -1 names the channel's
 *  latest thread, and a send to -1 starts a thread that copies its settings. */
async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname === '/api/providers/all') {
    if (modelList == null) return json({ error: 'boom' }, 500);
    const page = Number(url.searchParams.get('page'));
    const limit = Number(url.searchParams.get('limit'));
    listingPages.push(page);
    // A reader that never stops paging fails here instead of looping forever.
    if (page > 10) return json({ error: 'no such page' }, 500);
    const result = modelList.slice((page - 1) * limit, page * limit);
    return json({ success: true, result, pagination: { page, limit, total: claimedTotal ?? modelList.length } });
  }
  const settings = /^\/api\/threads\/settings\/(-?\d+)$/.exec(url.pathname);
  if (settings) {
    const named = Number(settings[1]);
    const turnId = named === -1 ? latestThread() : named;
    const saved = turnId == null ? undefined : threads.get(turnId);
    if (init?.method === 'POST') {
      const body = JSON.parse(String(init.body)) as Settings;
      settingsSaves.push({ path: url.pathname + url.search, body });
      if (saveFails) return json({ error: 'Provider not found' }, 422);
      if (turnId == null || !saved) return json({ error: 'Thread not found' }, 404);
      threads.set(turnId, body);
      return json({ success: true, result: { ...body, turn_id: turnId } });
    }
    settingsReads.push(settings[1]);
    if (named === -1 && turnId == null) {
      return json({ success: true, result: { provider_id: DEFAULT.id, thinking_level: 'auto', turn_id: null } });
    }
    if (turnId == null || !saved) return json({ error: 'Thread not found' }, 404);
    return json({ success: true, result: { ...saved, turn_id: turnId } });
  }
  const chat = /^\/api\/threads\/(-?\d+)$/.exec(url.pathname);
  if (init?.method === 'POST' && chat) {
    const form = init.body as FormData;
    sends.push({
      path: url.pathname,
      provider: form.get('provider_id') as string | null,
      level: form.get('thinking_level') as string | null,
    });
    let turnId = Number(chat[1]);
    if (turnId === -1) {
      const latest = latestThread();
      turnId = (latest ?? 0) + 1;
      threads.set(turnId, latest == null ? { provider_id: DEFAULT.id, thinking_level: 'auto' } : { ...threads.get(latest)! });
    }
    return json({ success: true, result: { turn_id: turnId, type: 'user' } });
  }
  return json({ error: 'unrouted' }, 404);
}

/** A socket that opens as soon as it is created, like a healthy server. The
 *  latest one is kept so a test can push a server frame into it. */
class OpenSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  static latest: OpenSocket | null = null;
  readyState = OpenSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  constructor() {
    OpenSocket.latest = this;
    queueMicrotask(() => {
      this.readyState = OpenSocket.OPEN;
      this.onopen?.();
    });
  }
  close(): void {
    this.readyState = OpenSocket.CLOSED;
    this.onclose?.();
  }
}

/** The frame the server pushes after each chat call, with the tokens it sent
 *  and the model's context window. */
async function pushUsage(turnId: number, tokens: number, window: number): Promise<void> {
  const frame = { status: 'context_usage', type: 'user', turn_id: turnId, tokens_input: tokens, context_window: window };
  OpenSocket.latest?.onmessage?.({ data: JSON.stringify(frame) });
  await flushPromises();
}

let wrapper: VueWrapper | null = null;

async function mountDock(turnId: number | null): Promise<VueWrapper> {
  wrapper = mount(InputDock, {
    props: { dockId: turnId == null ? 'main_spine_dock' : `thread_dock_${turnId}`, turnId },
    attachTo: document.body,
  });
  await flushPromises();
  return wrapper;
}

function shownLabel(dock: VueWrapper): string {
  return dock.get('#modelLabel').text();
}

async function openMenu(dock: VueWrapper): Promise<void> {
  await dock.get('#modelTrigger').trigger('click');
  await flushPromises();
}

async function pick(dock: VueWrapper, modelId: number, level: string): Promise<void> {
  await openMenu(dock);
  await dock.get(`[data-model="${modelId}"]`).trigger('click');
  await flushPromises();
  await dock.get(`[data-level="${level}"]`).trigger('click');
  await flushPromises();
}

async function send(dock: VueWrapper, text: string): Promise<void> {
  await dock.get('textarea').setValue(text);
  await dock.get('button[aria-label="Send message"]').trigger('click');
  await flushPromises();
}

/** Press a key where focus is, the way a keyboard does: on the focused element,
 *  bubbling up to the document. */
async function press(key: string): Promise<void> {
  document.activeElement?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  await flushPromises();
}

beforeEach(async () => {
  setActivePinia(createPinia());
  sends.length = 0;
  settingsReads.length = 0;
  settingsSaves.length = 0;
  threads = new Map();
  saveFails = false;
  modelList = [OPUS, SONNET];
  claimedTotal = null;
  listingPages.length = 0;
  vi.stubGlobal('fetch', fakeBackend);
  vi.stubGlobal('WebSocket', OpenSocket);
  // The same wiring the session gives the socket on boot: every server frame
  // goes through the drift router.
  getWebSocket().onDrift(dispatchDrift);
  getWebSocket().connect();
  await flushPromises();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  getWebSocket().close();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

function trigger(dock: VueWrapper) {
  return dock.get('#modelTrigger');
}

describe("the footer dock's model and level", () => {
  it('shows the default with nothing to change until the first message makes a thread', async () => {
    const dock = await mountDock(null);
    expect(shownLabel(dock)).toBe('claude-opus-4 · Auto');
    expect(trigger(dock).attributes('disabled')).toBeDefined();

    await send(dock, 'Plan my week');
    expect(sends).toEqual([{ path: '/api/threads/-1', provider: null, level: null }]);
    expect(settingsReads).toEqual(['-1', '-1']);
    expect(trigger(dock).attributes('disabled')).toBeUndefined();
    expect(shownLabel(dock)).toBe('claude-opus-4 · Auto');
  });

  it("saves a pick to the channel's latest thread at once and sends none with the next message", async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(null);
    expect(shownLabel(dock)).toBe('claude-opus-4 · High');

    await pick(dock, SONNET.id, 'medium');
    expect(settingsSaves).toEqual([
      { path: '/api/threads/settings/-1?type=user', body: { provider_id: SONNET.id, thinking_level: 'medium' } },
    ]);
    expect(shownLabel(dock)).toBe('claude-sonnet-4 · Medium');

    await send(dock, 'Plan my week');
    expect(sends).toEqual([{ path: '/api/threads/-1', provider: null, level: null }]);
    expect(shownLabel(dock)).toBe('claude-sonnet-4 · Medium');
  });
});

describe("a thread panel's model and level", () => {
  it("shows the thread's own settings and saves a change to the thread at once", async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(12);
    expect(settingsReads).toEqual(['12']);
    expect(shownLabel(dock)).toBe('claude-opus-4 · High');

    await pick(dock, SONNET.id, 'medium');
    expect(settingsSaves).toEqual([
      { path: '/api/threads/settings/12?type=user', body: { provider_id: SONNET.id, thinking_level: 'medium' } },
    ]);
    expect(shownLabel(dock)).toBe('claude-sonnet-4 · Medium');
    expect(useSessionStore().errorMessage).toBeNull();
  });

  it('sends a reply with neither, so the reply runs on the saved ones', async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(12);
    await send(dock, 'And Friday?');
    expect(sends).toEqual([{ path: '/api/threads/12', provider: null, level: null }]);
  });

  it('keeps the old settings and says so when the save fails', async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    saveFails = true;
    const dock = await mountDock(12);

    await pick(dock, SONNET.id, 'auto');
    expect(settingsSaves).toHaveLength(1);
    expect(shownLabel(dock)).toBe('claude-opus-4 · High');
    expect(useSessionStore().errorMessage).toBe('Could not change the model or thinking level');
  });
});

describe('the model menu', () => {
  it("lists the models with the current one marked, and a model's levels with a way back", async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(null);
    await openMenu(dock);

    const rows = dock.findAll('[data-model]');
    expect(rows.map((r) => r.text())).toEqual(['claude-opus-4', 'claude-sonnet-4']);
    expect(rows.map((r) => r.classes('active'))).toEqual([true, false]);
    expect(document.activeElement).toBe(rows[0].element);

    // Another model's levels: none is the current pick.
    await rows[1].trigger('click');
    await flushPromises();
    expect(dock.find('[data-model]').exists()).toBe(false);
    expect(dock.get('.model-select__item--back').text()).toBe('claude-sonnet-4');
    expect(dock.findAll('[data-level]').map((r) => r.attributes('aria-checked'))).toEqual(['false', 'false', 'false']);

    // Back to the models, then into the current one: its level is checked.
    await dock.get('.model-select__item--back').trigger('click');
    await flushPromises();
    await dock.get(`[data-model="${OPUS.id}"]`).trigger('click');
    await flushPromises();
    expect(dock.findAll('[data-level]').map((r) => r.attributes('aria-checked'))).toEqual(['false', 'false', 'true']);
    expect(document.activeElement).toBe(dock.get('[data-level="high"]').element);
  });

  it('shows a model by the model it runs, adding the provider only where another provider runs it too', async () => {
    const backup = { id: 5, name: 'Backup', model: OPUS.model, context_window: 200_000 };
    modelList = [OPUS, SONNET, backup];
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(null);
    expect(shownLabel(dock)).toBe('claude-opus-4 (Anthropic) · High');

    await openMenu(dock);
    expect(dock.findAll('[data-model]').map((r) => r.text())).toEqual([
      'claude-opus-4 (Anthropic)',
      'claude-sonnet-4',
      'claude-opus-4 (Backup)',
    ]);

    await dock.get(`[data-model="${backup.id}"]`).trigger('click');
    await flushPromises();
    expect(dock.get('.model-select__item--back').text()).toBe('claude-opus-4 (Backup)');
    expect(dock.get('#modelMenu').attributes('aria-label')).toBe('claude-opus-4 (Backup) thinking levels');

    await dock.get('[data-level="medium"]').trigger('click');
    await flushPromises();
    expect(shownLabel(dock)).toBe('claude-opus-4 (Backup) · Medium');
  });

  it('backs out of the levels on the first Escape and closes on the second, keeping it from the page', async () => {
    const pageKeys: string[] = [];
    const onPageKey = (e: KeyboardEvent): void => {
      pageKeys.push(e.key);
    };
    document.addEventListener('keydown', onPageKey);
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(null);
    await openMenu(dock);
    await dock.get(`[data-model="${SONNET.id}"]`).trigger('click');
    await flushPromises();

    await press('Escape');
    expect(dock.find('[data-level]').exists()).toBe(false);
    expect(dock.findAll('[data-model]')).toHaveLength(2);

    await press('Escape');
    expect(dock.find('#modelMenu').exists()).toBe(false);
    expect(document.activeElement).toBe(dock.get('#modelTrigger').element);
    expect(pageKeys).toEqual([]);
    document.removeEventListener('keydown', onPageKey);
  });

  it('moves between rows with the arrow keys, wrapping at both ends', async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(null);
    await openMenu(dock);
    const [opus, sonnet] = dock.findAll('[data-model]').map((r) => r.element);

    await press('ArrowUp');
    expect(document.activeElement).toBe(sonnet);
    await press('ArrowDown');
    expect(document.activeElement).toBe(opus);
    await press('ArrowDown');
    expect(document.activeElement).toBe(sonnet);
  });

  it('closes on a click outside it', async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(null);
    await openMenu(dock);
    expect(dock.find('#modelMenu').exists()).toBe(true);

    document.body.click();
    await flushPromises();
    expect(dock.find('#modelMenu').exists()).toBe(false);
  });

  it('lists every model, however many pages the listing takes', async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    modelList = Array.from({ length: 130 }, (_, i) => ({ id: i + 1, name: `Model ${i + 1}`, model: `model-${i + 1}`, context_window: null }));
    const dock = await mountDock(null);
    await openMenu(dock);
    expect(dock.findAll('[data-model]')).toHaveLength(130);
  });

  it('stops at the first empty page when models were removed while it read', async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    modelList = Array.from({ length: 130 }, (_, i) => ({ id: i + 1, name: `Model ${i + 1}`, model: `model-${i + 1}`, context_window: null }));
    claimedTotal = 160;
    const dock = await mountDock(null);
    await openMenu(dock);
    expect(dock.findAll('[data-model]')).toHaveLength(130);
    expect(listingPages).toEqual([1, 2, 3]);
    expect(dock.text()).not.toContain('Could not load the models');
  });

  it('offers to add a model when none is set up, opening the Brain', async () => {
    modelList = [];
    const openBrain = vi.spyOn(webPlatformAdapter, 'openBrain').mockImplementation(() => undefined);
    const dock = await mountDock(null);
    expect(dock.get('#modelTrigger').text()).toBe('Add a model');

    await dock.get('#modelTrigger').trigger('click');
    expect(openBrain).toHaveBeenCalledTimes(1);
    expect(dock.find('#modelMenu').exists()).toBe(false);
  });

  it('shows the level alone and says so when the models cannot be read', async () => {
    modelList = null;
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'high' });
    const dock = await mountDock(null);
    expect(shownLabel(dock)).toBe('High');
    expect(useSessionStore().errorMessage).toBe('Could not load the models');
  });
});

describe('the context ring', () => {
  function ring(dock: VueWrapper) {
    return dock.find('#contextDisplay');
  }

  it("fills with the thread's own reading and ignores another thread's", async () => {
    const dock = await mountDock(12);
    expect(ring(dock).exists()).toBe(false);

    await pushUsage(13, 150_000, 200_000);
    expect(ring(dock).exists()).toBe(false);

    await pushUsage(12, 50_000, 200_000);
    expect(ring(dock).attributes('aria-valuenow')).toBe('25');
    expect(ring(dock).get('.context-ring__fill').attributes('stroke-dashoffset')).toBe('75');
    expect(ring(dock).get('.context-ring__label').text()).toBe('25%');
  });

  it("follows the channel's latest reading in the footer dock", async () => {
    const dock = await mountDock(null);
    await pushUsage(12, 50_000, 200_000);
    expect(ring(dock).attributes('aria-valuenow')).toBe('25');

    await pushUsage(13, 150_000, 200_000);
    expect(ring(dock).attributes('aria-valuenow')).toBe('75');
  });

  it("measures the reading against the picked model's window, as the server's check does", async () => {
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'auto' });
    const dock = await mountDock(12);
    await pushUsage(12, 50_000, 200_000);
    expect(ring(dock).attributes('aria-valuenow')).toBe('25');

    // Sonnet's window is half of Opus's: the same reading fills twice as much
    // of it, shown before the next message is sent.
    await pick(dock, SONNET.id, 'auto');
    expect(ring(dock).attributes('aria-valuenow')).toBe('50');
  });

  it("measures against the reading's own window while the picked model's is unknown", async () => {
    modelList = [OPUS, { ...SONNET, context_window: null }];
    threads.set(12, { provider_id: OPUS.id, thinking_level: 'auto' });
    const dock = await mountDock(12);
    await pushUsage(12, 50_000, 200_000);

    await pick(dock, SONNET.id, 'auto');
    expect(ring(dock).attributes('aria-valuenow')).toBe('25');
  });

  it('shows its percentage on a tap and hides it on the next', async () => {
    const dock = await mountDock(null);
    await pushUsage(12, 50_000, 200_000);
    expect(ring(dock).classes('context-ring--open')).toBe(false);

    const toggle = dock.get('.context-ring__toggle');
    await toggle.trigger('click');
    expect(ring(dock).classes('context-ring--open')).toBe(true);
    expect(toggle.attributes('aria-pressed')).toBe('true');
    await toggle.trigger('click');
    expect(ring(dock).classes('context-ring--open')).toBe(false);
    expect(toggle.attributes('aria-pressed')).toBe('false');
  });
});

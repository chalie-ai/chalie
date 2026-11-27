// @vitest-environment happy-dom
/**
 * Quote blocks in the composer — feature spec.
 *
 * A quote made from one of Chalie's replies lands in that surface's composer as
 * a row above the message field: it can be opened to read in full and removed,
 * never edited, and it waits for the user to send. The send then carries every
 * block at the head of the typed text, in the exact form the chat reads back.
 *
 * Real throughout: the InputDock, its quote rows and Reply dialog, the Pinia
 * stores, the session send path and the shared WebSocketService. The quote
 * arrives as the selection bar hands it over — the selection itself needs a
 * real browser and is covered by quoteSelection.browser.spec.ts. The only fakes
 * are the network boundary: `fetch` (the backend) and the global `WebSocket`
 * class (an open socket, so the service reports itself connected).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { getWebSocket } from '@chalie/shared';
import InputDock from '../layout/InputDock.vue';
import { QUOTE_BLOCK_EVENT, type QuoteBlock } from '../../utils/quoteBlocks';

/** Every chat send's text, in order. */
const sent: string[] = [];
/** Browser storage, where the composer keeps its draft: the test runtime's own
 *  has no backing file, so an in-memory one stands in. */
const saved = new Map<string, string>();

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** The backend, at the network boundary: the dock's thread-settings read and
 *  the chat send, nothing else. */
async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname.startsWith('/api/threads/settings/')) {
    return json({
      success: true,
      result: { provider_id: 1, thinking_level: 'auto', turn_id: null },
    });
  }
  if (init?.method === 'POST' && /^\/api\/threads\/-?\d+$/.test(url.pathname)) {
    sent.push(String((init.body as FormData).get('text')));
    return json({ success: true, result: { turn_id: 7, type: 'user' } });
  }
  return json({ error: 'unrouted' }, 404);
}

/** A socket that opens as soon as it is created, like a healthy server. */
class OpenSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  readyState = OpenSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor() {
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

/** A quote handed to the composer of `dockId`, as the selection bar hands it. */
async function quote(dockId: string, block: QuoteBlock): Promise<void> {
  document.dispatchEvent(new CustomEvent(QUOTE_BLOCK_EVENT, { detail: { dockId, block } }));
  await flushPromises();
}

const rows = (): string[] =>
  [...document.querySelectorAll('.quote-rows__row')].map((row) => row.textContent?.trim() ?? '');

let wrapper: VueWrapper | null = null;

function mountDock(): VueWrapper {
  wrapper = mount(InputDock, { props: { dockId: 'main_spine_dock' }, attachTo: document.body });
  return wrapper;
}

beforeEach(async () => {
  setActivePinia(createPinia());
  saved.clear();
  sent.length = 0;
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => saved.get(key) ?? null,
    setItem: (key: string, value: string) => void saved.set(key, value),
    removeItem: (key: string) => void saved.delete(key),
  });
  vi.stubGlobal('fetch', fakeBackend);
  vi.stubGlobal('WebSocket', OpenSocket);
  getWebSocket().connect();
  await flushPromises();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  getWebSocket().close();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('quote blocks in the composer', () => {
  it('gathers quotes as rows, opens and removes them, and sends them at the head of the typed text', async () => {
    const dock = mountDock();
    await flushPromises();

    await quote('main_spine_dock', { quote: 'Paris is the capital', reply: 'and Lyon?' });
    await quote('main_spine_dock', { quote: 'two\nlines', reply: '' });
    // A quote made on another surface goes to that surface's composer, not this one.
    await quote('thread_view_dock', { quote: 'elsewhere', reply: 'no' });

    // One row per quote: the quote in curly quotes, a tag only where there is a reply.
    expect(rows()).toEqual(['“Paris is the capital”With reply', '“two\nlines”']);
    // Nothing went out: the quotes wait for the user to send.
    expect(sent).toEqual([]);

    // A row opens to read the block in full — the quote, then the reply.
    await dock.findAll('.quote-rows__open')[0].trigger('click');
    const dialog = document.querySelector<HTMLElement>('.quote-dialog__card')!;
    expect(dialog.getAttribute('role')).toBe('dialog');
    expect(dialog.querySelector('.quote-dialog__title')?.textContent).toBe('Reply');
    expect(dialog.querySelector('.quote-dialog__quote')?.textContent).toBe('> Paris is the capital');
    expect(dialog.querySelector('.quote-dialog__reply')?.textContent).toBe('and Lyon?');
    // Read-only: nothing in it can be typed into.
    expect(dialog.querySelectorAll('input, textarea, [contenteditable]')).toHaveLength(0);

    // Escape closes it and stops there, so a thread panel behind it stays open.
    const heard = vi.fn();
    document.addEventListener('keydown', heard);
    dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await flushPromises();
    document.removeEventListener('keydown', heard);
    expect(document.querySelector('.quote-dialog')).toBeNull();
    expect(heard).not.toHaveBeenCalled();

    // A multi-line quote reads line by line.
    await dock.findAll('.quote-rows__open')[1].trigger('click');
    expect(document.querySelector('.quote-dialog__quote')?.textContent).toBe('> two\n> lines');
    expect(document.querySelector('.quote-dialog__reply')).toBeNull();
    (document.querySelector('.quote-dialog__close') as HTMLButtonElement).click();
    await flushPromises();
    expect(document.querySelector('.quote-dialog')).toBeNull();

    // Removing a row drops that block only.
    await dock.findAll('.quote-rows__remove')[1].trigger('click');
    expect(rows()).toEqual(['“Paris is the capital”With reply']);

    // The user types alongside the block and sends: one ordinary message.
    await dock.get('textarea').setValue('  Also, what is the weather?  ');
    await dock.get('button[aria-label="Send message"]').trigger('click');
    await flushPromises();
    expect(sent).toEqual([
      '[quote_block]{"quote": "Paris is the capital", "reply": "and Lyon?"}[/quote_block]\n' +
        '\n' +
        'Also, what is the weather?',
    ]);
    // The composer is empty again.
    expect(rows()).toEqual([]);
    expect((dock.get('textarea').element as HTMLTextAreaElement).value).toBe('');
  });

  it('sends quotes alone, with nothing typed', async () => {
    const dock = mountDock();
    await flushPromises();
    const send = dock.get('button[aria-label="Send message"]');
    expect(send.attributes('disabled')).toBeDefined();

    await quote('main_spine_dock', { quote: 'first', reply: 'one' });
    await quote('main_spine_dock', { quote: 'second', reply: 'two' });
    expect(send.attributes('disabled')).toBeUndefined();

    await send.trigger('click');
    await flushPromises();
    expect(sent).toEqual([
      '[quote_block]{"quote": "first", "reply": "one"}[/quote_block]\n' +
        '[quote_block]{"quote": "second", "reply": "two"}[/quote_block]',
    ]);
  });

  it('keeps the quotes in the saved draft, so a reload brings the rows back', async () => {
    const dock = mountDock();
    await flushPromises();
    await quote('main_spine_dock', { quote: 'keep me', reply: 'please' });
    await dock.get('textarea').setValue('half-typed');
    await flushPromises();
    expect(saved.get('chalie:draft')).toBe(
      '[quote_block]{"quote": "keep me", "reply": "please"}[/quote_block]\n\nhalf-typed',
    );

    dock.unmount();
    const again = mountDock();
    await flushPromises();
    expect(rows()).toEqual(['“keep me”With reply']);
    expect((again.get('textarea').element as HTMLTextAreaElement).value).toBe('half-typed');
  });
});

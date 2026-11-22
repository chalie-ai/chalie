// @vitest-environment happy-dom
/**
 * Attaching any file to a chat message — feature spec.
 *
 * Dropping a file on the chat never cared about its type, and nothing after it
 * does either: the attachments store, the multipart send and the server all
 * take any file. The attach button's picker must offer the same — a file the
 * app has no special handling for (a router backup, say) is still a file the
 * user may send.
 *
 * Real throughout: the footer InputDock, its attachment strip, the Pinia
 * stores, the session send path and the shared WebSocketService. The only
 * fakes are the network boundary: `fetch` (the backend) and the global
 * `WebSocket` class (an open socket, so the service reports itself connected).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { getWebSocket } from '@chalie/shared';
import InputDock from './InputDock.vue';

/** One multipart POST to `/api/threads/<id>`: where it went, which files it carried. */
interface Post {
  path: string;
  files: File[];
}

const posts: Post[] = [];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** The backend, at the network boundary: the dock's thinking-level read and
 *  the chat send, nothing else. */
async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname.startsWith('/api/threads/thinking-level/')) {
    return json({ success: true, result: { level: 'auto' } });
  }
  if (init?.method === 'POST' && /^\/api\/threads\/-?\d+$/.test(url.pathname)) {
    const form = init.body as FormData;
    posts.push({ path: url.pathname, files: form.getAll('files') as File[] });
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

/** The bytes of a UniFi controller backup: opaque, no MIME type the browser knows. */
function unifiBackup(name: string): File {
  return new File([new Uint8Array([0x55, 0x4e, 0x46, 0x00, 0xff, 0x10])], name, { type: '' });
}

let wrapper: VueWrapper | null = null;

beforeEach(async () => {
  setActivePinia(createPinia());
  posts.length = 0;
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

describe('attaching a file through the attach button', () => {
  it('offers every file type, and a picked file the app has never heard of is sent like a dropped one', async () => {
    wrapper = mount(InputDock, { props: { dockId: 'main_spine_dock' }, attachTo: document.body });
    await flushPromises();

    // The attach button opens the file input, and that input narrows the OS
    // picker to nothing: no `accept`, so every file is selectable.
    const picker = document.querySelector<HTMLInputElement>('#attachFileInput')!;
    let opened = 0;
    picker.addEventListener('click', () => { opened += 1; });
    await wrapper.get('button[aria-label="Attach"]').trigger('click');
    expect(opened).toBe(1);
    expect(picker.hasAttribute('accept')).toBe(false);

    // The user picks the backup in the OS picker …
    const picked = new DataTransfer();
    picked.items.add(unifiBackup('UniFi.unf'));
    picker.files = picked.files;
    picker.dispatchEvent(new Event('change'));
    // … and drags a second one onto the chat.
    const dropped = new DataTransfer();
    dropped.items.add(unifiBackup('site-two.unf'));
    // happy-dom's DragEvent drops `dataTransfer` from its init, so set it on
    // the event the way a browser delivers it.
    const drop = new DragEvent('drop', { cancelable: true });
    Object.defineProperty(drop, 'dataTransfer', { value: dropped });
    document.dispatchEvent(drop);
    await flushPromises();

    // Both land in the strip the same way: a named file chip each.
    const chips = [...document.querySelectorAll('#imagePreview .image-preview__doc-name')].map((el) => el.textContent);
    expect(chips).toEqual(['UniFi.unf', 'site-two.unf']);

    // Sending posts both to the server as files, bytes intact, and clears the strip.
    await wrapper.get('button[aria-label="Send message"]').trigger('click');
    await flushPromises();
    expect(posts).toHaveLength(1);
    expect(posts[0].path).toBe('/api/threads/-1');
    expect(posts[0].files.map((f) => f.name)).toEqual(['UniFi.unf', 'site-two.unf']);
    expect([...new Uint8Array(await posts[0].files[0].arrayBuffer())]).toEqual([0x55, 0x4e, 0x46, 0x00, 0xff, 0x10]);
    expect(document.querySelectorAll('#imagePreview .image-preview__doc-name')).toHaveLength(0);
  });
});

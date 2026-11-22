// @vitest-environment happy-dom
/**
 * The browser panel's certificate warning — feature spec on the real panel.
 *
 * When the server's browser refuses a page for its certificate, the tab list
 * says so, and the panel shows a warning in place of the tab's page, as any
 * browser does: the site, what is wrong in words, the browser's own code, and
 * Go back or Proceed. The address bar shows the refused address. Only the
 * person's click on Proceed sends the site to the server; going back sends
 * back for that tab.
 *
 * The real BrowserPanel on the real browser store and REST client. The network
 * is the one stand-in: `fetch` serves the tab list and records each request
 * the panel sends, and the page view's stream socket only records the tab it
 * was opened for.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import BrowserPanel from './BrowserPanel.vue';
import { useBrowserStore } from '../../stores/browser';
import type { BrowserCertificateError, BrowserTab } from '../../api/browser';

const backend = {
  tabs: [] as BrowserTab[],
  sent: [] as { path: string; body: unknown }[],
  /** The server answers Proceed with this status; 204 clears the warning as the server does. */
  proceedStatus: 204,
  /** The tab each stream socket was opened for. */
  streams: [] as string[],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const path = new URL(String(input), 'http://chalie.test').pathname;
  if ((init?.method ?? 'GET') === 'GET' && path === '/api/browser/tabs') {
    return json({ success: true, result: { tabs: backend.tabs } });
  }
  if (init?.method === 'POST') {
    backend.sent.push({ path, body: init.body ? JSON.parse(String(init.body)) : null });
    if (path.startsWith('/api/browser/proceed/')) {
      if (backend.proceedStatus !== 204) return json({ error: 'No warning for that site' }, backend.proceedStatus);
      backend.tabs = backend.tabs.map((t) => ({ ...t, certificate_error: null }));
    }
    return new Response(null, { status: 204 });
  }
  return json({ error: 'unrouted' }, 404);
}

/** The page view's stream socket: it records the tab it streams and never connects. */
class StreamSocket {
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  binaryType = 'blob';
  constructor(url: string) {
    backend.streams.push(decodeURIComponent(url.split('/ws/browser/')[1] ?? ''));
  }
  send(): void {}
  close(): void {}
}

const SELF_SIGNED: BrowserCertificateError = {
  host: '127.0.0.1',
  port: 8443,
  error: 'ERR_CERT_AUTHORITY_INVALID',
  url: 'https://127.0.0.1:8443/admin',
};

function refusedTab(id: string, refused: BrowserCertificateError): BrowserTab {
  return {
    id,
    url: 'about:blank',
    title: 'about:blank',
    opener: 'user',
    origin: null,
    handoff: null,
    certificate_error: refused,
  };
}

let panel: VueWrapper | null = null;

/** The panel open on the tab the server lists, after the read that lists it. */
async function openOn(tab: BrowserTab): Promise<VueWrapper> {
  backend.tabs = [tab];
  const browser = useBrowserStore();
  panel = mount(BrowserPanel, { attachTo: document.body });
  browser.openPanel(tab.id);
  await flushPromises();
  return panel;
}

function button(w: VueWrapper, label: string) {
  const found = w.findAll('button').find((b) => b.text() === label || b.attributes('aria-label') === label);
  if (!found) throw new Error(`no button ${label}`);
  return found;
}

beforeEach(() => {
  backend.tabs = [];
  backend.sent = [];
  backend.proceedStatus = 204;
  backend.streams = [];
  setActivePinia(createPinia());
  vi.stubGlobal('fetch', fakeBackend);
  vi.stubGlobal('WebSocket', StreamSocket);
});

afterEach(() => {
  panel?.unmount();
  panel = null;
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('browser panel — certificate warning', () => {
  it('shows the warning in place of the page: the site, the reason in words, the code, the refused address, and Back', async () => {
    const w = await openOn(refusedTab('t1', SELF_SIGNED));

    const warning = w.get('.cert-warning');
    expect(warning.get('h2').text()).toBe("This site's certificate isn't trusted");
    expect(warning.text()).toContain('127.0.0.1:8443 has a certificate that no trusted authority issued; it may be self-signed');
    expect(warning.get('code').text()).toBe('ERR_CERT_AUTHORITY_INVALID');
    expect(button(w, 'Proceed to 127.0.0.1:8443').exists()).toBe(true);
    expect((w.get('input[aria-label="Address"]').element as HTMLInputElement).value).toBe(SELF_SIGNED.url);
    expect(button(w, 'Back').attributes('disabled')).toBeUndefined();
    expect(w.get('.browser-panel__tab-text').text()).toBe('Certificate warning');
    expect(backend.streams).toEqual([]);
  });

  it("Proceed sends that site for that tab, and the tab's page view comes back", async () => {
    const w = await openOn(refusedTab('t1', { ...SELF_SIGNED, host: 'router.example', port: 443, url: 'https://router.example/' }));

    await button(w, 'Proceed to router.example').trigger('click');
    await flushPromises();

    expect(backend.sent).toEqual([{ path: '/api/browser/proceed/t1', body: { host: 'router.example', port: 443 } }]);
    expect(w.find('.cert-warning').exists()).toBe(false);
    expect(backend.streams).toEqual(['t1']);
  });

  it('Go back sends back for that tab and proceeds nowhere', async () => {
    const w = await openOn(refusedTab('t1', SELF_SIGNED));

    await button(w, 'Go back').trigger('click');
    await flushPromises();

    expect(backend.sent).toEqual([{ path: '/api/browser/history/t1', body: { action: 'back' } }]);
  });

  it('a Proceed the server turns down leaves the warning up and says why', async () => {
    backend.proceedStatus = 409;
    const w = await openOn(refusedTab('t1', SELF_SIGNED));

    await button(w, 'Proceed to 127.0.0.1:8443').trigger('click');
    await flushPromises();

    expect(w.find('.cert-warning').exists()).toBe(true);
    expect(document.querySelector('.chalie-toast')?.textContent).toContain('No warning for that site');
  });
});

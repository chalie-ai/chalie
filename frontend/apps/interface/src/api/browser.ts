import { api, getHost } from '@chalie/shared';
import type { SingleEnvelope } from './conversation';
import type { PermissionOrigin } from './policies';

/** A pending hand-off: Chalie gave this tab to the user (a login, a captcha). */
export interface BrowserHandoff {
  /** Opaque ID; finished on POST /api/browser/handoffs/<id>. */
  id: string;
  /** Chalie's one-line reason, shown in the hand-off bar. */
  reason: string;
  /** The turn whose run handed the tab over. */
  origin: PermissionOrigin;
  /** The instant the hand-off was asked for (ISO-8601). */
  asked_at: string;
}

/** One open tab in the server's browser. */
export interface BrowserTab {
  id: string;
  /** 'about:blank' while the tab has nothing loaded. */
  url: string;
  title: string;
  /** Who opened it: Chalie's tabs close with the run that opened them, so only the user's can be closed here. */
  opener: 'chalie' | 'user';
  handoff: BrowserHandoff | null;
}

export type BrowserHistoryAction = 'back' | 'forward' | 'reload';

/*
 * Open, navigate and history return once the browser has the request, not once
 * the page loads; `browser_tabs_changed` reports where the tab lands.
 */
export const browser = {
  /** GET /api/browser/tabs — the open tabs; never starts the browser. */
  async tabs(): Promise<BrowserTab[]> {
    const body = await api.get<SingleEnvelope<{ tabs: BrowserTab[] }>>('/api/browser/tabs');
    return body.result.tabs;
  },

  /** POST /api/browser/tabs — open a user tab, blank or on `url`. */
  async open(url?: string): Promise<BrowserTab> {
    const body = await api.post<SingleEnvelope<{ tab: BrowserTab }>>(
      '/api/browser/tabs',
      url ? { url } : undefined,
    );
    return body.result.tab;
  },

  /** DELETE /api/browser/tabs/<id> — 409 for a browsing run's tab, 404 when unknown. */
  async close(tabId: string): Promise<void> {
    await api.del(`/api/browser/tabs/${encodeURIComponent(tabId)}`);
  },

  /** POST /api/browser/navigate/<id> — point the tab at `url`, taken as given. */
  async navigate(tabId: string, url: string): Promise<void> {
    await api.post(`/api/browser/navigate/${encodeURIComponent(tabId)}`, { url });
  },

  /** POST /api/browser/history/<id> — back, forward or reload. */
  async history(tabId: string, action: BrowserHistoryAction): Promise<void> {
    await api.post(`/api/browser/history/${encodeURIComponent(tabId)}`, { action });
  },

  /** POST /api/browser/handoffs/<id> — the user is done; Chalie carries on. 404 once it is already done. */
  async finishHandoff(handoffId: string): Promise<void> {
    await api.post(`/api/browser/handoffs/${encodeURIComponent(handoffId)}`);
  },

  /** The tab's live stream socket. The session cookie rides the handshake. */
  streamUrl(tabId: string): string {
    const base = getHost().replace(/\/$/, '') || globalThis.location.origin;
    return `${base.replace(/^http/, 'ws')}/ws/browser/${encodeURIComponent(tabId)}`;
  },
};

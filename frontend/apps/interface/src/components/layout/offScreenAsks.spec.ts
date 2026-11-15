// @vitest-environment happy-dom
/**
 * The input dock's off-screen asks — feature spec on the real dock.
 *
 * A subagent the chat started can ask for permission after the chat turn that
 * started it has ended: no live line shows that turn any more, so the ask
 * waits in the input dock. The spec loads the pending asks the way the app does
 * on every connect, from the backend's listing, mounts the real OffScreenAsks
 * on the real permissions and session stores, and asserts what a person reads
 * and can press:
 *
 *   - an ask no live line shows gets one row naming its turn and what it asks,
 *     with Deny, Allow and Open;
 *   - Allow and Deny each send their own answer for that ask alone, and its
 *     row goes;
 *   - an answer that fails to send brings its row back, since the ask still
 *     waits;
 *   - Open shows the ask's turn in the thread panel and answers nothing.
 *
 * The network is the one stand-in: the spec serves the pending listing and
 * reads each answer the dock sent.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import OffScreenAsks from './OffScreenAsks.vue';
import { usePermissionsStore } from '../../stores/permissions';
import { useSessionStore } from '../../stores/session';
import type { PendingPermission } from '../../api/policies';

/** The chat turn that started the subagent; it ended before the ask came. */
const CHAT_TURN = 7;

const backend = {
  pending: [] as PendingPermission[],
  answers: [] as unknown[],
  /** Every answer is received but fails, so its ask still waits. */
  answersFail: false,
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname === '/api/policies/pending' && (init?.method ?? 'GET') === 'GET') {
    return json({ success: true, result: backend.pending, pagination: { page: 1, limit: 50, total: backend.pending.length } });
  }
  if (url.pathname === '/api/policies/respond' && init?.method === 'POST') {
    backend.answers.push(JSON.parse(String(init.body)));
    return backend.answersFail ? json({ error: 'Could not answer the ask' }, 500) : new Response(null, { status: 204 });
  }
  return json({ error: 'unrouted' }, 404);
}

/** A subagent's ask on the chat turn that started it, as the backend lists it. */
function ask(request_id: string, action_id: string, summary: string): PendingPermission {
  return {
    request_id, action_id, summary, asked_at: new Date().toISOString(),
    origin: { type: 'user', turn_id: CHAT_TURN, forked: false },
  };
}

let wrapper: VueWrapper | null = null;

/** The dock with the backend's pending asks loaded, as on connect. */
async function mountDock(...asks: PendingPermission[]): Promise<VueWrapper> {
  backend.pending = asks;
  await usePermissionsStore().refreshPending();
  wrapper = mount(OffScreenAsks);
  await flushPromises();
  return wrapper;
}

const rows = (w: VueWrapper) => w.findAll('.offscreen-asks > .activity');
const labels = (w: VueWrapper): string[] => rows(w).map((r) => r.find('.activity__label').text());
const rowNamed = (w: VueWrapper, label: string) => {
  const found = rows(w).find((r) => r.find('.activity__label').text() === label);
  if (!found) throw new Error(`no dock row for "${label}"`);
  return found;
};

beforeEach(() => {
  setActivePinia(createPinia());
  backend.pending = [];
  backend.answers = [];
  backend.answersFail = false;
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('an ask no live line shows', () => {
  it('gets one row naming its turn and what it asks, with Deny, Allow and Open', async () => {
    const w = await mountDock(ask('r1', 'pim', 'Read the ferry booking email'));

    expect(rows(w)).toHaveLength(1);
    const row = rows(w)[0];
    expect(row.find('.activity__origin').text()).toBe(`Thread · #${CHAT_TURN} ·`);
    expect(row.find('.activity__label').text()).toBe('Read the ferry booking email');
    expect(row.find('.activity__ask').text()).toBe('Access Email, Calendar & Contacts');
    expect(row.text()).toContain('waiting for you');
    expect(row.findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow', 'Open']);
  });

  it('Allow sends approval for that ask alone, and its row goes', async () => {
    const w = await mountDock(
      ask('r1', 'pim', 'Read the ferry booking email'),
      ask('r2', 'code_agent', 'Write the fare table script'),
    );

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ request_id: 'r1', approved: true }]);
    expect(labels(w)).toEqual(['Write the fare table script']);
  });

  it('Deny sends a refusal for that ask alone, and its row goes', async () => {
    const w = await mountDock(
      ask('r1', 'pim', 'Read the ferry booking email'),
      ask('r2', 'code_agent', 'Write the fare table script'),
    );

    await rowNamed(w, 'Write the fare table script').find('.activity__answer--deny').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ request_id: 'r2', approved: false }]);
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it('an answer that fails to send brings its row back, since the ask still waits', async () => {
    const w = await mountDock(
      ask('r1', 'pim', 'Read the ferry booking email'),
      ask('r2', 'code_agent', 'Write the fare table script'),
    );
    backend.answersFail = true;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => { /* the failed answer's console line */ });

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ request_id: 'r1', approved: true }]);
    expect(warn).toHaveBeenCalledWith('[permissions] respond failed:', expect.anything());
    expect(labels(w)).toEqual(['Write the fare table script', 'Read the ferry booking email']);
    expect(rowNamed(w, 'Read the ferry booking email').findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow', 'Open']);
  });

  it('Open shows its turn in the thread panel and answers nothing', async () => {
    const w = await mountDock(ask('r1', 'pim', 'Read the ferry booking email'));
    const session = useSessionStore();

    await rowNamed(w, 'Read the ferry booking email').find('.activity__stop').trigger('click');
    await flushPromises();

    expect([session.panelThreadId, session.panelType]).toEqual([CHAT_TURN, 'user']);
    expect(backend.answers).toEqual([]);
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });
});

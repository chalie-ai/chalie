// @vitest-environment happy-dom
/**
 * A sign-in that waits on a code — feature spec on the real rows that answer it.
 *
 * When a UniFi tool call needs the user's 2FA code, the backend opens an ask
 * of kind `code`. It is answered wherever a permission ask is: on the live
 * line of the turn that asked, or in the input dock when no live line shows
 * it. The spec loads the open asks the way the app does on every connect,
 * from the backend's listing, mounts the real OffScreenAsks or TurnView on the
 * real stores, and asserts what a person reads, types and presses:
 *
 *   - the row says what it waits for and labels its one field; the field keeps
 *     ASCII digits only and asks a phone for its number pad and the code from
 *     its messages; Sign in stays off until six digits are in;
 *   - Enter sends a complete code, and nothing shorter;
 *   - the code leaves once per press, in the answer's body and nowhere else —
 *     no console line, toast, page text, request address or store holds it —
 *     whether the server takes it, refuses it or fails;
 *   - on the asking turn's live line the field and Sign in sit beside Stop,
 *     and the dock leaves that ask alone;
 *   - a second sign-in waiting on the same turn starts with an empty field, so
 *     the first code is never sent for it;
 *   - in the dock, a sign-in from a chat turn offers Open; one from a run no
 *     chat turn shows has no origin and no Open.
 *
 * The network is the one stand-in: the spec serves the open-asks listing and
 * reads each answer sent, and the server takes, refuses (400) or fails (500)
 * an answer as the test sets it.
 */
import { inspect } from 'node:util';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { DOMWrapper, VueWrapper } from '@vue/test-utils';
import { createPinia, getActivePinia, setActivePinia } from 'pinia';
import OffScreenAsks from './OffScreenAsks.vue';
import TurnView from '../conversation/TurnView.vue';
import { ConfigType, useAsksStore } from '@chalie/shared';
import type { AskOrigin } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import { clearAll } from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock } from '../../api/conversation';

/** The code the user reads off their authenticator: six digits, a leading zero. */
const CODE = '048217';
const HEADING = 'UniFi sign-in needs your 2FA code';

/** The chat turn whose UniFi call asked for the code. */
const TURN = 7;
const ON_TURN: AskOrigin = { type: 'user', turn_id: TURN, forked: false };

const backend = {
  open: [] as object[],
  /** Every address the app requested, in order. */
  requested: [] as string[],
  answers: [] as { path: string; body: unknown }[],
  /** How the server takes an answer: 204 closes the ask, 400 refuses the code, 500 fails. */
  replyStatus: 204,
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  backend.requested.push(String(input));
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname === '/api/asks/open' && (init?.method ?? 'GET') === 'GET') {
    return json({ success: true, result: backend.open, pagination: { page: 1, limit: 50, total: backend.open.length } });
  }
  if (url.pathname.startsWith('/api/asks/reply/') && init?.method === 'POST') {
    backend.answers.push({ path: url.pathname, body: JSON.parse(String(init.body)) });
    if (backend.replyStatus === 400) return json({ error: 'That code was not accepted' }, 400);
    if (backend.replyStatus === 500) return json({ error: 'Could not answer the ask' }, 500);
    const answered = decodeURIComponent(url.pathname.slice('/api/asks/reply/'.length));
    backend.open = backend.open.filter((a) => (a as { ask_id: string }).ask_id !== answered);
    return new Response(null, { status: 204 });
  }
  return json({ error: 'unrouted' }, 404);
}

/** A UniFi sign-in waiting on its code, as the backend lists it. */
function codeAsk(ask_id: string, origin: AskOrigin | null = ON_TURN) {
  return { ask_id, kind: 'code', asked_at: new Date().toISOString(), data: { service: 'unifi', origin } };
}

let mounted: VueWrapper[] = [];

/** The open asks loaded from the backend's listing, as on connect. */
async function connect(...asks: object[]): Promise<void> {
  backend.open = asks;
  await useAsksStore().refreshOpen();
}

async function mountDock(...asks: object[]): Promise<VueWrapper> {
  await connect(...asks);
  const wrapper = mount(OffScreenAsks);
  mounted.push(wrapper);
  await flushPromises();
  return wrapper;
}

const USER_ROW: ConversationMessage = {
  id: '41', role: 'user', content: 'Why is the garage camera offline?',
  timestamp: '2026-11-20 09:00:00', day: '2026-11-20', created_at: '2026-11-20T09:00:00Z', turn_id: TURN,
};

/** The asking turn, still working, as the chat shows it. */
async function mountTurn(): Promise<VueWrapper> {
  const block: ConversationTurnBlock = {
    turn_id: TURN, gist: null, preview: USER_ROW.content, last_activity_at: null,
    working: true, duration_ms: 0, messages: [USER_ROW], type: 'user',
  };
  const wrapper = mount(TurnView, { props: { block, type: ConfigType.USER, fullThread: false } });
  mounted.push(wrapper);
  await flushPromises();
  return wrapper;
}

type Found = ReturnType<VueWrapper['find']>;

const rows = (w: VueWrapper) => w.findAll('.offscreen-asks > .activity');
/** A row's buttons by what they say (a live line's thinking fold is not one of its controls). */
const buttons = (row: Found): string[] => row.findAll('button:not(.fold)').map((b) => b.text());
const field = (row: Found) => row.find('input') as DOMWrapper<HTMLInputElement>;
function signIn(row: Found): DOMWrapper<HTMLButtonElement> {
  const found = row.findAll('button').find((b) => b.text() === 'Sign in');
  if (!found) throw new Error('no Sign in button on the row');
  return found as DOMWrapper<HTMLButtonElement>;
}
const press = (key: string, row: Found) => field(row).trigger('keydown', { key });

beforeEach(() => {
  setActivePinia(createPinia());
  backend.open = [];
  backend.requested = [];
  backend.answers = [];
  backend.replyStatus = 204;
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  clearAll(ConfigType.USER);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('a sign-in that waits on a code', () => {
  it('its dock row says UniFi sign-in needs your 2FA code and labels its one field; the field keeps ASCII digits only and Sign in waits for six', async () => {
    const w = await mountDock(codeAsk('c1'));
    expect(rows(w)).toHaveLength(1);
    const row = rows(w)[0]!;

    expect(row.find('.activity__label').text()).toBe(HEADING);
    expect(row.find('label').text()).toBe('6-digit code');
    expect(row.find('label input').exists()).toBe(true);
    const input = field(row);
    expect([input.attributes('inputmode'), input.attributes('autocomplete')]).toEqual(['numeric', 'one-time-code']);
    // A browser checks the field's pattern against the whole value, ASCII digits only.
    const pattern = new RegExp(`^(?:${input.attributes('pattern')})$`, 'v');
    expect([CODE, '12345', '١٢٣٤٥٦'].map((v) => pattern.test(v))).toEqual([true, false, false]);

    const typed: [string, string, boolean][] = [];
    for (const keys of ['', '12345', '12a45', '١٢٣٤٥٦', CODE]) {
      await input.setValue(keys);
      typed.push([keys, input.element.value, signIn(row).element.disabled]);
    }
    expect(typed).toEqual([
      ['', '', true],
      ['12345', '12345', true],
      ['12a45', '1245', true],
      ['١٢٣٤٥٦', '', true],
      [CODE, CODE, false],
    ]);
  });

  it('Enter sends a complete code, and nothing shorter', async () => {
    const w = await mountDock(codeAsk('c1'));
    const row = rows(w)[0]!;

    await field(row).setValue('04821');
    await press('Enter', row);
    await flushPromises();
    expect(backend.answers).toEqual([]);
    expect(rows(w)).toHaveLength(1);

    await field(row).setValue(CODE);
    await press('Enter', row);
    await flushPromises();
    expect(backend.answers).toEqual([{ path: '/api/asks/reply/c1', body: { value: { code: CODE } } }]);
    expect(rows(w)).toHaveLength(0);
  });

  it('the code leaves once per press, in the answer body only — never logged, toasted, shown, put in an address or kept — whether the server takes it, refuses it or fails', async () => {
    const logged: { method: string; calls: unknown[][] }[] = [];
    for (const method of Object.keys(console)) {
      const fn = (console as unknown as Record<string, unknown>)[method];
      if (/^[a-z]/.test(method) && typeof fn === 'function') {
        const spy = vi.spyOn(console as unknown as Record<string, (...args: unknown[]) => unknown>, method);
        logged.push({ method, calls: spy.mock.calls });
      }
    }
    const w = await mountDock(codeAsk('c1'));

    /** Every place outside the answer's body that holds the code. */
    const holdingTheCode = (): string[] => [
      ...logged
        .filter(({ calls }) => calls.some((args) => args.some((a) => `${String(a)} ${inspect(a, { depth: null })}`.includes(CODE))))
        .map(({ method }) => `console.${method}`),
      ...(document.body.innerHTML.includes(CODE) ? ['page (where a toast shows)'] : []),
      ...(w.html().includes(CODE) ? ['dock'] : []),
      ...(JSON.stringify(getActivePinia()!.state.value).includes(CODE) ? ['stores'] : []),
      ...(backend.requested.some((u) => u.includes(CODE)) || window.location.href.includes(CODE) ? ['address'] : []),
    ];

    const after: [number, number, string[]][] = [];
    for (const status of [500, 400, 204]) {
      backend.replyStatus = status;
      const row = rows(w)[0]!;
      await field(row).setValue(CODE);
      await signIn(row).trigger('click');
      await flushPromises();
      after.push([status, rows(w).length, holdingTheCode()]);
    }

    // Refused or failed, the ask still waits, so its row is back, its field empty.
    expect(after).toEqual([
      [500, 1, []],
      [400, 1, []],
      [204, 0, []],
    ]);
    expect(backend.answers.map((a) => a.body)).toEqual([{ value: { code: CODE } }, { value: { code: CODE } }, { value: { code: CODE } }]);
    // The refused and failed answers were reported, so the console was watched.
    expect(logged.some(({ calls }) => calls.length > 0)).toBe(true);
  });

  it("on the asking turn's live line, it is named with its field and Sign in beside Stop, the dock leaves it alone, and Sign in hands the line back", async () => {
    await connect(codeAsk('other-turn', { type: 'user', turn_id: TURN + 1, forked: false }), codeAsk('c1'));
    const turn = await mountTurn();
    const dock = mount(OffScreenAsks);
    mounted.push(dock);
    await flushPromises();
    const live = () => turn.find('.activity');

    expect(live().find('.step__label').text()).toBe(HEADING);
    expect(live().text()).toContain('waiting for you');
    expect(live().find('label').text()).toBe('6-digit code');
    expect(buttons(live())).toEqual(['Sign in', 'Stop']);
    // Only the other turn's sign-in waits in the dock.
    expect(rows(dock).map((r) => r.find('.activity__origin').text())).toEqual([`Chat · #${TURN + 1} ·`]);

    await field(live()).setValue(CODE);
    await signIn(live()).trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/c1', body: { value: { code: CODE } } }]);
    expect(live().find('.step__label').text()).not.toBe(HEADING);
    expect(buttons(live())).toEqual(['Stop']);
  });

  it('a second sign-in waiting on the same turn starts with an empty field, so the first code is never sent for it', async () => {
    await connect(codeAsk('first'), codeAsk('second'));
    const turn = await mountTurn();
    const live = () => turn.find('.activity');

    await field(live()).setValue(CODE);
    await press('Enter', live());
    await flushPromises();
    expect(backend.answers).toEqual([{ path: '/api/asks/reply/first', body: { value: { code: CODE } } }]);
    expect(useAsksStore().queue.map((a) => a.ask_id)).toEqual(['second']);

    expect(field(live()).element.value).toBe('');
    expect(signIn(live()).element.disabled).toBe(true);
    await press('Enter', live());
    await flushPromises();
    expect(backend.answers).toHaveLength(1);
  });

  it('in the dock, a sign-in from a chat turn names it and offers Open; one from a run no chat turn shows has no origin and no Open', async () => {
    const w = await mountDock(codeAsk('scheduled', null), codeAsk('from-chat'));
    const origin = (r: Found) => (r.find('.activity__origin').exists() ? r.find('.activity__origin').text() : null);

    expect(rows(w).map((r) => [origin(r), buttons(r)])).toEqual([
      [null, ['Sign in']],
      [`Chat · #${TURN} ·`, ['Sign in', 'Open']],
    ]);

    await rows(w)[1]!.find('.activity__stop').trigger('click');
    const session = useSessionStore();
    expect([session.panelThreadId, session.panelType]).toEqual([TURN, 'user']);
    expect(backend.answers).toEqual([]);
  });
});

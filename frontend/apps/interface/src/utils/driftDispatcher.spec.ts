/**
 * driftDispatcher — feature spec for the WS-push discriminator + DOM router.
 *
 * Exercises the real `dispatchDrift` discrimination order (tool_call →
 * turn_signal → turn_execution → simple event) against hand-built frames,
 * and asserts the correct island/DOM effect is invoked for each branch.
 *
 * Boundaries mocked (network + DOM-effect modules, per the project's
 * established convention — see stores/session.spec.ts's "only WS/network is
 * mocked" note): api/conversation, utils/toast, utils/liveActTrail,
 * utils/turnDom, and the three Pinia stores the router touches. Because
 * `utils/cancelReconcile.ts` imports the SAME `./liveActTrail` and
 * `./turnDom` modules (vi.mock is keyed by resolved path, not import
 * specifier), the cancelled branch's real `reconcileCancelledTurn` runs
 * unmocked and its effects surface through those same mocks — no separate
 * mock of cancelReconcile itself is needed.
 *
 * This module no longer imports `stores/session.ts` (see its own docblock —
 * that would be a circular import); the handful of session-owned side
 * effects it still needs are reached through `SessionHooks`, registered via
 * `registerSessionHooks()`. A fake implementation is registered once, here,
 * standing in for `session.init()`.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { WsPushEvent } from '@chalie/shared';

vi.mock('@chalie/shared', () => ({
  ConfigType: { USER: 'user', SCHEDULED: 'scheduled', DISCOVERY: 'discovery' },
}));

const { threadMock } = vi.hoisted(() => ({ threadMock: vi.fn() }));
vi.mock('../api/conversation', () => ({
  conversation: { thread: threadMock },
}));

const { showToastMock } = vi.hoisted(() => ({ showToastMock: vi.fn() }));
vi.mock('./toast', () => ({ showToast: showToastMock }));

const { startLiveToolMock, finishLiveToolMock, clearLiveTurnMock } = vi.hoisted(() => ({
  startLiveToolMock: vi.fn(),
  finishLiveToolMock: vi.fn(),
  clearLiveTurnMock: vi.fn(),
}));
vi.mock('./liveActTrail', () => ({
  startLiveTool: startLiveToolMock,
  finishLiveTool: finishLiveToolMock,
  clearLiveTurn: clearLiveTurnMock,
}));

const { setTurnWorkingMock, setTurnDoneMock, removeTurnMock, upsertTurnToSurfacesMock, findTurnTypeMock } = vi.hoisted(() => ({
  setTurnWorkingMock: vi.fn(),
  setTurnDoneMock: vi.fn(),
  removeTurnMock: vi.fn(),
  upsertTurnToSurfacesMock: vi.fn(),
  // Defaults to "no rendered copy anywhere" — see beforeEach's reset. Real
  // DOM-backed resolution (a rendered node actually naming a turn's type) is
  // covered separately in driftDispatcher.typeResolution.spec.ts, which runs
  // against the REAL turnDom module rather than this mock.
  findTurnTypeMock: vi.fn(() => null as string | null),
}));
vi.mock('./turnDom', () => ({
  setTurnWorking: setTurnWorkingMock,
  setTurnDone: setTurnDoneMock,
  removeTurn: removeTurnMock,
  upsertTurnToSurfaces: upsertTurnToSurfacesMock,
  findTurnType: findTurnTypeMock,
}));

const { enqueueMock, removeMock } = vi.hoisted(() => ({ enqueueMock: vi.fn(), removeMock: vi.fn() }));
vi.mock('../stores/permissions', () => ({
  usePermissionsStore: () => ({ enqueue: enqueueMock, remove: removeMock }),
}));

const { recordVoiceMock } = vi.hoisted(() => ({ recordVoiceMock: vi.fn() }));
vi.mock('../stores/voiceTranscripts', () => ({
  useVoiceTranscriptsStore: () => ({ record: recordVoiceMock }),
}));

const { dispatchDrift, registerSessionHooks } = await import('./driftDispatcher');

// Fake stand-in for the SessionHooks a real `session.init()` would register —
// the only session-owned side effects `_dispatchTurnExecution` still needs.
// `panelThreadId`/`panelType` are mutable per-test state read live by
// `getPanelThreadId`/`getPanelType` — the D16 gate is the FULL (turn_id,
// type) pair (see `isOpenPanelTurn`'s own doc comment), so both must be
// independently controllable per test, not just the turn id.
const releasePendingSendMock = vi.fn();
const setErrorMessageMock = vi.fn();
const finishTurnMock = vi.fn();
const drainQueuesMock = vi.fn();
const hooksState = { panelThreadId: null as number | null, panelType: 'user' };
registerSessionHooks({
  releasePendingSend: releasePendingSendMock,
  getPanelThreadId: () => hooksState.panelThreadId,
  getPanelType: () => hooksState.panelType,
  setErrorMessage: setErrorMessageMock,
  finishTurn: finishTurnMock,
  drainQueues: drainQueuesMock,
});

function frame(data: Record<string, unknown>): WsPushEvent {
  return data as unknown as WsPushEvent;
}

beforeEach(() => {
  vi.clearAllMocks();
  hooksState.panelThreadId = null;
  hooksState.panelType = 'user';
  findTurnTypeMock.mockReturnValue(null);
  threadMock.mockResolvedValue({
    turn_id: 1,
    gist: null,
    preview: '',
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    messages: [],
    type: 'user',
  });
});

describe('dispatchDrift — discrimination order', () => {
  it('a frame carrying both tool_name AND status is claimed by the tool_call branch, never reaching turn_signal', () => {
    dispatchDrift(frame({ tool_name: 'search', status: 'updated', turn_id: 5, state: 'started', type: 'user' }));
    expect(startLiveToolMock).toHaveBeenCalled();
    expect(threadMock).not.toHaveBeenCalled();
    expect(upsertTurnToSurfacesMock).not.toHaveBeenCalled();
  });

  it('a frame with status but no tool_name routes to turn_signal', () => {
    dispatchDrift(frame({ status: 'updated', turn_id: 5, type: 'user' }));
    expect(startLiveToolMock).not.toHaveBeenCalled();
    expect(threadMock).toHaveBeenCalledWith(5, 'user');
  });

  it('a structural turn_execution frame (state/turn_id/started_at, no tool_name/status) routes there', () => {
    dispatchDrift(frame({ state: 'working', turn_id: 9, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    expect(setTurnWorkingMock).toHaveBeenCalledWith(9, 'user', true);
  });

  it('home_state_changed collision guard: a state-bearing frame with no turn_id/started_at is NOT treated as turn_execution', () => {
    dispatchDrift(frame({ type: 'home_state_changed', state: 'working' }));
    expect(setTurnWorkingMock).not.toHaveBeenCalled();
    expect(threadMock).not.toHaveBeenCalled();
  });
});

describe('dispatchDrift — tool_call frame', () => {
  it('state=started calls startLiveTool with (type, turn_id, id, tool_name, summary, transcript_row_id)', () => {
    dispatchDrift(frame({
      tool_name: 'calendar', state: 'started', turn_id: 3, id: 42,
      summary: 'checking calendar', transcript_row_id: 7, type: 'scheduled',
    }));
    expect(startLiveToolMock).toHaveBeenCalledWith('scheduled', 3, 42, 'calendar', 'checking calendar', 7);
  });

  it('state=done calls finishLiveTool with ok=true', () => {
    dispatchDrift(frame({ tool_name: 'calendar', state: 'done', turn_id: 3, id: 42, type: 'user' }));
    expect(finishLiveToolMock).toHaveBeenCalledWith('user', 3, 42, true);
  });

  it('state=error calls finishLiveTool with ok=false', () => {
    dispatchDrift(frame({ tool_name: 'calendar', state: 'error', turn_id: 3, id: 42, type: 'user' }));
    expect(finishLiveToolMock).toHaveBeenCalledWith('user', 3, 42, false);
  });

  it('a tool frame with turn_id null is dropped — claimed, but no island call', () => {
    dispatchDrift(frame({ tool_name: 'calendar', state: 'started', turn_id: null, type: 'user' }));
    expect(startLiveToolMock).not.toHaveBeenCalled();
    expect(finishLiveToolMock).not.toHaveBeenCalled();
  });
});

describe('dispatchDrift — turn_signal frame', () => {
  it('status=updated fetches the block once and fans it to surfaces', async () => {
    dispatchDrift(frame({ status: 'updated', turn_id: 11, type: 'user' }));
    await Promise.resolve();
    await Promise.resolve();
    expect(threadMock).toHaveBeenCalledTimes(1);
    expect(threadMock).toHaveBeenCalledWith(11, 'user');
    expect(upsertTurnToSurfacesMock).toHaveBeenCalledTimes(1);
  });

  // The send's POST-scoped busy hold is released on ANY signal that proves the
  // turn is now tracked by its own frames — an 'updated' signal is such proof,
  // exactly like a turn_execution frame. This is the recovery path for a turn
  // whose `turn_execution` frame is dropped or arrives before the turn is in
  // the DOM (unresolvable type): the turn renders and settles from 'updated'
  // frames alone. Without it, the SOLE releaser is the execution path and
  // `_reconcileWorking` never touches the hold, so the spine's `isSurfaceBusy`
  // gate strands forever and every later send silently queues.
  it('status=updated releases the send\'s POST-scoped busy hold only AFTER the refetch settles — releasing sooner would let a second queued send slip through against stale state', async () => {
    dispatchDrift(frame({ status: 'updated', turn_id: 11, type: 'user' }));
    expect(releasePendingSendMock).not.toHaveBeenCalled();
    await Promise.resolve();
    await Promise.resolve();
    expect(releasePendingSendMock).toHaveBeenCalledWith(11, 'user');
  });

  it('status=updated for an unresolvable-type frame drops WITHOUT releasing the hold — a wrong-channel release could free a different send', () => {
    // No `type` on the frame, nothing rendered, no matching open panel — the
    // channel can't be named, so the frame is dropped. Releasing here would
    // key off a guessed type and could clear a hold for a same-numbered turn
    // in another channel.
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* silence */ });
    dispatchDrift(frame({ status: 'updated', turn_id: 12 }));
    expect(releasePendingSendMock).not.toHaveBeenCalled();
    expect(threadMock).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('status=provider_retry shows a toast with the frame message', () => {
    dispatchDrift(frame({ status: 'provider_retry', message: 'ollama is slow', type: 'user' }));
    expect(showToastMock).toHaveBeenCalledWith('ollama is slow');
  });

  it('status=provider_retry falls back to the default retry copy when no message is present', () => {
    dispatchDrift(frame({ status: 'provider_retry', type: 'user' }));
    expect(showToastMock).toHaveBeenCalledWith('The AI provider had a problem — retrying…');
  });

  // Pre-synthesis pushes these as it moves a settled reply pending → ready/failed.
  // The frame is addressed by transcript row, so it carries no turn_id and no
  // type — and it must NOT be treated as a turn signal that needs either.
  it('status=voice_transcript records the state against its transcript row, with no refetch and no DOM effect', () => {
    dispatchDrift(frame({ status: 'voice_transcript', transcript_id: 412, state: 'ready' }));
    expect(recordVoiceMock).toHaveBeenCalledWith(412, 'ready');
    expect(threadMock).not.toHaveBeenCalled();
    expect(upsertTurnToSurfacesMock).not.toHaveBeenCalled();
  });

  it('a voice_transcript frame with no transcript_id is claimed and dropped — there is no row to paint', () => {
    dispatchDrift(frame({ status: 'voice_transcript', state: 'ready' }));
    expect(recordVoiceMock).not.toHaveBeenCalled();
  });

  // The store's state union is the contract; anything outside it would paint an
  // unstyled button forever, so it is dropped loudly rather than stored.
  it('a voice_transcript frame carrying an unknown state warns and stores nothing', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* silence */ });
    dispatchDrift(frame({ status: 'voice_transcript', transcript_id: 412, state: 'synthesizing' }));
    expect(recordVoiceMock).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});

describe('dispatchDrift — turn_execution frame', () => {
  it('state=working sets data-working and re-fetches/upserts', async () => {
    dispatchDrift(frame({ state: 'working', turn_id: 21, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    expect(setTurnWorkingMock).toHaveBeenCalledWith(21, 'user', true);
    await Promise.resolve();
    await Promise.resolve();
    expect(threadMock).toHaveBeenCalledWith(21, 'user');
  });

  it('state=completed clears data-working and sets data-done when the turn is NOT the open panel thread', () => {
    hooksState.panelThreadId = 999;
    dispatchDrift(frame({ state: 'completed', turn_id: 21, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    expect(setTurnWorkingMock).toHaveBeenCalledWith(21, 'user', false);
    expect(setTurnDoneMock).toHaveBeenCalledWith(21, 'user', true);
    expect(finishTurnMock).toHaveBeenCalledWith(21, 'user');
  });

  it('state=completed does NOT set data-done when the turn IS the open panel thread', () => {
    hooksState.panelThreadId = 21;
    dispatchDrift(frame({ state: 'completed', turn_id: 21, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    expect(setTurnDoneMock).not.toHaveBeenCalled();
    expect(finishTurnMock).toHaveBeenCalledWith(21, 'user');
  });

  // D16 pair gate — the panel identifies a turn by (turn_id, type) TOGETHER,
  // never turn_id alone (turn_id is only unique PER TYPE). A same-numbered
  // turn settling in a DIFFERENT channel than the one open in the panel must
  // still be marked done, exactly as if no panel were open at all — this is
  // the gate the old `getPanelType?.() === undefined` fallback would have
  // silently collapsed back to a turn_id-only match.
  it('D16 pair gate: a settled turn sharing the panel\'s turn_id but a DIFFERENT type is still marked done', () => {
    hooksState.panelThreadId = 5;
    hooksState.panelType = 'user';

    // Same turn_id as the open panel, but a different channel — the pair
    // does not match, so this settle is NOT the one open in the panel.
    dispatchDrift(frame({ state: 'completed', turn_id: 5, started_at: '2026-01-01T00:00:00Z', type: 'scheduled' }));
    expect(setTurnDoneMock).toHaveBeenCalledWith(5, 'scheduled', true);

    setTurnDoneMock.mockClear();

    // The FULL pair now matches (same turn_id AND same type as the panel) —
    // this one genuinely is the open turn, so done must NOT be stamped.
    dispatchDrift(frame({ state: 'completed', turn_id: 5, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    expect(setTurnDoneMock).not.toHaveBeenCalled();
  });

  it('resolveFrameType falls back to the open panel\'s type only when its turn_id matches — a frame for a DIFFERENT turn is dropped rather than guessed', () => {
    hooksState.panelThreadId = 5;
    hooksState.panelType = 'user';
    findTurnTypeMock.mockReturnValue(null); // nothing rendered for either turn yet

    // No `type` on the frame, and its turn_id matches the open panel's — the
    // panel's type is used to resolve it.
    dispatchDrift(frame({ status: 'updated', turn_id: 5 }));
    expect(threadMock).toHaveBeenCalledWith(5, 'user');

    threadMock.mockClear();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* silence */ });

    // No `type`, and this turn_id does NOT match the panel's — nothing can
    // name its channel, so the frame is dropped rather than defaulted to
    // the panel's (or any other) type.
    dispatchDrift(frame({ status: 'updated', turn_id: 6 }));
    expect(threadMock).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalled();

    warnSpy.mockRestore();
  });

  it('state=cancelled with a non-empty fetched block re-renders with force and clears working (version can shrink), then drains queues', async () => {
    threadMock.mockResolvedValue({
      turn_id: 33, gist: null, preview: 'x', last_activity_at: null, working: false, duration_ms: 0, type: 'user',
      messages: [{ id: '1', role: 'user', content: 'hi', timestamp: '2026-01-01 00:00:00', turn_id: 33 }],
    });
    dispatchDrift(frame({ state: 'cancelled', turn_id: 33, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(upsertTurnToSurfacesMock).toHaveBeenCalledWith(
      expect.objectContaining({ turn_id: 33 }), 'user', { force: true },
    );
    expect(setTurnWorkingMock).toHaveBeenCalledWith(33, 'user', false);
    expect(removeTurnMock).not.toHaveBeenCalled();
    expect(drainQueuesMock).toHaveBeenCalled();
  });

  it('state=cancelled with an empty fetched block removes the turn nodes, then drains queues', async () => {
    threadMock.mockResolvedValue({
      turn_id: 34, gist: null, preview: '', last_activity_at: null, working: false, duration_ms: 0, messages: [], type: 'user',
    });
    dispatchDrift(frame({ state: 'cancelled', turn_id: 34, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(removeTurnMock).toHaveBeenCalledWith(34, 'user');
    expect(upsertTurnToSurfacesMock).not.toHaveBeenCalled();
    // The cancel is terminal on EVERY branch — a remote-initiated cancel
    // (another tab/device) whose turn emptied must still clear the live
    // working record, or the thread's send gate stays wedged for the tab's
    // lifetime.
    expect(setTurnWorkingMock).toHaveBeenCalledWith(34, 'user', false);
    expect(drainQueuesMock).toHaveBeenCalled();
  });

  it('completed/cancelled/crashed release the send\'s POST-scoped busy hold synchronously', () => {
    for (const [i, state] of (['completed', 'cancelled', 'crashed'] as const).entries()) {
      dispatchDrift(frame({ state, turn_id: 50 + i, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
      expect(releasePendingSendMock).toHaveBeenCalledWith(50 + i, 'user');
    }
  });

  // A crash is the ONLY terminal state with nothing behind it: the backend
  // emits one lifecycle frame and stops, and a crashed turn frequently has no
  // assistant row, so TranscriptService never fires the 'updated' signal that
  // masks the same hole on 'completed'. Without a refetch here the rendered
  // block keeps the `working: true` it was last upserted with — TurnView keeps
  // appending its live-act row and renders 'thinking…' forever — and `crashed`,
  // which only ever arrives over REST, never reaches the crash note built for
  // exactly this case. Force, because a terminal frame outranks the version
  // guard.
  it('state=crashed refetches the settled block with force, so working clears and crashed can render', async () => {
    threadMock.mockResolvedValue({
      turn_id: 40, gist: null, preview: 'x', last_activity_at: null, working: false, crashed: true,
      duration_ms: 0, type: 'user',
      messages: [{ id: '1', role: 'user', content: 'hi', timestamp: '2026-01-01 00:00:00', turn_id: 40 }],
    });
    dispatchDrift(frame({ state: 'crashed', turn_id: 40, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(threadMock).toHaveBeenCalledWith(40, 'user');
    expect(upsertTurnToSurfacesMock).toHaveBeenCalledWith(
      expect.objectContaining({ turn_id: 40, working: false, crashed: true }), 'user', { force: true },
    );
    expect(setErrorMessageMock).toHaveBeenCalled();
  });

  // Same terminal path, deliberately: 'completed' must settle on its own frame
  // rather than depending on append_assistant's 'updated' winning a race it is
  // not guaranteed to win.
  it('state=completed refetches the settled block with force too — one common terminal path', async () => {
    threadMock.mockResolvedValue({
      turn_id: 41, gist: null, preview: 'x', last_activity_at: null, working: false,
      duration_ms: 0, type: 'user',
      messages: [{ id: '1', role: 'assistant', content: 'done', timestamp: '2026-01-01 00:00:00', turn_id: 41 }],
    });
    dispatchDrift(frame({ state: 'completed', turn_id: 41, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(upsertTurnToSurfacesMock).toHaveBeenCalledWith(
      expect.objectContaining({ turn_id: 41, working: false }), 'user', { force: true },
    );
    expect(setErrorMessageMock).not.toHaveBeenCalled();
  });

  // The settle bookkeeping is synchronous and must not become contingent on
  // the network: a refetch that never resolves cannot re-wedge the send gate.
  it('state=crashed still settles synchronously when the refetch rejects', async () => {
    threadMock.mockRejectedValue(new Error('network down'));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* silence */ });
    dispatchDrift(frame({ state: 'crashed', turn_id: 42, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    expect(releasePendingSendMock).toHaveBeenCalledWith(42, 'user');
    expect(setTurnWorkingMock).toHaveBeenCalledWith(42, 'user', false);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  // 'working' is the one state with an in-flight refetch — releasing the hold
  // synchronously here would reopen the spine's busy gate before that refetch
  // resolves, letting a second queued send slip through against stale
  // (pre-refetch) state. The release must wait for the refetch to
  // settle.
  it('state=working releases the send\'s POST-scoped busy hold only AFTER the refetch settles', async () => {
    dispatchDrift(frame({ state: 'working', turn_id: 60, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    expect(releasePendingSendMock).not.toHaveBeenCalled();
    await Promise.resolve();
    await Promise.resolve();
    expect(releasePendingSendMock).toHaveBeenCalledWith(60, 'user');
  });

  it('state=cancelled still clears data-working and drains queues when the refetch rejects', async () => {
    threadMock.mockRejectedValue(new Error('network down'));
    dispatchDrift(frame({ state: 'cancelled', turn_id: 35, started_at: '2026-01-01T00:00:00Z', type: 'user' }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(setTurnWorkingMock).toHaveBeenCalledWith(35, 'user', false);
    expect(drainQueuesMock).toHaveBeenCalled();
  });
});

describe('dispatchDrift — simple (content-free) events', () => {
  it('permission_request reaches the permissions store', () => {
    dispatchDrift(frame({ type: 'permission_request', request_id: 'r1', action_id: 'email.send' }));
    expect(enqueueMock).toHaveBeenCalled();
  });

  it('permission_resolved removes that gate\'s card from the permissions store', () => {
    dispatchDrift(frame({ type: 'permission_resolved', request_id: 'r1' }));
    expect(removeMock).toHaveBeenCalledWith('r1');
  });

  it('permission_resolved without a request_id touches nothing', () => {
    dispatchDrift(frame({ type: 'permission_resolved' }));
    expect(removeMock).not.toHaveBeenCalled();
  });

  it('an unknown type is a no-op — no throw, no store touched', () => {
    expect(() => dispatchDrift(frame({ type: 'something_nobody_emits' }))).not.toThrow();
    expect(enqueueMock).not.toHaveBeenCalled();
  });
});

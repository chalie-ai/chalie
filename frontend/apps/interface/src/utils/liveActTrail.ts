/**
 * liveActTrail — the live act-trail (transient pill) state.
 *
 * Per-feed map of live tool pills, keyed by transcript_row_id (the turn_id
 * anchor). A feed is a ConfigType, or a delegate turn's full channel — the
 * two never share a key, so a delegate's pills can't land on a same-id user
 * turn. Visual-only; no turn data. Reactive so Vue components re-render off it.
 */
import { reactive } from 'vue';
import type { DelegateRef } from '@chalie/shared';
import { isDelegateChannel } from './delegateChannel';

// ── Public surface types ─────────────────────────────────────────────────────

export interface LiveToolPill {
  /** Unique id — the opened tool_calls row id (matches finishLiveTool callId). */
  id: string;
  name: string;
  summary: string | undefined;
  startedAt: number;
  /** Frozen elapsed ms once resolved. */
  ms?: number;
  ok?: boolean;
  resolved: boolean;
  /** The transcript row this tool call anchors to (from WS frame transcript_row_id). */
  transcriptRowId: number | null;
  /** The delegate turn this call spawned — set by a later frame for the same call. */
  delegate: DelegateRef | null;
}

export interface LiveTrail {
  rowId: number;
  pills: LiveToolPill[];
}

// ── Per-type state ──────────────────────────────────────────────────────────

interface LiveToolState {
  /**
   * Live act-trail pills keyed by turn_id — the frame now carries transcript_row_id
   * (the anchor row the tool call lives on), so the pill knows its transcript row.
   * The turn REMAINS the visual anchor by design (consecutive tool-only steps
   * collapse onto one binder anyway). Visual-only; no turn data. Resolved pills are
   * dropped when the turn's persisted tool_calls arrive (upsertTurn §6.5 step 4).
   */
  liveTools: Record<number, LiveToolPill[]>;
  /** Single interval driving all live-pill elapsed timers for this type. */
  timerInterval: ReturnType<typeof setInterval> | null;
}

const _feeds = new Map<string, LiveToolState>();

function feedState(type: string): LiveToolState {
  let s = _feeds.get(type);
  if (!s) {
    s = reactive<LiveToolState>({
      liveTools: {},
      timerInterval: null,
    });
    _feeds.set(type, s);
  }
  return s;
}

// ── Timer loop ───────────────────────────────────────────────────────────────

function _ensureTimerRunning(s: LiveToolState): void {
  if (s.timerInterval !== null) return;
  s.timerInterval = setInterval(() => {
    for (const turnId of Object.keys(s.liveTools)) {
      const pills = s.liveTools[Number(turnId)];
      if (pills?.some((p) => !p.resolved)) s.liveTools[Number(turnId)] = [...pills];
    }
  }, 500);
}

function _maybeStopTimer(s: LiveToolState): void {
  const hasLive = Object.values(s.liveTools).some((pills) => pills.some((p) => !p.resolved));
  if (!hasLive && s.timerInterval !== null) {
    clearInterval(s.timerInterval);
    s.timerInterval = null;
  }
}

// ── Core pill operations ─────────────────────────────────────────────────────

export function startLiveTool(
  type: string,
  turnId: number,
  callId: number | null,
  name: string,
  summary?: string,
  transcriptRowId: number | null = null,
): void {
  if (callId == null) return;
  const s = feedState(type);
  const id = String(callId);
  // A delegate re-sends its call's frame once its child turn exists; the pill
  // is already on screen, so the repeat must not open a second one.
  if (s.liveTools[turnId]?.some((p) => p.id === id)) return;
  const pill: LiveToolPill = {
    id,
    name,
    summary,
    startedAt: Date.now(),
    ok: false,
    resolved: false,
    transcriptRowId,
    delegate: null,
  };
  s.liveTools[turnId] = [...(s.liveTools[turnId] ?? []), pill];
  _ensureTimerRunning(s);
}

/** Point a live pill at the delegate turn its call spawned. */
export function setLiveToolDelegate(type: string, turnId: number, callId: number | null, delegate: DelegateRef): void {
  if (callId == null) return;
  const s = feedState(type);
  const key = String(callId);
  const pills = s.liveTools[turnId];
  const pill = pills?.find((p) => p.id === key);
  if (!pills || !pill) return;
  if (pill.delegate?.channel === delegate.channel && pill.delegate.turn_id === delegate.turn_id) return;
  s.liveTools[turnId] = pills.map((p) =>
    p.id === key ? { ...p, delegate: { channel: delegate.channel, turn_id: delegate.turn_id } } : p,
  );
}

export function finishLiveTool(type: string, turnId: number, callId: number | null, ok: boolean): void {
  if (callId == null) return;
  const s = feedState(type);
  const key = String(callId);
  const pills = s.liveTools[turnId];
  if (!pills?.some((p) => p.id === key && !p.resolved)) return;
  s.liveTools[turnId] = pills.map((p) =>
    p.id === key ? { ...p, resolved: true, ok, ms: Date.now() - p.startedAt } : p,
  );
  _maybeStopTimer(s);
}

export function liveTrailsFor(type: string, turnId: number): LiveTrail[] {
  const s = feedState(type);
  const pills = s.liveTools[turnId];
  return pills ? [{ rowId: turnId, pills }] : [];
}

export function clearLiveTurn(type: string, turnId: number): void {
  const s = feedState(type);
  delete s.liveTools[turnId];
  _maybeStopTimer(s);
}

export function clearLiveTurnsForToolCallsResolved(type: string, turnId: number, hasToolCalls: boolean): void {
  if (!hasToolCalls) return;
  const s = feedState(type);
  const live = s.liveTools[turnId]?.filter((p) => !p.resolved);
  if (live?.length) s.liveTools[turnId] = live;
  else delete s.liveTools[turnId];
  _maybeStopTimer(s);
}

/** Tear down all live pills for a type — called when the feed is reset. */
export function clearAll(type: string): void {
  const s = feedState(type);
  s.liveTools = {};
  if (s.timerInterval !== null) {
    clearInterval(s.timerInterval);
    s.timerInterval = null;
  }
}

/** Tear down every delegate feed. A delegate's pills settle only on its own
 *  terminal frame, which a dropped socket loses for good. */
export function clearDelegateFeeds(): void {
  for (const feed of _feeds.keys()) {
    if (isDelegateChannel(feed)) clearAll(feed);
  }
}

/**
 * liveActTrail — the live act-trail (transient pill) state.
 *
 * Per-feed map of live tool pills, keyed by turn_id. A feed is a ConfigType,
 * or a delegate turn's full channel — the two never share a key, so a
 * delegate's pills can't land on a same-id user turn. Visual-only; no turn
 * data. Reactive so Vue components re-render off it.
 */
import { reactive, shallowReactive } from 'vue';
import type { DelegateRef, TurnExecutionState } from '@chalie/shared';
import { isDelegateChannel } from './delegateChannel';

// ── Public surface types ─────────────────────────────────────────────────────

export interface LiveToolPill {
  /** Unique id — the opened tool_calls row id (matches finishLiveTool callId). */
  id: string;
  name: string;
  /** The arguments the call was made with. */
  params: Record<string, unknown> | null;
  summary: string | undefined;
  ok?: boolean;
  resolved: boolean;
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
   * Live act-trail pills keyed by turn_id, the visual anchor. Visual-only; no
   * turn data. Resolved pills are dropped when the turn's persisted tool_calls
   * arrive (upsertTurn).
   */
  liveTools: Record<number, LiveToolPill[]>;
}

const _feeds = new Map<string, LiveToolState>();

function feedState(type: string): LiveToolState {
  let s = _feeds.get(type);
  if (!s) {
    s = reactive<LiveToolState>({ liveTools: {} });
    _feeds.set(type, s);
  }
  return s;
}

// ── Core pill operations ─────────────────────────────────────────────────────

export function startLiveTool(
  type: string,
  turnId: number,
  callId: number | null,
  name: string,
  summary?: string,
  params: Record<string, unknown> | null = null,
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
    params,
    summary,
    ok: false,
    resolved: false,
    delegate: null,
  };
  s.liveTools[turnId] = [...(s.liveTools[turnId] ?? []), pill];
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
    p.id === key ? { ...p, resolved: true, ok } : p,
  );
}

export function liveTrailsFor(type: string, turnId: number): LiveTrail[] {
  const s = feedState(type);
  const pills = s.liveTools[turnId];
  return pills ? [{ rowId: turnId, pills }] : [];
}

export function clearLiveTurn(type: string, turnId: number): void {
  const s = feedState(type);
  delete s.liveTools[turnId];
}

export function clearLiveTurnsForToolCallsResolved(type: string, turnId: number, hasToolCalls: boolean): void {
  if (!hasToolCalls) return;
  const s = feedState(type);
  const live = s.liveTools[turnId]?.filter((p) => !p.resolved);
  if (live?.length) s.liveTools[turnId] = live;
  else delete s.liveTools[turnId];
}

/** Tear down all live pills for a type — called when the feed is reset. */
export function clearAll(type: string): void {
  feedState(type).liveTools = {};
}

// ── Delegate turn run state ─────────────────────────────────────────────────

/**
 * Each delegate turn's run state as its own latest execution frame gave it,
 * keyed `${channel}#${turnId}`. An ended turn can work again: a follow-up
 * message to a subtask reopens the same turn, and its frames mark it working.
 */
const _delegateStates = reactive(new Map<string, TurnExecutionState>());

/** Record a delegate turn's run state from its own execution frame. */
export function setDelegateState(channel: string, turnId: number, state: TurnExecutionState): void {
  _delegateStates.set(`${channel}#${turnId}`, state);
}

/** How the delegate turn `delegate` names stands — the one status the
 *  Subagents dock and a call's mark both read: its own execution
 *  frames once one was seen, else the REST read its ref came from. A live
 *  frame's ref carries no run state; its subagent has just started. */
export function delegateState(delegate: DelegateRef): TurnExecutionState {
  const seen = _delegateStates.get(`${delegate.channel}#${delegate.turn_id}`);
  if (seen) return seen;
  if (delegate.working ?? true) return 'working';
  if (delegate.crashed) return 'crashed';
  return delegate.cancelled ? 'cancelled' : 'completed';
}

/** Tear down every delegate feed. A delegate's pills settle only on its own
 *  terminal frame, which a dropped socket loses for good. The run map loses
 *  its still-working entries the same way — the session re-reads every turn
 *  that showed a running subagent, so the REST read decides again; ended
 *  entries stay. */
export function clearDelegateFeeds(): void {
  for (const feed of _feeds.keys()) {
    if (isDelegateChannel(feed)) clearAll(feed);
  }
  for (const [key, state] of _delegateStates) {
    if (state === 'working') _delegateStates.delete(key);
  }
}

// ── Running subagents, across every turn on screen ──────────────────────────

/** A subagent still running, with the one-line summary of the call that started it. */
export interface RunningSubagent {
  delegate: DelegateRef;
  summary: string;
}

/**
 * Every running subagent some mounted turn view showed, keyed
 * `${channel}#${turnId}` in first-seen order, with the views that showed it —
 * so a turn drawn twice keeps its entry while one copy is still mounted.
 */
const _running = shallowReactive(new Map<string, { subagent: RunningSubagent; views: Set<symbol> }>());

/** Set the running subagents `view` shows, replacing what it showed before. */
export function publishRunningSubagents(view: symbol, running: readonly RunningSubagent[]): void {
  const keys = new Set(running.map((s) => `${s.delegate.channel}#${s.delegate.turn_id}`));
  for (const [key, held] of _running) {
    if (keys.has(key)) continue;
    held.views.delete(view);
    if (held.views.size === 0) _running.delete(key);
  }
  for (const subagent of running) {
    const key = `${subagent.delegate.channel}#${subagent.delegate.turn_id}`;
    const held = _running.get(key);
    if (held) held.views.add(view);
    else _running.set(key, { subagent, views: new Set([view]) });
  }
}

/** Drop everything `view` showed — called as its turn view unmounts. */
export function withdrawRunningSubagents(view: symbol): void {
  publishRunningSubagents(view, []);
}

/** Every running subagent on screen, once each, in the order first seen. */
export function runningSubagents(): RunningSubagent[] {
  return [..._running.values()].map((held) => held.subagent);
}

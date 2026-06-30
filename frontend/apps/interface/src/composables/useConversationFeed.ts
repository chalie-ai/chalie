/**
 * useConversationFeed — the JS state/API layer for rendering turn blocks.
 *
 * DOCTRINE (authoritative — all later phases obey this file's contract):
 *   1. The buffer is filled ONLY by API responses via upsertTurn / fetchTurn /
 *      loadRecent / loadMore. Never by WS payloads.
 *   2. WS handlers MAY call: fetchTurn (updated/done), startLiveTool /
 *      finishLiveTool (tool signals), setWorking (working/done). That is ALL.
 *   3. No method in this file mutates turn data from a WS payload.
 *
 * Channel-keyed factory — call useConversationFeed('user') or with no arg for
 * the main user feed; pass a different channel key for scheduled/other surfaces.
 * Module-level singletons are eliminated; each channel owns its own FeedState.
 */
import { reactive, computed } from 'vue';
import type { ComputedRef } from 'vue';
import type { ConversationTurnBlock, ConversationThread } from '../api/conversation';
import { conversation as convoApi } from '../api/conversation';
import { messagePlaintext } from '../utils/speech';

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
}

export interface LiveTrail {
  rowId: number;
  pills: LiveToolPill[];
}

// ── Per-channel state ─────────────────────────────────────────────────────────

interface FeedState {
  /** Ordered turn blocks keyed by turn_id. Reactive. */
  blocks: Record<number, ConversationTurnBlock>;
  /** Highest message row id seen per turn — the monotonic version guard. */
  versions: Record<number, number>;
  /** Transient working-spinner flags per turn_id. NOT turn data. */
  working: Record<number, boolean>;
  /**
   * turn_ids whose reply has settled (`done`) but the user hasn't viewed yet —
   * the Activity dock's "done" (blue) state. Independent of the block map so
   * fetchTurn/upsertTurn never clear a standing notification.
   */
  done: Set<number>;
  /**
   * Live act-trail pills keyed by transcript_row_id. Visual-only; no turn data.
   * Cleared per-row when the row's tool_calls persist (upsertTurn §6.5 step 4).
   */
  liveTools: Record<number, { turnId: number; pills: LiveToolPill[] }>;
  /** Pagination cursor (count of thread items seen so far). */
  offset: number;
  hasMore: boolean;
  /** Single interval driving all live-pill elapsed timers for this channel. */
  timerInterval: ReturnType<typeof setInterval> | null;
}

const _channels = new Map<string, FeedState>();

function feedState(channel: string): FeedState {
  let s = _channels.get(channel);
  if (!s) {
    s = reactive<FeedState>({
      blocks: {},
      versions: {},
      working: {},
      done: new Set(),
      liveTools: {},
      offset: 0,
      hasMore: true,
      timerInterval: null,
    });
    _channels.set(channel, s);
  }
  return s;
}

// ── Timer loop ───────────────────────────────────────────────────────────────

function _ensureTimerRunning(s: FeedState): void {
  if (s.timerInterval !== null) return;
  s.timerInterval = setInterval(() => {
    for (const rowId of Object.keys(s.liveTools)) {
      const trail = s.liveTools[Number(rowId)];
      if (trail?.pills.some((p) => !p.resolved)) {
        s.liveTools[Number(rowId)] = { ...trail, pills: [...trail.pills] };
      }
    }
  }, 500);
}

function _maybeStopTimer(s: FeedState): void {
  const hasLive = Object.values(s.liveTools).some((t) => t.pills.some((p) => !p.resolved));
  if (!hasLive && s.timerInterval !== null) {
    clearInterval(s.timerInterval);
    s.timerInterval = null;
  }
}

// ── Version guard ─────────────────────────────────────────────────────────────

function _blockVersion(block: ConversationTurnBlock): number {
  let v = 0;
  for (const m of block.messages) {
    const n = parseInt(m.id, 10);
    if (n > v) v = n;
  }
  return v;
}

// ── Core buffer operations ────────────────────────────────────────────────────

function _upsertTurn(s: FeedState, block: ConversationTurnBlock): void {
  const incoming = _blockVersion(block);
  if ((s.versions[block.turn_id] ?? -1) > incoming) return;
  s.versions[block.turn_id] = incoming;
  s.blocks[block.turn_id] = block;

  // §6.5 step 4 — persisted tool_calls supersede live pills for that row.
  for (const m of block.messages) {
    if (m.tool_calls?.length) delete s.liveTools[parseInt(m.id, 10)];
  }
  _maybeStopTimer(s);
}

async function _fetchTurn(s: FeedState, turnId: number, channel: string): Promise<void> {
  _upsertTurn(s, await convoApi.thread(turnId, channel));
}

// ── Pagination ────────────────────────────────────────────────────────────────

async function _loadRecent(s: FeedState, channel: string): Promise<void> {
  s.offset = 0;
  s.hasMore = true;
  const limit = 20;
  const { threads, has_more } = await convoApi.threads(limit, 0, undefined, channel);
  s.hasMore = has_more;
  s.offset = threads.length;

  if (!threads.length) return;
  const ids = threads.map((t) => t.turn_id).filter((id): id is number => id != null);
  if (!ids.length) return;
  const { blocks } = await convoApi.batch(ids, channel);
  for (const b of blocks) _upsertTurn(s, b);
}

async function _loadMore(s: FeedState, channel: string): Promise<void> {
  if (!s.hasMore) return;
  const limit = 20;
  const { threads, has_more } = await convoApi.threads(limit, s.offset, undefined, channel);
  s.hasMore = has_more;
  s.offset += threads.length;

  const ids = threads.map((t) => t.turn_id).filter((id): id is number => id != null);
  if (!ids.length) return;
  const { blocks } = await convoApi.batch(ids, channel);
  for (const b of blocks) _upsertTurn(s, b);
}

// ── Visual-only state ─────────────────────────────────────────────────────────

function _setWorking(s: FeedState, turnId: number, on: boolean): void {
  if (on) {
    s.working[turnId] = true;
    s.done.delete(turnId);
  } else {
    delete s.working[turnId];
    _clearLiveTurn(s, turnId);
  }
}

/** Settle a background thread reply: stop spinner, drop live pills, leave a
 *  standing `done` card until the user opens the thread. */
function _markThreadDone(s: FeedState, turnId: number): void {
  delete s.working[turnId];
  s.done.add(turnId);
  _clearLiveTurn(s, turnId);
}

/** The user opened (or is viewing) the thread — dismiss its standing `done` card. */
function _seenThread(s: FeedState, turnId: number): void {
  s.done.delete(turnId);
}

function _startLiveTool(
  s: FeedState,
  turnId: number,
  rowId: number | null,
  callId: number | null,
  name: string,
  summary?: string,
): void {
  if (rowId == null || callId == null) return;
  const pill: LiveToolPill = {
    id: String(callId),
    name,
    summary,
    startedAt: Date.now(),
    ok: false,
    resolved: false,
  };
  const trail = s.liveTools[rowId] ?? { turnId, pills: [] };
  s.liveTools[rowId] = { turnId, pills: [...trail.pills, pill] };
  _ensureTimerRunning(s);
}

function _finishLiveTool(s: FeedState, callId: number | null): void {
  if (callId == null) return;
  const key = String(callId);
  for (const rowId of Object.keys(s.liveTools)) {
    const trail = s.liveTools[Number(rowId)];
    if (!trail?.pills.some((p) => p.id === key && !p.resolved)) continue;
    s.liveTools[Number(rowId)] = {
      turnId: trail.turnId,
      pills: trail.pills.map((p) =>
        p.id === key ? { ...p, resolved: true, ok: true, ms: Date.now() - p.startedAt } : p,
      ),
    };
    _maybeStopTimer(s);
    return;
  }
}

function _clearLiveTurn(s: FeedState, turnId: number): void {
  for (const rowId of Object.keys(s.liveTools)) {
    if (s.liveTools[Number(rowId)]?.turnId === turnId) delete s.liveTools[Number(rowId)];
  }
  _maybeStopTimer(s);
}

/** Tear down an aborted/superseded live turn — its block, version, and all
 *  signal state — leaving the feed clean. */
function _dropLiveTurn(s: FeedState, turnId: number): void {
  delete s.blocks[turnId];
  delete s.versions[turnId];
  delete s.working[turnId];
  s.done.delete(turnId);
  _clearLiveTurn(s, turnId);
}

// ── Derived helpers ───────────────────────────────────────────────────────────

/** True while `turnId`'s `working` signal is on (spinner anchor visible). */
function _isTurnWorking(s: FeedState, turnId: number): boolean {
  return !!s.working[turnId];
}

/** A forked thread carries reply rows past settle0 (`thread_message` flag). */
function _isForkedThread(s: FeedState, turnId: number): boolean {
  return s.blocks[turnId]?.messages.some((m) => m.thread_message) ?? false;
}

/** A forked thread's Activity phase: working → done → null (seen / no activity). */
function _threadPhase(s: FeedState, turnId: number): 'working' | 'done' | null {
  if (s.working[turnId]) return 'working';
  if (s.done.has(turnId)) return 'done';
  return null;
}

/** True once a turn's block is in the buffer (hydrated on load or fetched). */
function _isHydrated(s: FeedState, turnId: number): boolean {
  return turnId in s.blocks;
}

/**
 * True when a thread's last activity was within the 1-hour active window.
 * Display/ordering only — no behavioral branch (spec §4.B).
 */
function _isThreadActive(lastActivityAt: string | null): boolean {
  if (!lastActivityAt) return false;
  // SQLite stores naive UTC ("YYYY-MM-DD HH:MM:SS"); mark it as UTC.
  const ts = new Date(`${lastActivityAt.replace(' ', 'T')}Z`).getTime();
  if (Number.isNaN(ts)) return false;
  return Date.now() - ts < 3_600_000;
}

/** Derive collapsed thread metadata from the block buffer (layer-2 getter). */
function _threadList(s: FeedState): ConversationThread[] {
  return Object.keys(s.blocks)
    .map(Number)
    .sort((a, b) => a - b)
    .map((id) => {
      const b = s.blocks[id] as ConversationTurnBlock;
      return {
        turn_id: b.turn_id,
        last_activity_at: b.last_activity_at,
        row_count: b.messages.length,
        preview: b.preview,
        gist: b.gist,
      };
    });
}

/** TTS plaintext of every assistant row in a buffered block. */
function _turnSpeechText(s: FeedState, turnId: number): string {
  const block = s.blocks[turnId];
  if (!block) return '';
  return block.messages
    .filter((m) => m.role === 'assistant' && !!(m.content || m.segments?.length))
    .map(messagePlaintext)
    .filter(Boolean)
    .join(' ');
}

/** Live trails for a given turnId, for the template to render. */
function _liveTrailsFor(s: FeedState, turnId: number): LiveTrail[] {
  return Object.entries(s.liveTools)
    .filter(([, v]) => v.turnId === turnId)
    .map(([rowId, v]) => ({ rowId: Number(rowId), pills: v.pills }));
}

async function _searchThreads(channel: string, query: string): Promise<ConversationThread[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  return (await convoApi.threads(5, undefined, trimmed, channel)).threads;
}

// ── Public API ────────────────────────────────────────────────────────────────

export interface ConversationFeedApi {
  /** The sorted turn blocks, reactive — bind directly to v-for. */
  readonly sortedBlocks: ComputedRef<ConversationTurnBlock[]>;
  /** The raw block map, reactive — keyed by turn_id. */
  readonly blocks: Record<number, ConversationTurnBlock>;
  /** Working-spinner flags (transient visual), reactive. */
  readonly working: Record<number, boolean>;
  /** Live tool-trail map (transient visual), reactive. */
  readonly liveTools: Record<number, { turnId: number; pills: LiveToolPill[] }>;
  readonly hasMore: boolean;

  upsertTurn(block: ConversationTurnBlock): void;
  fetchTurn(turnId: number): Promise<void>;
  loadRecent(): Promise<void>;
  loadMore(): Promise<void>;

  setWorking(turnId: number, on: boolean): void;
  markThreadDone(turnId: number): void;
  seenThread(turnId: number): void;
  dropLiveTurn(turnId: number): void;

  startLiveTool(turnId: number, rowId: number | null, callId: number | null, name: string, summary?: string): void;
  finishLiveTool(callId: number | null): void;
  liveTrailsFor(turnId: number): LiveTrail[];

  isTurnWorking(turnId: number): boolean;
  isForkedThread(turnId: number): boolean;
  threadPhase(turnId: number): 'working' | 'done' | null;
  isHydrated(turnId: number): boolean;
  isThreadActive(lastActivityAt: string | null): boolean;
  /** Layer-2: thread metadata derived from the block buffer. */
  threadList(): ConversationThread[];
  searchThreads(query: string): Promise<ConversationThread[]>;

  turnSpeechText(turnId: number): string;
}

function makeApi(channel: string): ConversationFeedApi {
  const s = feedState(channel);

  const sortedBlocks = computed<ConversationTurnBlock[]>(() =>
    Object.keys(s.blocks)
      .map(Number)
      .sort((a, b) => a - b)
      .map((id) => s.blocks[id] as ConversationTurnBlock),
  );

  return {
    sortedBlocks,
    get blocks() { return s.blocks; },
    get working() { return s.working; },
    get liveTools() { return s.liveTools; },
    get hasMore() { return s.hasMore; },

    upsertTurn: (block) => _upsertTurn(s, block),
    fetchTurn: (turnId) => _fetchTurn(s, turnId, channel),
    loadRecent: () => _loadRecent(s, channel),
    loadMore: () => _loadMore(s, channel),

    setWorking: (turnId, on) => _setWorking(s, turnId, on),
    markThreadDone: (turnId) => _markThreadDone(s, turnId),
    seenThread: (turnId) => _seenThread(s, turnId),
    dropLiveTurn: (turnId) => _dropLiveTurn(s, turnId),

    startLiveTool: (turnId, rowId, callId, name, summary) =>
      _startLiveTool(s, turnId, rowId, callId, name, summary),
    finishLiveTool: (callId) => _finishLiveTool(s, callId),
    liveTrailsFor: (turnId) => _liveTrailsFor(s, turnId),

    isTurnWorking: (turnId) => _isTurnWorking(s, turnId),
    isForkedThread: (turnId) => _isForkedThread(s, turnId),
    threadPhase: (turnId) => _threadPhase(s, turnId),
    isHydrated: (turnId) => _isHydrated(s, turnId),
    isThreadActive: (lastActivityAt) => _isThreadActive(lastActivityAt),
    threadList: () => _threadList(s),
    searchThreads: (query) => _searchThreads(channel, query),

    turnSpeechText: (turnId) => _turnSpeechText(s, turnId),
  };
}

const _apis = new Map<string, ConversationFeedApi>();

/** Channel-keyed factory — safe to call outside setup(). */
export function useConversationFeed(channel = 'user'): ConversationFeedApi {
  let api = _apis.get(channel);
  if (!api) {
    api = makeApi(channel);
    _apis.set(channel, api);
  }
  return api;
}

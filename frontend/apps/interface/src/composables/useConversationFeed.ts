/**
 * useConversationFeed — the JS state/API layer for rendering turn blocks.
 *
 * DOCTRINE (authoritative — all later phases obey this file's contract):
 *   1. The buffer is filled ONLY by API responses via upsertTurn / fetchTurn /
 *      loadRecent / loadMore. Never by WS payloads.
 *   2. WS handlers MAY call: fetchTurn (updated + turn_execution working),
 *      setWorking (turn_execution working/terminal). That is ALL.
 *   3. No method in this file mutates turn data from a WS payload.
 *
 * Type-keyed factory — call useConversationFeed(ConfigType.USER) or with no arg
 * for the main user feed; pass a different type key for scheduled/other surfaces.
 * Module-level singletons are eliminated; each type owns its own FeedState.
 */
import type { ComputedRef } from 'vue';
import { computed, reactive } from 'vue';
import { ConfigType } from '@chalie/shared';
import type { ConversationThread, ConversationTurnBlock } from '../api/conversation';
import { conversation as convoApi } from '../api/conversation';
import { messagePlaintext } from '../utils/speech';
import { clearLiveTurn, clearLiveTurnsForToolCallsResolved } from '../utils/liveActTrail';

// ── Per-type state ──────────────────────────────────────────────────────────

interface FeedState {
  /** The ConfigType key this state was created under (liveActTrail scoping). */
  type: string;
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
  /** Pagination cursor (count of thread items seen so far). */
  offset: number;
  hasMore: boolean;
}

const _feeds = new Map<string, FeedState>();

function feedState(type: string): FeedState {
  let s = _feeds.get(type);
  if (!s) {
    s = reactive<FeedState>({
      type,
      blocks: {},
      versions: {},
      working: {},
      done: new Set(),
      offset: 0,
      hasMore: true,
    });
    _feeds.set(type, s);
  }
  return s;
}

// ── Version guard ─────────────────────────────────────────────────────────────

function _blockVersion(block: ConversationTurnBlock): number {
  let v = 0;
  for (const m of block.messages) {
    const n = Number.parseInt(m.id, 10);
    if (n > v) v = n;
  }
  return v;
}

// ── Core buffer operations ────────────────────────────────────────────────────

/** The actual buffer write, shared by the guarded and unguarded entry points. */
function _writeTurn(s: FeedState, block: ConversationTurnBlock, version: number): void {
  s.versions[block.turn_id] = version;
  s.blocks[block.turn_id] = block;

  // §6.5 step 4 — the turn's persisted tool_calls supersede its resolved live
  // pills; an unresolved pill (a step still running) is left to finish live.
  clearLiveTurnsForToolCallsResolved(
    s.type,
    block.turn_id,
    block.messages.some((m) => m.tool_calls?.length),
  );
}

function _upsertTurn(s: FeedState, block: ConversationTurnBlock): void {
  const incoming = _blockVersion(block);
  if ((s.versions[block.turn_id] ?? -1) > incoming) return;
  _writeTurn(s, block, incoming);
}

/**
 * Unconditional write, bypassing `_upsertTurn`'s monotonic version guard.
 * The ONE legitimate case where the freshest server read can be SMALLER than
 * what's already buffered: a cancelled turn's trailing orphan row(s) were
 * stripped server-side (`api.threads.serialize_turn`), so a post-cancel
 * re-fetch is authoritative even though its content shrank — the version
 * guard exists to reject a stale/out-of-order WS-triggered fetch racing a
 * newer one, which does not apply here (this write follows a cancel the
 * caller just confirmed). Used by `session.ts`'s `_handleCancelled`.
 */
function _forceUpsertTurn(s: FeedState, block: ConversationTurnBlock): void {
  _writeTurn(s, block, _blockVersion(block));
}

async function _fetchTurn(s: FeedState, turnId: number, type: string): Promise<void> {
  _upsertTurn(s, await convoApi.thread(turnId, type));
}

// ── Pagination ────────────────────────────────────────────────────────────────

async function _loadRecent(s: FeedState, type: string): Promise<void> {
  s.offset = 0;
  s.hasMore = true;
  const limit = 20;
  const { threads, has_more } = await convoApi.threads(limit, 0, undefined, type);
  s.hasMore = has_more;
  s.offset = threads.length;

  if (!threads.length) return;
  const ids = threads.map((t) => t.turn_id).filter((id): id is number => id != null);
  if (!ids.length) return;
  const { blocks } = await convoApi.batch(ids, type);
  for (const b of blocks) _upsertTurn(s, b);
}

async function _loadMore(s: FeedState, type: string): Promise<void> {
  if (!s.hasMore) return;
  const limit = 20;
  const { threads, has_more } = await convoApi.threads(limit, s.offset, undefined, type);
  s.hasMore = has_more;
  s.offset += threads.length;

  const ids = threads.map((t) => t.turn_id).filter((id): id is number => id != null);
  if (!ids.length) return;
  const { blocks } = await convoApi.batch(ids, type);
  for (const b of blocks) _upsertTurn(s, b);
}

// ── Visual-only state ─────────────────────────────────────────────────────────

function _setWorking(s: FeedState, turnId: number, on: boolean): void {
  if (on) {
    s.working[turnId] = true;
    s.done.delete(turnId);
  } else {
    delete s.working[turnId];
    clearLiveTurn(s.type, turnId);
  }
}

/** Settle a background thread reply: stop spinner, leave a standing `done`
 *  card until the user opens the thread. */
function _markThreadDone(s: FeedState, turnId: number): void {
  delete s.working[turnId];
  s.done.add(turnId);
  clearLiveTurn(s.type, turnId);
}

/** The user opened (or is viewing) the thread — dismiss its standing `done` card. */
function _seenThread(s: FeedState, turnId: number): void {
  s.done.delete(turnId);
}

/** Tear down an aborted/superseded live turn — its block, version, and all
 *  signal state — leaving the feed clean. */
function _dropLiveTurn(s: FeedState, turnId: number): void {
  delete s.blocks[turnId];
  delete s.versions[turnId];
  delete s.working[turnId];
  s.done.delete(turnId);
  clearLiveTurn(s.type, turnId);
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

async function _searchThreads(type: string, query: string): Promise<ConversationThread[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  return (await convoApi.threads(5, undefined, trimmed, type)).threads;
}

// ── Public API ────────────────────────────────────────────────────────────────

export interface ConversationFeedApi {
  /** The sorted turn blocks, reactive — bind directly to v-for. */
  readonly sortedBlocks: ComputedRef<ConversationTurnBlock[]>;
  /** The raw block map, reactive — keyed by turn_id. */
  readonly blocks: Record<number, ConversationTurnBlock>;
  /** Working-spinner flags (transient visual), reactive. */
  readonly working: Record<number, boolean>;
  readonly hasMore: boolean;

  upsertTurn(block: ConversationTurnBlock): void;
  /** Unguarded write — see `_forceUpsertTurn`. Only for the post-cancel reconcile path. */
  forceUpsertTurn(block: ConversationTurnBlock): void;
  fetchTurn(turnId: number): Promise<void>;
  loadRecent(): Promise<void>;
  loadMore(): Promise<void>;

  setWorking(turnId: number, on: boolean): void;
  markThreadDone(turnId: number): void;
  seenThread(turnId: number): void;
  dropLiveTurn(turnId: number): void;

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

function makeApi(type: string): ConversationFeedApi {
  const s = feedState(type);

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
    get hasMore() { return s.hasMore; },

    upsertTurn: (block) => _upsertTurn(s, block),
    forceUpsertTurn: (block) => _forceUpsertTurn(s, block),
    fetchTurn: (turnId) => _fetchTurn(s, turnId, type),
    loadRecent: () => _loadRecent(s, type),
    loadMore: () => _loadMore(s, type),

    setWorking: (turnId, on) => _setWorking(s, turnId, on),
    markThreadDone: (turnId) => _markThreadDone(s, turnId),
    seenThread: (turnId) => _seenThread(s, turnId),
    dropLiveTurn: (turnId) => _dropLiveTurn(s, turnId),

    isTurnWorking: (turnId) => _isTurnWorking(s, turnId),
    isForkedThread: (turnId) => _isForkedThread(s, turnId),
    threadPhase: (turnId) => _threadPhase(s, turnId),
    isHydrated: (turnId) => _isHydrated(s, turnId),
    isThreadActive: (lastActivityAt) => _isThreadActive(lastActivityAt),
    threadList: () => _threadList(s),
    searchThreads: (query) => _searchThreads(type, query),

    turnSpeechText: (turnId) => _turnSpeechText(s, turnId),
  };
}

const _apis = new Map<string, ConversationFeedApi>();

/** Type-keyed factory — safe to call outside setup(). */
export function useConversationFeed(type: string = ConfigType.USER): ConversationFeedApi {
  let api = _apis.get(type);
  if (!api) {
    api = makeApi(type);
    _apis.set(type, api);
  }
  return api;
}

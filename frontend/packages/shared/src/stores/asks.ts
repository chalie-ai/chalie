/**
 * Asks store — everything Chalie is waiting on the user to answer: a
 * permission ask (Deny/Allow), a question (pick an answer, or type one) or a
 * browser hand-off (Done on its tab).
 *
 * Two feeds, one queue keyed by `ask_id`: `route(frame)` takes each
 * `ask_opened` and `ask_closed` frame off the socket, and `refreshOpen()`
 * runs on every WS connect, so a reload or a dropped socket restores the asks
 * the frames would have queued and drops the ones whose close was missed (the
 * backend keeps waiting regardless — the frame is only the visual trigger).
 * Queueing dedupes on `ask_id`, so the live frame and the REST listing
 * overlapping is harmless. An ask without an origin (a run no chat turn
 * shows) is kept: its row has no origin prefix and no Open.
 *
 * In the chat app a permission ask or a question is shown and answered on the
 * live activity line of its origin's turn while that turn works. Each mounted
 * live line calls `markShown(askId)` on mount and `unmarkShown(askId)` on unmount — a
 * count, since the same turn can render twice, on the main chat and in the
 * thread panel — and the input dock lists `offScreen`, where each ask can be
 * answered too: the queued asks no mounted line is showing (a scheduled task,
 * a thread whose panel is closed, or a subagent asking after the turn that
 * started it ended).
 */
import { defineStore } from 'pinia';
import { api } from '../composables/useApiClient';
import { HttpError } from '../services/ApiClient';
import type { WsPushEvent } from '../services/WebSocketService';

/** The turn an ask belongs to, as the backend stamps it on the ask. */
export interface AskOrigin {
  /** ConfigType of the turn: `user` or `scheduled`. */
  type: string;
  /** The turn the asking call runs under (a schedule's item id on `scheduled`). */
  turn_id: number;
  /** True for a reply sent into a thread (the panel's turn), false for a main-spine turn. */
  forked: boolean;
}

export interface PermissionAskData {
  /** Permission key the policy gated, e.g. "bash.modify_file" or "email.send". */
  action_id: string;
  /** The model's one-line summary of the gated action; empty when the backend sent none. */
  summary: string;
  origin: AskOrigin | null;
}

export interface QuestionAnswer {
  /** The answer as the model wrote it; sent back as-is when picked. */
  label: string;
  /** One line explaining the answer; empty when the model gave none. */
  description: string;
}

export interface QuestionAskData {
  /** The question, as the user reads it. */
  question: string;
  /** The 2 to 4 answers to pick from, in the model's order. */
  answers: QuestionAnswer[];
  /** True lets the user pick more than one answer. */
  multi_select: boolean;
  origin: AskOrigin | null;
}

/** A question's answer: the labels picked, or the user's own words. */
export type QuestionReply = { labels: string[] } | { other: string };

export interface HandoffAskData {
  /** The browser tab handed to the user. */
  tab_id: string;
  /** What the user is asked to do there; empty when the backend sent none. */
  reason: string;
  origin: AskOrigin | null;
}

interface AskBase {
  /** Opaque ID; answers the ask on POST /api/asks/reply/<ask_id>. */
  ask_id: string;
  /** The instant the ask opened (ISO-8601 UTC with offset) — the "waiting" clock ages from it; empty when absent. */
  asked_at: string;
}

export interface PermissionAsk extends AskBase {
  kind: 'permission';
  data: PermissionAskData;
}

export interface QuestionAsk extends AskBase {
  kind: 'question';
  data: QuestionAskData;
}

export interface HandoffAsk extends AskBase {
  kind: 'handoff';
  data: HandoffAskData;
}

/** One open ask — an `ask_opened` frame or open-asks listing item as kept here. */
export type Ask = PermissionAsk | QuestionAsk | HandoffAsk;

interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

/**
 * Asks an open-asks read must not bring back, since its listing is a snapshot
 * that can predate their close: each read in flight collects the ids closed
 * while it runs, and `replying` holds the asks whose answer is still on its
 * way to the server.
 */
const closedDuringRead = new Set<Set<string>>();
const replying = new Set<string>();

function noteClosed(askId: string): void {
  for (const closed of closedDuringRead) closed.add(askId);
}

/**
 * How a failed reply is told to the user. This package has no toast, so each
 * app registers its own once at startup; until one does, a failed reply is
 * only logged.
 */
let reportReplyFailure: (message: string) => void = () => {};

export function registerReplyFailureReporter(report: (message: string) => void): void {
  reportReplyFailure = report;
}

/** The server's own words for a failed reply, or a short line for when it sent none. */
function replyFailureMessage(err: unknown, kind: Ask['kind'] | undefined): string {
  if (err instanceof HttpError && err.error) return err.error;
  return kind === 'handoff' ? "Couldn't finish the hand-off" : "Couldn't send your answer";
}

const idOf = (item: unknown): unknown => (item as { ask_id?: unknown } | null)?.ask_id;

/** The `origin` object as sent, or null when absent/malformed — never a guessed one. */
export function readOrigin(raw: unknown): AskOrigin | null {
  if (raw == null || typeof raw !== 'object') return null;
  const o = raw as Partial<AskOrigin>;
  if (typeof o.type !== 'string' || typeof o.turn_id !== 'number') return null;
  return { type: o.type, turn_id: o.turn_id, forked: o.forked === true };
}

const text = (v: unknown): string => (typeof v === 'string' ? v : '');

/** A question's answers as sent, or null unless there are 2 to 4 and each has a label. */
function readAnswers(raw: unknown): QuestionAnswer[] | null {
  if (!Array.isArray(raw) || raw.length < 2 || raw.length > 4) return null;
  const answers: QuestionAnswer[] = [];
  for (const item of raw) {
    const a = (item ?? {}) as { label?: unknown; description?: unknown };
    if (typeof a.label !== 'string' || !a.label) return null;
    answers.push({ label: a.label, description: text(a.description) });
  }
  return answers;
}

/**
 * An ask as sent, or null when it lacks what its row needs: an id, a known
 * kind, and its action, question and answers, or tab.
 */
function readAsk(raw: unknown): Ask | null {
  if (raw == null || typeof raw !== 'object') return null;
  const r = raw as { ask_id?: unknown; kind?: unknown; data?: unknown; asked_at?: unknown };
  if (typeof r.ask_id !== 'string' || !r.ask_id || r.data == null || typeof r.data !== 'object')
    return null;
  const d = r.data as Record<string, unknown>;
  const base = { ask_id: r.ask_id, asked_at: text(r.asked_at) };
  const origin = readOrigin(d.origin);
  if (r.kind === 'permission' && typeof d.action_id === 'string' && d.action_id) {
    return {
      ...base,
      kind: 'permission',
      data: { action_id: d.action_id, summary: text(d.summary), origin },
    };
  }
  const answers = r.kind === 'question' ? readAnswers(d.answers) : null;
  if (answers && typeof d.question === 'string' && d.question) {
    return {
      ...base,
      kind: 'question',
      data: { question: d.question, answers, multi_select: d.multi_select === true, origin },
    };
  }
  if (r.kind === 'handoff' && typeof d.tab_id === 'string' && d.tab_id) {
    return { ...base, kind: 'handoff', data: { tab_id: d.tab_id, reason: text(d.reason), origin } };
  }
  return null;
}

export const useAsksStore = defineStore('asks', {
  state: () => ({
    queue: [] as Ask[],
    /** ask_id → how many mounted live lines show that ask. */
    shown: {} as Record<string, number>,
    /**
     * ask_id → what the user has typed as their own answer to a question. Kept
     * here, not in the row, so it is still there when the row comes back after
     * an answer fails to send.
     */
    otherDrafts: {} as Record<string, string>,
  }),

  getters: {
    /** The permission asks and questions — the kinds answered on their own row — in queue order. */
    permissionsAndQuestions(state): (PermissionAsk | QuestionAsk)[] {
      return state.queue.filter((a): a is PermissionAsk | QuestionAsk => a.kind !== 'handoff');
    },
    /** The queued permission asks and questions no mounted live line is showing, in queue order. */
    offScreen(state): (PermissionAsk | QuestionAsk)[] {
      return this.permissionsAndQuestions.filter((a) => (state.shown[a.ask_id] ?? 0) <= 0);
    },
    handoffs(state): HandoffAsk[] {
      return state.queue.filter((a): a is HandoffAsk => a.kind === 'handoff');
    },
    /** The open hand-off on a browser tab, if any. */
    handoffOn(): (tabId: string) => HandoffAsk | null {
      return (tabId) => this.handoffs.find((a) => a.data.tab_id === tabId) ?? null;
    },
  },

  actions: {
    /** Take an `ask_opened` or `ask_closed` frame; any other frame is not an ask's. */
    route(frame: WsPushEvent): void {
      if (frame.type === 'ask_opened') this.enqueue(frame);
      else if (frame.type === 'ask_closed' && typeof frame.ask_id === 'string')
        this.remove(frame.ask_id);
    },

    /** Queue an ask unless it is already queued; one that cannot be read is dropped loudly. */
    enqueue(raw: unknown): void {
      const ask = readAsk(raw);
      if (ask == null) {
        console.error(
          '[asks] unreadable ask dropped:',
          (raw as { ask_id?: unknown } | null)?.ask_id,
        );
        return;
      }
      if (this.queue.some((a) => a.ask_id === ask.ask_id)) return;
      this.queue.push(ask);
    },

    /** A live line showing this ask mounted. */
    markShown(askId: string): void {
      this.shown[askId] = (this.shown[askId] ?? 0) + 1;
    },

    /** A live line showing this ask unmounted. */
    unmarkShown(askId: string): void {
      const n = (this.shown[askId] ?? 0) - 1;
      if (n <= 0) delete this.shown[askId];
      else this.shown[askId] = n;
    },

    /** The user's own answer to a question, as typed so far. */
    typeOther(askId: string, text: string): void {
      if (text) this.otherDrafts[askId] = text;
      else delete this.otherDrafts[askId];
    },

    /** Drop an ask — answered, cancelled or dismissed, here or elsewhere. No-op if absent. */
    remove(askId: string): void {
      this.queue = this.queue.filter((a) => a.ask_id !== askId);
      delete this.otherDrafts[askId];
      noteClosed(askId);
    },

    /**
     * Reconcile the queue with the open asks over REST — called on every WS
     * connect (initial load and reconnect) and after a failed `reply`. An ask
     * queued before the fetch that the listing no longer has is dropped (its
     * `ask_closed` frame was missed while the socket was down); a frame that
     * arrives while the fetch is in flight is not in that snapshot, so it is
     * never pruned; the listing's asks are queued, except those that closed
     * while the read ran or whose answer is still on its way. Best-effort: a
     * failed fetch leaves the queue as it is; the next connect retries.
     */
    async refreshOpen(): Promise<void> {
      const queuedBefore = new Set(this.queue.map((a) => a.ask_id));
      const closed = new Set<string>();
      closedDuringRead.add(closed);
      let open: unknown[];
      try {
        open = (await api.get<ListingEnvelope<unknown>>('/api/asks/open')).result;
      } catch (err) {
        console.warn('[asks] open asks fetch failed:', err);
        return;
      } finally {
        closedDuringRead.delete(closed);
      }
      const listed = new Set(open.map(idOf));
      this.queue = this.queue.filter((a) => listed.has(a.ask_id) || !queuedBefore.has(a.ask_id));
      for (const item of open) {
        const id = idOf(item);
        if (typeof id === 'string' && (closed.has(id) || replying.has(id))) continue;
        this.enqueue(item);
      }
    },

    /**
     * Answer an ask — a permission with true/false, a question with the
     * labels picked or the user's own words, a hand-off with null.
     * Optimistic: the row leaves before the round-trip. A 404 means the ask
     * was already answered or closed elsewhere (the first reply wins), so it
     * stays gone quietly. Any other failure is told to the user through the
     * registered reporter and re-reads the open asks, which brings the row
     * back only if the ask is still open, with the user's own answer still
     * typed in it.
     */
    async reply(askId: string, value: boolean | null | QuestionReply): Promise<void> {
      const kind = this.queue.find((a) => a.ask_id === askId)?.kind;
      const draft = this.otherDrafts[askId];
      this.remove(askId);
      replying.add(askId);
      let failure: unknown = null;
      try {
        await api.post(`/api/asks/reply/${encodeURIComponent(askId)}`, { value });
      } catch (err) {
        if (!(err instanceof HttpError && err.status === 404)) failure = err;
      } finally {
        replying.delete(askId);
      }
      if (failure == null) {
        noteClosed(askId);
        return;
      }
      console.warn('[asks] reply failed:', failure);
      if (draft) this.otherDrafts[askId] = draft;
      reportReplyFailure(replyFailureMessage(failure, kind));
      void this.refreshOpen();
    },
  },
});

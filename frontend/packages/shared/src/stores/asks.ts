/**
 * Asks store — everything Chalie is waiting on the user to answer: a
 * permission ask (Deny/Allow), a question (pick an answer, type one, or skip it), a
 * sign-in code ask (the code typed on its row) or a browser hand-off (Done on
 * its tab).
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
 * In the chat app a permission ask or a code ask is shown and answered on
 * the live activity line of its origin's turn while that turn works, and a
 * question waits there too. Each mounted live line calls `markShown(askId)` on
 * mount and `unmarkShown(askId)` on unmount — a count, since the same turn can
 * render twice, on the main chat and in the thread panel — and the input dock
 * lists `offScreen`, where each ask can be answered too: the permission and
 * code asks no mounted line is showing (a scheduled task, a thread whose panel
 * is closed, or a subagent asking after the turn that started it ended).
 * Every open question is answered on one card, `questions` paged through it,
 * above the input dock in the chat and pinned to the Brain's bottom edge.
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

/** A code ask's answer: the code the user typed. */
export type CodeReply = { code: string };

export interface HandoffAskData {
  /** The browser tab handed to the user. */
  tab_id: string;
  /** What the user is asked to do there; empty when the backend sent none. */
  reason: string;
  origin: AskOrigin | null;
}

export interface CodeAskData {
  /** The service whose sign-in waits on the code; its row's words are built from it. */
  service: 'unifi';
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

export interface CodeAsk extends AskBase {
  kind: 'code';
  data: CodeAskData;
}

/** One open ask — an `ask_opened` frame or open-asks listing item as kept here. */
export type Ask = PermissionAsk | QuestionAsk | HandoffAsk | CodeAsk;

/** An ask a turn's live line waits on: a permission, a question or a sign-in code. */
export type InlineAsk = PermissionAsk | QuestionAsk | CodeAsk;

/** An ask shown as a row of its own: every kind but a question, which has its card. */
export type RowAsk = PermissionAsk | HandoffAsk | CodeAsk;

interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

/**
 * Asks an open-asks read must not bring back, since its listing is a snapshot
 * that can predate their close: each read in flight collects the ids closed
 * while it runs, and `replying` holds the asks whose answer or dismissal is
 * still on its way to the server.
 */
const closedDuringRead = new Set<Set<string>>();
const replying = new Set<string>();

function noteClosed(askId: string): void {
  for (const closed of closedDuringRead) closed.add(askId);
}

/**
 * How a failed reply or dismissal is told to the user. This package has no
 * toast, so each app registers its own once at startup; until one does, a
 * failure is only logged.
 */
let reportReplyFailure: (message: string) => void = () => {};

export function registerReplyFailureReporter(report: (message: string) => void): void {
  reportReplyFailure = report;
}

/** The server's own words for a failed reply or dismiss, or `fallback` when it sent none. */
function failureMessage(err: unknown, fallback: string): string {
  return err instanceof HttpError && err.error ? err.error : fallback;
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
 * kind, and its action, question and answers, tab, or service.
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
  if (r.kind === 'code' && d.service === 'unifi') {
    return { ...base, kind: 'code', data: { service: 'unifi', origin } };
  }
  return null;
}

export const useAsksStore = defineStore('asks', {
  state: () => ({
    queue: [] as Ask[],
    /** ask_id → how many mounted live lines show that ask. */
    shown: {} as Record<string, number>,
    /**
     * ask_id → a question's answer as the user has it so far: the answers
     * picked on a multi-select question, or their own words as typed. Kept
     * here, not on the card, so it is still there when the question comes
     * back after its answer fails to send, or on another screen's card.
     */
    drafts: {} as Record<string, QuestionReply>,
  }),

  getters: {
    /** The permission asks, questions and code asks — the kinds a turn's live line waits on — in queue order. */
    inline(state): InlineAsk[] {
      return state.queue.filter((a): a is InlineAsk => a.kind !== 'handoff');
    },
    /** Every open ask but the questions, in queue order. */
    rows(state): RowAsk[] {
      return state.queue.filter((a): a is RowAsk => a.kind !== 'question');
    },
    /** The open questions, in queue order — the pages of their one card. */
    questions(state): QuestionAsk[] {
      return state.queue.filter((a): a is QuestionAsk => a.kind === 'question');
    },
    /** The permission asks and code asks no mounted live line is showing, in queue order. */
    offScreen(state): (PermissionAsk | CodeAsk)[] {
      return this.rows.filter(
        (a): a is PermissionAsk | CodeAsk => a.kind !== 'handoff' && (state.shown[a.ask_id] ?? 0) <= 0,
      );
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

    /** A question's answer as the user has it so far; none picked or typed clears it. */
    draft(askId: string, answer: QuestionReply): void {
      if ('labels' in answer ? answer.labels.length : answer.other) this.drafts[askId] = answer;
      else delete this.drafts[askId];
    },

    /** Drop an ask — answered, cancelled or dismissed, here or elsewhere. No-op if absent. */
    remove(askId: string): void {
      this.queue = this.queue.filter((a) => a.ask_id !== askId);
      delete this.drafts[askId];
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
     * labels picked or the user's own words, a code ask with the code as
     * typed (sent in the body only, never kept or logged here), a hand-off
     * with null.
     */
    async reply(askId: string, value: boolean | null | QuestionReply | CodeReply): Promise<void> {
      const kind = this.queue.find((a) => a.ask_id === askId)?.kind;
      await this.settle(
        askId,
        `/api/asks/reply/${encodeURIComponent(askId)}`,
        { value },
        kind === 'handoff' ? "Couldn't finish the hand-off" : "Couldn't send your answer",
      );
    },

    /** Close a question unanswered — Skip or the close button on its card. */
    async dismiss(askId: string): Promise<void> {
      await this.settle(askId, `/api/asks/dismiss/${encodeURIComponent(askId)}`, undefined, "Couldn't skip the question");
    },

    /**
     * The one round-trip `reply` and `dismiss` share. Optimistic: the ask
     * leaves before it. A 404 means the ask was already answered or closed
     * elsewhere (the first reply wins), so it stays gone quietly. Any other
     * failure is told to the user through the registered reporter — the
     * server's reason, else `fallback` — and re-reads the open asks, which
     * brings the ask back only if it is still open, with a question's answer
     * still picked or typed in it.
     */
    async settle(askId: string, path: string, body: unknown, fallback: string): Promise<void> {
      const draft = this.drafts[askId];
      this.remove(askId);
      replying.add(askId);
      let failure: unknown = null;
      try {
        await api.post(path, body);
      } catch (err) {
        if (!(err instanceof HttpError && err.status === 404)) failure = err;
      } finally {
        replying.delete(askId);
      }
      if (failure == null) {
        noteClosed(askId);
        return;
      }
      console.warn(`[asks] ${path} failed:`, failure);
      if (draft) this.drafts[askId] = draft;
      reportReplyFailure(failureMessage(failure, fallback));
      void this.refreshOpen();
    },
  },
});

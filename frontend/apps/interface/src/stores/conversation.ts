/**
 * Conversation spine store — the ordered `forms` array of discriminated-union
 * items (user / chalie / act). Errors are not feed rows: every error — turn,
 * action, voice — funnels to `session.errorMessage` and renders as the one
 * dock toast. Components render the list; this store owns all mutations.
 */
import { defineStore } from 'pinia';
import type { ConversationAttachment, ConversationMessage, ConversationSegment, ConversationThread } from '../api/conversation';
import { conversation as convoApi } from '../api/conversation';
import { extractText } from '../composables/useMarkup';

export interface AttachmentPreview {
  filename: string;
  objectUrl: string | null;
  isImage: boolean;
}

export interface ChalieMeta {
  topic?: string;
  exchange_id?: string;
  mode?: string;
  confidence?: number;
  segments?: ConversationSegment[];
  ts?: string;
  duration_ms?: number;
  /** Event type from drift (thought, response, escalation, drift). */
  type?: string;
}

export interface ToolPill {
  id: string;
  name: string;
  summary: string | undefined;
  startedAt: number;
  ms?: number;
  ok?: boolean;
  resolved: boolean;
}

/**
 * The backend turn this form belongs to. Set on forms reconstructed from
 * history; the `turns` getter groups by it (so a compaction-continuation turn
 * with no leading user row is still its own group). Live forms leave it unset
 * and fall back to user-form-boundary grouping.
 */
type TurnTagged = { turnId?: number | null };

export interface UserForm extends TurnTagged {
  kind: 'user';
  id: number;
  text: string;
  attachments?: AttachmentPreview[];
  /** Whether the turn is within working memory (faded if false). */
  inWorkingMemory?: boolean;
}

export interface ChalieForm extends TurnTagged {
  kind: 'chalie';
  id: number;
  text?: string;
  segments?: ConversationSegment[];
  meta: ChalieMeta;
  escalation?: boolean;
  inWorkingMemory?: boolean;
}

export interface ActForm extends TurnTagged {
  kind: 'act';
  id: number;
  tools: ToolPill[];
  /**
   * True once the step is superseded — its tools are done AND the next prose
   * bubble (or the final reply) has landed. Collapsed groups render summary-only:
   * the tool-name pill and running timer are dropped, leaving the act summaries.
   */
  collapsed: boolean;
}

export type ConversationForm = UserForm | ChalieForm | ActForm;

/** A turn: a user form plus the assistant rows / act groups it produced. */
export interface Turn {
  id: number;
  forms: ConversationForm[];
}

/** Collapsed thread row in the thread-list feed. */
export interface ThreadListItem extends ConversationThread {
  /** True once the thread's full rows have been loaded into `forms`. */
  expanded: boolean;
  loading: boolean;
}

/**
 * Plain-text (TTS / remember) for one Chalie form — segments concatenated, else
 * the single text path, both markup-stripped. Shared by the speak/remember
 * buttons and turn-level speech aggregation so the two never diverge.
 */
export function chalieFormPlaintext(form: ChalieForm): string {
  const segs = form.meta.segments ?? form.segments;
  if (segs?.length) {
    return extractText(segs.map((s) => s.content ?? s.synthesis ?? '').join(' '));
  }
  return extractText(form.text ?? '');
}

let _nextId = 1;
function nextId(): number {
  return _nextId++;
}

export const useConversationStore = defineStore('conversation', {
  state: () => ({
    forms: [] as ConversationForm[],
    /** Collapsed thread metadata from /conversation/threads — the thread feed. */
    threads: [] as ThreadListItem[],
    /** Total threads seen across all loaded pages — advances pagination. */
    threadsOffset: 0,
    threadsExhausted: false,
  }),

  getters: {
    /**
     * Collapsed turn_ids (thread list items not expanded). Forms belonging to
     * these are excluded from `turns` so collapsed threads render as gist rows.
     */
    collapsedTurnIds(state): Set<number | null> {
      const ids = new Set<number | null>();
      for (const t of state.threads) {
        if (!t.expanded && t.turn_id != null) ids.add(t.turn_id);
      }
      return ids;
    },
    /**
     * Group forms into turns. History forms carry `turnId`, so a boundary is any
     * change of it — this keeps a compaction-continuation turn (assistant rows,
     * no leading user row) as its own group. Live forms have no `turnId` and
     * fall back to: a turn starts at each user form. Forms belonging to a
     * collapsed thread list item are excluded — they render as gist rows.
     */
    turns(state): Turn[] {
      const collapsed = this.collapsedTurnIds;
      const groups: Turn[] = [];
      let key: number | null | undefined;
      for (const f of state.forms) {
        if (f.turnId != null && collapsed.has(f.turnId)) continue;
        const boundary =
          groups.length === 0 ||
          (f.turnId != null ? f.turnId !== key : f.kind === 'user');
        if (boundary) groups.push({ id: f.id, forms: [] });
        key = f.turnId;
        groups[groups.length - 1].forms.push(f);
      }
      return groups;
    },
    /** Id of the most recent act form, or null. */
    activeActFormId(state): number | null {
      for (let i = state.forms.length - 1; i >= 0; i--) {
        const f = state.forms[i];
        if (f.kind === 'act') return f.id;
      }
      return null;
    },
    /**
     * True when a thread's last activity was within the 1-hour active window.
     * Display/ordering only — no behavioral branch (spec §4.B).
     */
    isThreadActive(): (lastActivityAt: string | null) => boolean {
      return (lastActivityAt) => {
        if (!lastActivityAt) return false;
        const ts = new Date(lastActivityAt).getTime();
        if (Number.isNaN(ts)) return false;
        return Date.now() - ts < 3_600_000;
      };
    },
    /** Id of the most recent chalie form, or null. */
    activeChalieFormId(state): number | null {
      for (let i = state.forms.length - 1; i >= 0; i--) {
        const f = state.forms[i];
        if (f.kind === 'chalie') return f.id;
      }
      return null;
    },
    /**
     * True when `formId` is the LAST Chalie row in its turn. A turn can hold
     * several assistant rows; the per-turn remember/speak controls appear once,
     * on that final row, so they act on the whole turn.
     */
    isLastChalieInTurn(): (formId: number) => boolean {
      return (formId) => {
        const turn = this.turns.find((t) => t.forms.some((f) => f.id === formId));
        const chalie = turn?.forms.filter((f) => f.kind === 'chalie') ?? [];
        return chalie.length > 0 && chalie[chalie.length - 1].id === formId;
      };
    },
  },

  actions: {
    appendUser(
      text: string,
      attachments?: AttachmentPreview[],
      opts?: { inWorkingMemory?: boolean; turnId?: number | null },
    ): number {
      const id = nextId();
      const form: UserForm = {
        kind: 'user',
        id,
        text,
        attachments: attachments ?? [],
        inWorkingMemory: opts?.inWorkingMemory ?? true,
        turnId: opts?.turnId,
      };
      this.forms.push(form);
      return id;
    },

    /** When `meta.segments` is set the form carries segments; otherwise `text`. */
    appendChalie(
      textOrContent: string,
      meta: ChalieMeta,
      opts?: { escalation?: boolean; inWorkingMemory?: boolean },
    ): number {
      const id = nextId();
      const form: ChalieForm = {
        kind: 'chalie',
        id,
        text: meta.segments ? undefined : textOrContent,
        segments: meta.segments,
        meta,
        escalation: opts?.escalation ?? false,
        inWorkingMemory: opts?.inWorkingMemory ?? true,
      };
      this.forms.push(form);
      return id;
    },

    /** Create a new (live, expanded) ACT tool-group and return its id. */
    appendAct(): number {
      const id = nextId();
      const form: ActForm = {
        kind: 'act',
        id,
        tools: [],
        collapsed: false,
      };
      this.forms.push(form);
      return id;
    },

    /**
     * Settle a step's tool group when its prose / final reply lands: an empty
     * group — the up-front "thinking…" placeholder that never ran a tool — is
     * evicted; one that ran tools collapses to summary-only.
     */
    resolveAct(actId: number): void {
      const idx = this.forms.findIndex((f) => f.id === actId);
      const form = idx === -1 ? undefined : this.forms[idx];
      if (form?.kind !== 'act') return;
      if (form.tools.length) form.collapsed = true;
      else this.forms.splice(idx, 1);
    },

    appendToolPill(actId: number, id: string, name: string, summary?: string): void {
      const form = this._findAct(actId);
      if (!form) return;
      if (!id) return; // guard matches renderer.js
      const pill: ToolPill = {
        id,
        name,
        summary,
        startedAt: Date.now(),
        resolved: false,
      };
      form.tools.push(pill);
    },

    /**
     * Enforces a 150ms minimum visible duration. When `ms` is 0, falls back to
     * client-measured elapsed (from pill.startedAt).
     */
    resolveToolPill(pillId: string, ms: number, ok: boolean): void {
      if (!pillId) return;
      for (const form of this.forms) {
        if (form.kind !== 'act') continue;
        const pill = form.tools.find((t) => t.id === pillId);
        if (!pill) continue;

        const elapsed = pill.startedAt ? Date.now() - pill.startedAt : 200;
        const wait = Math.max(0, 150 - elapsed);
        const effectiveMs = ms > 0 ? ms : (pill.startedAt ? Date.now() - pill.startedAt : 0);

        setTimeout(() => {
          // Re-locate: forms array may have shifted.
          for (const f of this.forms) {
            if (f.kind !== 'act') continue;
            const p = f.tools.find((t) => t.id === pillId);
            if (p) {
              p.ms = effectiveMs;
              p.ok = ok;
              p.resolved = true;
              break;
            }
          }
        }, wait);
        break;
      }
    },

    /**
     * Replace an ACT form with the final Chalie response. A no-content message
     * simply removes the ACT form.
     */
    replaceActWithResponse(
      actId: number,
      data: {
        content?: string;
        topic?: string;
        exchange_id?: string;
        mode?: string;
        confidence?: number;
        segments?: ConversationSegment[];
        timestamp?: string;
        duration_ms?: number;
      },
    ): number | null {
      const idx = this.forms.findIndex((f) => f.id === actId);
      if (idx === -1) return null;

      const content = data.content ?? '';
      if (!content && !data.segments?.length) {
        this.forms.splice(idx, 1);
        return null;
      }

      const meta: ChalieMeta = {
        topic: data.topic,
        exchange_id: data.exchange_id,
        mode: data.mode ?? '',
        confidence: data.confidence ?? 0,
        segments: data.segments ?? undefined,
        ts: data.timestamp ?? '',
        duration_ms: data.duration_ms,
      };

      const id = nextId();
      const chalie: ChalieForm = {
        kind: 'chalie',
        id,
        text: meta.segments ? undefined : content,
        segments: meta.segments,
        meta,
        inWorkingMemory: true,
      };
      this.forms.splice(idx, 1, chalie);
      return id;
    },

    /**
     * Concatenated plaintext of every Chalie form in `formId`'s turn — so speak
     * on any bubble plays the whole turn, not just the row clicked.
     */
    turnSpeechText(formId: number): string {
      const turn = this.turns.find((t) => t.forms.some((f) => f.id === formId));
      if (!turn) return '';
      return turn.forms
        .filter((f): f is ChalieForm => f.kind === 'chalie')
        .map(chalieFormPlaintext)
        .filter(Boolean)
        .join(' ');
    },

    _findAct(actId: number): ActForm | undefined {
      const form = this.forms.find((f) => f.id === actId);
      return form?.kind === 'act' ? form : undefined;
    },

    _attachmentsFor(msg: ConversationMessage): AttachmentPreview[] {
      return (msg.attachments ?? []).map((a: ConversationAttachment) => ({
        filename: a.filename,
        objectUrl: a.url,
        isImage: a.is_image,
      }));
    },

    /**
     * Forms for one assistant row in display order: prose bubble (when content/
     * segments) then a collapsed tool group (when the row drove tools). Each row
     * owns its own tools, so refresh reconstructs the live path's prose-then-
     * collapsed-summaries shape. A tools-only step yields just the act group.
     */
    _assistantForms(msg: ConversationMessage, inWorkingMemory: boolean): ConversationForm[] {
      const out: ConversationForm[] = [];
      if (msg.content || msg.segments?.length) {
        const meta: ChalieMeta = { ts: msg.timestamp };
        if (msg.segments) meta.segments = msg.segments;
        out.push({
          kind: 'chalie',
          id: nextId(),
          text: msg.segments ? undefined : msg.content || '',
          segments: msg.segments,
          meta,
          inWorkingMemory,
          turnId: msg.turn_id,
        });
      }
      if (msg.tool_calls?.length) {
        const tools: ToolPill[] = msg.tool_calls.map((c) => ({
          id: `hist-${nextId()}`,
          name: c.tool_name,
          summary: c.summary || undefined,
          startedAt: 0,
          ok: true,
          resolved: true,
        }));
        out.push({ kind: 'act', id: nextId(), tools, collapsed: true, turnId: msg.turn_id });
      }
      return out;
    },

    // ---- Thread-list feed (workstream F) ----

    /** Append collapsed thread metadata from a /conversation/threads page. */
    appendThreadList(items: ConversationThread[]): void {
      for (const t of items) {
        this.threads.push({ ...t, expanded: false, loading: false });
      }
    },

    /** Prepend collapsed thread metadata from a paginated /conversation/threads page. */
    prependThreadList(items: ConversationThread[]): void {
      for (let i = items.length - 1; i >= 0; i--) {
        this.threads.unshift({ ...items[i], expanded: false, loading: false });
      }
    },

    /** Expand a collapsed thread: fetch its full rows and insert into `forms`. */
    async expandThread(turnId: number): Promise<void> {
      const item = this.threads.find((t) => t.turn_id === turnId);
      if (!item || item.expanded || item.loading) return;
      item.loading = true;
      try {
        const data = await convoApi.thread(turnId);
        const messages = data.messages ?? [];
        // Insert the thread's forms at the correct position — after the last
        // form of the previous expanded thread and before the first form of the
        // next expanded thread. Find insertion index by scanning `forms` for the
        // boundary: we insert before the first form whose turnId is greater than
        // turnId (or at the end).
        const forms: ConversationForm[] = [];
        for (const msg of messages) {
          if (msg.role === 'user') {
            const attachments = this._attachmentsFor(msg);
            forms.push({
              kind: 'user',
              id: nextId(),
              text: msg.content,
              attachments,
              inWorkingMemory: true,
              turnId: msg.turn_id,
            });
          } else {
            for (const f of this._assistantForms(msg, true)) forms.push(f);
          }
        }
        // Insert forms into the forms array at the right position.
        const insertIdx = this._threadInsertIndex(turnId);
        this.forms.splice(insertIdx, 0, ...forms);
        item.expanded = true;
      } finally {
        item.loading = false;
      }
    },

    /** Collapse an expanded thread: remove its forms from `forms`. */
    collapseThread(turnId: number): void {
      const item = this.threads.find((t) => t.turn_id === turnId);
      if (!item || !item.expanded) return;
      this.forms = this.forms.filter((f) => f.turnId !== turnId);
      item.expanded = false;
    },

    /** Toggle expand/collapse for a thread. */
    async toggleThread(turnId: number | null): Promise<void> {
      if (turnId == null) return;
      const item = this.threads.find((t) => t.turn_id === turnId);
      if (!item) return;
      if (item.expanded) this.collapseThread(turnId);
      else await this.expandThread(turnId);
    },

    /** True when a live turn is in-flight (forms exist without a matching thread list item). */
    hasLiveTurn(): boolean {
      // A live turn has forms with turnId not present in any thread list item.
      const known = new Set(this.threads.map((t) => t.turn_id).filter((t): t is number => t != null));
      return this.forms.some((f) => f.turnId != null && !known.has(f.turnId));
    },

    /**
     * Find the insertion index in `forms` for a thread's rows. Threads render in
     * the same order as the thread list (most-recent first). Forms for a thread
     * go before any form whose turnId maps to a thread that's earlier in the
     * list (higher recency).
     */
    _threadInsertIndex(turnId: number): number {
      // Build a map of turnId → list position for expanded threads.
      const expandedOrder = new Map<number, number>();
      for (let i = 0; i < this.threads.length; i++) {
        const t = this.threads[i];
        if (t.turn_id != null && t.expanded) expandedOrder.set(t.turn_id, i);
      }
      const myOrder = expandedOrder.get(turnId) ?? this.threads.findIndex((t) => t.turn_id === turnId);
      // Find the first form belonging to a thread that comes AFTER this thread
      // in the list — insert before it.
      for (let i = 0; i < this.forms.length; i++) {
        const fTurn = this.forms[i].turnId;
        if (fTurn == null) continue;
        const fOrder = expandedOrder.get(fTurn);
        if (fOrder != null && fOrder > myOrder) return i;
      }
      return this.forms.length;
    },
  },
});

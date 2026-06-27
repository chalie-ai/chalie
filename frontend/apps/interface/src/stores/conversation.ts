/**
 * Conversation spine store — the ordered `forms` array of discriminated-union
 * items (user / chalie / act). Errors are not feed rows: every error — turn,
 * action, voice — funnels to `session.errorMessage` and renders as the one
 * dock toast. Components render the list; this store owns all mutations.
 */
import { defineStore } from 'pinia';
import type { ConversationAttachment, ConversationMessage, ConversationSegment, ConversationThread, ConversationTurnBlock } from '../api/conversation';
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
type TurnTagged = { turnId?: number | null; threadMessage?: boolean };

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
  /**
   * Sub-text for a live working anchor — the transient `tool_invoked` summary
   * ("Searching files…"), shown under the spinner while the turn runs. Set only
   * on the synthetic anchor TurnView renders for a working turn; falls back to
   * "thinking…" when empty.
   */
  placeholder?: string;
}

export type ConversationForm = UserForm | ChalieForm | ActForm;

/** A turn: a user form plus the assistant rows / act groups it produced. */
export interface Turn {
  id: number;
  forms: ConversationForm[];
}

/** Thread row in the feed — metadata for the opener; the rows themselves live
 *  in `forms`. */
export type ThreadListItem = ConversationThread;

/**
 * Plain-text (TTS) for one Chalie form — segments concatenated, else the single
 * text path, both markup-stripped. Shared by the speak button and turn-level
 * speech aggregation so the two never diverge.
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
    /** Collapsed thread metadata from /api/threads — the thread feed. */
    threads: [] as ThreadListItem[],
    /** Total threads seen across all loaded pages — advances pagination. */
    threadsOffset: 0,
    threadsExhausted: false,
    /** turn_ids currently in flight (the `working` signal is on, no `done` yet) —
     *  drives the spinner anchor. Independent of the rows, which arrive via fetch. */
    workingTurnIds: new Set<number>(),
    /** Transient per-turn `tool_invoked` summary ("Searching files…"), shown under
     *  the working anchor. Dropped when the turn settles (setWorking false). */
    liveSummaries: {} as Record<number, string>,
    /** Highest row id rendered per turn — the monotonic guard that drops a stale
     *  or out-of-order refetch so a slow response can't overwrite a newer one. */
    turnVersions: {} as Record<number, number>,
  }),

  getters: {
    /**
     * Group forms into turns. History forms carry `turnId`, so a boundary is any
     * change of it — this keeps a compaction-continuation turn (assistant rows,
     * no leading user row) as its own group. Live forms have no `turnId` and
     * fall back to: a turn starts at each user form.
     */
    turns(state): Turn[] {
      const groups: Turn[] = [];
      let key: number | null | undefined;
      for (const f of state.forms) {
        const boundary =
          groups.length === 0 ||
          (f.turnId != null ? f.turnId !== key : f.kind === 'user');
        if (boundary) groups.push({ id: f.id, forms: [] });
        key = f.turnId;
        groups[groups.length - 1].forms.push(f);
      }
      return groups;
    },
    /** True while `turnId`'s `working` signal is on (spinner anchor visible). */
    isTurnWorking(state): (turnId: number | null) => boolean {
      return (turnId) => turnId != null && state.workingTurnIds.has(turnId);
    },
    /** Transient `tool_invoked` summary for `turnId` (empty when none). */
    liveSummaryFor(state): (turnId: number | null) => string {
      return (turnId) => (turnId != null ? state.liveSummaries[turnId] ?? '' : '');
    },
    /**
     * True when a thread's last activity was within the 1-hour active window.
     * Display/ordering only — no behavioral branch (spec §4.B).
     */
    isThreadActive(): (lastActivityAt: string | null) => boolean {
      return (lastActivityAt) => {
        if (!lastActivityAt) return false;
        // SQLite stores naive UTC ("YYYY-MM-DD HH:MM:SS"); mark it as UTC so the
        // browser doesn't read it as local time.
        const ts = new Date(`${lastActivityAt.replace(' ', 'T')}Z`).getTime();
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
      opts?: { escalation?: boolean; inWorkingMemory?: boolean; turnId?: number | null },
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
        turnId: opts?.turnId,
      };
      this.forms.push(form);
      return id;
    },

    /** Create a new (live, expanded) ACT tool-group and return its id. A thread
     *  reply tags it with the thread's turn_id so the live trail renders inside
     *  that thread; a new thread leaves it null until bindLiveTurn assigns one. */
    appendAct(turnId: number | null = null): number {
      const id = nextId();
      const form: ActForm = {
        kind: 'act',
        id,
        tools: [],
        collapsed: false,
        turnId,
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

    // ---- Turn lifecycle signals ----

    /** Flip a turn's spinner on/off. Settling also drops its transient summary. */
    setWorking(turnId: number, on: boolean): void {
      if (on) {
        this.workingTurnIds.add(turnId);
      } else {
        this.workingTurnIds.delete(turnId);
        delete this.liveSummaries[turnId];
      }
    },

    /** Write the transient `tool_invoked` summary straight onto the anchor — no
     *  fetch: a step in progress never triggers a refetch. */
    setLiveSummary(turnId: number, summary: string): void {
      this.liveSummaries[turnId] = summary;
    },

    /** Project one turn block into ordered forms — the user row then each
     *  assistant row's prose + collapsed tool group. Shared by live refetch and
     *  thread expand so both paths render identically. */
    parseTurn(block: ConversationTurnBlock): ConversationForm[] {
      const forms: ConversationForm[] = [];
      for (const msg of block.messages) {
        const made: ConversationForm[] =
          msg.role === 'user'
            ? [{
                kind: 'user',
                id: nextId(),
                text: msg.content,
                attachments: this._attachmentsFor(msg),
                inWorkingMemory: true,
                turnId: msg.turn_id,
              }]
            : this._assistantForms(msg, true);
        // Every row past settle0 is a thread reply — the spine drops these; their
        // presence is what makes the turn a thread (the opener appears).
        if (msg.thread_message) for (const f of made) f.threadMessage = true;
        forms.push(...made);
      }
      return forms;
    },

    /**
     * Atomic block replacement for one turn: drop the turn's current forms and
     * splice in the freshly-parsed block at its turn_id-ordered slot. A monotonic
     * row-id guard drops a stale/out-of-order payload so a slow fetch can't undo a
     * newer one; an equal version re-applies (idempotent self-heal). The matching
     * thread item's collapsed metadata is refreshed from the same block. Working
     * state is untouched — that is the `working`/`done` signals' job.
     */
    upsertTurn(block: ConversationTurnBlock): void {
      let version = 0;
      for (const m of block.messages) {
        const n = parseInt(m.id, 10);
        if (n > version) version = n;
      }
      if ((this.turnVersions[block.turn_id] ?? -1) > version) return;
      this.turnVersions[block.turn_id] = version;

      const incoming = this.parseTurn(block);
      this.forms = this.forms.filter((f) => f.turnId !== block.turn_id);
      this.forms.splice(this._turnInsertIndex(block.turn_id), 0, ...incoming);

      const item = this.threads.find((t) => t.turn_id === block.turn_id);
      if (item) {
        item.last_activity_at = block.last_activity_at;
        item.preview = block.preview;
        item.gist = block.gist;
      }
    },

    /** Fetch one turn block and upsert it — the single read behind both the WS
     *  `turn_updated` refetch and thread expand. */
    async refetchTurn(turnId: number): Promise<void> {
      this.upsertTurn(await convoApi.thread(turnId));
    },

    /** Tear down an aborted/superseded live turn — its forms, thread shell and
     *  all signal state — leaving the feed clean. */
    dropLiveTurn(turnId: number): void {
      this.forms = this.forms.filter((f) => f.turnId !== turnId);
      this.threads = this.threads.filter((t) => t.turn_id !== turnId);
      this.workingTurnIds.delete(turnId);
      delete this.liveSummaries[turnId];
      delete this.turnVersions[turnId];
    },

    /** Slot for a turn's forms in turn_id order (spine order is turn_id, not
     *  arrival) — before the first form of a later turn or any unbound live
     *  form, else at the end. */
    _turnInsertIndex(turnId: number): number {
      for (let i = 0; i < this.forms.length; i++) {
        const t = this.forms[i].turnId;
        if (t == null || t > turnId) return i;
      }
      return this.forms.length;
    },

    // ---- Thread-list feed (workstream F) ----

    /**
     * Append a /api/threads page (newest-first from the API) as the
     * initial feed. The feed reads chronologically — oldest at the top, newest at
     * the bottom — so the page is reversed before it is pushed.
     */
    appendThreadList(items: ConversationThread[]): void {
      for (const t of [...items].reverse()) {
        this.threads.push({ ...t });
      }
    },

    /**
     * Promote the just-finished live turn into its own thread: tag every still-
     * untagged form with the allocated `turnId` (the backend `done` event carries
     * it) and register a ThreadListItem. Without this, every live turn's forms
     * stay turn_id-less and collapse into one flat `liveTurn` with no reply box —
     * so the feed reads as one giant thread. Idempotent: a reply turn whose forms
     * already carry `turnId` only ensures the item exists.
     */
    bindLiveTurn(turnId: number): void {
      let preview = '';
      for (const f of this.forms) {
        if (f.turnId == null) {
          f.turnId = turnId;
          if (f.kind === 'user' && !preview) preview = f.text;
        }
      }
      if (this.threads.some((t) => t.turn_id === turnId)) return;
      // SQLite naive-UTC shape ("YYYY-MM-DD HH:MM:SS") so isThreadActive reads it.
      const nowUtc = new Date().toISOString().slice(0, 19).replace('T', ' ');
      this.threads.push({
        turn_id: turnId,
        last_activity_at: nowUtc,
        last_row_id: 0,
        row_count: 0,
        preview,
        gist: null,
      });
    },

    /**
     * Prepend an older /api/threads page (newest-first from the API)
     * above the current feed, keeping the chronological order: the page's oldest
     * thread lands at the very top.
     */
    prependThreadList(items: ConversationThread[]): void {
      for (const t of items) {
        this.threads.unshift({ ...t });
      }
    },

    /** True once a turn's rows are in `forms` (batch-hydrated on load or fetched
     *  for a deep-link open). */
    isHydrated(turnId: number): boolean {
      return this.forms.some((f) => f.turnId === turnId);
    },
  },
});

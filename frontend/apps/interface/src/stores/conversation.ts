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
    /** turn_ids whose reply has settled (`done`) but the user hasn't viewed yet —
     *  the Activity dock's "done" (blue) state. Cleared when the thread is opened
     *  (`seenThread`) or re-activated by a new `working`. */
    doneTurnIds: new Set<number>(),
    /**
     * Live act-trail pills for the in-flight step, keyed by their
     * `transcript_row_id` — the assistant row driving the calls (spec §6.5). Pushed
     * on `tool_called`, resolved on `tool_done`, shown with a running timer. The
     * live↔persisted link IS this row id: once the row's `tool_calls` persist,
     * `upsertTurn` drops its entry and the refetched summaries take over — a
     * data-driven handoff, never a state machine. `turnId` rides each entry so the
     * turn can collect its live trails for render without a row→turn lookup.
     */
    liveTools: {} as Record<number, { turnId: number; pills: ToolPill[] }>,
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
    /** A forked thread carries reply rows past settle0 (`thread_message` forms) —
     *  the same test the feed uses to decide a turn earns a Thread pill. Only these
     *  surface in the Activity dock; a plain spine turn (watched inline) does not. */
    isForkedThread(state): (turnId: number | null) => boolean {
      return (turnId) =>
        turnId != null && state.forms.some((f) => f.turnId === turnId && f.threadMessage);
    },
    /** A forked thread's Activity phase: `working` (reply streaming, pink) →
     *  `done` (settled but unseen, blue) → null (seen / no activity). Drives both
     *  the dock card and the spine pill colour. */
    threadPhase(state): (turnId: number | null) => 'working' | 'done' | null {
      return (turnId) => {
        if (turnId == null) return null;
        if (state.workingTurnIds.has(turnId)) return 'working';
        if (state.doneTurnIds.has(turnId)) return 'done';
        return null;
      };
    },
    /** Live act-trail trails for `turnId`'s in-flight step — one entry per
     *  transcript row carrying unsettled pills (spec §6.5). Empty when none. The
     *  in-flight row is always the latest, so the view paints these at the turn's
     *  tail, exactly where the persisted ActForm will land on the next `updated`. */
    liveTrailsFor(state): (turnId: number | null) => { rowId: number; pills: ToolPill[] }[] {
      return (turnId) =>
        turnId == null
          ? []
          : Object.entries(state.liveTools)
              .filter(([, v]) => v.turnId === turnId)
              .map(([rowId, v]) => ({ rowId: Number(rowId), pills: v.pills }));
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

    /** Flip a turn's spinner on/off. Settling also drops any live act-trail pills
     *  the persisting refetch did not already clear (the §6.5 tail floor). */
    setWorking(turnId: number, on: boolean): void {
      if (on) {
        this.workingTurnIds.add(turnId);
        this.doneTurnIds.delete(turnId);
      } else {
        this.workingTurnIds.delete(turnId);
        this._clearLiveTurn(turnId);
      }
    },

    /** Settle a background thread reply into the Activity dock's `done` (blue)
     *  state: stop its spinner, drop its live pills, and leave a standing card
     *  until the user opens the thread. The notify-on-reply path `setWorking`
     *  can't carry. */
    markThreadDone(turnId: number): void {
      this.workingTurnIds.delete(turnId);
      this.doneTurnIds.add(turnId);
      this._clearLiveTurn(turnId);
    },

    /** The user opened (or is viewing) the thread — clear its Activity state so the
     *  dock card and pill colour fall back to idle. */
    seenThread(turnId: number): void {
      this.workingTurnIds.delete(turnId);
      this.doneTurnIds.delete(turnId);
    },

    /** Push a live act-trail pill the instant a tool call begins (spec §6.5 step 2),
     *  starting its running timer — NO fetch. Keyed by `transcript_row_id`: the row
     *  the pill attaches to and whose persisted summary later supersedes it. `id` is
     *  the opened `tool_calls` row that the matching `tool_done` resolves. */
    startLiveTool(turnId: number, rowId: number | null, id: number | null, name: string, summary?: string): void {
      if (rowId == null || id == null) return;
      const pill: ToolPill = {
        id: String(id), name, summary, startedAt: Date.now(), ok: false, resolved: false,
      };
      const trail = this.liveTools[rowId] ?? { turnId, pills: [] };
      this.liveTools[rowId] = { turnId, pills: [...trail.pills, pill] };
    },

    /** Resolve a live pill on `tool_done` (spec §6.5 step 3) — stop its timer and
     *  freeze the measured duration. Matched by the unique `tool_calls` row `id`, so
     *  no row id is needed. No-op if already cleared by the persisting refetch. */
    finishLiveTool(id: number | null): void {
      if (id == null) return;
      const key = String(id);
      for (const [rowId, trail] of Object.entries(this.liveTools)) {
        if (!trail.pills.some((p) => p.id === key && !p.resolved)) continue;
        this.liveTools[Number(rowId)] = {
          turnId: trail.turnId,
          pills: trail.pills.map((p) =>
            p.id === key ? { ...p, resolved: true, ok: true, ms: Date.now() - p.startedAt } : p,
          ),
        };
        return;
      }
    },

    /** Drop every live trail belonging to a turn — the settle/abort floor; the
     *  normal handoff is the per-row drop in `upsertTurn` (spec §6.5 step 4). */
    _clearLiveTurn(turnId: number): void {
      for (const [rowId, v] of Object.entries(this.liveTools)) {
        if (v.turnId === turnId) delete this.liveTools[Number(rowId)];
      }
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

      // §6.5 step 4 — the live↔persisted handoff. Any row whose tool_calls now
      // persist no longer needs its live pills: the just-rendered summaries are
      // authoritative, so drop that row's trail. Driven by the data arriving, not
      // by a signal or a manual dedup — so nothing ever double-renders.
      for (const m of block.messages) {
        if (m.tool_calls?.length) delete this.liveTools[parseInt(m.id, 10)];
      }

      const item = this.threads.find((t) => t.turn_id === block.turn_id);
      if (item) {
        item.last_activity_at = block.last_activity_at;
        item.preview = block.preview;
        item.gist = block.gist;
      }
    },

    /** Fetch one turn block and upsert it — the single read behind both the WS
     *  `updated` refetch and thread expand. */
    async refetchTurn(turnId: number): Promise<void> {
      this.upsertTurn(await convoApi.thread(turnId));
    },

    /** Tear down an aborted/superseded live turn — its forms, thread shell and
     *  all signal state — leaving the feed clean. */
    dropLiveTurn(turnId: number): void {
      this.forms = this.forms.filter((f) => f.turnId !== turnId);
      this.threads = this.threads.filter((t) => t.turn_id !== turnId);
      this.workingTurnIds.delete(turnId);
      this.doneTurnIds.delete(turnId);
      this._clearLiveTurn(turnId);
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
     * Thread search — the same /api/threads getter with a `q` filter (capped at
     * 5 server-side). Returns matches without touching the feed; the search
     * overlay renders them ephemerally and discards them on close.
     */
    async searchThreads(query: string): Promise<ThreadListItem[]> {
      const trimmed = query.trim();
      if (!trimmed) return [];
      return (await convoApi.threads(5, undefined, trimmed)).threads;
    },

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

/**
 * Conversation spine store — the ordered `forms` array of discriminated-union
 * items (user / chalie / act / error). Components render the list; this store
 * owns all mutations.
 */
import { defineStore } from 'pinia';
import type { ConversationAttachment, ConversationMessage, ConversationSegment } from '../api/conversation';
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

export interface ErrorForm extends TurnTagged {
  kind: 'error';
  id: number;
  message: string;
}

export type ConversationForm = UserForm | ChalieForm | ActForm | ErrorForm;

/** A turn: a user form plus the assistant rows / act groups it produced. */
export interface Turn {
  id: number;
  forms: ConversationForm[];
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
    /** Id of the most recent act form, or null. */
    activeActFormId(state): number | null {
      for (let i = state.forms.length - 1; i >= 0; i--) {
        const f = state.forms[i];
        if (f.kind === 'act') return f.id;
      }
      return null;
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

    appendError(message: string): number {
      const id = nextId();
      this.forms.push({ kind: 'error', id, message });
      return id;
    },

    /**
     * Collapse a step's tool group to summary-only — called when the step is
     * superseded (next interim prose bubble or the final reply has landed).
     */
    collapseAct(actId: number): void {
      const form = this._findAct(actId);
      if (form) form.collapsed = true;
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

    replaceActWithError(actId: number, message: string): void {
      const idx = this.forms.findIndex((f) => f.id === actId);
      if (idx === -1) return;
      const id = nextId();
      this.forms.splice(idx, 1, { kind: 'error', id, message });
    },

    /** Append history turns to the END of the spine (initial load). */
    appendTurns(messages: ConversationMessage[]): void {
      for (const msg of messages) {
        this._appendMessage(msg, true);
      }
    },

    /**
     * Prepend history turns to the START of the spine (scroll-up pagination).
     * Iterated in reverse so indices stay stable.
     */
    prependTurns(messages: ConversationMessage[]): void {
      for (let i = messages.length - 1; i >= 0; i--) {
        this._prependMessage(messages[i], false);
      }
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

    _appendMessage(msg: ConversationMessage, inWorkingMemory: boolean): void {
      if (msg.role === 'user') {
        const attachments = this._attachmentsFor(msg);
        this.appendUser(msg.content, attachments, { inWorkingMemory, turnId: msg.turn_id });
        return;
      }
      for (const form of this._assistantForms(msg, inWorkingMemory)) {
        this.forms.push(form);
      }
    },

    _prependMessage(msg: ConversationMessage, inWorkingMemory: boolean): void {
      if (msg.role === 'user') {
        const attachments = this._attachmentsFor(msg);
        const id = nextId();
        this.forms.unshift({
          kind: 'user',
          id,
          text: msg.content,
          attachments,
          inWorkingMemory,
          turnId: msg.turn_id,
        });
        return;
      }
      // Unshift in reverse so the prose bubble ends up before its tool group.
      const forms = this._assistantForms(msg, inWorkingMemory);
      for (let i = forms.length - 1; i >= 0; i--) {
        this.forms.unshift(forms[i]);
      }
    },
  },
});
